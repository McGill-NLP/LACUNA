#!/usr/bin/env python3
"""Compute precision metrics (ROC AUC, precision@k) for unlearning experiments.

Two-pass streaming design per experiment:
  1. Stats pass: per-(layer,component) stds for normalization
  2. Scores pass: compute all scores, stream into adaptive histograms (no accumulation)

Memory: ~101GB (shared weights + tiny histograms). No score accumulation.
Uses $SLURM_TMPDIR for fast local I/O. Resumable via per-experiment cache.

Scoring functions
-----------------
  raw         — |W_unlearned - W_trained|  (absolute weight change)
  qtile       — quantile-rank of |delta| within each parameter tensor
  compnorm    — |delta| / mean(|delta_unmask|)  per parameter tensor
  contrast    — |delta_mask| - |delta_unmask|  per parameter (subtraction)
  contrastnorm— (|delta_mask| - |delta_unmask|) / (|delta_mask| + |delta_unmask|)  symmetric contrast index [-1,1]
  contrastln  — contrast score normalized by std within same (layer, component) group
  signrev     — -(injection * unlearn)  sign-reversal of injection direction
  layernorm   — |delta| / std(|delta|) within same (layer, component) group
  reversal    — (|inj| - |W_unl - W_pre|) / (|inj| + ε)  fractional reversal toward pretrained
  dirreversal — -(unlearn · sign(inj)) / (|inj| + ε)  directional reversal, normalized
  eratio      — |delta| / (|delta_unmask| + ε)  per-parameter ratio (finer than compnorm)
  crossfield  — min of quantile-ranked |delta| across all 4 PII fields (cross-field consistency)
  composite   — cross-validated logistic regression on all features above
"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

# ── Config ──────────────────────────────────────────────────────
FIELDS = ['Email_Address', 'Phone_Number', 'Birth_City', 'Drivers_License']
METHODS = ['SimNPO', 'MemFlex', 'AlphaEdit', 'OracleGrad']
FORGET_ONLY = {'OracleGrad'}
NO_UNMASK = {'OracleGrad'}
SKIP = {'norm', 'embedding', 'lm_head'}
DIR_NAMES = {'OracleGrad': 'GradDiff_OracleGrad'}
SCORE_METRICS = ['raw', 'qtile', 'compnorm', 'layernorm', 'signrev', 'reversal',
                 'dirreversal', 'contrast', 'contrastnorm', 'eratio', 'contrastln']
ALL_METRICS = SCORE_METRICS + ['composite', 'crossfield']
COMPOSITE_SAMPLE = 2_000_000
N_BINS = 1_000_000

# ── Globals ─────────────────────────────────────────────────────
BASE = UNMASK_BASE = CACHE_DIR = LOCAL_TMPDIR = None
orig_w = {}
pre_w = {}
unmask_orig_w = {}
gt_masks = {}        # 'forget'|'unified' -> {name -> bool array}
unmask_mean = {}     # (field, method) -> {name -> float}
param_names = []
_staged = {}


# ── Helpers ─────────────────────────────────────────────────────
def _upath(f, m):
    return BASE / 'unlearned_models' / f / DIR_NAMES.get(m, m) / 'model.safetensors'

def _umpath(f, m):
    return UNMASK_BASE / 'unlearned_models' / f / DIR_NAMES.get(m, m) / 'model.safetensors'

def stage(src, label=''):
    src = Path(src)
    if LOCAL_TMPDIR is None or not src.exists():
        return src
    key = str(src)
    if key in _staged:
        return _staged[key]
    dest = LOCAL_TMPDIR / 'staged' / label / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.exists() and dest.stat().st_size == src.stat().st_size):
        t0 = time.time()
        shutil.copy2(src, dest)
        print(f'  Staged {label}/{src.name} ({src.stat().st_size/1e9:.1f}GB, {time.time()-t0:.0f}s)', flush=True)
    _staged[key] = dest
    return dest

def local(src):
    return _staged.get(str(src), src)

def classify(name):
    if 'layers.' in name:
        p = name.split('.')
        i = int(p[p.index('layers') + 1])
        if 'self_attn' in name: return i, 'attention'
        if 'mlp' in name: return i, 'mlp'
        if 'norm' in name: return i, 'norm'
        return i, 'other'
    if 'embed' in name: return -1, 'embedding'
    if 'lm_head' in name: return -1, 'lm_head'
    return -1, 'other'

def quantile_ranks(a):
    order = np.argsort(a, kind='stable')
    r = np.empty_like(order, dtype=np.float32)
    r[order] = np.arange(len(a), dtype=np.float32)
    if len(a) > 1:
        r /= len(a) - 1
    return r

def rss_gb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return float('nan')


# ── Score computation ───────────────────────────────────────────
def compute_all_scores(name, li, comp, h, um_h, um_keys, gt, lc_std, cl_std, comp_key):
    """Compute all 11 score arrays for one parameter. Returns dict of metric->array."""
    o = orig_w[name]
    u = h.get_tensor(name).float().numpy().ravel()
    diff = u - o
    delta = np.abs(diff)
    del u

    p = pre_w.get(name)
    ls = lc_std.get((li, comp), 1e-12)
    cs = cl_std.get((li, comp), 1e-12)
    um = unmask_mean.get(comp_key, {}).get(name, 0.0)

    s = {}
    s['raw'] = delta
    s['qtile'] = quantile_ranks(delta)
    s['layernorm'] = delta / ls
    s['compnorm'] = delta / max(um, 1e-12) if um > 1e-12 else delta.copy()

    if p is not None:
        inj = o - p
        inj_abs = np.abs(inj)
        s['signrev'] = -(inj * diff)
        s['reversal'] = np.divide(inj_abs - np.abs(diff + inj), inj_abs + 1e-10, dtype=np.float32)
        s['dirreversal'] = np.divide(-(diff * np.sign(inj)), inj_abs + 1e-10, dtype=np.float32)
    else:
        z = np.zeros(delta.size, np.float32)
        s['signrev'] = z
        s['reversal'] = z.copy()
        s['dirreversal'] = z.copy()

    has_um = um_h is not None
    dum = None
    if has_um and name in um_keys and name in unmask_orig_w:
        u2 = um_h.get_tensor(name).float().numpy().ravel()
        dum = np.abs(u2 - unmask_orig_w[name])
        del u2

    if dum is not None:
        s['contrast'] = delta - dum
        den = delta + dum
        s['contrastnorm'] = np.divide(delta - dum, den, out=np.zeros_like(delta), where=den > 1e-12)
        s['eratio'] = np.divide(delta, dum + 1e-10, dtype=np.float32)
        s['contrastln'] = (delta - dum) / cs
    else:
        s['contrast'] = delta.copy()
        s['contrastnorm'] = np.ones_like(delta)
        s['eratio'] = delta.copy()
        s['contrastln'] = delta / cs

    return s, diff, delta, dum


# ── Histogram-based AUC ─────────────────────────────────────────
def auc_from_histograms(pos_hist, neg_hist):
    """Compute AUC + downsampled ROC + precision@k from pos/neg bin histograms.

    Histograms: bin 0 = lowest scores, bin N_BINS-1 = highest scores.
    ROC sweeps threshold from high to low (standard convention).
    """
    n_pos = pos_hist.sum()
    n_neg = neg_hist.sum()
    if n_pos == 0 or n_neg == 0:
        return float('nan'), np.array([0., 1.], dtype=np.float16), \
               np.array([0., 1.], dtype=np.float16), {}

    # Cumulative counts from the RIGHT (highest bin down).
    # cum_pos_from_right[i] = positives in bins [i, N-1] = positives with score >= bin i
    cum_pos = np.cumsum(pos_hist[::-1])[::-1]
    cum_neg = np.cumsum(neg_hist[::-1])[::-1]

    # ROC points: sweep threshold from highest bin (index N-1) to lowest (index 0).
    # At threshold = bin i: TPR = cum_pos[i] / n_pos, FPR = cum_neg[i] / n_neg
    # Reverse so points go from (0,0) → ... → (1,1) for standard ROC convention.
    tpr_pts = (cum_pos / n_pos)[::-1]   # now: low threshold first = high TPR first → reversed to ascending
    fpr_pts = (cum_neg / n_neg)[::-1]
    tpr = np.concatenate(([0.0], tpr_pts))
    fpr = np.concatenate(([0.0], fpr_pts))
    auc_val = float(np.trapezoid(tpr, fpr))

    # Downsample to 1000 points
    ix = np.linspace(0, len(fpr) - 1, min(1000, len(fpr))).astype(int)
    fpr_d = fpr[ix].astype(np.float16)
    tpr_d = tpr[ix].astype(np.float16)

    # Precision@k: find top-k elements by scanning from highest bin down.
    # cum_pos[i] and cum_neg[i] give counts with score >= bin i.
    # cum_total[i] = cum_pos[i] + cum_neg[i] = total elements with score >= bin i.
    cum_total = cum_pos + cum_neg  # already cumulated from right
    pk = {}
    n_total = int(n_pos + n_neg)
    for kp in [0.1, 1, 5, 10]:
        k = max(1, int(n_total * kp / 100))
        # Find largest bin index i such that cum_total[i] >= k
        # (i.e., the highest threshold where we still have >= k elements above it)
        # cum_total is decreasing: cum_total[0] = N (all), cum_total[N-1] = top-bin count
        candidates = np.where(cum_total >= k)[0]
        if len(candidates) == 0:
            pk[f'p@{kp}%'] = float(n_pos / n_total)
            continue
        bin_idx = candidates[-1]  # largest i where cum_total[i] >= k

        total_at = int(cum_total[bin_idx])
        pos_at = int(cum_pos[bin_idx])

        if total_at == k:
            # Exact: top k elements are those with score >= bin_idx
            pk[f'p@{kp}%'] = float(pos_at / k)
        else:
            # total_at > k: the boundary bin has more elements than needed.
            # Elements strictly above bin_idx:
            if bin_idx + 1 < len(cum_total):
                total_above = int(cum_total[bin_idx + 1])
                pos_above = int(cum_pos[bin_idx + 1])
            else:
                total_above = 0
                pos_above = 0
            # Elements in boundary bin = total_at - total_above
            total_in_bin = total_at - total_above
            pos_in_bin = pos_at - pos_above
            # We need (k - total_above) elements from this bin
            remaining = k - total_above
            if total_in_bin > 0 and remaining > 0:
                frac = remaining / total_in_bin
                pos_from_bin = pos_in_bin * frac
            else:
                pos_from_bin = 0
            pk[f'p@{kp}%'] = float((pos_above + pos_from_bin) / k)

    return auc_val, fpr_d, tpr_d, pk


class AdaptiveHistogram:
    """Streaming histogram that expands its range as new data arrives.

    When new scores fall outside the current [lo, hi] range, the existing
    bin counts are re-distributed into a wider range. Re-binning is O(n_bins)
    and takes microseconds. With n_bins=1M, precision loss from re-binning
    is negligible (AUC error remains ≤ 0.5/n_bins).
    """
    __slots__ = ('n_bins', 'pos', 'neg', 'lo', 'hi')

    def __init__(self, n_bins=N_BINS):
        self.n_bins = n_bins
        self.pos = np.zeros(n_bins, dtype=np.int64)
        self.neg = np.zeros(n_bins, dtype=np.int64)
        self.lo = None
        self.hi = None

    def add(self, scores, gt):
        """Add one parameter's scores and GT labels to the histogram."""
        mn, mx = float(scores.min()), float(scores.max())
        if mn == mx:
            # All scores identical — put everything in one bin
            pos_count = int((gt > 0.5).sum())
            neg_count = len(gt) - pos_count
            if self.lo is None:
                self.lo = mn - 0.5
                self.hi = mn + 0.5
            mid_bin = self.n_bins // 2
            self.pos[mid_bin] += pos_count
            self.neg[mid_bin] += neg_count
            return

        if self.lo is None:
            # First data — initialize range
            self.lo, self.hi = mn, mx
        elif mn < self.lo or mx > self.hi:
            # Range needs expansion — re-bin existing counts
            new_lo = min(mn, self.lo)
            new_hi = max(mx, self.hi)
            self._rebin(new_lo, new_hi)

        # Bin scores into current range
        bins = np.clip(
            ((scores - self.lo) / (self.hi - self.lo) * (self.n_bins - 1)).astype(np.int32),
            0, self.n_bins - 1)
        pos_mask = gt > 0.5
        self.pos += np.bincount(bins[pos_mask], minlength=self.n_bins)
        self.neg += np.bincount(bins[~pos_mask], minlength=self.n_bins)

    def _rebin(self, new_lo, new_hi):
        """Re-distribute existing bin counts into a wider range."""
        if self.lo is None or (self.pos.sum() == 0 and self.neg.sum() == 0):
            self.lo, self.hi = new_lo, new_hi
            return
        # Map old bin centers to new bin indices
        old_centers = np.linspace(self.lo, self.hi, self.n_bins)
        new_bins = np.clip(
            ((old_centers - new_lo) / (new_hi - new_lo) * (self.n_bins - 1)).astype(np.int32),
            0, self.n_bins - 1)
        new_pos = np.zeros(self.n_bins, dtype=np.int64)
        new_neg = np.zeros(self.n_bins, dtype=np.int64)
        np.add.at(new_pos, new_bins, self.pos)
        np.add.at(new_neg, new_bins, self.neg)
        self.pos, self.neg = new_pos, new_neg
        self.lo, self.hi = new_lo, new_hi


# ── Save ────────────────────────────────────────────────────────
def save_experiment(mask_type, field, method, r):
    d = CACHE_DIR / mask_type / field / method
    d.mkdir(parents=True, exist_ok=True)
    la = {f'{k[0]}_{k[1]}': v for k, v in r['layer_aucs'].items()}
    metrics = {'layer_aucs': la}
    for m in ALL_METRICS:
        metrics[f'auc_{m}'] = r.get(f'auc_{m}', float('nan'))
        metrics[f'pk_{m}'] = r.get(f'pk_{m}', {})
    with open(d / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    roc = {}
    for m in ALL_METRICS:
        for p in ('fpr', 'tpr'):
            k = f'{p}_{m}'
            if k in r:
                roc[k] = r[k]
    np.savez_compressed(d / 'roc_curves.npz', **roc)
    r['param_stats'].to_parquet(d / 'param_stats.parquet', index=False)


# ── Open helpers ────────────────────────────────────────────────
def _open_unlearned(path, um_path, has_um):
    """Open unlearned + optionally unmask safetensors, return (h, um_h, um_keys)."""
    h = safe_open(str(path), framework='pt', device='cpu')
    um_h = None
    um_keys = set()
    if has_um and um_path is not None:
        um_h = safe_open(str(um_path), framework='pt', device='cpu')
        um_keys = set(um_h.keys())
    return h, um_h, um_keys


# ── Experiment ──────────────────────────────────────────────────
def run_experiment(field, method, mask_type):
    """Two-pass streaming computation of all metrics."""
    cache_dir = CACHE_DIR / mask_type / field / method
    if (cache_dir / 'metrics.json').exists():
        print(f'  {field}/{method}[{mask_type}]: cached', flush=True)
        return

    t0 = time.time()
    wid = f'{field}/{method}[{mask_type}]'
    print(f'  [{wid}] started', flush=True)

    path = local(_upath(field, method))
    gt = gt_masks[mask_type]
    has_um = method not in NO_UNMASK
    um_path = None
    if has_um:
        raw_um = _umpath(field, method)
        um_path = local(raw_um)
        if not Path(um_path).exists():
            has_um = False
    comp_key = (field, method)

    with safe_open(str(path), framework='pt', device='cpu') as h:
        unl_keys = set(h.keys())

    # Active params
    active = []
    for name in param_names:
        if name not in unl_keys:
            continue
        li, comp = classify(name)
        if comp in SKIP:
            continue
        g = gt[name]
        if g.any() and not g.all():
            active.append((name, li, comp))

    # ── Pass 1: running stats + param_stats ──
    lc_run, cl_run = {}, {}
    pstats = []
    # Count non-skipped params for progress
    all_params = [(n, *classify(n)) for n in param_names
                  if n in unl_keys and classify(n)[1] not in SKIP]
    n_all = len(all_params)

    h, um_h, um_keys = _open_unlearned(path, um_path, has_um)
    for pi, (name, li, comp) in enumerate(all_params):
        u = h.get_tensor(name).float().numpy().ravel()
        delta = np.abs(u - orig_w[name])
        del u

        pstats.append({
            'param': name, 'layer': li, 'component': comp,
            'mean_delta': float(delta.mean()), 'std_delta': float(delta.std()),
            'max_delta': float(delta.max()),
            'frac_nonzero': float((delta > 1e-10).mean()),
            'numel': int(delta.size),
        })

        key = (li, comp)
        if key not in lc_run:
            lc_run[key] = [0.0, 0.0, 0]
        s = lc_run[key]
        s[0] += float(delta.sum())
        s[1] += float(np.sum(delta.astype(np.float64) ** 2))
        s[2] += int(delta.size)

        if has_um and name in um_keys and name in unmask_orig_w:
            u2 = um_h.get_tensor(name).float().numpy().ravel()
            c = delta - np.abs(u2 - unmask_orig_w[name])
            del u2
            if key not in cl_run:
                cl_run[key] = [0.0, 0.0, 0]
            r = cl_run[key]
            r[0] += float(c.sum())
            r[1] += float(np.sum(c.astype(np.float64) ** 2))
            r[2] += int(c.size)
            del c
        del delta

        if (pi + 1) % 50 == 0 or pi + 1 == n_all:
            print(f'    [{wid}] pass 1: {pi+1}/{n_all} params ({time.time()-t0:.0f}s)', flush=True)

    del h
    if um_h is not None:
        del um_h
    gc.collect()

    lc_std, cl_std = {}, {}
    for key, (s, sq, n) in lc_run.items():
        m = s / n
        lc_std[key] = max(np.sqrt(max(sq / n - m * m, 0.0)), 1e-12)
    for key, (s, sq, n) in cl_run.items():
        m = s / n if n else 0
        cl_std[key] = max(np.sqrt(max(sq / n - m * m, 0.0)), 1e-12) if n else 1e-12
    del lc_run, cl_run
    print(f'  [{wid}] pass 1 stats ({time.time()-t0:.0f}s) RSS={rss_gb():.0f}GB', flush=True)

    # ── Pass 2: stream scores into adaptive histograms + composite features ──
    # Single pass replaces the old pass 2 (ranges) + pass 3 (histograms).
    hists = {m: AdaptiveHistogram() for m in SCORE_METRICS}
    layer_hists = {}  # (layer, comp) -> AdaptiveHistogram for qtile

    comp_feat, comp_lab = [], []
    total_active = sum(gt[n].size for n, _, _ in active)
    sfrac = min(1.0, COMPOSITE_SAMPLE / max(total_active, 1))
    crng = np.random.RandomState(42)
    n_active = len(active)

    h, um_h, um_keys = _open_unlearned(path, um_path, has_um)
    for pi, (name, li, comp) in enumerate(active):
        scores, diff, delta, dum = compute_all_scores(
            name, li, comp, h, um_h, um_keys, gt, lc_std, cl_std, comp_key)
        gf = gt[name].astype(np.float32)

        # Stream into adaptive histograms
        for m in SCORE_METRICS:
            hists[m].add(scores[m], gf)

        # Layer-level qtile histogram
        key = (li, comp)
        if key not in layer_hists:
            layer_hists[key] = AdaptiveHistogram()
        layer_hists[key].add(scores['qtile'], gf)

        # Composite features (subsampled — tiny memory)
        p = pre_w.get(name)
        n = delta.size
        ns = max(1, int(n * sfrac))
        ix = crng.choice(n, ns, replace=False)

        feats = [delta[ix], delta[ix] / lc_std.get((li, comp), 1e-12)]
        um_val = unmask_mean.get(comp_key, {}).get(name, 0.0)
        if p is not None:
            inj = orig_w[name] - p
            ia = np.abs(inj[ix])
            feats.append(-(inj[ix] * diff[ix]))
            feats.append(np.divide(ia - np.abs(diff[ix] + inj[ix]), ia + 1e-10, dtype=np.float32))
            feats.append(np.divide(-(diff[ix] * np.sign(inj[ix])), ia + 1e-10, dtype=np.float32))
        else:
            feats.extend([np.zeros(ns, np.float32)] * 3)
        if dum is not None:
            cr = delta[ix] - dum[ix]
            den = delta[ix] + dum[ix]
            feats.append(cr)
            feats.append(np.divide(cr, den, out=np.zeros(ns, np.float32), where=den > 1e-12))
            feats.append(cr / cl_std.get((li, comp), 1e-12))
            feats.append(np.divide(delta[ix], dum[ix] + 1e-10, dtype=np.float32))
        else:
            feats.extend([np.zeros(ns, np.float32)] * 4)
        feats.append(delta[ix] / max(um_val, 1e-12) if um_val > 1e-12 else delta[ix].copy())
        comp_feat.append(np.column_stack(feats))
        comp_lab.append(gt[name][ix].astype(np.float32))

        del scores, diff, delta, dum

        if (pi + 1) % 25 == 0 or pi + 1 == n_active:
            print(f'    [{wid}] pass 2: {pi+1}/{n_active} params ({time.time()-t0:.0f}s)', flush=True)

    del h
    if um_h is not None:
        del um_h
    gc.collect()
    print(f'  [{wid}] pass 2 done ({time.time()-t0:.0f}s) RSS={rss_gb():.0f}GB', flush=True)

    # ── Compute AUCs from histograms ──
    result = {'param_stats': pd.DataFrame(pstats)}

    for m in SCORE_METRICS:
        ah = hists[m]
        auc, fpr, tpr, pk = auc_from_histograms(ah.pos, ah.neg)
        result[f'auc_{m}'] = auc
        result[f'fpr_{m}'] = fpr
        result[f'tpr_{m}'] = tpr
        result[f'pk_{m}'] = pk
        print(f'    [{wid}] {m} AUC={auc:.4f}', flush=True)

    # Layer AUCs
    layer_aucs = {}
    for key, ah in layer_hists.items():
        auc, _, _, _ = auc_from_histograms(ah.pos, ah.neg)
        layer_aucs[key] = auc
    result['layer_aucs'] = layer_aucs
    del hists, layer_hists

    # ── Composite ──
    if comp_feat:
        X = np.concatenate(comp_feat)
        y = np.concatenate(comp_lab)
        del comp_feat, comp_lab
        X = StandardScaler().fit_transform(X)
        yp = cross_val_predict(
            LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0),
            X, y, cv=5, method='predict_proba')[:, 1]
        del X
        try:
            fpr_c, tpr_c, _ = roc_curve(y, yp)
            auc_c = sk_auc(fpr_c, tpr_c)
            ix = np.linspace(0, len(fpr_c) - 1, min(1000, len(fpr_c))).astype(int)
            result['fpr_composite'] = fpr_c[ix].astype(np.float16)
            result['tpr_composite'] = tpr_c[ix].astype(np.float16)
        except ValueError:
            auc_c = float('nan')
            result['fpr_composite'] = result['tpr_composite'] = np.array([0., 1.], dtype=np.float16)
        si = np.argsort(-yp)
        pk = {}
        for kp in [0.1, 1, 5, 10]:
            k = max(1, int(len(yp) * kp / 100))
            pk[f'p@{kp}%'] = float(y[si[:k]].mean())
        result['auc_composite'] = auc_c
        result['pk_composite'] = pk
        del y, yp
        print(f'    [{wid}] composite AUC={auc_c:.4f}', flush=True)
    else:
        result['auc_composite'] = float('nan')
        result['fpr_composite'] = result['tpr_composite'] = np.array([0., 1.], dtype=np.float16)
        result['pk_composite'] = {}

    gc.collect()
    save_experiment(mask_type, field, method, result)
    print(f'  [{wid}] DONE ({time.time()-t0:.0f}s) RSS={rss_gb():.0f}GB', flush=True)


# ── Crossfield ──────────────────────────────────────────────────
def run_crossfield(method, mask_type):
    """Min-of-quantile-ranks across all fields."""
    cd0 = CACHE_DIR / mask_type / FIELDS[0] / method
    if cd0.exists() and (cd0 / 'metrics.json').exists():
        with open(cd0 / 'metrics.json') as f:
            if 'auc_crossfield' in json.load(f):
                print(f'  {method}[{mask_type}]: crossfield cached', flush=True)
                return

    gt = gt_masks[mask_type]
    fpaths = [(fld, local(_upath(fld, method)))
              for fld in FIELDS if _upath(fld, method).exists()]
    if len(fpaths) < 2:
        return

    # Pass 1: find min/max of running-min qtile ranks
    running_min = {}
    for fi, (fld, fp) in enumerate(fpaths):
        with safe_open(str(fp), framework='pt', device='cpu') as h:
            uk = set(h.keys())
            for name in param_names:
                if name not in uk:
                    continue
                _, comp = classify(name)
                if comp in SKIP:
                    continue
                g = gt[name]
                if not (g.any() and not g.all()):
                    continue
                u = h.get_tensor(name).float().numpy().ravel()
                q = quantile_ranks(np.abs(u - orig_w[name]))
                del u
                if fi == 0:
                    running_min[name] = q
                elif name in running_min:
                    np.minimum(running_min[name], q, out=running_min[name])
        gc.collect()

    # Histogram
    ah = AdaptiveHistogram()
    for name, v in running_min.items():
        gf = gt[name].astype(np.float32)
        ah.add(v, gf)
    del running_min
    gc.collect()

    auc, fpr, tpr, pk = auc_from_histograms(ah.pos, ah.neg)
    print(f'  {method}[{mask_type}]: crossfield AUC={auc:.4f}', flush=True)

    for fld in FIELDS:
        cd = CACHE_DIR / mask_type / fld / method
        mj = cd / 'metrics.json'
        if not mj.exists():
            continue
        with open(mj) as f:
            data = json.load(f)
        data['auc_crossfield'] = auc
        data['pk_crossfield'] = pk
        with open(mj, 'w') as f:
            json.dump(data, f, indent=2)
        rp = cd / 'roc_curves.npz'
        roc = dict(np.load(rp)) if rp.exists() else {}
        roc['fpr_crossfield'] = fpr
        roc['tpr_crossfield'] = tpr
        np.savez_compressed(rp, **roc)


# ── Main ────────────────────────────────────────────────────────
def main(skip_crossfield=False):
    global orig_w, pre_w, unmask_orig_w, gt_masks, param_names, unmask_mean

    t_start = time.time()
    print(f'RSS: {rss_gb():.0f}GB', flush=True)

    # Stage shared files
    print('Staging shared files...', flush=True)
    lo = stage(BASE / 'intruction_tuned' / 'model.safetensors', 'shared')
    lp = stage(BASE / 'model.safetensors', 'shared/pretrained')
    lm = stage(BASE / 'mask.pt', 'shared')
    lu = stage(UNMASK_BASE / 'intruction_tuned' / 'model.safetensors', 'shared/unmask')
    for gt_name in ['gt_mask_forget_cache.pt', 'gt_mask_unified_cache.pt']:
        p = BASE / gt_name
        if p.exists():
            stage(p, 'shared')

    # Stage unlearned models
    print('Staging unlearned models...', flush=True)
    st = time.time()
    for f in FIELDS:
        for m in METHODS:
            p = _upath(f, m)
            if p.exists():
                stage(p, f'unlearned/{f}/{m}')
            if m not in NO_UNMASK:
                p2 = _umpath(f, m)
                if p2.exists():
                    stage(p2, f'unmask/{f}/{m}')
    print(f'Models staged ({time.time()-st:.0f}s)', flush=True)

    # Load param names + GT masks
    with safe_open(str(lo), framework='pt', device='cpu') as h:
        model_keys = set(h.keys())
    mask = torch.load(str(lm), map_location='cpu')
    param_names = sorted(set(mask.keys()) & model_keys)
    print(f'Params: {len(param_names)}', flush=True)

    forget_bits = sum(1 << g for g in [0, 2, 4])
    unified_bits = sum(1 << g for g in [0, 1, 2, 3, 4, 5])

    def _load_gt(bits, cache_name, label):
        cp = BASE / cache_name
        lcp = local(cp)
        if Path(lcp).exists():
            print(f'Loading cached GT mask ({label})...', flush=True)
            raw = torch.load(str(lcp), map_location='cpu', weights_only=True)
        else:
            print(f'Computing GT mask ({label})...', flush=True)
            raw = {n: ((mask[n] & bits) > 0).bool() for n in param_names}
            torch.save(raw, str(cp))
        return {n: raw[n].numpy().ravel() for n in param_names}

    gt_masks['forget'] = _load_gt(forget_bits, 'gt_mask_forget_cache.pt', 'forget')
    gt_masks['unified'] = _load_gt(unified_bits, 'gt_mask_unified_cache.pt', 'unified')
    del mask
    gc.collect()

    # Load shared weights
    print('Loading shared weights...', flush=True)
    with safe_open(str(lo), framework='pt', device='cpu') as h:
        for n in param_names:
            orig_w[n] = h.get_tensor(n).float().numpy().ravel()
    with safe_open(str(lp), framework='pt', device='cpu') as h:
        pk = set(h.keys())
        for n in param_names:
            if n in pk:
                pre_w[n] = h.get_tensor(n).float().numpy().ravel()
    with safe_open(str(lu), framework='pt', device='cpu') as h:
        uk = set(h.keys())
        for n in param_names:
            if n in uk:
                unmask_orig_w[n] = h.get_tensor(n).float().numpy().ravel()
    print(f'Weights loaded, RSS: {rss_gb():.0f}GB', flush=True)

    # UnMask mean deltas
    print('Computing unmask mean deltas...', flush=True)
    for f in FIELDS:
        for m in METHODS:
            if m in NO_UNMASK:
                continue
            up = _umpath(f, m)
            if not up.exists():
                continue
            means = {}
            with safe_open(str(local(up)), framework='pt', device='cpu') as h:
                ukeys = set(h.keys())
                for n in param_names:
                    if n not in ukeys or n not in unmask_orig_w:
                        continue
                    _, c = classify(n)
                    if c in SKIP:
                        continue
                    u = h.get_tensor(n).float().numpy().ravel()
                    means[n] = float(np.abs(u - unmask_orig_w[n]).mean())
                    del u
            unmask_mean[(f, m)] = means
            print(f'  {f}/{m}: {len(means)} params', flush=True)
    gc.collect()

    # Build tasks
    tasks = []
    for f in FIELDS:
        for m in METHODS:
            if not _upath(f, m).exists():
                print(f'  SKIP {f}/{m}: model not found', flush=True)
                continue
            for mt in (['forget'] if m in FORGET_ONLY else ['unified', 'forget']):
                tasks.append((f, m, mt))

    print(f'\n{len(tasks)} experiments to compute', flush=True)
    for i, (f, m, mt) in enumerate(tasks, 1):
        print(f'\n[{i}/{len(tasks)}]', flush=True)
        run_experiment(f, m, mt)

    # Crossfield
    if skip_crossfield:
        print('\nCrossfield: skipped', flush=True)
    else:
        print('\nCrossfield post-processing...', flush=True)
        for m in METHODS:
            if not _upath(FIELDS[0], m).exists():
                continue
            for mt in (['forget'] if m in FORGET_ONLY else ['unified', 'forget']):
                run_crossfield(m, mt)

    print(f'\nTotal wall time: {time.time()-t_start:.0f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=True)
    parser.add_argument('--unmask-base', type=str, required=True)
    parser.add_argument('--fields', type=str, default=None,
                        help='Comma-separated fields to process (default: all)')
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated methods to process (default: all)')
    parser.add_argument('--skip-crossfield', action='store_true',
                        help='Skip crossfield post-processing')
    args = parser.parse_args()

    if args.fields:
        FIELDS = [f.strip() for f in args.fields.split(',')]
    if args.methods:
        METHODS = [m.strip() for m in args.methods.split(',')]

    tmpdir = os.environ.get('SLURM_TMPDIR', '')
    if not tmpdir:
        raise RuntimeError('SLURM_TMPDIR not set. Run inside a SLURM job.')
    LOCAL_TMPDIR = Path(tmpdir) / 'precision_metrics'
    LOCAL_TMPDIR.mkdir(parents=True, exist_ok=True)

    BASE = Path(args.base)
    UNMASK_BASE = Path(args.unmask_base)
    CACHE_DIR = BASE / 'cached_notebook_files' / 'precision_metrics'

    main(skip_crossfield=args.skip_crossfield)
