#!/usr/bin/env python3
"""Precision metrics restricted to ACTIVE parameter tensors only.

Same scoring functions as precision_metrics.py, but each parameter tensor is
included only if it has at least one weight that changed (|delta| > threshold)
between the instruction-tuned and unlearned models.  This answers: "among the
parameters the method actually touched, how precisely did it target the GT mask?"

Saves to a separate cache:
  {BASE}/cached_notebook_files/precision_metrics_active/{mask_type}/{field}/{method}/
    metrics.json   — AUC per scoring function
    roc_curves.npz — downsampled ROC curve points (float16)
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse
import gc
import json
import time
import traceback
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.metrics import roc_curve, auc as sklearn_auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

# ──────────────────────────── Config ────────────────────────────

BASE = Path.home() / 'OLMoBenchOutputs/saves/Train_95frozen_FullSubset'
UNMASK_BASE = Path.home() / 'OLMoBenchOutputs/saves/Train_UnMask_FullSubset'
ORIGINAL_PATH = BASE / 'intruction_tuned' / 'model.safetensors'
PRETRAINED_PATH = BASE / 'model.safetensors'
UNMASK_ORIGINAL_PATH = UNMASK_BASE / 'intruction_tuned' / 'model.safetensors'
MASK_PATH = BASE / 'mask.pt'
CACHE_DIR = BASE / 'cached_notebook_files' / 'precision_metrics_active'

FIELDS = ['Email_Address', 'Phone_Number', 'Birth_City', 'Drivers_License']
METHODS = ['SimNPO', 'MemFlex', 'AlphaEdit', 'OracleGrad']
FORGET_ONLY_METHODS = {'OracleGrad'}
UNMASK_EXCLUDED_METHODS = {'OracleGrad'}
UNMASK_METRICS = {'compnorm', 'contrast', 'contrastnorm', 'contrastln', 'eratio'}

SKIP_COMPONENTS = {'norm', 'embedding', 'lm_head'}
FORGET_GROUPS = {0, 2, 4}
RETAIN_GROUPS = {1, 3, 5}
NUM_WORKERS = 4
COMPOSITE_SAMPLE = 2_000_000
ACTIVE_THRESHOLD = 1e-10  # a param tensor is "active" if any |delta| exceeds this

# ──────────────────────── Global state (shared via fork COW) ────────────────────────

orig_weights = {}
pretrained_weights = {}
unmask_orig_weights = {}
gt_mask_unified_np = {}
gt_mask_forget_np = {}
unmask_comp_mean_delta = {}
param_names = []

ALL_METRICS = {'raw', 'qtile', 'compnorm', 'contrast', 'contrastnorm', 'contrastln',
               'signrev', 'layernorm', 'reversal', 'dirreversal', 'eratio',
               'crossfield', 'composite'}
WORKER_METRICS = ALL_METRICS - {'crossfield'}

METHOD_DIR_NAMES = {'OracleGrad': 'GradDiff_OracleGrad'}

def unlearned_path(field, method):
    dir_name = METHOD_DIR_NAMES.get(method, method)
    return BASE / 'unlearned_models' / field / dir_name / 'model.safetensors'

def unmask_unlearned_path(field, method):
    dir_name = METHOD_DIR_NAMES.get(method, method)
    return UNMASK_BASE / 'unlearned_models' / field / dir_name / 'model.safetensors'


# ──────────────────────────── Helpers ────────────────────────────

def classify_param(name):
    if 'layers.' in name:
        parts = name.split('.')
        layer_idx = int(parts[parts.index('layers') + 1])
        if 'self_attn' in name: return layer_idx, 'attention'
        elif 'mlp' in name:     return layer_idx, 'mlp'
        elif 'norm' in name:    return layer_idx, 'norm'
        return layer_idx, 'other'
    elif 'embed' in name:   return -1, 'embedding'
    elif 'lm_head' in name: return -1, 'lm_head'
    return -1, 'other'


def to_quantile_ranks(delta_flat):
    order = np.argsort(delta_flat, kind='stable')
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(delta_flat), dtype=np.float32)
    ranks /= (len(delta_flat) - 1) if len(delta_flat) > 1 else 1.0
    return ranks


def compute_roc_auc(scores_list, gt_list):
    """Concatenate, compute ROC AUC and downsampled curve."""
    d = np.concatenate(scores_list)
    g = np.concatenate(gt_list)
    try:
        fpr, tpr, _ = roc_curve(g, d)
        auc_val = sklearn_auc(fpr, tpr)
        idx = np.linspace(0, len(fpr) - 1, min(1000, len(fpr))).astype(int)
        fpr_p, tpr_p = fpr[idx], tpr[idx]
        del fpr, tpr
    except ValueError:
        auc_val = float('nan')
        fpr_p, tpr_p = np.array([0., 1.]), np.array([0., 1.])
    del d, g
    gc.collect()
    return auc_val, fpr_p, tpr_p


def _nan_result():
    return float('nan'), np.array([0., 1.]), np.array([0., 1.])


# ──────────────────────── Cache detection ─────────────────────────

def is_fully_cached(cache_dir):
    metrics_path = cache_dir / 'metrics.json'
    roc_path = cache_dir / 'roc_curves.npz'
    if not (metrics_path.exists() and roc_path.exists()):
        return False
    with open(metrics_path) as f:
        metrics = json.load(f)
    roc = np.load(roc_path)
    roc_keys = set(roc.files)
    for m in ALL_METRICS:
        if f'auc_{m}' not in metrics or f'fpr_{m}' not in roc_keys:
            return False
    return True


def load_cached_result(cache_dir):
    with open(cache_dir / 'metrics.json') as f:
        metrics = json.load(f)
    roc = np.load(cache_dir / 'roc_curves.npz')
    result = {}
    for m in ALL_METRICS:
        result[f'auc_{m}'] = metrics.get(f'auc_{m}', float('nan'))
        result[f'fpr_{m}'] = roc[f'fpr_{m}'].astype(np.float32) if f'fpr_{m}' in roc else np.array([0., 1.])
        result[f'tpr_{m}'] = roc[f'tpr_{m}'].astype(np.float32) if f'tpr_{m}' in roc else np.array([0., 1.])
    result['n_active'] = metrics.get('n_active', 0)
    result['n_total'] = metrics.get('n_total', 0)
    return result


# ──────────────────────────── Worker ────────────────────────────

def _worker_fn(field, method, mask_type, param_names_list, skip_components, conn):
    """Worker: compute all metrics for one (field, method, mask_type), filtering
    to active-only parameter tensors."""
    try:
        path = unlearned_path(field, method)
        mask_src = gt_mask_forget_np if mask_type == 'forget' else gt_mask_unified_np
        comp_mean_key = (field, method)
        has_compnorm_data = comp_mean_key in unmask_comp_mean_delta

        has_contrast_data = (method not in UNMASK_EXCLUDED_METHODS)
        unmask_upath = unmask_unlearned_path(field, method) if has_contrast_data else None
        if unmask_upath and not unmask_upath.exists():
            has_contrast_data = False

        result = {}

        # ── Pre-scan: identify active params ──
        # A param is "active" if: (1) has both GT mask classes, AND
        # (2) at least one weight changed above threshold
        with safe_open(str(path), framework='pt', device='cpu') as h:
            unl_keys = set(h.keys())

        active_params = []  # (name, layer_idx, comp)
        n_eligible = 0  # params that pass mask check but may not be active
        with safe_open(str(path), framework='pt', device='cpu') as unl_h:
            for name in param_names_list:
                if name not in unl_keys:
                    continue
                layer_idx, comp = classify_param(name)
                if comp in skip_components:
                    continue
                gt_flat = mask_src[name]
                if not (gt_flat.any() and not gt_flat.all()):
                    continue
                n_eligible += 1
                u = unl_h.get_tensor(name).float().numpy().ravel()
                delta = np.abs(u - orig_weights[name])
                if (delta > ACTIVE_THRESHOLD).any():
                    active_params.append((name, layer_idx, comp))
                del u, delta

        result['n_active'] = len(active_params)
        result['n_total'] = n_eligible

        if not active_params:
            for m in WORKER_METRICS:
                auc, fpr, tpr = _nan_result()
                result[f'auc_{m}'], result[f'fpr_{m}'], result[f'tpr_{m}'] = auc, fpr, tpr
            conn.send(('ok', field, method, mask_type, result))
            return

        def _iter_active():
            for name, layer_idx, comp in active_params:
                yield name, layer_idx, comp

        # ── Pass A: raw + qtile + running stats ──
        gt_list = []
        raw_list = []
        qtile_list = []
        layercomp_running = {}
        contrast_lc_running = {}

        with safe_open(str(path), framework='pt', device='cpu') as unl_h:
            unmask_h_a = None
            unmask_keys_a = set()
            if has_contrast_data:
                unmask_h_a = safe_open(str(unmask_upath), framework='pt', device='cpu')
                unmask_keys_a = set(unmask_h_a.keys())

            for name, layer_idx, comp in _iter_active():
                o = orig_weights[name]
                u = unl_h.get_tensor(name).float().numpy().ravel()
                delta = np.abs(u - o)
                del u
                gt_flat = mask_src[name]

                gt_list.append(gt_flat.astype(np.float32))
                raw_list.append(delta)
                qtile_list.append(to_quantile_ranks(delta))

                key = (layer_idx, comp)
                if key not in layercomp_running:
                    layercomp_running[key] = [0.0, 0.0, 0]
                s = layercomp_running[key]
                s[0] += float(delta.sum())
                s[1] += float(np.sum(delta.astype(np.float64) ** 2))
                s[2] += int(delta.size)

                if has_contrast_data:
                    if name in unmask_keys_a and name in unmask_orig_weights:
                        u_um = unmask_h_a.get_tensor(name).float().numpy().ravel()
                        c_raw = delta - np.abs(u_um - unmask_orig_weights[name])
                        del u_um
                        if key not in contrast_lc_running:
                            contrast_lc_running[key] = [0.0, 0.0, 0]
                        rs = contrast_lc_running[key]
                        rs[0] += float(c_raw.sum())
                        rs[1] += float(np.sum(c_raw.astype(np.float64) ** 2))
                        rs[2] += int(c_raw.size)
                        del c_raw

                del delta

            if unmask_h_a is not None:
                del unmask_h_a

        auc, fpr, tpr = compute_roc_auc(raw_list, gt_list)
        result['auc_raw'], result['fpr_raw'], result['tpr_raw'] = auc, fpr, tpr
        del raw_list

        auc, fpr, tpr = compute_roc_auc(qtile_list, gt_list)
        result['auc_qtile'], result['fpr_qtile'], result['tpr_qtile'] = auc, fpr, tpr
        del qtile_list
        gc.collect()

        # ── Pass B: signrev + compnorm + reversal + dirreversal ──
        signrev_list = []
        compnorm_list = []
        reversal_list = []
        dirreversal_list = []

        with safe_open(str(path), framework='pt', device='cpu') as unl_h:
            for name, layer_idx, comp in _iter_active():
                o = orig_weights[name]
                u = unl_h.get_tensor(name).float().numpy().ravel()
                p = pretrained_weights.get(name)

                signed_delta = u - o
                signrev_list.append(-((o - p) * signed_delta) if p is not None
                                    else np.zeros(u.size, dtype=np.float32))

                delta = np.abs(signed_delta)
                if has_compnorm_data:
                    unmask_mean = unmask_comp_mean_delta[comp_mean_key].get(name, 0.0)
                    compnorm_list.append(delta / unmask_mean if unmask_mean > 1e-12 else delta)
                else:
                    compnorm_list.append(delta)

                if p is not None:
                    injection = o - p
                    injection_abs = np.abs(injection)
                    dist_to_pre = np.abs(u - p)
                    reversal_list.append(
                        np.divide(injection_abs - dist_to_pre,
                                  injection_abs + 1e-10, dtype=np.float32))
                    dirreversal_list.append(
                        np.divide(-(signed_delta * np.sign(injection)),
                                  injection_abs + 1e-10, dtype=np.float32))
                    del injection, injection_abs, dist_to_pre
                else:
                    reversal_list.append(np.zeros(u.size, dtype=np.float32))
                    dirreversal_list.append(np.zeros(u.size, dtype=np.float32))

                del u, signed_delta, delta

        for metric, slist in [('signrev', signrev_list), ('compnorm', compnorm_list),
                               ('reversal', reversal_list), ('dirreversal', dirreversal_list)]:
            if slist:
                auc, fpr, tpr = compute_roc_auc(slist, gt_list)
            else:
                auc, fpr, tpr = _nan_result()
            result[f'auc_{metric}'], result[f'fpr_{metric}'], result[f'tpr_{metric}'] = auc, fpr, tpr
        del signrev_list, compnorm_list, reversal_list, dirreversal_list
        gc.collect()

        # ── Pass C: contrast + contrastnorm + eratio ──
        contrast_list = []
        contrastnorm_list = []
        eratio_list = []

        if has_contrast_data:
            with safe_open(str(path), framework='pt', device='cpu') as unl_h:
                with safe_open(str(unmask_upath), framework='pt', device='cpu') as um_h:
                    um_keys = set(um_h.keys())
                    for name, layer_idx, comp in _iter_active():
                        o = orig_weights[name]
                        u = unl_h.get_tensor(name).float().numpy().ravel()
                        delta = np.abs(u - o)
                        del u
                        if name in um_keys and name in unmask_orig_weights:
                            u_um = um_h.get_tensor(name).float().numpy().ravel()
                            delta_um = np.abs(u_um - unmask_orig_weights[name])
                            del u_um
                            contrast_raw = delta - delta_um
                            contrast_list.append(contrast_raw)
                            denom = delta + delta_um
                            cn = np.divide(contrast_raw, denom,
                                           out=np.zeros_like(contrast_raw),
                                           where=denom > 1e-12)
                            contrastnorm_list.append(cn)
                            eratio_list.append(
                                np.divide(delta, delta_um + 1e-10, dtype=np.float32))
                            del contrast_raw, delta_um, denom, cn
                        else:
                            contrast_list.append(delta.copy())
                            contrastnorm_list.append(np.ones_like(delta))
                            eratio_list.append(delta.copy())
                        del delta

        for metric, slist in [('contrast', contrast_list), ('contrastnorm', contrastnorm_list),
                               ('eratio', eratio_list)]:
            if slist:
                auc, fpr, tpr = compute_roc_auc(slist, gt_list)
            else:
                auc, fpr, tpr = _nan_result()
            result[f'auc_{metric}'], result[f'fpr_{metric}'], result[f'tpr_{metric}'] = auc, fpr, tpr
        del contrast_list, contrastnorm_list, eratio_list
        gc.collect()

        # ── Pass D: layernorm + contrastln + composite ──
        layercomp_std = {}
        for key, (s, sq, n) in layercomp_running.items():
            mean = s / n
            var = sq / n - mean ** 2
            layercomp_std[key] = max(np.sqrt(max(var, 0.0)), 1e-12)
        del layercomp_running

        contrast_lc_std = {}
        for key, (s, sq, n) in contrast_lc_running.items():
            if n > 0:
                mean = s / n
                var = sq / n - mean ** 2
                contrast_lc_std[key] = max(np.sqrt(max(var, 0.0)), 1e-12)
            else:
                contrast_lc_std[key] = 1e-12
        del contrast_lc_running

        layernorm_list = []
        contrastln_list = []
        comp_features = []
        comp_labels = []

        total_bc = sum(gt.size for gt in gt_list)
        sample_frac = min(1.0, COMPOSITE_SAMPLE / max(total_bc, 1))
        composite_rng = np.random.RandomState(42)

        with safe_open(str(path), framework='pt', device='cpu') as unl_h:
            unmask_h_d = None
            unmask_keys_d = set()
            if has_contrast_data:
                unmask_h_d = safe_open(str(unmask_upath), framework='pt', device='cpu')
                unmask_keys_d = set(unmask_h_d.keys())

            for name, layer_idx, comp in _iter_active():
                o = orig_weights[name]
                u = unl_h.get_tensor(name).float().numpy().ravel()
                signed_d = u - o
                d = np.abs(signed_d)
                del u
                std = layercomp_std.get((layer_idx, comp), 1e-12)

                layernorm_list.append(d / std)

                contrast_raw = None
                delta_um = None
                if has_contrast_data:
                    if unmask_h_d is not None and name in unmask_keys_d and name in unmask_orig_weights:
                        u_um = unmask_h_d.get_tensor(name).float().numpy().ravel()
                        delta_um = np.abs(u_um - unmask_orig_weights[name])
                        contrast_raw = d - delta_um
                        del u_um

                c_std = contrast_lc_std.get((layer_idx, comp), 1e-12)
                contrastln_list.append((contrast_raw / c_std) if contrast_raw is not None
                                       else (d / c_std))

                # Composite features
                n = d.size
                n_sample = max(1, int(n * sample_frac))
                idx = composite_rng.choice(n, n_sample, replace=False)

                f1 = d[idx]
                f2 = d[idx] / std
                p = pretrained_weights.get(name)
                if p is not None:
                    inj = o - p
                    inj_idx = inj[idx]
                    f3 = -(inj_idx * signed_d[idx])
                else:
                    inj = None
                    inj_idx = None
                    f3 = np.zeros(n_sample, dtype=np.float32)
                f4 = contrast_raw[idx] if contrast_raw is not None else np.zeros(n_sample, dtype=np.float32)
                if has_compnorm_data:
                    um = unmask_comp_mean_delta[comp_mean_key].get(name, 0.0)
                    f5 = d[idx] / max(um, 1e-12)
                else:
                    f5 = f1.copy()
                if contrast_raw is not None and delta_um is not None:
                    denom = d[idx] + delta_um[idx]
                    f6 = np.divide(contrast_raw[idx], denom,
                                   out=np.zeros(n_sample, dtype=np.float32),
                                   where=denom > 1e-12)
                else:
                    f6 = np.zeros(n_sample, dtype=np.float32)
                f7 = (contrast_raw[idx] / c_std) if contrast_raw is not None else np.zeros(n_sample, dtype=np.float32)
                if inj is not None:
                    inj_abs = np.abs(inj_idx)
                    u_at_idx = signed_d[idx] + o[idx]
                    dist_pre = np.abs(u_at_idx - p[idx])
                    f8 = np.divide(inj_abs - dist_pre, inj_abs + 1e-10, dtype=np.float32)
                    f9 = np.divide(-(signed_d[idx] * np.sign(inj_idx)),
                                   inj_abs + 1e-10, dtype=np.float32)
                    del inj_abs, dist_pre, u_at_idx
                else:
                    f8 = np.zeros(n_sample, dtype=np.float32)
                    f9 = np.zeros(n_sample, dtype=np.float32)
                if delta_um is not None:
                    f10 = np.divide(d[idx], delta_um[idx] + 1e-10, dtype=np.float32)
                else:
                    f10 = np.zeros(n_sample, dtype=np.float32)

                gt_flat = mask_src[name]
                comp_features.append(np.column_stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]))
                comp_labels.append(gt_flat[idx].astype(np.float32))
                del inj, d, signed_d, contrast_raw, delta_um

            if unmask_h_d is not None:
                del unmask_h_d

        auc, fpr, tpr = compute_roc_auc(layernorm_list, gt_list)
        result['auc_layernorm'], result['fpr_layernorm'], result['tpr_layernorm'] = auc, fpr, tpr
        del layernorm_list

        auc, fpr, tpr = compute_roc_auc(contrastln_list, gt_list)
        result['auc_contrastln'], result['fpr_contrastln'], result['tpr_contrastln'] = auc, fpr, tpr
        del contrastln_list
        gc.collect()

        # Composite
        if comp_features:
            X = np.concatenate(comp_features)
            y = np.concatenate(comp_labels)
            del comp_features, comp_labels

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            del X

            lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
            y_prob = cross_val_predict(lr, X_scaled, y, cv=5, method='predict_proba')[:, 1]
            del X_scaled

            try:
                fpr_c, tpr_c, _ = roc_curve(y, y_prob)
                auc_c = sklearn_auc(fpr_c, tpr_c)
                idx_c = np.linspace(0, len(fpr_c) - 1, min(1000, len(fpr_c))).astype(int)
                fpr_cp, tpr_cp = fpr_c[idx_c], tpr_c[idx_c]
            except ValueError:
                auc_c, fpr_cp, tpr_cp = _nan_result()
            result['auc_composite'], result['fpr_composite'], result['tpr_composite'] = auc_c, fpr_cp, tpr_cp
            del y_prob, y
        else:
            auc, fpr, tpr = _nan_result()
            result['auc_composite'], result['fpr_composite'], result['tpr_composite'] = auc, fpr, tpr

        del gt_list
        gc.collect()

        conn.send(('ok', field, method, mask_type, result))
    except Exception:
        conn.send(('error', field, method, mask_type, traceback.format_exc()))
    finally:
        conn.close()


# ──────────────────────── Parallel runner ─────────────────────────

def run_parallel(tasks):
    pending = []
    task_iter = iter(tasks)
    results = {}
    done_count = 0
    total = len(tasks)
    t0 = time.time()

    def launch_next():
        try:
            field, method, mask_type = next(task_iter)
        except StopIteration:
            return False
        parent_conn, child_conn = mp.Pipe(duplex=False)
        p = mp.Process(target=_worker_fn,
                       args=(field, method, mask_type, param_names, SKIP_COMPONENTS, child_conn))
        p.start()
        child_conn.close()
        pending.append((p, parent_conn, field, method, mask_type))
        return True

    for _ in range(min(NUM_WORKERS, total)):
        launch_next()

    while pending:
        for i, (p, conn, field, method, mask_type) in enumerate(pending):
            if conn.poll(0.5):
                try:
                    msg = conn.recv()
                except EOFError:
                    conn.close()
                    p.join()
                    for pp, cc, *_ in pending:
                        cc.close()
                        pp.kill()
                        pp.join()
                    raise RuntimeError(
                        f'Worker died: {field} / {method} [{mask_type}] '
                        f'(exit code {p.exitcode}) — likely OOM.')
                conn.close()
                p.join()

                if msg[0] == 'ok':
                    _, f, m, mt, r = msg
                    results.setdefault(mt, {}).setdefault(f, {})[m] = r
                    done_count += 1
                    parts = []
                    for metric in sorted(ALL_METRICS - {'crossfield'}):
                        auc_key = f'auc_{metric}'
                        if auc_key in r and not np.isnan(r[auc_key]):
                            parts.append(f'{metric}={r[auc_key]:.4f}')
                    auc_str = ', '.join(parts)
                    active_pct = 100 * r['n_active'] / max(r['n_total'], 1)
                    print(f'  [{done_count}/{total}] {f} / {m} [{mt}]: '
                          f'active={r["n_active"]}/{r["n_total"]} ({active_pct:.1f}%)  '
                          f'AUC {auc_str}  ({time.time()-t0:.0f}s)', flush=True)
                else:
                    _, f, m, mt, tb = msg
                    done_count += 1
                    print(f'  [{done_count}/{total}] {f} / {m} [{mt}]: FAILED\n{tb}', flush=True)

                pending.pop(i)
                launch_next()
                break
        else:
            for i, (p, conn, field, method, mask_type) in enumerate(pending):
                if not p.is_alive() and not conn.poll(0):
                    conn.close()
                    p.join()
                    for pp, cc, *_ in pending:
                        cc.close()
                        pp.kill()
                        pp.join()
                    raise RuntimeError(
                        f'Worker died: {field} / {method} [{mask_type}] '
                        f'(exit code {p.exitcode}) — likely OOM.')

    print(f'\nAll {total} experiments processed in {time.time()-t0:.1f}s.', flush=True)
    return results


# ──────────────────────── Cache saving ──────────────────────────

def save_results_cache(results):
    for mask_type, mresults in results.items():
        for field, field_results in mresults.items():
            for method, r in field_results.items():
                out_dir = CACHE_DIR / mask_type / field / method
                out_dir.mkdir(parents=True, exist_ok=True)

                metrics = {
                    'n_active': r.get('n_active', 0),
                    'n_total': r.get('n_total', 0),
                }
                roc_data = {}
                for m in ALL_METRICS:
                    metrics[f'auc_{m}'] = r.get(f'auc_{m}', float('nan'))
                    fpr_key = f'fpr_{m}'
                    tpr_key = f'tpr_{m}'
                    if fpr_key in r and tpr_key in r:
                        roc_data[fpr_key] = r[fpr_key].astype(np.float16)
                        roc_data[tpr_key] = r[tpr_key].astype(np.float16)

                with open(out_dir / 'metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=2)
                np.savez_compressed(out_dir / 'roc_curves.npz', **roc_data)

    print(f'\nCache saved to {CACHE_DIR}', flush=True)


def print_auc_summary(results):
    for mask_type, mresults in sorted(results.items()):
        mask_label = 'Forget | Retain' if mask_type == 'unified' else 'Forget Only'
        rows = []
        for field in FIELDS:
            for method in METHODS:
                if field not in mresults or method not in mresults[field]:
                    continue
                r = mresults[field][method]
                row = {'Field': field.replace('_', ' '), 'Method': method,
                       'Active': f'{r.get("n_active", "?")}/{r.get("n_total", "?")}'}
                for m in sorted(ALL_METRICS):
                    row[f'AUC ({m})'] = r.get(f'auc_{m}', float('nan'))
                rows.append(row)
        if not rows:
            continue
        prec_df = pd.DataFrame(rows)
        print(f'\n=== AUC Summary — Active Params Only (GT Mask: {mask_label}) ===')
        print(prec_df.to_string(index=False))


# ──────────────────────────── Main ──────────────────────────────

def main():
    global orig_weights, pretrained_weights, gt_mask_unified_np, gt_mask_forget_np
    global param_names, unmask_orig_weights, unmask_comp_mean_delta

    t_start = time.time()

    with safe_open(str(ORIGINAL_PATH), framework='pt', device='cpu') as f:
        model_keys = set(f.keys())

    print('Loading mask.pt...', flush=True)
    full_mask = torch.load(str(MASK_PATH), map_location='cpu')
    mask_keys = set(full_mask.keys())
    param_names = sorted(mask_keys & model_keys)
    print(f'Shared params: {len(param_names)}', flush=True)

    def build_gt_bits(groups):
        bits = 0
        for g in groups:
            bits |= (1 << g)
        return bits

    def load_or_compute_mask(groups, cache_path, label):
        if cache_path.exists():
            print(f'Loading cached GT mask ({label})...', flush=True)
            return torch.load(str(cache_path), map_location='cpu', weights_only=True)
        print(f'Computing GT mask ({label})...', flush=True)
        bits = build_gt_bits(groups)
        mask = {name: ((full_mask[name] & bits) > 0).to(torch.bool) for name in param_names}
        torch.save(mask, str(cache_path))
        return mask

    gt_mask_forget = load_or_compute_mask(
        FORGET_GROUPS, BASE / 'gt_mask_forget_cache.pt', 'forget only')
    gt_mask_unified = load_or_compute_mask(
        FORGET_GROUPS | RETAIN_GROUPS, BASE / 'gt_mask_unified_cache.pt',
        'unified (forget | retain)')
    del full_mask
    gc.collect()

    print('Converting GT masks to numpy...', flush=True)
    for name in param_names:
        gt_mask_unified_np[name] = gt_mask_unified[name].numpy().ravel()
        gt_mask_forget_np[name] = gt_mask_forget[name].numpy().ravel()
    del gt_mask_unified, gt_mask_forget
    gc.collect()

    print('Pre-loading original model weights...', flush=True)
    with safe_open(str(ORIGINAL_PATH), framework='pt', device='cpu') as f:
        for name in param_names:
            orig_weights[name] = f.get_tensor(name).float().numpy().ravel()
    print(f'Original weights: {sum(v.nbytes for v in orig_weights.values())/1e9:.2f} GB', flush=True)

    print('Pre-loading pretrained model weights...', flush=True)
    with safe_open(str(PRETRAINED_PATH), framework='pt', device='cpu') as f:
        pt_keys = set(f.keys())
        for name in param_names:
            if name in pt_keys:
                pretrained_weights[name] = f.get_tensor(name).float().numpy().ravel()
    print(f'Pretrained weights: {sum(v.nbytes for v in pretrained_weights.values())/1e9:.2f} GB', flush=True)

    print('Loading UnMask original weights...', flush=True)
    with safe_open(str(UNMASK_ORIGINAL_PATH), framework='pt', device='cpu') as f:
        for name in param_names:
            if name in f.keys():
                unmask_orig_weights[name] = f.get_tensor(name).float().numpy().ravel()

    print('Computing UnMask per-component mean deltas...', flush=True)
    for field in FIELDS:
        for method in METHODS:
            if method in UNMASK_EXCLUDED_METHODS:
                continue
            upath = unmask_unlearned_path(field, method)
            if not upath.exists():
                print(f'  SKIP {field}/{method}: UnMask model not found', flush=True)
                continue
            comp_means = {}
            with safe_open(str(upath), framework='pt', device='cpu') as unl_h:
                unl_keys = set(unl_h.keys())
                for name in param_names:
                    if name not in unl_keys or name not in unmask_orig_weights:
                        continue
                    _, comp = classify_param(name)
                    if comp in SKIP_COMPONENTS:
                        continue
                    u = unl_h.get_tensor(name).float().numpy().ravel()
                    comp_means[name] = float(np.abs(u - unmask_orig_weights[name]).mean())
                    del u
            unmask_comp_mean_delta[(field, method)] = comp_means
            print(f'  {field}/{method}: {len(comp_means)} params', flush=True)
    gc.collect()

    # Build tasks
    tasks = []
    results = {}
    n_skipped = 0
    for field in FIELDS:
        for method in METHODS:
            mask_types = ['forget'] if method in FORGET_ONLY_METHODS else ['unified', 'forget']
            for mt in mask_types:
                cache_dir = CACHE_DIR / mt / field / method
                if is_fully_cached(cache_dir):
                    r = load_cached_result(cache_dir)
                    results.setdefault(mt, {}).setdefault(field, {})[method] = r
                    n_skipped += 1
                else:
                    tasks.append((field, method, mt))

    if n_skipped:
        print(f'\n{n_skipped} experiments fully cached — skipped.', flush=True)
    if tasks:
        print(f'Launching {len(tasks)} experiments across {NUM_WORKERS} workers...', flush=True)
        worker_results = run_parallel(tasks)
        for mt, mt_results in worker_results.items():
            for f, f_results in mt_results.items():
                for m, r in f_results.items():
                    results.setdefault(mt, {}).setdefault(f, {})[m] = r

    # Crossfield post-processing
    print('\nComputing crossfield metrics...', flush=True)
    for method in METHODS:
        mask_types = ['forget'] if method in FORGET_ONLY_METHODS else ['unified', 'forget']
        for mt in mask_types:
            # Check cache
            already_cached = True
            for field in FIELDS:
                r = results.get(mt, {}).get(field, {}).get(method)
                if r is not None and not np.isnan(r.get('auc_crossfield', float('nan'))):
                    continue
                already_cached = False
                break
            if already_cached:
                print(f'  {method} [{mt}]: crossfield cached', flush=True)
                continue

            mask_src = gt_mask_forget_np if mt == 'forget' else gt_mask_unified_np

            field_paths = []
            for field in FIELDS:
                p = unlearned_path(field, method)
                if p.exists():
                    field_paths.append((field, p))
            if len(field_paths) < 2:
                print(f'  {method} [{mt}]: <2 fields, skipping crossfield', flush=True)
                continue

            # Identify active params: must be active in at least one field
            active_in_any = set()
            for field, fpath in field_paths:
                with safe_open(str(fpath), framework='pt', device='cpu') as unl_h:
                    unl_keys = set(unl_h.keys())
                    for name in param_names:
                        if name not in unl_keys or name in active_in_any:
                            continue
                        _, comp = classify_param(name)
                        if comp in SKIP_COMPONENTS:
                            continue
                        gt_flat = mask_src[name]
                        if not (gt_flat.any() and not gt_flat.all()):
                            continue
                        u = unl_h.get_tensor(name).float().numpy().ravel()
                        delta = np.abs(u - orig_weights[name])
                        if (delta > ACTIVE_THRESHOLD).any():
                            active_in_any.add(name)
                        del u, delta

            running_min_scores = {}
            running_gt = {}
            for fi, (field, fpath) in enumerate(field_paths):
                with safe_open(str(fpath), framework='pt', device='cpu') as unl_h:
                    unl_keys = set(unl_h.keys())
                    for name in param_names:
                        if name not in active_in_any or name not in unl_keys:
                            continue
                        o = orig_weights[name]
                        u = unl_h.get_tensor(name).float().numpy().ravel()
                        delta = np.abs(u - o)
                        del u
                        qtile = to_quantile_ranks(delta)
                        del delta
                        if fi == 0:
                            running_min_scores[name] = qtile
                            running_gt[name] = mask_src[name].astype(np.float32)
                        elif name in running_min_scores:
                            np.minimum(running_min_scores[name], qtile,
                                       out=running_min_scores[name])
                        del qtile
                gc.collect()

            scores_list = [running_min_scores[n] for n in param_names if n in running_min_scores]
            gt_list_cf = [running_gt[n] for n in param_names if n in running_gt]
            del running_min_scores, running_gt

            if scores_list:
                auc_cf, fpr_cf, tpr_cf = compute_roc_auc(scores_list, gt_list_cf)
            else:
                auc_cf, fpr_cf, tpr_cf = _nan_result()
            del scores_list, gt_list_cf
            gc.collect()

            print(f'  {method} [{mt}]: crossfield AUC = {auc_cf:.4f}', flush=True)

            for field in FIELDS:
                r = results.get(mt, {}).get(field, {}).get(method)
                if r is not None:
                    r['auc_crossfield'] = auc_cf
                    r['fpr_crossfield'] = fpr_cf
                    r['tpr_crossfield'] = tpr_cf

    save_results_cache(results)
    print_auc_summary(results)
    print(f'\nTotal wall time: {time.time()-t_start:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Precision metrics restricted to active parameter tensors only.')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                        help='Number of parallel worker processes')
    args = parser.parse_args()
    NUM_WORKERS = args.workers
    mp.set_start_method('fork')
    main()
