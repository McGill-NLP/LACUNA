from evals.base import Evaluator


class PanoramaEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("Panorama", eval_cfg, **kwargs)