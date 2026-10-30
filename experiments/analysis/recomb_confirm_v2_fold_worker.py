"""Use the verified frozen predictor implementation with a separate run namespace."""
from pathlib import Path
import recomb_benign_fold_worker_20261017 as worker

worker.BASE=Path("reports/recomb_confirm_v2_span_structures")
worker.CAT=Path("reports/recomb_confirm_v2_span_catalog")
worker.FREEZE=Path("configs/recomb_confirm_v2_span_prediction_freeze.json")

if __name__=="__main__":worker.main()
