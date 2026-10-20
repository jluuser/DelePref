"""Check inference import isolation before any GPU retry; no model forward."""
import hashlib
import json
import sys
import pkg_resources
import torch
from delepref_v8_baseline import ROOT,write
from delepref_v8_scisor_inference_import_v2 import load_native_class


def main():
    before=torch.cuda.is_available();require=pkg_resources.require
    sys.path.insert(0,'/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit')
    cls=load_native_class()
    if cls.__name__!='ShorteningSCUD' or pkg_resources.require is not require:
        raise RuntimeError('inference import or package resolver restoration failed')
    if torch.cuda.is_available()!=before:raise RuntimeError('CUDA availability was modified')
    if sys.modules.get('deepspeed') is not None:raise RuntimeError('unused DeepSpeed training package was imported')
    path=ROOT/'analysis/delepref_v8_scisor_inference_import_v2.py'
    report=dict(status='INFERENCE_ONLY_CLASS_IMPORT_CHECK_PASS',shim_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        package_resolver_restored=True,CUDA_availability_unchanged=True,
        scoped_change='optional DeepSpeed training plugin hidden only in this process',
        model_initialized=False,protein_sequence_forward_run=False,GPU_calls=0,shared_environment_changed=False,paper_changes=False)
    write(ROOT/'reports/delepref_v8_scisor_inference_check.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
