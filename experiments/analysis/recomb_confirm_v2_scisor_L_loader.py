"""Load complete official L state without redundantly loading a base ESM checkpoint."""
import copy
import importlib
from pathlib import Path
import torch
from recomb_confirm_v2_diagnostics import read,sha
from delepref_v8_scisor_inference_import_v2 import load_native_class
from delepref_v8_scisor_compat import patch_tokenizer


def load(root):
    root=Path(root);ckpt=root/'models/scisor_u90_l/SCISOR_U90_L.ckpt'
    expected='6925f185f82b9916698d03cc7b32b42166e5272d460f599eb9c64e8902442fe7'
    if sha(ckpt)!=expected:raise ValueError('Official L weights differ')
    config=root/'models/scisor_u90_l/esm2_650M_config'
    identity=read(config/'identity.json')
    if sha(config/'config.json')!=identity['sha256'] or identity['sha256']!='539095c22efc52a09d6147074ba4ca119f76a890df5901213b2b55f7d2f96b2b':
        raise ValueError('Official base architecture configuration differs')
    cls=load_native_class();native=importlib.import_module('SCISOR.faesm')
    payload=torch.load(ckpt,map_location='cpu',weights_only=False)
    hparams=copy.deepcopy(payload['hyper_parameters']);hparams['x0_model_class']=native.FAESM_Base
    nn=dict(hparams['nn_params'])
    if nn.get('hf_model_name')!='esm2_t33_650M_UR50D':raise ValueError('Unexpected L backbone')
    # This switch affects only initialization; strict loading supplies every trained tensor.
    nn['pretrained']=False;hparams['nn_params']=nn
    resolve=native.resolve_local_hf_snapshot
    def local_configuration(name,*args,**kwargs):
        if name!='facebook/esm2_t33_650M_UR50D':raise ValueError('Unexpected model initialization request')
        return str(config)
    native.resolve_local_hf_snapshot=local_configuration
    try:model=cls(**hparams)
    finally:native.resolve_local_hf_snapshot=resolve
    model.load_state_dict(payload['state_dict'],strict=True);model.eval();patch_tokenizer(model.tokenizer)
    return model,payload,dict(checkpoint_sha256=expected,architecture_config_sha256=identity['sha256'],
        constructor_preload_skipped=True,all_checkpoint_state_loaded_strictly=True,
        shared_source_and_environment_changed=False,extra_backbone_weights_downloaded=False)
