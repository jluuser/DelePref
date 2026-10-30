"""Check the server SDPA layer against standard ESM on synthetic hidden states."""
import json
from pathlib import Path
import sys
import torch
from transformers import EsmConfig
from transformers.models.esm.modeling_esm import EsmLayer
from delepref_v8_baseline import ROOT,write


def main():
    torch.set_num_threads(2);sys.path.insert(0,'/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit')
    from SCISOR.esm import FAEsmLayer
    config=EsmConfig(hidden_size=480,num_attention_heads=20,intermediate_size=1920,
        position_embedding_type='rotary',attention_probs_dropout_prob=0.,hidden_dropout_prob=0.)
    config.use_fa=False
    differences=[]
    with torch.inference_mode():
        for seed in (13,29):
            torch.manual_seed(seed)
            local=FAEsmLayer(config).eval();reference=EsmLayer(config).eval()
            reference.load_state_dict({k:local.state_dict()[k] for k in reference.state_dict()},strict=True)
            hidden=torch.randn(2,17,480)
            for padding in (0,5):
                mask=torch.zeros(2,1,1,17)
                if padding:mask[1,:,:,-padding:]=-10000.
                a=local(hidden_states=hidden,attention_mask=mask)[0]
                b=reference(hidden_states=hidden,attention_mask=mask)[0]
                error=float((a-b).abs().max());differences.append(error)
                if not torch.allclose(a,b,rtol=1e-5,atol=2e-6):raise ValueError('SDPA layer differs from ESM reference: '+str(error))
    report=dict(status='SDPA_COMPATIBILITY_LAYER_CHECK_PASS',synthetic_hidden_states=True,
        reference='transformers EsmLayer, same weights, rotary encoding, eval mode, padded/unpadded masks',
        cases=4,max_absolute_difference=max(differences),protein_sequence_forward_run=False,GPU_calls=0,
        checkpoint_weights_changed=False,new_structure_predictions=0,paper_changes=False)
    write(ROOT/'reports/delepref_v8_scisor_attention_check.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
