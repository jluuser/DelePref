import pytest
from adapters.esmc_loader import ESMCSequenceTokenizer


def test_special_tokens_padding_and_exact_residue_ids():
    tokenizer=ESMCSequenceTokenizer(max_length=20)
    result=tokenizer(['ACD','W'])
    assert result['input_ids'].tolist()==[[0,5,23,13,2],[0,22,2,1,1]]
    assert result['attention_mask'].tolist()==[[True]*5,[True,True,True,False,False]]
    assert result['input_ids'].device.type=='cpu'


def test_no_silent_truncation_or_unequal_unpadded_batch():
    tokenizer=ESMCSequenceTokenizer(max_length=3)
    with pytest.raises(ValueError,match='exceeds'):tokenizer('ACDE')
    with pytest.raises(ValueError,match='padding'):tokenizer(['ACD','W'],padding=False)
    with pytest.raises(ValueError,match='empty'):tokenizer([])
