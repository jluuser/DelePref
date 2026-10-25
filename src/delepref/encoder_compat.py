"""Frozen ESM tokenization without modifying an installed ESM/Transformers.

The constructor follows the official ESM character BPE tokenizer (MIT; see
THIRD_PARTY_NOTICES.md). HF owns its special-token properties here, matching
the constructor used by the frozen research runtime on Transformers 4.45.
"""


def make_sequence_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast
    from esm.tokenization.tokenizer_base import EsmTokenizerBase
    from esm.utils.constants import esm3 as constants

    class CompatibleSequenceTokenizer(PreTrainedTokenizerFast,EsmTokenizerBase):
        model_input_names=['sequence_tokens','attention_mask']

        def __init__(self):
            vocab={token:i for i,token in enumerate(constants.SEQUENCE_VOCAB)}
            tokenizer=Tokenizer(BPE(vocab,merges=[],unk_token='<unk>'))
            tokenizer.add_special_tokens(['<cls>','<pad>','<mask>','<eos>','|'])
            tokenizer.post_processor=TemplateProcessing(single='<cls> $A <eos>',
                special_tokens=[('<cls>',tokenizer.token_to_id('<cls>')),('<eos>',tokenizer.token_to_id('<eos>'))])
            self.cb_token='|'
            super().__init__(tokenizer_object=tokenizer,unk_token='<unk>',cls_token='<cls>',
                pad_token='<pad>',mask_token='<mask>',eos_token='<eos>',additional_special_tokens=['|'])

        @property
        def bos_token(self):return self.cls_token
        @property
        def bos_token_id(self):return self.cls_token_id
        @property
        def chain_break_token(self):return self.cb_token
        @property
        def chain_break_token_id(self):return self.convert_tokens_to_ids(self.cb_token)
        @property
        def all_token_ids(self):return list(range(self.vocab_size))
        @property
        def special_token_ids(self):return self.all_special_ids

    tokenizer=CompatibleSequenceTokenizer()
    expected={'<cls>':0,'<pad>':1,'<eos>':2,'<unk>':3,'<mask>':32}
    if any(tokenizer.convert_tokens_to_ids(k)!=v for k,v in expected.items()):
        raise ValueError('ESMC vocabulary differs from the frozen token IDs')
    return tokenizer
