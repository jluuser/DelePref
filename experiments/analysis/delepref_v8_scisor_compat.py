"""Original V3-25 tokenizer compatibility patch, isolated from shared sources."""


def patch_tokenizer(tokenizer):
    values={'_unk_token':'<unk>','_cls_token':'<cls>','_bos_token':'<cls>',
        '_eos_token':'<eos>','_sep_token':'<eos>','_pad_token':'<pad>',
        '_mask_token':'<mask>','_additional_special_tokens':[]}
    for name,value in values.items():
        if not hasattr(tokenizer,name):setattr(tokenizer,name,value)
    expected={'<cls>':0,'<pad>':1,'<eos>':2,'<unk>':3,'<mask>':32}
    for token,index in expected.items():
        if tokenizer.convert_tokens_to_ids(token)!=index:
            raise ValueError('SCISOR special-token vocabulary differs: '+token)
