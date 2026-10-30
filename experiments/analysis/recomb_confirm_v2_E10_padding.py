"""Process-local token padding matching the measured original input layout."""


class ScientificPaddingTokenizer:
    def __init__(self, tokenizer, token_width):
        if type(token_width) is not int or not 2 <= token_width <= 1024:
            raise ValueError('Scientific token width must retain the original model input range')
        self.tokenizer = tokenizer
        self.token_width = token_width

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        import torch
        encoded = self.tokenizer(*args, **kwargs)
        current = int(encoded['input_ids'].shape[1])
        extra = self.token_width-current
        if extra < 0:
            raise ValueError('A scientific-padding profile cannot truncate a query')
        result = {}
        for key, value in encoded.items():
            fill = int(self.tokenizer.pad_token_id) if key == 'input_ids' else (1 if key == 'special_tokens_mask' else 0)
            result[key] = torch.nn.functional.pad(value, (0, extra), value=fill) if value.ndim == 2 else value
        return result
