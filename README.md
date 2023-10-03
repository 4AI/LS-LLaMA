# LS-LLaMA: Label Supervised LLaMA Finetuning


## Usage

Load Pretrained Models

```python
from transformers import AutoTokenizer
from modeling_llama import (
    LlamaForSequenceClassification, LlamaForTokenClassification,
    UnmaskingLlamaForSequenceClassification, UnmaskingLlamaForTokenClassification,
)


model_id = 'meta-llama/Llama-2-7b'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = LlamaForSequenceClassification.from_pretrained(model_id).bfloat16()
model = LlamaForTokenClassification.from_pretrained(model_id).bfloat16()
model = UnmaskingLlamaForSequenceClassification.from_pretrained(model_id).bfloat16()
model = UnmaskingLlamaForTokenClassification.from_pretrained(model_id).bfloat16()
```

More usage please refer to `unllama_seq_clf.py`, `unllama_token_clf.py`, `llama_seq_clf.py`, `llama_token_clf.py`.
