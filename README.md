# LS-LLaMA: Label Supervised LLaMA Finetuning

<p align="center">

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/label-supervised-llama-finetuning/named-entity-recognition-on-conll03-4)](https://paperswithcode.com/sota/named-entity-recognition-on-conll03-4?p=label-supervised-llama-finetuning)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/label-supervised-llama-finetuning/named-entity-recognition-on-ontonotes-5-0-1)](https://paperswithcode.com/sota/named-entity-recognition-on-ontonotes-5-0-1?p=label-supervised-llama-finetuning)
</p>


<p align='center'>
<img src='./docs/lsllama.png'/>
</p>

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
