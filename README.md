# SQuAD
NLP Final Project：SQuAD

## Model
- RoBERTa-large
- RoBERTa-large + Post-Pretraing(textbook corpus, spanmask)
- RoBERTa-large + Post-Pretraing(textbook corpus+SQuAD corpus, spanmask)
- RoBERTa-large + NewsQA Finetune
- RoBERTa-large + Post-Pretraing(textbook corpus, spanmask) + NewsQA Finetune
- RoBERTa-large + Post-Pretraing(textbook corpus+SQuAD corpus, spanmask) + NewsQA Finetune


## Experiment Settings
SQuAD Finetune: max_len=180, lr=1e-6, batch_size=32

NewsQA Finetune: max_len=512, lr=1e-6, batch_size=8

## Result

- RoBERTa-large

batchsize=16

res_v6

|2-stage| epoch=7 |  |
|------ | ------- | -------|
|  | **EM** | **F1-score** |
| **Dev** | 81.47 | 84.66 |
| **Test** | - | - |

****

- RoBERTa-large + Post-Pretraing(textbook corpus, spanmask)

batchsize=16

|3-stage| epoch=8 |  |
|------ | ------- | -------|
|  | **EM** | **F1-score** |
| **Dev** | 81.07 | 84.34 |
| **Test** | - | - |

****

- RoBERTa-large + NewsQA Finetune (e3)

batchsize=16

res_without_postpt_e

|3-stage| epoch=5 | newsqa_without_postpt_e3 |
|------ | ------- | -------|
|  | **EM** | **F1-score** |
| **Dev** | 81.73 | 84.76 |
| **Test** | - | - |

****

- RoBERTa-large + Post-Pretraing(textbook corpus, spanmask) + NewsQA Finetune (e2)

batchsize=16 (better than 32)

res_v5

|4-stage| epoch=5 | newsqa_e2 |
|------ | ------- | -------|
|  | **EM** | **F1-score** |
| **Dev** | 81.82 | 84.79 |
| **Test** | - | - |

- RoBERTa-large + Post-Pretraing(textbook corpus, spanmask) + NewsQA Finetune (e3)

batchsize=16

|4-stage| epoch=8 | newsqa_e3 |
|------ | ------- | -------|
|  | **EM** | **F1-score** |
| **Dev** | 81.85 | 84.91 |
| **Test** | - | - |


- RoBERTa-large + Post-Pretraing(textbook corpus, spanmask) + NewsQA Finetune (e4)

res_v4

batchsize = 16

|4-stage| epoch=8 | newsqa_e4 |
|------ | ------- | -------|
|  | **EM** | **F1-score** |
| **Dev** | 81.87 | 84.89 |
| **Test** | - | - |
