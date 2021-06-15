# SQuAD
NLP Final Project：SQuAD

## Model
- RoBERTa-large + Post-Pretraing(textbook corpus, spanmask)
- RoBERTa-large + Post-Pretraing(textbook corpus, spanmask) + NewsQA Finetune


## Experiment Settings
max_len = 180

learning rate = 1e-6

## Result

|3-stage| epoch=8 |  |
|------ | ------- | -------|
|  | **EM** | **F1-score** |
| **Dev** | 81.07 | 84.34 |
| **Test** | - | - |

|4-stage| epoch=3 |  |
|------ | ------- | -------|
|  | **EM** | **F1-score** |
| **Dev** | 81.22 | 84.21 |
| **Test** | - | - |
