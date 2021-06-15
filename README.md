# SQuAD
NLP Final Project：SQuAD

## Model
- RoBERTa-large + Post-Pretraing(textbook corpus, spanmask)
- RoBERTa-large + Post-Pretraing(textbook corpus, spanmask) + NewsQA Finetune


## Experiment Settings
max_len = 180

learning rate = 1e-6

## Result

| 栏目1 | 栏目2 |
| ----- | ----- |
| 内容1 | 内容2 |
Dev:  **81.07/84.34** (EM/F1-score)

Test: blind
