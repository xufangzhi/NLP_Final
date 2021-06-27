import random
from tqdm import tqdm
import numpy as np
import math
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast, RobertaTokenizer
from transformers import RobertaForMaskedLM, RobertaForMultipleChoice
from transformers import LineByLineTextDataset, AdamW
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class PreTrianDataset(torch.utils.data.Dataset):
    def __init__(self, file_, tokenizer):
        self.texts = self.read_file(file_)
        self.tokenizer = tokenizer
        self.mask_id = 50264
        self.lower = 1  # span lower
        self.upper = 10  # span_upper
        self.lens = list(range(self.lower, self.upper + 1))
        self.p = 0.2  # geometric_p
        self.len_distrib = [self.p * (1 - self.p) ** (i - self.lower) for i in
                            range(self.lower, self.upper + 1)] if self.p >= 0 else None
        self.len_distrib = [x / (sum(self.len_distrib)) for x in self.len_distrib]

    def read_file(self, file_name):
        text_list = []
        with open(file_name, 'r') as f:
            for i, line in enumerate(f):
                line_ = line.strip()
                if len(line_) > 0:
                    text_list.append(line_)
        return text_list

    def mask_method(self, sentence):  # mask_method spanbert
        tokenizer_temp = self.tokenizer(sentence, return_offsets_mapping=True)
        tokens = tokenizer_temp.tokens()  # ['<s>', 'hello', ',', '?my', '?name', '?is', '?Nancy', '.', '</s>']
        input_ids = tokenizer_temp['input_ids']
        target_ids = input_ids.copy()  # correct ids
        attention_mask = tokenizer_temp['attention_mask']
        offset_mapping = tokenizer_temp['offset_mapping']
        wordstart = self.is_wordstart(input_ids, offset_mapping)
        # print("wordstart length:", len(wordstart))
        pure_tokens = []
        for i in range(len(tokens)):
            if wordstart[i] == 1:
                pure_tokens.append(input_ids[i])

        sent_length = len(tokens)
        mask_num = math.ceil(sent_length * 0.15)  # mask ratio is 0.15
        mask = set()

        spans = []
        while len(mask) < mask_num:
            span_len = np.random.choice(self.lens, p=self.len_distrib)
            anchor = np.random.choice(sent_length)

            if anchor in mask:
                continue
            # find word start, end
            left1, right1 = self.get_word_start(input_ids, anchor, wordstart), self.get_word_end(input_ids, anchor, wordstart)
            spans.append([left1, left1])
            for i in range(left1, right1):
                if len(mask) >= mask_num:
                    break
                mask.add(i)
                spans[-1][-1] = i
            num_words = 1
            right2 = right1
            while num_words < span_len and right2 < len(input_ids) and len(mask) < mask_num:
                # complete current word
                left2 = right2
                right2 = self.get_word_end(input_ids, right2, wordstart)
                num_words += 1
                for i in range(left2, right2):
                    if len(mask) >= mask_num:
                        break
                    mask.add(i)
                    spans[-1][-1] = i
        # mask process
        spans = merge_intervals(spans)
        for start, end in spans:
            rand = np.random.random()
            for i in range(start, end + 1):
                if rand < 0.8:
                    input_ids[i] = self.mask_id
                elif rand < 0.9:
                    input_ids[i] = np.random.choice(pure_tokens)
        return input_ids, attention_mask, target_ids

    def is_wordstart(self, input_ids, offset_mapping):
        wordstarts = [0]
        wordstart_index = []
        pre_index = 0
        for i in range(1, len(input_ids) - 1):
            if i == 1:
                wordstarts.append(1)
                wordstart_index.append(i)
                pre_index = offset_mapping[i][1]
            else:
                if offset_mapping[i][0] == pre_index:
                    wordstarts.append(0)
                else:
                    wordstarts.append(1)
                    wordstart_index.append(i)
                pre_index = offset_mapping[i][1]
        wordstarts.append(0)
        return wordstarts

    def get_word_start(self, sentence, anchor, wordstart):
        left = anchor
        while left > 0 and wordstart[left] == False:
            left -= 1
        return left

    # word end is next word start
    def get_word_end(self, sentence, anchor, wordstart):
        right = anchor + 1
        while right < len(sentence) and wordstart[right] == False:
            right += 1
        return right

    def __getitem__(self, idx):
        text = self.texts[idx]
        return self.mask_method(text)

    def __len__(self):
        return len(self.texts)


def merge_intervals(intervals):  # process and avoid overlap
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] + 1 < interval[0]:
            merged.append(interval)
        else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged

def load_model():
    # lm_model = RobertaForMaskedLM.from_pretrained('roberta-base').cuda()
    lm_model = torch.load("/data/linqika/xufangzhi/ISAAQ/checkpoints/pretrain_physics+tqa_spanmask_RACE_e2.pth")
    mc_model = RobertaForMaskedLM.from_pretrained('roberta-large')

    mc_model.roberta.embeddings.load_state_dict(lm_model.roberta.embeddings.state_dict())
    mc_model.roberta.encoder.load_state_dict(lm_model.roberta.encoder.state_dict())

    return mc_model
    
def train_LM():
    # from datasets import load_dataset
    # datasets = load_dataset('wikitext.py', 'wikitext-2-raw-v1')

    # 50265
    device = torch.device("cuda:1")
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "roberta-large")  # ['<s>', '</s>', '<unk>', '<pad>', '<mask>'] [0, 2, 3, 1, 50264]
    max_length = tokenizer.model_max_length

    # max_length = 800

    # max_length = 100

    def my_collate(batch):
        # print(batch)
        # print(len(batch))
        input_ids = [item[0] for item in batch]
        attention_mask = [item[1] for item in batch]
        target_ids = [item[2] for item in batch]

        # print(len(input_ids))
        # print(len(attention_mask))
        # print(len(target_ids))

        max_len = max([len(item) for item in input_ids])
        max_len = max_len if max_len < max_length else max_length
        # print(max_len)
        input_ids_new = []
        attention_mask_new = []
        target_ids_new = []
        for i in range(len(input_ids)):
            temp_input = [1] * max_len
            temp_mask = [0] * max_len
            temp_target = [1] * max_len

            temp_input[:len(input_ids[i])] = input_ids[i][:max_len]
            temp_mask[:len(attention_mask[i])] = attention_mask[i][:max_len]
            temp_target[:len(input_ids[i])] = target_ids[i][:max_len]

            input_ids_new.append(temp_input)
            attention_mask_new.append(temp_mask)
            target_ids_new.append(temp_target)

        return input_ids_new, attention_mask_new, target_ids_new

    epochs = 10
    train_dataset = PreTrianDataset("/data/linqika/xufangzhi/SQUAD/jsons/squad_text.txt", tokenizer)
    
    #model = RobertaForMaskedLM.from_pretrained("roberta-large").cuda()
    model = load_model().to(device)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=my_collate)
    optim = AdamW(model.parameters(), lr=3e-6)

    for epoch in range(epochs):
        print('--' * 50)
        print('Trianing Epoch:', epoch+1)
        loss_list = []
        pbar = tqdm(train_loader)
        for batch in pbar:
            #print(batch)
            optim.zero_grad() 
            inputs = torch.tensor(batch[0]).to(device)
            attention_mask = torch.tensor(batch[1]).to(device)
            labels = torch.tensor(batch[2]).to(device)
            # print(inputs.size())
            # print(attention_mask.size())
            # print(labels.size())
            outputs = model(inputs, attention_mask, labels=labels)
            loss = outputs[0]
            # print(loss.item())
            loss_list.append(loss.item())
            loss.backward()
            optim.step()
            
            train_loss = np.mean(loss_list)

            pbar.set_description("loss {0:.4f}".format(train_loss))
            
        #print('Training Loss:', np.mean(loss_list))
        from transformers import WEIGHTS_NAME, CONFIG_NAME
        output_dir = "/data/linqika/xufangzhi/SQUAD/train_lm/checkpoints/e" + str(epoch+1)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model.state_dict(), output_model_file)
        model.config.to_json_file(output_config_file)
        torch.save(model, "/data/linqika/xufangzhi/SQUAD/train_lm/checkpoints/physics+tqa+squad_spanmask_" + str(epoch + 1) + ".pth")


train_LM()
