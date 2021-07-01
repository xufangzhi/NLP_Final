from transformers import AdamW, RobertaForQuestionAnswering, RobertaTokenizerFast,  PreTrainedTokenizer
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
import torch
import json
import sys
import argparse
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import os
from model_methods import SQuAD_QA, SQuAD_self_attention
import torch.nn.functional as F
#os.environ['CUDA_VISIBLE_DEVICES']= '0,3'


def load_model():
    # lm_model = RobertaForMaskedLM.from_pretrained('roberta-base').cuda()
    lm_model = torch.load("/data/linqika/xufangzhi/ISAAQ/checkpoints/pretrain_physics+tqa_spanmask_RACE_e2.pth")
    mc_model = RobertaForQuestionAnswering.from_pretrained('roberta-large')

    mc_model.roberta.embeddings.load_state_dict(lm_model.roberta.embeddings.state_dict())
    mc_model.roberta.encoder.load_state_dict(lm_model.roberta.encoder.state_dict())

    return mc_model


def form_pred_json(b_input_ids, start_logits, end_logits, tokenizer, train_pred_json, num):
    start_pred = np.argmax(start_logits, axis=-1).flatten()
    end_pred = np.argmax(end_logits, axis=-1).flatten()
    
    for s_pred, e_pred, input_id in zip(start_pred, end_pred, b_input_ids):
        if input_id[s_pred]==1 or input_id[e_pred]==1:
            ans_span = ''
        else:
            if s_pred==0 and e_pred==0:
                ans_span = ''
            else:
                ans_span = tokenizer.decode(input_id[s_pred:e_pred+1]).strip()
        train_pred_json[num] = ans_span
    return train_pred_json, num



def flat_accuracy(start_logits, end_logits, start_position, end_position):
    start_pred = np.argmax(start_logits, axis=-1).flatten()
    end_pred = np.argmax(end_logits, axis=-1).flatten()
    start_real = np.argmax(start_position, axis=-1).flatten()
    end_real = np.argmax(end_position, axis=-1).flatten()
    correct = 0
    error = 0
    for s_pred, s_real in zip(start_pred, start_real):
        if s_pred==s_real:
            correct += 1
        else:
            error += 1
    for e_pred, e_real in zip(end_pred, end_real):
        if e_pred==e_real:
            correct += 1
        else:
            error += 1

    return correct, error

def flat_pred(start_logits, end_logits):
    s_logits = np.argmax(start_logits, axis=-1).flatten()
    e_logits = np.argmax(end_logits, axis=-1).flatten()
    
    pred = []
    for s,e in zip(s_logits, e_logits):
        pred.append((s,e))
    return pred 

def evaluate(pred_list, dev_qids_list, tokenizer, process_data_dev):
    input_ids = []
    dev_pred_json = {}
    for batch in process_data_dev:
        for i in range(batch[0].shape[0]): 
            input_ids.append(batch[0][i])
    for (s_pred,e_pred), qid, input_id in zip(pred_list, dev_qids_list, input_ids):
        #print(input_id)
        if input_id[s_pred]==1 or input_id[e_pred]==1:
            ans_span = ''
        else:
            if s_pred==0 and e_pred==0:
                ans_span = ''
            else:
                ans_span = tokenizer.decode(input_id[s_pred:e_pred+1]).strip()
        dev_pred_json[qid] = ans_span
    return dev_pred_json
    
    
def get_data(split, tokenizer, max_len):
    cls_index = tokenizer.cls_token_id
    input_ids_list=[]
    att_mask_list=[]
    start_positions_list=[]
    end_positions_list=[]
    dev_qids_list = []
    
    with open("jsons/squad_v1.1.json", "r", encoding="utf-8", errors="surrogatepass") as file:
        dataset = json.load(file)

    dataset = [doc for doc in dataset if doc["split"] == split]

    for doc in tqdm(dataset):
        question, text = doc["question"], doc["text"]
        encoded = tokenizer.encode_plus(question, text, max_length=max_len, pad_to_max_length=True, return_offsets_mapping=True)
        input_ids = encoded["input_ids"]
        att_mask = encoded["attention_mask"]
        offset_mapping = encoded['offset_mapping']
        
        sequence_ids = encoded.sequence_ids(0)

        start_char = doc['answers']['answer_start']
        end_char = doc['answers']['answer_end']
        token_start_index = 0      
        while sequence_ids[token_start_index] != 1:    #1:text,0:question
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        

        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

         # Detect if the answer is out of the span
        if not (offset_mapping[token_start_index][0] <= start_char and offset_mapping[token_end_index][1] >= end_char):
            start_pos = cls_index
            end_pos = cls_index
        else:
            while token_start_index < len(offset_mapping) and offset_mapping[token_start_index][0] <= start_char:
                token_start_index += 1
            start_pos = token_start_index - 1
            
            
            
            while token_end_index>0 and offset_mapping[token_end_index][1] >= end_char:
                token_end_index -= 1
                #print("offset length:",len(offset_mapping))
                #print("end_char:", end_char)
                #print("offset",offset_mapping)
                #print("token_end_index:",token_end_index)
            end_pos = token_end_index + 1
    
        input_ids_list.append(input_ids)
        att_mask_list.append(att_mask)
        start_positions_list.append(start_pos)
        end_positions_list.append(end_pos)
        if split=="dev":
            dev_qids_list.append(doc['id'])
    if split=="dev":
        return [input_ids_list, att_mask_list, start_positions_list, end_positions_list], dev_qids_list
    else:
        return [input_ids_list, att_mask_list, start_positions_list, end_positions_list]
        


def process_data(raw_data, batch_size, split):
    input_ids_list, att_mask_list, start_positions_list, end_positions_list = raw_data
    inputs = torch.tensor(input_ids_list)
    masks = torch.tensor(att_mask_list)
    start_posi = torch.tensor(start_positions_list)
    end_posi = torch.tensor(end_positions_list)

    if split=="train":
        data = TensorDataset(inputs, masks, start_posi, end_posi)
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    else:
        data = TensorDataset(inputs, masks, start_posi, end_posi)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    
    return dataloader


def train(model, tokenizer, process_data_train, process_data_dev, dev_qids_list, optimizer, scheduler, epochs, batch_size, device, save_model):
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Reset the total loss for this epoch.
        total_points = 0
        total_errors = 0
        train_loss_list = []
        train_pred_json = {}
        num = 0

        model.train()
        pbar = tqdm(process_data_train)
        for batch in pbar:  
            model.train()
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_start_positions = batch[2].to(device)
            b_end_positions = batch[3].to(device)
            
            outputs = model(b_input_ids, attention_mask=b_input_mask, start_positions=b_start_positions, end_positions=b_end_positions)
            loss, start_logits, end_logits = outputs[0], outputs[1], outputs[2]

            # Move logits and labels to CPU
            start_logits = start_logits.detach().cpu().numpy()
            end_logits = end_logits.detach().cpu().numpy()
            start_position = b_start_positions.to('cpu').numpy()
            end_position = b_end_positions.to('cpu').numpy()
            
            train_pred, n = form_pred_json(b_input_ids, start_logits, end_logits, tokenizer, train_pred_json, num)
            train_pred_json, num = train_pred, n
            
            
            # Calculate the accuracy for this batch of test sentences.

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            train_loss_list.append(loss.item())
            train_loss = np.mean(train_loss_list)

            pbar.set_description("loss {0:.4f}".format(train_loss))

        if save_model:
            torch.save(model, "checkpoints/v1.1_newsqa_e"+str(epoch_i+1)+".pth")
        
        res_json = dev(model, tokenizer, process_data_dev, dev_qids_list, device)
        with open("res_v1.1_final_e"+str(epoch_i+1)+".json",'w') as file:
            json.dump(res_json,file,indent=4)
            
    print("")
    print("Training complete!")


def dev(model, tokenizer, process_data_dev, dev_qids_list, device):

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running DEV...")

    total_points = 0
    total_errors = 0
    val_loss_list = []
    pred_list = []
    dev_pred_json = {}
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Evaluate data for one epoch
    sum_aux = 0
    total_aux = 0

    for batch in tqdm(process_data_dev):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_start_positions, b_end_positions = batch

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():        
            outputs = model(b_input_ids, attention_mask=b_input_mask, start_positions=b_start_positions, end_positions=b_end_positions)

        loss, start_logits, end_logits = outputs[0], outputs[1], outputs[2]

        # Move logits and labels to CPU
        start_logits = start_logits.detach().cpu().numpy()
        end_logits = end_logits.detach().cpu().numpy()
        
        start_position = b_start_positions.to('cpu').numpy()
        end_position = b_end_positions.to('cpu').numpy()
        
        pred = flat_pred(start_logits, end_logits)
        pred_list += pred

        val_loss_list.append(loss.item())

    val_loss = np.mean(val_loss_list)
    
    dev_pred_json = evaluate(pred_list, dev_qids_list, tokenizer, process_data_dev)
    print("val_loss {0:.4f}".format(val_loss))
    #print(dev_pred_json)
    return dev_pred_json

def main(argv):
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'], help='device to train the model with. Options: cpu or gpu. Default: gpu')
    parser.add_argument('-b', '--batchsize', default= 16, type=int, help='size of the batches. Default: 1')
    parser.add_argument('-x', '--maxlen', default= 180, type=int, help='max sequence length. Default: 180')
    parser.add_argument('-l', '--lr', default= 1e-6, type=float, help='learning rate. Default: 1e-6')
    parser.add_argument('-e', '--epochs', default= 8, type=int, help='number of epochs. Default: 4')
    parser.add_argument('-s', '--save', default=False, help='save model at the end of the training', action='store_true')
    args = parser.parse_args()
    print(args)
    
    #model = RobertaForQuestionAnswering.from_pretrained("/data/linqika/xufangzhi/SQUAD/train_lm/checkpoints/e2/")
    #model = RobertaForQuestionAnswering.from_pretrained("roberta-large")
    #model.roberta = torch.load("/data/linqika/xufangzhi/ISAAQ/checkpoints/pretrain_physics+tqa_spanmask_RACE_e2.pth").roberta
    #model = torch.load("/data/linqika/xufangzhi/SQUAD/checkpoints/baseline_e8.pth")
    model = torch.load("/data/linqika/xufangzhi/NewsQA/checkpoints/newsqa_postpte2_e3.pth")
    #model = SQuAD_QA()
    print("using roberta model")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")

    if args.device=="gpu":
        device = torch.device("cuda:0")
        model.to(device)
    if args.device=="cpu":
        device = torch.device("cpu") 
        model.to(device)
    print(device)
    model.zero_grad()
    
    batch_size = args.batchsize
    max_len = args.maxlen
    lr = args.lr
    epochs = args.epochs
    save_model = args.save

    raw_data_train = get_data("train", tokenizer, max_len)
    raw_data_dev, dev_qids_list = get_data("dev", tokenizer, max_len)

    process_data_train = process_data(raw_data_train, batch_size, "train")
    process_data_dev = process_data(raw_data_dev, batch_size, "dev")

    optimizer = AdamW(model.parameters(), lr = lr, eps = 1e-8)
    total_steps = len(raw_data_train[-1]) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
    
    train(model, tokenizer, process_data_train, process_data_dev, dev_qids_list, optimizer, scheduler, epochs, batch_size, device, save_model)
if __name__ == "__main__":
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    main(sys.argv[1:])