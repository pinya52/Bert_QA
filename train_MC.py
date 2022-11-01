import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
from pathlib import Path
from argparse import ArgumentParser, Namespace
import json

from datasets import DatasetDict, load_dataset, Dataset
# from accelerate import Accelerator
from transformers import (AutoTokenizer,
                          default_data_collator, 
                          get_cosine_schedule_with_warmup,
                          AutoModelForMultipleChoice,
                          SchedulerType,
                          set_seed)
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

# logger = logging.getLogger(__name__)


def preprocess(dataset, context):
    def preprocess_function(dataset):
        first_sentences = [[question]*4 for question in dataset["question"]]
        second_sentences = [[context[idx] for idx in idxs] for idxs in dataset["paragraphs"]]
        labels = [paragraph.index(dataset["relevant"][idx]) for idx, paragraph in enumerate(dataset["paragraphs"])]

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        
        tokenized_dataset = tokenizer(first_sentences, second_sentences, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=True, max_length=args.max_length)
        tokenized_inputs = {k:[v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_dataset.items()}
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

    return tokenizer, tokenized_dataset


def read_dataset():
    # Get Context
    with open(args.context_file, encoding="utf-8") as f:
        context = json.load(f)
    
    # load train and valid json
    dataset_dict = dict()
    with open(args.train_file, encoding="utf-8") as f:
        tmp_json = json.load(f)
        pd_dict_train = pd.DataFrame.from_dict(tmp_json)
        for idx, data in enumerate(tmp_json):
            tmp_json[idx]["context"] = context[data["relevant"]]

        pd_dict_train = pd.DataFrame.from_dict(tmp_json)
                
    with open(args.valid_file, encoding="utf-8") as f:
        tmp_json = json.load(f)
        for idx, data in enumerate(tmp_json):
            tmp_json[idx]["context"] = context[data["relevant"]]
            
        pd_dict_val = pd.DataFrame.from_dict(tmp_json)

    pd_dataset_train = Dataset.from_pandas(pd_dict_train)
    pd_dataset_val = Dataset.from_pandas(pd_dict_val)
    
    dataset_dict["train"] = pd_dataset_train
    dataset_dict["valid"] = pd_dataset_val
    dataset = DatasetDict(dataset_dict)

    return context, dataset


def train():
    print('\nModel using now: %s\n'%(args.model_name))
    # set_seed()
    print('Reading Dataset\n')
    context, dataset = read_dataset()
    model_name = str(args.model_name).replace('/', '_')
    
    # preprocess
    print('Preprocessing Raw Data\n')
    tokenizer, processed_datasets = preprocess(dataset, context)
    
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["valid"]

    # tokenizer.save_pretrained("./models/tokenizer/")
    print('Prepare for training\n')
    tokenizer_path = str(args.tokenizer_path) + '/%s'%(model_name)
    tokenizer.save_pretrained(tokenizer_path)
    # tokenizer2 = AutoTokenizer.from_pretrained("./models/tokenizer/")
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.batch_size)

    
    # Train
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # print(len(train_dataloader))
    # print(len(train_dataset))
    total_step = len(train_dataset) * args.num_epoch // (args.batch_size * args.accum_steps)
    warmup_step = total_step * 0.06
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_step, total_step)
    # model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, scheduler)
    
    best_dev_loss = 1e10
    print("Start Training")
    for epoch in range(args.num_epoch):
        model.train()
        
        print(f"\nEpoch: {epoch+1} / {args.num_epoch}")
        train_loss, train_acc = 0, 0
        for batch_step, batch_datas in enumerate(tqdm(train_dataloader, desc="Train")):
            input_ids, token_type_ids, attention_mask, labels = [b_data.to(args.device) for b_data in batch_datas.values()]
            # input_ids, token_type_ids, attention_mask, labels = batch_data.values()
            # outputs = model(**batch_data)
            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            train_loss += loss.detach().float()
            loss = loss / args.accum_steps
            
            # accelerator.backward(loss)
            loss.backward()
            
            if batch_step % args.accum_steps == 0 or batch_step == len(train_dataloader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            predictions = outputs.logits.argmax(dim=-1)
            train_acc += (predictions == labels).cpu().sum().item()
        
        train_loss /= (batch_step * args.accum_steps)
        train_acc /= len(train_dataset)

        model.eval()
        dev_acc, dev_loss = 0, 0
        for batch_step, batch_datas in enumerate(tqdm(eval_dataloader, desc="Valid")):
            with torch.no_grad():
                # input_ids, token_type_ids, attention_mask, labels = batch_data.values()
                input_ids, token_type_ids, attention_mask, labels = [b_data.to(args.device) for b_data in batch_datas.values()]

                outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
                dev_loss += loss.detach().float()

                predictions = outputs.logits.argmax(dim=-1)
                # print("pred", predictions)
                # print("labels", labels)

                dev_acc += (predictions == labels).cpu().sum().item()

        dev_loss /= (batch_step * args.accum_steps)
        dev_acc /= len(eval_dataset)
        
        print(f"TRAIN LOSS:{train_loss} ACC:{train_acc}  | EVAL LOSS:{dev_loss} ACC:{dev_acc}")

        if dev_loss < best_dev_loss:
            best_dev_loss = best_dev_loss
            # best_state_dict = deepcopy(model.state_dict())
            if args.model_path is not None:
                save_name = str(args.model_path) + '/%s'%(model_name)
                model.save_pretrained(save_name)

    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="model name",
        # default="bert-base-chinese",
        default="hfl/chinese-roberta-wwm-ext",
        # default='luhua/chinese_pretrain_mrc_macbert_large'
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        help="Tokenizer name",
        # default="bert-base-chinese",
        default="hfl/chinese-roberta-wwm-ext",
        # default='luhua/chinese_pretrain_mrc_macbert_large'
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        help="Directory to save the model.",
        default="./ckpt/MC/models/",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        help="Path to save the tokenizer.",
        default="./ckpt/tokenizer/MC",
    )
    parser.add_argument(
        "--context_file",
        type=Path,
        help="Context json file",
        default="./data/context.json",
    )
    parser.add_argument(
        "--train_file",
        type=Path,
        help="Context json file",
        default="./data/train.json",
    )
    parser.add_argument(
        "--valid_file",
        type=Path,
        help="Validation json file",
        default="./data/valid.json",
    )
    # data
    parser.add_argument("--max_length", type=int, default=384, help="Tokenize max length")

    # model
    # parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)
    # parser.add_argument(
    #     "--max_train_steps",
    #     type=int,
    #     default=None,
    #     help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    # )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    # data loader
    parser.add_argument("--batch_size", type=int, default=2)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=3)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    args.model_path.mkdir(parents=True, exist_ok=True)
    
    train()
    
    