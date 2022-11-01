from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import logging
from pathlib import Path
from argparse import ArgumentParser, Namespace
import json
import math

from datasets import DatasetDict, load_dataset, Dataset
from transformers import (AutoModelForMultipleChoice, AutoModelForQuestionAnswering, AutoTokenizer,
                          default_data_collator, 
                          get_cosine_schedule_with_warmup,
                          SchedulerType, get_linear_schedule_with_warmup, pipeline)
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np
import pandas as pd


def read_dataset():
    # load Context json
    with open(args.context_json, encoding="utf-8") as f:
        context = json.load(f)
    
    # load test json
    with open(args.test_json, encoding="utf-8") as f:
        tmp_json = json.load(f)
        pd_dict = pd.DataFrame.from_dict(tmp_json)

    dataset = Dataset.from_pandas(pd_dict)
    # dataset = Dataset.from_dict(tmp_json)
    return context, dataset


def preprocess_multiple_choice(dataset, context, model_name):
    def preprocess_function(examples):
        """
        The preprocessing function needs to do:

        1. Make four copies of the sent1 field so you can combine each of them with sent2 to 
        recreate how a sentence starts.
        2. Combine sent2 with each of the four possible sentence endings.
        3. Flatten these two lists so you can tokenize them, and then unflatten them afterward 
        so each example has a corresponding input_ids, attention_mask, and labels field.
        """
        first_sentences = [[question]*4 for question in examples["question"]]
        second_sentences = [[context[idx] for idx in idxs] for idxs in examples["paragraphs"]]

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        
        tokenized_examples = tokenizer(first_sentences, second_sentences, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=True, max_length=args.max_seq_len)
        tokenized_inputs = {k:[v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        return tokenized_inputs

    tokenizer = str(args.tokenizer) + '/%s'%(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    return tokenizer, tokenized_dataset

def collate_fn(features: List[Dict]):
    first = features[0]
    batch = {}

    for k, v in first.items():
        if k not in ("id", "question") and v is not None:
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ("id", "question") and v is not None:
            batch[k] = [f[k] for f in features]

    return batch

def main():
    print('\nModel using now: %s\n'%(args.model_name))
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    print('Reading Dataset\n')
    context, dataset = read_dataset()
    model_name = str(args.model_name).replace('/', '_')
    print('./pred/%s.csv'%(model_name))
    
    print('Preprocessing Raw Data\n')
    tokenizer, processed_dataset =  preprocess_multiple_choice(dataset, context, model_name)

    # data_collator = default_data_collator
    dataloader = DataLoader(processed_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)

    QA_model = str(args.QA_model) + '/%s'%(model_name)
    MC_model = str(args.MC_model) + '/%s'%(model_name)
    model_QA = AutoModelForQuestionAnswering.from_pretrained(QA_model).to(args.device)
    model_MC = AutoModelForMultipleChoice.from_pretrained(MC_model).to(args.device)

    device = -1 if args.device == "cpu" else 0
    pipe = pipeline("question-answering", model=model_QA, tokenizer=tokenizer, device=device)

    model_QA.eval()
    model_MC.eval()

    ans_dict = {}
    index = []
    answers = []
    print('\nStart Predicting')
    for batch_step, batch_data in enumerate(tqdm(dataloader, desc="Test")):
        with torch.no_grad():
            ids, questions, paragraphs, input_ids, token_type_ids, attention_mask = batch_data.values()
            input_ids, token_type_ids, attention_mask = input_ids.to(args.device), token_type_ids.to(args.device), attention_mask.to(args.device)

            outputs = model_MC(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            predictions = outputs.logits.argmax(dim=-1)
            
            qa_context = [context[paragraph[pred]] for paragraph, pred in list(zip(paragraphs.numpy(), predictions.cpu().numpy()))]

            ans = pipe(question=questions, context=qa_context, max_seq_len=args.max_seq_len, doc_stride=args.doc_stride, max_question_len=args.max_question_len, max_answer_len=args.max_answer_len)
            for i, id in enumerate(ids):
                index.append(id)
                if isinstance(ans, List):
                    # ans_dict[id] = ans[i]["answer"]
                    answers.append(ans[i]["answer"])
                else:
                    # ans_dict[id] = ans["answer"]
                    answers.append(ans["answer"])
    
    # write prediction to file (args.pred_file)
    ans_dict = {"id": index, "answer": answers}
    args.pred_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(ans_dict).to_csv('./pred/%s.csv'%(model_name), index=False)

    # ans = pipe(doc_stride=args.doc_stride, max_seq_len=args.max_length, question=["舍本和誰的數據能推算出連星的恆星的質量？","在關西鎮以什麼方言為主？"], context=["在19世紀雙星觀測所獲得的成就使重要性也增加了。在1834年，白塞爾觀測到天狼星自行的變化，因而推測有一顆隱藏的伴星；愛德華·皮克林在1899年觀測開陽週期性分裂的光譜線時發現第一顆光譜雙星，週期是104天。天文學家斯特魯維和舍本·衛斯里·伯納姆仔細的觀察和收集了許多聯星的資料，使得可以從被確定的軌道要素推算出恆星的質量。第一個獲得解答的是1827年由菲利克斯·薩瓦里透過望遠鏡的觀測得到的聯星軌道。對恆星的科學研究在20世紀獲得快速的進展，相片成為天文學上很有價值的工具。卡爾·史瓦西發現經由比較視星等和攝影星等的差別，可以得到恆星的顏色和它的溫度。1921年，光電光度計的發展可以在不同的波長間隔上非常精密的測量星等。阿爾伯特·邁克生在虎克望遠鏡第一次使用干涉儀測量出恆星的直徑。","新竹縣是中華民國臺灣省的縣，位於臺灣本島西北部，北臨桃園市，南接苗栗縣，東南以雪山山脈與宜蘭縣、臺中市相連，西部面向台灣海峽，西接與新竹市交界。全縣總面積約1,427平方公里，除鳳山溪、頭前溪中下游沖積平原外，其餘大多為丘陵、台地及山地。早期新竹縣郊區多務農，1970年代工業技術研究院創設於新竹市，1980年代新竹科學工業園區設立於新竹市東區及新竹縣寶山鄉，1990年代位於湖口鄉的新竹工業區也逐漸從傳統產業聚落轉型為新興高科技產業聚落，使得新竹縣成為北台灣的高科技產業重鎮，而人口也在近幾年急速增加。本縣方言於絕大部分地區使用海陸客家話，竹北市及新豐鄉沿海地區部分使用泉州腔閩南話較多，關西鎮及峨眉鄉部分使用四縣腔客家話為主。"])
    # print(ans)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--QA_model",
        type=Path,
        help="Question Answering model name or path",
        default="./ckpt/QA/models/",
        # default="./ckpt/QA/hfl_chinese-roberta-wwm-ext/"
    )
    parser.add_argument(
        "--MC_model",
        type=Path,
        help="Multiple Choice model name or path",
        default="./ckpt/MC/models/",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name",
        default="bert-base-chinese",
        # default="hfl/chinese-roberta-wwm-ext",
        # default='luhua/chinese_pretrain_mrc_macbert_large'
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Tokenizer name",
        default="./ckpt/tokenizer/QA/",
    )
    parser.add_argument(
        "--context_json",
        type=Path,
        help="Path of context.json.",
        default="./data/context.json",
    )
    parser.add_argument(
        "--test_json",
        type=Path,
        help="Path of test.json",
        default="./data/test.json",
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        help="Path of prediction file.",
        default="./pred.csv",
    )

    
    # data
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--max_answer_len", type=int, default=50, help="Max answer length")
    parser.add_argument("--max_question_len", type=int, default=35, help="Max question length")
    parser.add_argument("--doc_stride", type=int, default=192, help="The authorized overlap between two part of the context when splitting it is needed.")

    # model
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")

    # # optimizer
    # parser.add_argument("--lr", type=float, default=3e-5)
    # parser.add_argument(
    #     "--max_train_steps",
    #     type=int,
    #     default=None,
    #     help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    # )
    # parser.add_argument(
    #     "--accum_steps",
    #     type=int,
    #     default=4,
    #     help="Number of updates steps to accumulate before performing a backward/update pass.",
    # )

    # data loader
    parser.add_argument("--batch_size", type=int, default=4)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=3)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main()
    
    