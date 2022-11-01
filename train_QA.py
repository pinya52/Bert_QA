import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import sys
from dataclasses import dataclass, field
from typing import Optional

from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import trange, tqdm
from matplotlib import pyplot as plt


import datasets
from datasets import DatasetDict, load_dataset, Dataset, load_metric

# import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from utils import postprocess_qa_predictions, create_and_fill_np_array

TRAIN = "train"
DEV = "valid"
SPLITS = [TRAIN, DEV]


def read_dataset():
    
    with open('./data/context.json', encoding="utf-8") as f:
        context = json.load(f)
    
    # load train and valid json
    dataset_dict = dict()
    with open('./data/train.json', encoding="utf-8") as f:
        tmp_json = json.load(f)
        pd_dict_train = pd.DataFrame.from_dict(tmp_json)
        for idx, data in enumerate(tmp_json):
            tmp_json[idx]["context"] = context[data["relevant"]]

        pd_dict_train = pd.DataFrame.from_dict(tmp_json)
                
    with open('./data/valid.json', encoding="utf-8") as f:
        tmp_json = json.load(f)
        for idx, data in enumerate(tmp_json):
            tmp_json[idx]["context"] = context[data["relevant"]]
            
        pd_dict_val = pd.DataFrame.from_dict(tmp_json)

    pd_dataset_train = Dataset.from_pandas(pd_dict_train)
    pd_dataset_val = Dataset.from_pandas(pd_dict_val)
    
    dataset_dict["train"] = pd_dataset_train
    dataset_dict["valid"] = pd_dataset_val
    dataset = DatasetDict(dataset_dict)
    
    return dataset   


def preprocess(train_dataset, val_dataset, tokenizer):
    
    def preprocess_train(dataset):
        # Train preprocessing
        questions = [q.lstrip() for q in dataset["question"]]
        contexts = dataset["context"]
        
        tokenized_dataset = tokenizer(
            questions,
            contexts,
            max_length=args.max_length,
            stride=args.doc_stride,
            truncation="only_second",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            padding=True,
        )
        
        offset_mapping = tokenized_dataset.pop("offset_mapping")
        sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")

        # label those examples
        tokenized_dataset["start_positions"] = []
        tokenized_dataset["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_dataset["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_dataset.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = dataset["answer"][sample_index]
            # If no answers are given, set the cls_index as answer.
            # if len(answers["start"]) == 0:
            if not isinstance(answers["start"], int):
                tokenized_dataset["start_positions"].append(cls_index)
                tokenized_dataset["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["start"]
                end_char = start_char + len(answers["text"])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_dataset["start_positions"].append(cls_index)
                    tokenized_dataset["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_dataset["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_dataset["end_positions"].append(token_end_index + 1)

        return tokenized_dataset
    
    def prepare_validation(dataset):
        # Validation preprocessing
        questions = [q.lstrip() for q in dataset["question"]]
        contexts = dataset["context"]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_dataset = tokenizer(
            questions,
            contexts,
            # examples[question_column_name if pad_on_right else context_column_name],
            # examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second",
            max_length=args.max_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        # offset_mapping = tokenized_examples.pop("offset_mapping")
        offset_mapping = tokenized_dataset["offset_mapping"]
        sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")

        # label those examples
        tokenized_dataset["start_positions"] = []
        tokenized_dataset["end_positions"] = []
        
        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_dataset["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_dataset.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = dataset["answer"][sample_index]
            # If no answers are given, set the cls_index as answer.
            # if len(answers["start"]) == 0:
            if not isinstance(answers["start"], int):
                tokenized_dataset["start_positions"].append(cls_index)
                tokenized_dataset["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["start"]
                end_char = start_char + len(answers["text"])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_dataset["start_positions"].append(cls_index)
                    tokenized_dataset["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_dataset["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_dataset["end_positions"].append(token_end_index + 1)
        
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_dataset["example_id"] = []

        for i in range(len(tokenized_dataset["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_dataset.sequence_ids(i)
            context_index = 1

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_dataset["example_id"].append(dataset["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_dataset["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_dataset["offset_mapping"][i])
            ]

        return tokenized_dataset
    
    tokenized_train_dataset = train_dataset.map(preprocess_train, batched=True, remove_columns=train_dataset.column_names)
    tokenized_valid_dataset = val_dataset.map(prepare_validation, batched=True, remove_columns=val_dataset.column_names)

    return tokenized_train_dataset, tokenized_valid_dataset


def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        null_score_diff_threshold=args.null_score_diff_threshold,
        output_dir=args.output_dir,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": {"text": [ex["answer"]["text"]], "answer_start": [ex["answer"]["start"]]}} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def plot_curve(curve, title, legend, output_name):
    plt.plot(curve)
    plt.legend([legend])
    plt.title(title)
    plt.xlabel("Epoch")
    plt.xticks([epoch for epoch in range(args.num_epoch) if epoch % 10 == 0])
    plt.savefig(args.plot_dir / output_name)
    plt.clf()


def plot_EM_loss_curve(em_curve, loss_curve, model_name):
    plot_curve(em_curve, "EM curve", "exact match", "EM_CURVE%s.png"%(model_name))
    plot_curve(loss_curve, "Loss curve", "Loss", "LOSS_CURVE%s.png"%(model_name))


def main(args):
    print('\nModel using now: %s\n'%(args.model_name))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print('Reading Dataset\n')
    dataset = read_dataset()
    model_name = str(args.model_name).replace('/', '_')

    print('Preprocessing Raw Data\n')
    train_dataset, eval_dataset = preprocess(dataset['train'], dataset['valid'], tokenizer)
    tokenizer_path = str(args.tokenizer_path) + '/%s'%(model_name)
    tokenizer.save_pretrained(tokenizer_path)
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size)
    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader(eval_dataset_for_model, shuffle=False, collate_fn=default_data_collator, batch_size=args.batch_size)

    print('Prepare for training\n')
    model = AutoModelForQuestionAnswering.from_pretrained('bert-base-chinese')
    model.resize_token_embeddings(len(tokenizer))
    model.to('cuda')

    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    total_step = len(train_dataset) * args.num_epoch // (args.batch_size * args.accum_steps)
    warmup_step = total_step * 0.06
    
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_step, total_step)
    metric = load_metric("squad")
    
    print("Start Training")
    best_exact_match = 0.0
    best_dev_loss = float('inf')
    EM_curve = []
    loss_curve = []
    for epoch in range(args.num_epoch):
        model.train()
        print(f"\nEpoch: {epoch+1} / {args.num_epoch}")
        for batch_step, batch_datas in enumerate(tqdm(train_dataloader, desc="Train")):
            # input_ids, token_type_ids, attention_mask, labels = batch_datas.values()
            input_ids, token_type_ids, attention_mask, start_pos, end_pos = [b_data.to(args.device) for b_data in batch_datas.values()]

            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, start_positions=start_pos, end_positions=end_pos)
            loss = outputs.loss
            
            # train_loss += loss.detach().float()
            loss = loss / args.accum_steps
            loss.backward()

            # # Choose the most probable start position / end position
            # start_index = torch.argmax(outputs.start_logits, dim=1)
            # end_index = torch.argmax(outputs.end_logits, dim=1)

            if batch_step % args.accum_steps == 0 or batch_step == len(train_dataloader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        model.eval()
        dev_loss = 0.0
        all_start_logits = []
        all_end_logits = []
        for batch_step, batch_datas in enumerate(tqdm(eval_dataloader, desc="Valid")):
            with torch.no_grad():
                # input_ids, token_type_ids, attention_mask, labels = batch_data.values()
                input_ids, token_type_ids, attention_mask, start_pos, end_pos = [b_data.to(args.device) for b_data in batch_datas.values()]

                outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, start_positions= start_pos, end_positions=end_pos)
                loss = outputs.loss
                dev_loss += loss.detach().item()
                
                # outputs start end
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                all_start_logits.append(start_logits.cpu().numpy())
                all_end_logits.append(end_logits.cpu().numpy())


        max_len = max([x.shape[1] for x in all_start_logits])
        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)
        
        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits
        outputs_numpy = (start_logits_concat, end_logits_concat)
        
        prediction = post_processing_function(dataset["valid"], eval_dataset, outputs_numpy)
        predict_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        dev_loss /= (batch_step * args.accum_steps)

        loss_curve.append(dev_loss)
        EM_curve.append(predict_metric["exact_match"])
        print("eval matrix:", predict_metric)
        print("loss:", dev_loss)
        
        plot_EM_loss_curve(em_curve=EM_curve, loss_curve=loss_curve, model_name=model_name)

        save_name = str(args.model_path) + '/%s'%(model_name)
        model.save_pretrained(save_name)

        if best_exact_match < predict_metric["exact_match"]:
            best_exact_match = predict_metric["exact_match"]
            if args.model_path is not None:
                save_name = str(args.model_path) + '/%s_em'%(model_name)
                model.save_pretrained(save_name)
        
        if best_dev_loss < dev_loss:
            best_dev_loss = dev_loss
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
    # parser.add_argument(
    #     "--tokenizer_name",
    #     type=str,
    #     help="Tokenizer name",
    #     default="bert-base-chinese",
    #     # default="hfl/chinese-roberta-wwm-ext",
    #     # default='luhua/chinese_pretrain_mrc_macbert_large'
    # )
    parser.add_argument(
        "--model_path",
        type=Path,
        help="Directory to save the model.",
        default="./ckpt/QA/models/",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        help="Path to save the tokenizer.",
        default="./ckpt/tokenizer/QA",
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
    parser.add_argument(
        "--plot_dir",
        type=Path,
        help="Directory to store EM Loss Curve.",
        default="./plot/"
    )
    
    # data
    parser.add_argument("--max_length", type=int, default=512, help="The maximum length of a feature (question and context)")
    parser.add_argument("--doc_stride", type=int, default=192, help="The authorized overlap between two part of the context when splitting it is needed.")

    # model
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    # data loader
    parser.add_argument("--batch_size", type=int, default=4)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=3)
    
    # post processing
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="The threshold used to select the null answer: if the best answer has a score that is less than "
        "the score of the null answer minus this threshold, the null answer is selected for this example. "
        "Only useful when `version_2_with_negative=True`.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=50,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print()
    # args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    main(args)