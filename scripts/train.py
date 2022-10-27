import os
import time
from datetime import datetime
import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
import transformers
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed
)

from custom_args import (
    DataArguments,
    ModelArguments
)
from utils import (
    set_other_seeds,
    seq2seqDataset
)
from weightedSeq2SeqTrainer import weightedSeq2SeqTrainer

from tw_rouge import get_rouge
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

# Set up logger
logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_info()

# Get args
parser = HfArgumentParser(
    (ModelArguments, DataArguments, Seq2SeqTrainingArguments)
)
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Set up wandb
if training_args.report_to == 'wandb':
    import wandb
    wandb.login()

# Set seed
set_seed(training_args.seed)
set_other_seeds(training_args.seed)

# load model, tokenizer
tokenizer = MT5Tokenizer.from_pretrained(model_args.model_name_or_path)
model = MT5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

training_args.generation_max_length = data_args.max_target_length

# Compute metrics for mT5, MTL mT5 
def compute_metrics(eval_pred):
    
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Compute rouge metrics
    results = get_rouge(decoded_preds, decoded_labels, avg=True, ignore_empty=True)

    # Extract f-measure results
    results = {key: value['f']*100 for key, value in results.items()}
        
    # Add mean generated length to results
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    results["gen_len"] = np.mean(prediction_lens)        

    # Round all results
    results = {k: round(v, 4) for k, v in results.items()}
            
    return results

# Compute metrics for DMTL mT5 
def med_atten_compute_metrics(eval_pred):
    
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Compute rouge metrics
    results = get_rouge(decoded_preds, decoded_labels, avg=True, ignore_empty=True)

    # Extract f-measure results
    results = {key: value['f']*100 for key, value in results.items()}
        
    # Add mean generated length to results
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    results["gen_len"] = np.mean(prediction_lens)        

    # Compute pecision, recall, f-score, and accuracy
    
    expln_decoded_preds = []
    expln_decoded_labels = []
    
    for pred, label in zip(decoded_preds, decoded_labels):
        if label[-1] == '0' or label[-1] == '1':
            expln_decoded_preds.append(pred)
            expln_decoded_labels.append(label)

    preds_binary = np.array([pred[-1] for pred in expln_decoded_preds])
    labels_binary = np.array([label[-1] for label in expln_decoded_labels])

    prfs = precision_recall_fscore_support(labels_binary, preds_binary, labels=['1', '0'])
    acc = accuracy_score(labels_binary, preds_binary)

    # Add pecision, recall, f-score, and accuracy to results
    results['accuracy'] = acc
    results['precision'] = prfs[0][0]
    results['recall'] = prfs[1][0]
    results['fscore'] = prfs[2][0]

    # Round all results
    results = {k: round(v, 4) for k, v in results.items()}
            
    return results

# Prepare dataset
if training_args.do_train:
    train_df = pd.read_csv(data_args.train_data_path)
    train_dataset = seq2seqDataset(
        train_df,
        tokenizer,
        data_args.max_source_length,
        data_args.max_target_length
    )

if training_args.do_eval:
    valid_df = pd.read_csv(data_args.valid_data_path)
    valid_dataset = seq2seqDataset(
        valid_df,
        tokenizer,
        data_args.max_source_length,
        data_args.max_target_length
    )

# For DMTL mT5, we use a custom Trainer with:
#   1. modified weighted loss
#   2. custom scheduler
# for the LM head of the binary label (l) when generating explanations
if model_args.discourse_aware:
    trainer = weightedSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=valid_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=med_atten_compute_metrics,
    )
else:
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=valid_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

if training_args.do_train:
    results = trainer.train()
    metrics = results.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

elif training_args.do_eval:
    results = trainer.evaluate()
    metrics = results.metrics
    metrics["eval_samples"] = len(train_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

trainer.save_state()
