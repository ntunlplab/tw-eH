'''
This code is based on https://github.com/allenai/label_rationale_association/blob/main/custom_args.py
'''
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: str = field(
        default='google/mt5-base',
        metadata={
            "help": "HF pretrained model"
        },
    )

    discourse_aware: bool = field(
        default=False,
        action='store_true',
        metadata={
            "help": "Set to True when using medical attention dataset (R3)"
        }
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    # task_name: str = field(
    #     metadata={"help": "The name of the task to train on."}
    # )

    train_data_path: str = field(
        required=True,
        meta_data={"help": "Path to training data csv file"}
    )

    valid_data_path: str = field(
        required=True,
        meta_data={"help": "Path to evaluation data csv file"}
    )

    max_source_length: int = field(
        default=512,
        metadata={"help": "max length for input text"},
    )

    max_target_length: int = field(
        default=128,
        metadata={"help": "max length for reference text"},
    )