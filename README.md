# Learning to Generate Explanation from e-Hospital Services for Medical Suggestion
This is the official repository for our paper [*Learning to Generate Explanation from e-Hospital Services for Medical Suggestion*](https://aclanthology.org/2022.coling-1.260/), *COLING* 2022.

### [Update 10/28] Add code implementation! :tada: (Readme will be updated shortly)
---
## Dataset
You can download our dataset from [here](https://drive.google.com/u/0/uc?id=1yB933kGHt-ai45H5rxOfEfsM2LlDAn6r&export=download).

The folder structure for the dataset is shown below, we saperate suggestion (recmd) and explanation (expln) in test data of R2 and R3 for convinience to perform evaluation.

  ```
  data
  ├── train
  │   ├── R1_train.csv
  │   ├── R2_train.csv
  │   └── R3_train.csv
  │
  └── test
      ├── R1_test.csv
      ├── R2_test.csv
      ├── R3_test.csv
      │
      ├── recmd
      │   ├── R2_test_recmd.csv
      │   └── R3_test_recmd.csv
      │
      └── expln
          ├── R2_test_expln.csv
          └── R2_test_expln.csv
  ```
 
- Note
  > The dataset is originally collected from [here](https://sp1.hso.mohw.gov.tw/doctor/).


## Code
See ```scripts/``` for our code.
All training and evaluation can be done by the setting the correct arguments in ```train.sh``` and run
```
bash train.sh
```
### How to set arguments
- Train and Evaluate mT5
  
  ```
  python train.py \
      --model_name_or_path google/mt5-base \
      --train_data_path train/R1_train.csv \
      --valid_data_path test/R1_test.csv \
      --save_path results \
      --do_train \
      --do_eval \
      --seed 42 \
      --num_train_epochs 3 \
      --per_device_train_batch_size 8 \
      --per_device_eval_batch_size 32 \
      --learning_rate 1e-4 \
      --warmup_ratio 0.15 \
      --weight_decay 0.01 \
      --save_strategy epoch \
      --logging_strategy steps \
      --logging_steps 500 \
      --evaluation_strategy epoch \
      --metric_for_best_model rouge-1 \
      --save_total_limit 1 \
      --load_best_model_at_end \
      --predict_with_generate True \
      --generation_num_beams 2 \
  ```
- Train and Evaluate MTL mT5
  
  ```
  python train.py \
      --model_name_or_path google/mt5-base \
      --train_data_path train/R2_train.csv \
      --valid_data_path test/R2_test.csv \
      --save_path results \
      --do_train \
      --do_eval \
      --seed 42 \
      --num_train_epochs 3 \
      --per_device_train_batch_size 8 \
      --per_device_eval_batch_size 32 \
      --learning_rate 1e-4 \
      --warmup_ratio 0.15 \
      --weight_decay 0.01 \
      --save_strategy epoch \
      --logging_strategy steps \
      --logging_steps 500 \
      --evaluation_strategy epoch \
      --metric_for_best_model rouge-1 \
      --save_total_limit 1 \
      --load_best_model_at_end \
      --predict_with_generate True \
      --generation_num_beams 2 \
  ```
- Train and Evaluate DMTL mT5

  ```
  python train.py \
      --model_name_or_path google/mt5-base \
      --train_data_path train/R3_train.csv \
      --valid_data_path test/R3_test.csv \
      --save_path results \
      --do_train \
      --do_eval \
      --seed 42 \
      --num_train_epochs 5 \
      --per_device_train_batch_size 8 \
      --per_device_eval_batch_size 32 \
      --learning_rate 1e-4 \
      --warmup_ratio 0.15 \
      --weight_decay 0.01 \
      --save_strategy epoch \
      --logging_strategy steps \
      --logging_steps 500 \
      --evaluation_strategy epoch \
      --metric_for_best_model rouge-1 \
      --save_total_limit 1 \
      --load_best_model_at_end \
      --predict_with_generate True \
      --generation_num_beams 2 \
  ```
Our arguments are directly compatible with [the huggingface Seq2SeqTrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments), you can adjust according to your own need. E.g., (un)set ```--do_train``` or ```do_eval``` if you only want to train or evaluate trained model.

