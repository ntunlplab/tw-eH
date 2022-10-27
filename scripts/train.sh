export CUDA_VISIBLE_DEVICES=1

python train.py \
    --model_name_or_path google/mt5-base \
    --discourse_aware \
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
    # --report_to wandb