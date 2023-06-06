np=$1
clip=$2
pergpu_bs=$3
accumu=$4
lr=$5
python run_glue_no_trainer.py \
  --model_name_or_path bert-base-uncased \
  --train_file ../data/sst2/filter_train_nonewline.json \
  --validation_file ../data/sst2/filter_train_nonewline.json \
  --max_length 128 \
  --per_device_train_batch_size $pergpu_bs \
  --gradient_accumulation_steps $accumu \
  --learning_rate $lr \
  --num_train_epochs 3 \
  --seed 42 --lr_scheduler_type constant \
  --output_dir ./sst2/domain_classifier_output --fp16 --noise_multiplier $np --dp_grad_clip $clip --filter_sentences --apply_lora --lora_r 32 --lora_alpha 16