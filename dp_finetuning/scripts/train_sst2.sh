model=$1
np=$2
clip=$3
pergpu_bs=$4
accumu=$5
lr=$6
epochs=$7
seed=$8
python run_glue_no_trainer.py \
  --model_name_or_path $model \
  --task_name sst2 \
  --max_length 512 \
  --per_device_train_batch_size $pergpu_bs \
  --gradient_accumulation_steps $accumu \
  --learning_rate $lr \
  --num_train_epochs $epochs \
  --seed $seed \
  --output_dir sst2/dp_finetune_output/$model-lr$lr-seed$seed --fp16 --noise_multiplier $np --dp_grad_clip $clip