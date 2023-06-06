train_file=$1
model_size=$2
lr=$3
max_steps=$4
per_gpu_batch_size=$5
numgpus=$6
accumu=$7
output_dir='results/'$train_file'_lr'$lr'_maxsteps'$max_steps'_'$model_size
python -m torch.distributed.launch --nproc_per_node $numgpus --nnodes=1 run_pretraining.py \
    --model_type bert \
    --tokenizer_name bert-base-uncased \
    --model_size=$model_size \
    --train_file ../data/sst2/$train_file \
    --per_device_train_batch_size $per_gpu_batch_size \
    --per_device_eval_batch_size 8 \
    --do_train \
    --save_steps 10000 \
    --output_dir $output_dir \
    --max_seq_length 512 --gradient_accumulation_steps $accumu  --fp16 --learning_rate $lr --warmup_ratio 0.06 --weight_decay 0.01 --adam_beta2 0.98 --adam_epsilon 1e-6 --max_grad_norm 0. --max_steps $max_steps --ddp_timeout 16200 --dataloader_num_workers 1 \
