This repo implements our framework on SST-2. Please see the example commands below for details.
The code requires GPUs that support half-precision training. Please first install the following python packages.

torch==1.12.1
datasets
evaluate
accelerate
tqdm


The first stage: selective pre-training.

1. First download Wikipedia and BookCorpus and split the documents into natural sentences. The resulting dataset is saved at "data/flatten_wiki_book_sentences.ds".

cd pretraining_data_selection
python splitting_into_sentences.py

2. Prepare the data for training the domain classifier. The results are saved at "data/sst2/filter_train_nonewline.json" and "data/sst2/filter_val_nonewline.json".

python build_classifier_data_for_sst2.py

3. Train the domain classifier with DP-Adam and compute the logits of the trained classifier on all pre-training sentences. The logits are saved at "dp_finetuning/sst2/domain_classifier_output".

cd ../dp_finetuning
# Please install the local transformers directory, in which we implement per-example gradient computation ("transformers/src/transformers/models/grad_sample_utils.py"), 
# and clipping + noising ("line 709-744 in dp_finetuning/run_glue_no_trainer.py").
# This implementation only supports single GPU training for dp fine-tuning
cd transformers
pip install --editable .
cd ..

# The arguments are: noise_multiplier, clip, pergpu_bs, gradient accumulation steps, and learning rate.
# We compute the logits in the function 'filter_pretraining_data' in "run_glue_no_trainer.py".
# In this implementation, only 1 GPU is used for training the domain classifier and computing the logits.
bash scripts/train_domain_classifier.sh 1.4 1 64 32 1e-3

4. Use the logits to select pre-training sentences.

cd ../pretraining_data_selection
# The argument is target number of pre-training tokens (in M).
python sampling_with_logits.py --num_tokens 40

# You can also select random sentences.
python sampling_with_logits.py --num_tokens 40 --random

5. Pre-training on selected data.

cd ../pretraining
# Install standard transformers package, i.e., without pre-example gradients computation.
# This also enables pre-training with multi gpus.
cd transformers
pip install --editable .
cd ..

# The arguments are: pre-training data path, model size (tiny=5M), lr, pre-training steps, per-gpu-bs, num_gpus, gradient accumulation.
bash scripts/pretraining.sh pretraining_data_40m.ds tiny 3e-4 1000000 32 8 1

# You can also pre-train on random data.
bash scripts/pretraining.sh pretraining_data_random_40m.ds tiny 3e-4 1000000 32 8 1


Finally, the second stage, private fine-tuning.

6. Private fine-tuning on sst-2

# Don't forget installing dp enabled transformers package.
cd ../dp_finetuning
cd transformers
pip install --editable .
cd ..

# The argumentments are: pre-trained model path, noise_multiplier, clip, pergpu_bs, gradient accumulation, lr, epochs, seed.
# Replace checkpoint-XXXX with your checkpoint.
bash scripts/train_sst2.sh ../pretraining/results/pretraining_data_random_40m.ds_lr3e-4_maxsteps1000000_tiny/checkpoint-XXXX 1.4 1 32 64 1e-3 30 0



