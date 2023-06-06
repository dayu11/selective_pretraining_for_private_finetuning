from datasets import Dataset
import torch
import numpy as np
import time

import argparse
import os

parser = argparse.ArgumentParser(description='data selection for sst2 based on logits')


parser.add_argument('--num_tokens', default=40, type=int, help='how many tokens (in Million) to select')
parser.add_argument('--random', action='store_true', help='randomly select sentences')
args = parser.parse_args()




path_to_save = f'../data/sst2/'

os.makedirs(path_to_save, exist_ok=True)

wikipedia_book_sentences = Dataset.load_from_disk('../data/flatten_wiki_book_sentences.ds')['text']

num_sentences = len(wikipedia_book_sentences)
# total number of sentences
print(f'total num sentences: {num_sentences/1e3:.2f}K')

if args.random:
    idx = np.arange(num_sentences)
    np.random.shuffle(idx)
    sorted_idx = np.array(idx)
else:
    logits = torch.load('../dp_finetuning/sst2/domain_classifier_output/flatten_wiki_book_sentences.ds_filtering_logits.tsr')
    probs = torch.softmax(logits, dim=1)
    pos_probs = probs[:, 1]
    sorted_idx = torch.argsort(pos_probs, descending=True).cpu()

 
start = time.time()
print(f'start filtering for sst2')


keep_sentences = []


num_tokens_to_keep = args.num_tokens * 1e6
num_selected_tokens = 0
for cnt, idx in enumerate(sorted_idx):

    _current_sentence = wikipedia_book_sentences[idx]

    # remove sentences that are too short
    length_checker = len(_current_sentence.split())
    smallest_length = 6
    if(length_checker < smallest_length):
        continue

    keep_sentences.append(_current_sentence)
    num_selected_tokens += length_checker

    # print progress and time every 1000K sentences
    if(cnt % 1e6 == 0):
        print(f'finshed {cnt/1e6:.2f}M sentences, time elapsed {time.time()-start:.2f}s')

    if(num_selected_tokens > num_tokens_to_keep):
        break


# shuffle selected sentences
np.random.shuffle(keep_sentences)


print(f'selected {len(keep_sentences)/1e3:.2f}K sentences')

dataset = Dataset.from_dict({'text': keep_sentences})
if args.random:
    path_to_save = f'../data/sst2/pretraining_data_random_{args.num_tokens}m.ds'
else:
    path_to_save = f'../data/sst2/pretraining_data_{args.num_tokens}m.ds'

# save the dataset
dataset.save_to_disk(path_to_save)


current = time.time()
print(f'finished in {current-start:.2f} seconds')
