from datasets import load_dataset, Dataset
import torch
import numpy as np
import argparse
import json

import os

import string

parser = argparse.ArgumentParser(description='data filtering for GLUE')

parser.add_argument('--od_data_size', default=5, type=float, help='how many times od data is larger than id data')
parser.add_argument('--output_dir', default='../data', type=str, help='output dictionary')

args = parser.parse_args()


def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))


def get_indomain_sentences(dataset):

    train_examples = dataset['train']
    in_domain_sentences = train_examples['sentence']

    valid_examples = dataset['validation']
    test_examples = dataset['test']
    in_domain_val_sentences = valid_examples['sentence'] + test_examples['sentence']

    return in_domain_sentences, in_domain_val_sentences

os.makedirs("/".join([args.output_dir, 'sst2']), exist_ok=True)

in_domain_dataset = load_dataset("glue", 'sst2')

out_domain_dataset = Dataset.load_from_disk('../data/flatten_wiki_book_sentences.ds')

# print number of sentences in out of domain dataset
print('Number of out domain sentences, ', len(out_domain_dataset['text']))
in_domain_sentences, in_domain_val_sentences = get_indomain_sentences(in_domain_dataset)

# out domain data before sampling
full_out_domain_sentences = out_domain_dataset['text']


id_data_size = len(in_domain_sentences)
od_data_size = int(args.od_data_size * id_data_size)

id_val_data_size = len(in_domain_val_sentences)
od_val_data_size = int(args.od_data_size * id_val_data_size)

## select subset from od dataset
full_od_data_size = len(full_out_domain_sentences)

## select subset from pretraining data
selected_idx = np.random.choice(full_od_data_size, od_data_size + od_val_data_size, replace=False)

selected_out_domain_sentences = []
for idx in selected_idx[0:od_data_size]:
    selected_out_domain_sentences.append(full_out_domain_sentences[idx])

selected_out_domain_val_sentences = []
for idx in selected_idx[od_data_size:]:
    selected_out_domain_val_sentences.append(full_out_domain_sentences[idx])


# combine in domain and out domain data + create labels
all_sentences = in_domain_sentences + selected_out_domain_sentences
pos_data_size = len(in_domain_sentences)
neg_data_size = len(selected_out_domain_sentences)
full_size = pos_data_size + neg_data_size
full_label = np.ones(pos_data_size).astype(np.int32).tolist() + np.zeros(neg_data_size).astype(np.int32).tolist()

print('number of postive/negative training examples: ', pos_data_size, neg_data_size)

all_val_sentences = in_domain_val_sentences + selected_out_domain_val_sentences
pos_val_data_size = len(in_domain_val_sentences)
neg_val_data_size = len(selected_out_domain_val_sentences)
full_val_size = pos_val_data_size + neg_val_data_size
full_val_label = np.ones(pos_val_data_size).astype(np.int32).tolist() + np.zeros(neg_val_data_size).astype(np.int32).tolist()

print('number of postive/negative validation examples: ', pos_val_data_size, neg_val_data_size)



# save to disk
train_file_name = "/".join([args.output_dir, 'sst2', 'filter_train_nonewline.json'])
val_file_name = "/".join([args.output_dir, 'sst2', 'filter_val_nonewline.json'])

train_data_dict = []
for i in  range(full_size):
    _cur_dict = {}
    _cur_dict['sentence'] = remove_punctuation(all_sentences[i])
    _cur_dict['label'] = full_label[i]
    train_data_dict.append(_cur_dict)



with open(train_file_name, "w") as outfile:
    json.dump(train_data_dict, outfile)

validation_data_dict = []
for i in range(full_val_size):
    _cur_dict = {}
    _cur_dict['sentence'] = remove_punctuation(all_val_sentences[i])
    _cur_dict['label'] = full_val_label[i]
    validation_data_dict.append(_cur_dict)

with open(val_file_name, "w") as outfile:
    json.dump(validation_data_dict, outfile)
