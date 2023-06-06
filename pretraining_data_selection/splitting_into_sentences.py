
# this code splits the documents in Wikipedia and BookCorpus into natural sentences


import datasets
import os
import nltk.data

os.system('python -m nltk.downloader punkt')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

wikipedia = datasets.load_dataset('wikipedia', '20220301.en', split='train')
bookcorpus = datasets.load_dataset('bookcorpus', split='train')

bookcorpus_sentences = bookcorpus['text']
wikipedia_documents = wikipedia['text']
wikipedia_sentences = []
for i in range(len(wikipedia_documents)):
    sentences = [s.lower() for s in tokenizer.tokenize(wikipedia_documents[i]) if len(s) > 1]
    wikipedia_sentences.extend(sentences)
    if(i % 1000 == 0):
        print('splitting wikipedia document into sentences, ', (i/len(wikipedia_documents))*100, '% documents processed')


wiki_book_sentences = wikipedia_sentences + bookcorpus_sentences

wiki_book_sentences = datasets.Dataset.from_dict({'text': wiki_book_sentences})

#print the first 10 sentences in the datasets
print(wiki_book_sentences['text'][:10])

wiki_book_sentences.save_to_disk('../data/flatten_wiki_book_sentences.ds')







