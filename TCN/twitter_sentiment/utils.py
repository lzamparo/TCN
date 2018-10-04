import os
import sys
import re
import csv
import string
import pickle
import torch
import h5py
import numpy as np

from torch.utils.data import Dataset
from torch.autograd import Variable
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk import FreqDist


def data_generator(args):
    if os.path.exists(args.data + "/corpus") and not args.corpus:
        corpus = pickle.load(open(args.data + '/corpus', 'rb'))
    else:
        corpus = TwitterCorpus(args)
        pickle.dump(corpus, open(args.data + '/corpus', 'wb'))
    return corpus


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class TwitterCorpus(object):
    def __init__(self, args):
        self.dictionary = Dictionary()
        self.dictionary.add_word("<<<padding>>>")
        self.padding_value = self.dictionary.word2idx["<<<padding>>>"]
        
        self.fdist = FreqDist()
        self.file_prepared = False
        self.username_re = re.compile("\@[\w]+")
        self.url_re = re.compile("http[s]?://[\w|\.|\?|\/]+")
        self.www_re = re.compile("www.[^ ]+")
        self.emoticon_re = re.compile("(;D)|(:D)|(:/)|(=\))|(:-D)|(;-D)|(:\()|(=\()|(:\s{1}\()")
        self.run_on_re = re.compile(r"(\w)\1{2,}", re.DOTALL)
        self.negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                         "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                        "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                        "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                        "mustn't":"must not"}
        self.neg_pattern = re.compile(r'\b(' + '|'.join(self.negations_dic.keys()) + r')\b')        
        
        self.datafile = os.path.join(args.data,"tweet_data.h5")
        self.data_handle = h5py.File(os.path.join(args.data,"tweet_data.h5"), 'w')
        
        self.prepare_dataset(args.training, 'training')
        self.prepare_dataset(args.testing, 'testing')
        self.data_handle.close()
        self.file_prepared = True
        
    def __getstate__(self):
        ''' Do not pickle the handle to the h5 file '''
        state = self.__dict__.copy()
        del state['data_handle']
        return state


    def __setstate__(self, state):
        
        self.__dict__.update(state)
        if os.path.exists(self.datafile):
            self.file_prepared = True
        else:
            self.file_prepared = False
     
    def get_padding_idx(self):
        return self.padding_value
    
    def get_data_file(self):
        if self.file_prepared:
            return self.datafile
        else:
            print('File is not prepared.  Re-build TwitterCorpus object properly.', file=sys.stderr)
    
    def prepare_dataset(self, path, data_split):
        """ Preprocess the dataset in `path` 
        data_split \in ['training','testing'] """
        
        outpath = path.replace(".csv",".prepared.csv")
        self._make_freqdist(path)
        tokens, max_len, num_tweets = self._preprocess_and_build_dictionary(path, outpath, fdist)
        self._pack_to_h5(outpath, data_split, tokens, max_len, num_tweets)
    
    def _process_tweet(self, tweet):
        """ Apply feature transformations to each tweet in the dataset """
   
        # unique tokens with this, no depunct: 755992
        # removing single char tokens, expanding contractions: 277990
        
        tweet = tweet.strip()
        tweet = BeautifulSoup(tweet, 'lxml').get_text()
        tweet = tweet.replace(u"\ufffd", "?")
        # @usernames -> USERNAME
        tweet = re.sub(self.username_re, lambda x: "USERNAME", tweet)
        # URLS -> URL
        tweet = re.sub(self.url_re, lambda x: "URL", tweet)
        # www. URLs -> URL
        tweet = re.sub(self.www_re, lambda x: "URL", tweet)
        # expand negation contractions
        tweet = re.sub(self.neg_pattern, lambda x: self.negations_dic[x.group()], tweet)
        
        # standardize emoticons
        tweet = re.sub(self.emoticon_re, lambda x: "", tweet)
        # shrink extended runs of any char
        tweet = re.sub(self.run_on_re, r"\1\1", tweet)  # result = re.sub("(\d+) (\w+)", r"\2 \1")
        
        return tweet
    
    def _make_freqdist(self, path):
        """ Read all the tweets, calculate the frequencies of 
        the words appearing in processed tweets """
        
        #fdist1 |= fdist2
        tok = WordPunctTokenizer()
        
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            tweet_reader = csv.reader(f, delimiter=',', quotechar='"')
            for i,parts in enumerate(tweet_reader):
                tweet = parts[-1]   
                clean_tweet = self._process_tweet(tweet)
                lc_clean_tweet = clean_tweet.lower()
                words = [w for w in tok.tokenize(lc_clean_tweet) if len(w) > 1 and w not in stopwords.words('english')]                
                tweet_fdist = FreqDist(words)
                self.fdist |= tweet_fdist
                   
    
    def _preprocess_and_build_dictionary(self, inpath, outpath, depunct=True):
        """ Preprocess the Twitter Sentiment data set in `inpath`,
        build the dictionary, and write the sanitized output to 
        `outpath`.  In addition, return how many unique tokens
        we see in the corpus. 
        
        return the number of unique tokens seen, the largest number
        of words seen in a single tweet, and the number of tweets 
        in this file.
        """
        
        assert os.path.exists(inpath)
        if depunct:
            tok = WordPunctTokenizer()
            
        with open(inpath, 'r', encoding='utf-8-sig', errors='replace') as in_f, open(outpath,'w', encoding='utf-8') as out_f:
            tweet_reader = csv.reader(in_f, delimiter=',', quotechar='"')
            tweet_writer = csv.writer(out_f, delimiter=',', quotechar='"')
            max_len = 0
            
            for i, parts in enumerate(tweet_reader):
                if (i % 10000) == 0:
                    print("Finished tweet ", i)
                tweet = parts[-1]
                clean_tweet = self._process_tweet(tweet)
                lc_clean_tweet = clean_tweet.lower()
                words = [w for w in tok.tokenize(lc_clean_tweet) if len(w) > 1 and w not in stopwords.words('english')]
                max_len = len(words) if len(words) > max_len else max_len
                
                for word in words:
                    self.dictionary.add_word(word)
                
                clean_line = parts[:-1] + [" ".join(words)]
                tweet_writer.writerow(clean_line)
                
        unique_tokens = len(self.dictionary)
        return unique_tokens, max_len, i+1    
    
    def _tweet_to_list(self, parts, max_len):
        label, tweet = parts[0], parts[-1]
        
        try:
            label = int(label)
        except ValueError:
            print('Cannot coerce ', label, ' to int ')
            label = -1
        words = tweet.split()
        encoded_words = [self.dictionary.word2idx[word] for word in words]
        encoded_words = encoded_words + [self.padding_value for i in range(max_len - len(
            words))]
        assert(len(encoded_words) == max_len)
        return encoded_words, label   


    def _calculate_amount_to_write(self, chunk, chunk_size, num_examples):
        amount_to_write = num_examples - (chunk * chunk_size)
        if amount_to_write < 0:
            amount_to_write = num_examples
        if amount_to_write < chunk_size:
            return amount_to_write
        else:   
            return chunk_size
    
    def _pack_to_h5(self, path, group, tokens, max_len, num_examples):
        """ Build the word2idx data structure for the Twitter Sentiment data set in `path` 
        I'll use an hdf5 file to store the embedded seqs, labels.
        
        path := path to cleaned up tweet file
        tokens := number of tokens to encode
        group := 'training' or 'testing', which group in the h5 file
                do we encode the data from `path`
        max_len := most number of words observed in a tweet
        num_examples := number of tweets in this file in `path`
        """
        
        assert os.path.exists(path)
        
        # create groups for data, labels
        group_name = '/' + group
        this_group = self.data_handle.create_group(group_name)
        data_name = group + "_data"
        label_name = group + "_labels"
        
        chunk = 0  
        chunk_size = 10000        
        buffer_size = self._calculate_amount_to_write(chunk, chunk_size, 
                                                  num_examples)
    
        data = this_group.create_dataset(data_name, shape=(num_examples,max_len), chunks=(buffer_size,max_len), dtype=np.int32)
        labels = this_group.create_dataset(label_name, shape=(num_examples,1), dtype=np.int32)
        
        # parse, encode words in each tweet, write to h5file
        temp_array = np.empty((chunk_size,max_len),dtype=np.int32)
        temp_labels = np.empty((chunk_size,1), dtype=np.int32)        
        
        
        
        with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            tweet_reader = csv.reader(f, delimiter=',', quotechar='"')
            for i, parts in enumerate(tweet_reader):
                embedded_list, label = self._tweet_to_list(parts, max_len)
                temp_array[i % chunk_size,:] = np.array(embedded_list)
                temp_labels[i % chunk_size,0] = label
                if (i + 1) % buffer_size == 0:
                    # write the buffer to the h5file
                    data[chunk*chunk_size : chunk*chunk_size + buffer_size, :] = temp_array[0:buffer_size,:]
                    labels[chunk*chunk_size : chunk*chunk_size + buffer_size, 0] = temp_labels[0:buffer_size,0]
                    chunk += 1
                    buffer_size = self._calculate_amount_to_write(chunk, chunk_size, 
                                                                          num_examples)                          


class PaddingTransformer(object):
    """ Pad the features to a given length with the padding token """
    
    def __init__(self, length, token):
        self.length = length
        self.token = token
        
    def __call__(self, features):
        cols = features.shape[0]
        to_pad = self.length - cols
        padded_features = np.zeros((self.length,))
        padded_features[0:cols] = features[:]
        padded_features[cols:] = self.token
        
        return padded_features
    
        
class RecodeLabel(object):
    """ Re-code the Twitter sentiment labels to be contiguous integer labels 
    between zero and two, as there are only three classes """
    
    def __init__(self):
        self.labelcode = {0: 0, 2: 1, 4: 2}
        
    def __call__(self, label):
        recoded = np.zeros_like(label)
        for i,l in enumerate(label):
            recoded[i] = self.labelcode[l]
        
        return recoded


class TwitterCorpus_Training(Dataset):
    """ Load up the encoded Twitter corpus dataset for training"""
    
    def __init__(self, h5_filepath, transform=None, label_transform=None):
        
        self.h5f = h5py.File(h5_filepath, 'r', libver='latest', swmr=True)
        self.num_entries, self.max_length = self.h5f['/training/training_data'].shape
        self.transform = transform
        self.label_transform = label_transform
        
    def __getitem__(self, index):
        
        features = self.h5f['/training/training_data'][index]
        labels = self.h5f['/training/training_labels'][index]
        #np.place(features,features == -1, [self.padding_idx])
        
        if self.transform:
            features = self.transform(features)
        features = torch.from_numpy(features)
        features = features.long()
        
        if self.label_transform:
            labels = self.label_transform(labels)
        labels = torch.from_numpy(labels)
        labels = labels.long()

        return features, labels
    
    def __len__(self):
        return self.num_entries
    
    def get_max_length(self):
        return self.max_length
    
    def close(self):
        self.h5f.close()
        
class TwitterCorpus_Testing(Dataset):
    """ Load up the encoded Twitter corpus dataset for training"""
    
    def __init__(self, h5_filepath, transform=None, label_transform=None):
        
        self.h5f = h5py.File(h5_filepath, 'r', libver='latest', swmr=True)
        self.num_entries, self.max_length = self.h5f['/testing/testing_data'].shape
        self.transform = transform
        self.label_transform = label_transform
        
    def __getitem__(self, index):
        
        features = self.h5f['/testing/testing_data'][index]
        labels = self.h5f['/testing/testing_labels'][index]
        #np.place(features,features == -1, [self.padding_idx])
        
        if self.transform:
            features = self.transform(features)        
        features = torch.from_numpy(features)
        features = features.long()
        
        if self.label_transform:
            labels = self.label_transform(labels)
        labels = torch.from_numpy(labels)
        labels = labels.long()
        
        return features, labels
    
    def __len__(self):
        return self.num_entries
    
    def close(self):
        self.h5f.close()