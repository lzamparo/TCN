import os
import re
import csv
import string
import pickle
import torch
import h5py
import numpy as np

from torch.autograd import Variable
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer

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
        
        self.datafile = h5py.File(os.path.join(args.data,"tweet_data.h5"), 'w')
        self.prepare_dataset(args.training, 'training')
        self.prepare_dataset(args.testing, 'testing')
              
    
    def prepare_dataset(self, path, data_split):
        """ Preprocess the dataset in `path` 
        data_split \in ['training','testing'] """
        
        outpath = path.replace(".csv",".prepared.csv")
        tokens, max_len, num_tweets = self._preprocess_and_build_dictionary(path, outpath)
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
    
    def _detect_charset(self, path):
        """ Use chardet to parse the file in path, 
        and try to best guess the charset.  Clearly not 
        ascii or utf-8. """
        
        from chardet.universaldetector import UniversalDetector
        detector = UniversalDetector()
        
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            tweet_reader = csv.reader(f, delimiter=',', quotechar='"')
            for i,parts in enumerate(tweet_reader):
                tweet = parts[-1]            
                detector.feed(tweet.encode('utf-8',errors='replace'))
                if detector.done: break
        detector.close()
        return detector.result    
    
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
                words = [w for w in tok.tokenize(lc_clean_tweet) if len(w) > 1]
                max_len = len(words) if len(words) > max_len else max_len
                
                for word in words:
                    self.dictionary.add_word(word)
                
                clean_line = parts[:-1] + [" ".join(words)]
                tweet_writer.writerow(clean_line)
                
        unique_tokens = len(self.dictionary)
        return unique_tokens, max_len, i    
    

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

        def _tweet_to_list(self, parts, max_len):
            label, tweet = parts[0], parts[-1]
            words = tweet.split()
            encoded_words = [self.dictionary.word2idx[word] for word in words]
            encoded_words = encoded_words + [-1 for i in range(max_len - len(
                words))]
            assert(len(encoded_words) == max_len)
            return encoded_words, label
        
        assert os.path.exists(path)
        
        # create groups for data, labels
        group_name = '/' + group
        this_group = self.datafile.create_group()
        data_name = group + "_data"
        label_name = group + "_labels"
        data = this_group.create_dataset(data_name, shape=(num_examples,max_len), chunks=(10000,max_len), dtype=np.i4)
        labels = this_group.create_dataset(label_name, shape=(num_examples,1), dtype=np.i4)
        
        # parse, encode words in each tweet, write to h5file
        chunk = 0  
        chunk_size = 10000
        temp_array = np.empty((10000,max_len))
        temp_labels = np.empty((10000,1))
        
        with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            tweet_reader = csv.reader(f, delimiter=',', quotechar='"')
            for i, parts in enumerate(tweet_reader):
                embedded_list, label = self._tweet_to_list(parts, max_len)
                temp_array[:,i] = np.array(embedded_list)
                temp_labels[i,0] = label
                if i % 10000 == 0:
                    # write to the h5file
                    data[chunk*chunk_size : (chunk+1) * chunk_size, :] = temp_array
                    labels[chunk * chunk_size : (chunk+1) * chunk_size,0] = temp_labels
                    chunk += 1
    

def batchify(data, batch_size, args):
    """The output should have size [L x batch_size], where L could be a long sequence length"""
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1)
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.seq_len, source.size(1) - 1 - i)
    data = Variable(source[:, i:i+seq_len], volatile=evaluation)
    target = Variable(source[:, i+1:i+1+seq_len])     # CAUTION: This is un-flattened!
    return data, target
