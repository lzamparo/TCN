import os
import re
import csv
import string
import pickle
import torch

from torch.autograd import Variable
from bs4 import BeautifulSoup


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


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class TwitterCorpus(object):
    def __init__(self, args):
        self.dictionary = Dictionary()
        
        self.username_re = re.compile("\@[\w]+")
        self.url_re = re.compile("http[s]?://[\w|\.|\?|\/]+")
        self.emoticon_re = re.compile("(;D)|(:D)|(:/)|(=\))|(:-D)|(;-D)|(:\()|(=\()|(:\s{1}\()")
        self.run_on_re = re.compile(r"(\w)\1{2,}", re.DOTALL)        
        
        self.train_tokens, self.train_ids = self.prepare_dataset(args.training)
        self.test_tokens, self.test_ids = self.prepare_dataset(args.testing)
              
    
    def prepare_dataset(self, path):
        """ Preprocess the dataset in `path` """
        outpath = path.replace(".csv",".prepared.csv")
        tokens = self._preprocess_and_build_dictionary(path, outpath)
        ids = self._setup_embedding_layer(path, tokens)
        return tokens, ids
    
    def _process_tweet(self, tweet, depunct):
        """ Apply feature transformations to each tweet in the dataset 
        Remove punctuation if depunct """
        
        
        #example1 = BeautifulSoup(df.text[279], 'lxml')
        #print example1.get_text()        
        
        tweet = tweet.strip()
        tweet = BeautifulSoup(tweet, 'lxml').get_text()
        tweet = tweet.replace(u"\ufffd", "?")
        # @usernames -> USERNAME
        tweet = re.sub(self.username_re, lambda x: "USERNAME", tweet)
        # URLS -> URL
        tweet = re.sub(self.url_re, lambda x: "URL", tweet)
        # standardize emoticons
        tweet = re.sub(self.emoticon_re, lambda x: "", tweet)
        # shrink extended runs of any char
        tweet = re.sub(self.run_on_re, r"\1\1", tweet)  # result = re.sub("(\d+) (\w+)", r"\2 \1")
        
        if depunct:
            depunctuate = str.maketrans('', '', string.punctuation)
            return tweet.translate(depunctuate)
        
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
    
    def _preprocess_and_build_dictionary(self, inpath, outpath, depunct=False):
        """ Preprocess the Twitter Sentiment data set in `inpath`,
        build the dictionary, and write the sanitized output to 
        `outpath`.  In addition, return how many tokens
        we see in the corpus. """
        
        assert os.path.exists(inpath)
        
        #charset = self._detect_charset(inpath)
        
        with open(inpath, 'r', encoding='utf-8-sig', errors='replace') as in_f, open(outpath,'w', encoding='utf-8') as out_f:
            tokens = 0
            tweet_reader = csv.reader(in_f, delimiter=',', quotechar='"')
            for i, parts in enumerate(tweet_reader):
                if (i % 10000) == 0:
                    print("Finished tweet ", i)
                tweet = parts[-1]
                clean_tweet = self._process_tweet(tweet, depunct)
                lc_clean_tweet = clean_tweet.lower()
                words = lc_clean_tweet.split() + ['<eos>']
                clean_words = [w for w in words if len(w) > 0]
                tokens += len(clean_words)
                for word in clean_words:
                    self.dictionary.add_word(word)
                clean_line = parts[:-1] + [lc_clean_tweet]
                print(','.join(clean_line), file=out_f)
        
        return tokens    
        
    def _setup_embedding_layer(self, path, tokens):
        """ Build the word2idx dict for the Twitter Sentiment data set in `path` """
        
        assert os.path.exists(path)
        
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            tweet_reader = csv.reader(f, delimiter=',', quotechar='"')
            for parts in tweet_reader:
                tweet = parts[-1]
                words = tweet.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids
    
    


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
