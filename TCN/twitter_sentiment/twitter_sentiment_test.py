import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import pickle
import yaml

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from random import randint
from tensorboardX import SummaryWriter

from utils import *
from model import DCNN


parser = argparse.ArgumentParser(description='Sequence Modeling - Twitter sentiment prediction repro')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA (default: False)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to top-most fully connected layer (default: 0.5)')
parser.add_argument('--max_vocab_size', type=int, default=76643,
                    help='limit the vocabulary to <size> most common words (default: 76643)')
parser.add_argument('--emb_dropout', type=float, default=0.0,
                    help='dropout applied to the embedded layer (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1.0,
                    help='gradient clip, -1 means no clip (default: 0.35)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--data', type=str, default='./data/sentiment',
                    help='location of the data corpus (default: ./data/sentiment)')
parser.add_argument('--training', type=str, default='./data/training.1600000.processed.noemoticon.csv',
                    help='location of the training data csv file')
parser.add_argument('--testing', type=str, default='./data/testdata.manual.2009.06.14.csv',
                    help='location of the test data csv file')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate (default: 0.1)')
parser.add_argument('--wd', type=float, default=0.1,
                    help='regularization value for all parameters')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--optim', type=str, default='adagrad',
                    help='optimizer type (default: Adagrad)')
parser.add_argument('--corpus', action='store_true',
                    help='force re-make the corpus (default: False)')
parser.add_argument('--load', action='store_true',
                    help='load the last savepoint from the model (default: False)')
parser.add_argument('--evalonly', action='store_true',
                    help='eval only, no training. Must be used in conjunction with --load (default: False)')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)
with open('./dcnn_repro.yaml','r') as f:
    repro_model_params = yaml.load(f)
    num_maps = repro_model_params['num_maps']
    k_top = repro_model_params['k_top']
    output_size = repro_model_params['output_size']
    kernel_sizes = repro_model_params['kernel_sizes']
    embedding_size = repro_model_params['embedding_size']
    dropout = repro_model_params['dropout']
    
corpus = data_generator(args)
n_words = len(corpus.dictionary)
h5_file = corpus.get_data_file()
eval_batch_size = 32

recoding_transform = RecodeLabel()
train_data = TwitterCorpus_Training(h5_file, label_transform=recoding_transform)

max_length = train_data.get_max_length()
padding_transform = PaddingTransformer(max_length, corpus.get_padding_idx())
valid_data = TwitterCorpus_Testing(h5_file, transform=padding_transform, label_transform=recoding_transform)

train_loader = DataLoader(train_data, 
                          batch_size=args.batch_size,
                          shuffle=True, 
                         pin_memory=True)

valid_loader = DataLoader(valid_data,
                          batch_size=args.batch_size,
                          shuffle=True,
                          pin_memory=True)


if os.path.exists("./model.pt") and args.load:
    with open("model.pt", 'rb') as f:
        model = torch.load(f, map_location=lambda storage, loc: storage)
else:
    model = DCNN(embedding_size=embedding_size,
                 vocab_size=n_words,
                 num_maps=num_maps, 
                 kernel_sizes=kernel_sizes, 
                 k_top=k_top,
                 output_size=output_size,
                 dropout=dropout,
                 padding_idx=corpus.get_padding_idx())

if args.cuda:
    print("Send model to device")
    model.cuda()

# May use adaptive softmax to speed up training
criterion = nn.CrossEntropyLoss()
optim_dict = {'adagrad': torch.optim.Adagrad, 'sgd': torch.optim.SGD, 'adam': torch.optim.Adam}
lr = args.lr
optimizer = optim_dict[args.optim](model.parameters(), lr=lr, weight_decay=args.wd)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)

def evaluate_count():
    model.eval()
    total_errors = 0
    
    for batch_idx, (x, y) in enumerate(valid_loader):
        
        x = x.long()
        y = y.squeeze()
        
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
        output = model(x)
        
        # take positional max of output
        value, max_index = output.max(dim=1, keepdim=True)
        diffs = y - max_index.squeeze()
        errors = diffs != 0
        total_errors += errors.sum()
        
    return total_errors.item()

def evaluate():
    model.eval()
    total_loss = 0
    
    for batch_idx, (x, y) in enumerate(valid_loader):
        
        x = x.long()
        y = y.squeeze()
        
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
        output = model(x)
        loss = criterion(output, y)

        # Note that we don't add TAR loss here
        total_loss += loss.data
    return total_loss.item() 


def train():
    # Turn on training mode which enables dropout.
    global writer
    
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (x, y) in enumerate(train_loader):
        
        x = x.long()
        y = y.squeeze()
        
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
    
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                  'loss {:5.2f}'.format(
                      epoch, batch_idx, len(train_data) // args.batch_size, lr,
                elapsed * 1000 / args.log_interval, cur_loss))
            writer.add_scalar('loss', cur_loss, batch_idx + 1)
            writer.add_scalar('learning rate', lr)
            total_loss = 0
            start_time = time.time()


if __name__ == "__main__":
    best_vloss = 1e8

    # for Tensorbard logging
    writer = SummaryWriter('./logs')
    
    if args.evalonly and args.load:
        print("Evaluating absolute number of erros for the model ...")
        errors = evaluate_count()
        print("Total test set errors: ", errors)
        sys.exit(1)
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        all_vloss = []
        
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate()

            writer.add_scalar('valid loss', val_loss)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '.format(
                epoch, (time.time() - epoch_start_time), val_loss))
            print('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_vloss:
                with open("model.pt", 'wb') as f:
                    print('Save model!\n')
                    torch.save(model, f)
                best_vloss = val_loss

            # Anneal the learning rate if the validation loss plateaus
            scheduler.step(val_loss)
            all_vloss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open("model.pt", 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate()
    print('=' * 89)
    print('| End of training | test loss {:5.2f}'.format(
        test_loss))
    print('=' * 89)
