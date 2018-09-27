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
from utils import *
from model import DCNN

from random import randint

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Sequence Modeling - Twitter sentiment prediction repro')

parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false', default=False,
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to top-most fully connected layer (default: 0.2)')
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
parser.add_argument('--lr', type=float, default=4,
                    help='initial learning rate (default: 4)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--optim', type=str, default='adagrad',
                    help='optimizer type (default: Adagrad)')
parser.add_argument('--corpus', action='store_true',
                    help='force re-make the corpus (default: False)')
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
eval_batch_size = 10

train_data = TwitterCorpus_Training(h5_file)
test_data = TwitterCorpus_Testing(h5_file)

model = DCNN(embedding_size=embedding_size,
             vocab_size=n_words,
             num_maps=num_maps, 
             kernel_sizes=kernel_sizes, 
             k_top=k_top,
             output_size=output_size,
             dropout=dropout)

if args.cuda:
    model.cuda()

# May use adaptive softmax to speed up training
criterion = nn.CrossEntropyLoss()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def evaluate(data_source):
    model.eval()
    total_loss = 0
    processed_data_size = 0
    for i in range(0, data_source.size(1) - 1, args.validseqlen):
        if i + args.seq_len - args.validseqlen >= data_source.size(1) - 1:
            continue
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output = model(data)

        # Discard the effective history, just like in training
        eff_history = args.seq_len - args.validseqlen
        final_output = output[:, eff_history:].contiguous().view(-1, n_words)
        final_target = targets[:, eff_history:].contiguous().view(-1)

        loss = criterion(final_output, final_target)

        # Note that we don't add TAR loss here
        total_loss += (data.size(1) - eff_history) * loss.data
        processed_data_size += data.size(1) - eff_history
    return total_loss.item() / processed_data_size


def train():
    # Turn on training mode which enables dropout.
    global writer
    global train_data
    global write_graph
    model.train()
    total_loss = 0
    start_time = time.time()
    for batch_idx, i in enumerate(range(0, train_data.size(1) - 1, args.validseqlen)):
        if i + args.seq_len - args.validseqlen >= train_data.size(1) - 1:
            continue
        data, targets = get_batch(train_data, i, args)
        optimizer.zero_grad()
        if write_graph:
            writer.add_graph(model, data)
            write_graph = False

        output = model(data)

        # Discard the effective history part
        eff_history = args.seq_len - args.validseqlen
        if eff_history < 0:
            raise ValueError("Valid sequence length must be smaller than sequence length!")
        final_target = targets[:, eff_history:].contiguous().view(-1)
        final_output = output[:, eff_history:].contiguous().view(-1, n_words)
        loss = criterion(final_output, final_target)

        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch_idx, train_data.size(1) // args.validseqlen, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            writer.add_scalar('loss', cur_loss, batch_idx + 1)
            writer.add_scalar('perplexity', math.exp(cur_loss), batch_idx + 1)
            writer.add_scalar('learning rate', lr)
            total_loss = 0
            start_time = time.time()


if __name__ == "__main__":
    best_vloss = 1e8

    # for Tensorbard logging
    write_graph = True # flag flipped once first batch processed
    writer = SummaryWriter('./logs')


    # At any point you can hit Ctrl + C to break out of training early.
    try:
        all_vloss = []
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            test_loss = evaluate(test_data)
            test_errors = evaluate(test_data)

            writer.add_scalar('test loss', test_loss)
            writer.add_scalar('test errors', )

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                  'test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            test_loss, math.exp(test_loss)))
            print('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_vloss:
                with open("model.pt", 'wb') as f:
                    print('Save model!\n')
                    torch.save(model, f)
                best_vloss = val_loss

            # Anneal the learning rate if the validation loss plateaus
            if epoch > 5 and val_loss >= max(all_vloss[-5:]):
                lr = lr / 2.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            all_vloss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open("model.pt", 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
