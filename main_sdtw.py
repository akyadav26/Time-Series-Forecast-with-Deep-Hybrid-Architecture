import argparse
import math
import time
import os
import io

import nntools
from soft_dtw import SoftDTW

import torch
import torch.nn as nn
from models import LSTNet
import numpy as np;
import importlib

import matplotlib.pyplot as plt

from utils import *;
import Optim

def plot_separate(history, epoch, output_dir, plot_type):
    global args
    if plot_type == args.loss:
        plt.clf()
        plt.xlabel('Epoch')
        plt.ylabel('Training ' + plot_type.upper())
        plt.plot([history[k][0][plot_type] for k in range(epoch)], label="training_{}".format(plot_type))
        plt.legend()
        plots_path = os.path.join(output_dir, "plots")
        if not os.path.exists(plots_path):
            os.mkdir(plots_path)
        plt.savefig(plots_path + '/plot_training_{}.png'.format(plot_type))
    
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel('Validation ' + plot_type.upper())
    plt.plot([history[k][1][plot_type] for k in range(epoch)], label="validation_{}".format(plot_type))
    plt.legend()
    plots_path = os.path.join(output_dir, "plots")
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)
    plt.savefig(plots_path + '/plot_validation_{}.png'.format(plot_type))


def evaluate(data, X, Y, model, evaluateSDTW, evaluateL2, evaluateL1, batch_size):
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    total_sdtw = 0
    n_samples = 0;
    predict = None;
    test = None;
    
    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X);
        if predict is None:
            predict = output;
            test = Y;
        else:
            predict = torch.cat((predict,output));
            test = torch.cat((test, Y));
        
        total_sdtw += evaluateSDTW(output, Y).item()
        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()#data[0]
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()#data[0]
        n_samples += (output.size(0) * data.m);
    rse = math.sqrt(total_loss / n_samples)/data.rse
    rae = (total_loss_l1/n_samples)/data.rae
    sdtw = total_sdtw / n_samples
    
    predict = predict.data.cpu().numpy();
    Ytest = test.data.cpu().numpy();
    sigma_p = (predict).std(axis = 0);
    sigma_g = (Ytest).std(axis = 0);
    mean_p = predict.mean(axis = 0)
    mean_g = Ytest.mean(axis = 0)
    index = (sigma_g!=0);
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g);
    correlation = (correlation[index]).mean();
    return sdtw, rse, rae, correlation;

def train(data, X, Y, model, criterion, optim, batch_size):
    model.train();
    total_loss = 0;
    n_samples = 0;
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad();
        output = model(X);
        scale = data.scale.expand(output.size(0), data.m)
        #loss = criterion(output * scale, Y * scale);
        loss = criterion(output, Y)
        loss.backward();
        grad_norm = optim.step();
        total_loss += loss.item()#data[0];
        n_samples += (output.size(0) * data.m);
    return total_loss / n_samples



def checkpoint(checkpoint_path, model, optimizer, history):
    state = {'Net': model.state_dict(),
                'Optimizer': optimizer.state_dict(),
                'History': history}
#     print('Saving to', checkpoint_path)
    torch.save(state, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer, history):
    global device
    checkpt = []
    with open(checkpoint_path, 'rb') as f:
        checkpt = torch.load(checkpoint_path, map_location = device)
    model.load_state_dict(checkpt['Net'])
    optimizer.load_state_dict(checkpt['Optimizer'])
    history = checkpt['History']
    return model, optimizer, history


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, required=True,
                    help='location of the data file')
parser.add_argument('--model', type=str, default='LSTNet',
                    help='')
parser.add_argument('--hidCNN', type=int, default=100,
                    help='number of CNN hidden units')
parser.add_argument('--hidRNN', type=int, default=100,
                    help='number of RNN hidden units')
parser.add_argument('--window', type=int, default=24 * 7,
                    help='window size')
parser.add_argument('--CNN_kernel', type=int, default=6,
                    help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=24,
                    help='The window size of the highway component')
parser.add_argument('--clip', type=float, default=10.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,
                    help='random seed')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='../LSTNetLogs/save/',
                    help='path to save the final model')
# parser.add_argument('--best', type=str,  default='../LSTNetLogs/save/best_model.pt',
#                     help='path to save the best model')
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--horizon', type=int, default=24)
parser.add_argument('--skip', type=float, default=0)
parser.add_argument('--hidSkip', type=int, default=5)
parser.add_argument('--loss', type=str, default='sdtw')
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='sigmoid')

'''python main_sdtw.py --loss l2 --save ../LSTNetLogs/elec-l2-f-24 --data data/electricity.txt
   python main_sdtw.py --loss sdtw --save ../LSTNetLogs/elec-sdtw-f-24 --data data/electricity.txt
   python main_sdtw.py --loss sdtw --horizon 12 --save ../LSTNetLogs/elec-sdtw-f-12 --data data/electricity.txt
   python main_sdtw.py --loss sdtw --horizon 6 --save ../LSTNetLogs/elec-sdtw-f-6 --data data/electricity.txt
   python main_sdtw.py --loss sdtw --horizon 3 --save ../LSTNetLogs/elec-sdtw-f-3 --data data/electricity.txt
   
   python main_sdtw.py --loss sdtw --save ../LSTNetLogs/elec-sdtw-lstm-24 --data data/electricity.txt
   python main_sdtw.py --loss sdtw --save ../LSTNetLogs/elec-sdtw-var-24 --data data/electricity.txt
   python main_sdtw.py --loss sdtw --save ../LSTNetLogs/elec-sdtw-novar-24 --data data/electricity.txt
   
'''
args = parser.parse_args()

args.cuda = args.gpu is not None
device = 'cpu'
if args.cuda:
    torch.cuda.set_device(args.gpu)
    device = 'cuda'
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

Data = Data_utility(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize);
print(Data.rse);

model = eval(args.model).Model(args, Data);

if args.cuda:
    model.cuda()
    
nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

if args.loss == 'rae':
    criterion = nn.L1Loss(size_average=False);
elif args.loss == 'rse':
    criterion = nn.MSELoss(size_average=False);
elif args.loss == 'sdtw':
    criterion = SoftDTW()
evaluateL2 = nn.MSELoss(size_average=False);
evaluateL1 = nn.L1Loss(size_average=False)
evaluateSDTW = SoftDTW()
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda();
    evaluateL2 = evaluateL2.cuda();
    
best_val = 10000000;
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip,
)


history = []
if(os.path.exists(args.save)):
    if os.path.exists(os.path.join(args.save, 'best_model.pt')):
        model, optim.optimizer, history = load_checkpoint(os.path.join(args.save,'best_model.pt'), model, optim.optimizer, history)
    elif os.path.exists(os.path.join(args.save, 'model.pt')):
        model, optim.optimizer, history = load_checkpoint(os.path.join(args.save, 'model.pt'), model, optim.optimizer, history)
else:
    os.makedirs(args.save)
# At any point you can hit Ctrl + C to break out of training early.

start_epoch = len(history)
try:
    print('begin training');
    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
        val_sdtw, val_rse, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateSDTW, evaluateL2, evaluateL1, args.batch_size);
        
        #Save the model, optim, history and val_history
        history.append(({args.loss:train_loss}, {'sdtw' : val_sdtw, 'rae' : val_rae,'rse' : val_rse, 'corr' : val_corr}))
        checkpoint(os.path.join(args.save, 'model.pt'), model, optim.optimizer, history)
        #plot the statistics
        plot_separate(history, epoch, output_dir = args.save, plot_type='sdtw')
        plot_separate(history, epoch, output_dir = args.save, plot_type='rse')
        plot_separate(history, epoch, output_dir = args.save, plot_type='rae')
        plot_separate(history, epoch, output_dir = args.save, plot_type='corr')
        # Save the model if the validation loss is the best we've seen so far.
        if val_sdtw < best_val:
            checkpoint(os.path.join(args.save, 'best_model.pt'), model, optim.optimizer, history)
            best_val = val_sdtw
            
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid sdtw {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_sdtw, val_rse, val_rae, val_corr))
        
        if epoch % 5 == 0:
            test_sdtw, test_rse, test_rae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateSDTW, evaluateL2, evaluateL1, args.batch_size);
            test_stats = dict()
            test_stats = {'test_sdtw':test_sdtw, 'test_rae':test_rae,'test_rse':test_rse, 'test_corr': test_corr}
            test_stat_path = os.path.join(args.save, 'test_stat.pt')
            torch.save(test_stats,test_stat_path)
            print ("test sdtw {:5.4f} | test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_sdtw, test_rse, test_rae, test_corr))
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.

model, optim, history = load_checkpoint(os.path.join(args.save, 'best_model.pt'), model, optim.optimizer, history)
test_sdtw, test_rse, test_rae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateSDTW, evaluateL2, evaluateL1, args.batch_size);
test_stats = dict()
test_stats = {'test_sdtw':test_sdtw, 'test_rae':test_rae,'test_rse':test_rse, 'test_corr': test_corr}
test_stat_path = os.path.join(args.save, 'test_stat.pt')
torch.save(test_stats,test_stat_path)
print ("test sdtw {:5.4f} | test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_sdtw, test_rse, test_rae, test_corr))
