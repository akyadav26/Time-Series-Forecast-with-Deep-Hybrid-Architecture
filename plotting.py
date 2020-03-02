# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:17:35 2020

@author: yadav
"""

import argparse
import math
import time
import os
import io

import nntools

import torch
import torch.nn as nn
from models import LSTNet
import numpy as np;
import importlib

import matplotlib.pyplot as plt

from plot_utils import *;
import Optim


def load_checkpoint(checkpoint_path, model, optimizer, history):

    checkpt = []
    with open(checkpoint_path, 'rb') as f:
        checkpt = torch.load(checkpoint_path)
    model.load_state_dict(checkpt['Net'])
    optimizer.load_state_dict(checkpt['Optimizer'])
    history = checkpt['History']
    return model, optimizer, history

def preds(data, X, Y, model, batch_size):
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;
#    count = 0
    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X);
#        print (X.shape)
#        print (Y.shape)
#        print (output.shape)
#        count+=1
#        print(count)
        if predict is None:
            predict = output;
            test = Y;
        else:
            predict = torch.cat((predict,output));
            test = torch.cat((test, Y));
        
#        scale = data.scale.expand(output.size(0), data.m)
#        total_loss += evaluateL2(output * scale, Y * scale).item()#data[0]
#        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()#data[0]
#        n_samples += (output.size(0) * data.m);
#    rse = math.sqrt(total_loss / n_samples)/data.rse
#    rae = (total_loss_l1/n_samples)/data.rae
    
    predict = predict.data.cpu().numpy();
    Ytest = test.data.cpu().numpy();
#    sigma_p = (predict).std(axis = 0);
#    sigma_g = (Ytest).std(axis = 0);
#    mean_p = predict.mean(axis = 0)
#    mean_g = Ytest.mean(axis = 0)
#    index = (sigma_g!=0);
#    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g);
#    correlation = (correlation[index]).mean();
#    return rse, rae, correlation;
    return predict, Ytest


#parser.add_argument('--save', type=str,  default='../LSTNetLogs/save/model.pt',
#                    help='path to save the final model')
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
parser.add_argument('--save', type=str,  default='../syn-f-rse-24',
                    help='path to save the final model')
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--horizon', type=int, default=24)
parser.add_argument('--skip', type=float, default=24)#24
parser.add_argument('--hidSkip', type=int, default=5)
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='sigmoid')

args = parser.parse_args()

args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

Data = Data_utility(args.data, 1, 0, args.cuda, args.horizon, args.window, args.normalize);
print(Data.rse);

model = eval(args.model).Model(args, Data);

if args.cuda:
    model.cuda()
    
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip,
)

history = []

model, optim, history = load_checkpoint(os.path.join(args.save, 'model.pt'), model, optim.optimizer, history)
pred, truth  = preds(Data, Data.train[0], Data.train[1], model, 300);


print (pred.shape)
print (truth.shape)

x_axs = range(2000,pred.shape[0])
plt.plot(x_axs,pred[2000:, 0], color = 'b', label = 'prediction', linewidth=0.7)
plt.plot(x_axs,truth[2000:, 0], color = 'y', label = 'ground truth', linewidth=0.7)
# plt.title('LSTNet w/o AR on Synthetic Data')
plt.title('Full LSTNet Model on Synthetic Data')
plt.legend()
plt.savefig('full-lstnet-synthetic.png')