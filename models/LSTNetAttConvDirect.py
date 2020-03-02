import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window;
        self.m = data.m
        self.hidR = args.hidCNN; #not self.hidRNN, since we are constrained to keep it the same as conv outs
        self.hidC = args.hidCNN;
        self.hidS = args.hidSkip;
       
        self.Ck = args.CNN_kernel;
        self.skip = 0;#int(args.skip);
#         self.pt = (self.P - self.Ck)/self.skip
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m));
#         self.GRU1 = nn.GRU(self.hidC, self.hidR);
        
        #linear layer to convert conv and gru output and get attention map across conv
        
        self.LSTM_cell = nn.LSTMCell(self.hidC, self.hidR)
        self.LSTM_cell = self.LSTM_cell.cuda()
        
        self.conv_out_dim = self.P - self.Ck + 1
        #attention is only applied upto the second last timestep
        self.att_layer = nn.Linear(self.hidC, 1)
        self.att_layer = self.att_layer.cuda()
        self.att_softmax = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(p = args.dropout);
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(int(self.hidR + self.skip * self.hidS), self.m);
        else:
            self.linear1 = nn.Linear(self.hidR, self.m);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;
 
    def forward(self, x):
        batch_size = x.size(0);
        
        #CNN
        
#         print('Input Shape: ', x.shape)
        c = x.view(-1, 1, self.P, self.m);
        c = F.relu(self.conv1(c));
        c = self.dropout(c);
        c = torch.squeeze(c, 3);
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous();
#         print("Shape after Contiguous =",r.shape)
#         _, r = self.GRU1(r);
        
        r = r.cuda()
        batch_size = r.shape[1]
        output = torch.zeros(r.shape[0],batch_size, self.hidR)
#         print(output.shape)
        c = torch.randn(batch_size,self.hidR).cuda()
        h = torch.randn(batch_size,self.hidR).cuda()
        
        #feed the input per timestep until the second last timestep, and compute the hidden state/output per ts:
        for i in range(r.shape[0]):
            h, c = self.LSTM_cell(r[i], (h, c))
            output[i] = h
#         print(output.shape) #163,128,100
        output = output.permute(1,0,2).contiguous()
        output = output.cuda()
        #compute the attention weighted sum of all the past hidden states:
        attn_out = self.att_layer(output)
        attn_out_sm = self.att_softmax(attn_out) #128, 163

        #now multiply this across all 100 dimensions of the cnn output and sum over the timestep part
        #(get weighted sum of all timesteps ke outputs)

        attn_out = (output * attn_out_sm).sum(1)
        out = attn_out
        #add the attention weighted sum of all past hidden states to the last hidden state
#         combined = attn_out + output[:,output.shape[1] - 1,:]
        
        #feed the combined sum to the LSTM cell to get the final output
#         out, c = self.LSTM_cell(combined,(h,c))
        
#         r = self.dropout(torch.squeeze(r,0));

        
        #skip-rnn
        
        if (self.skip > 0):
            self.pt = (self.P - self.Ck)//self.skip
#             print('pt = ',self.pt)
            s = c[:,:, int(-self.pt * self.skip):].contiguous();
#             print(s.shape)
            s = s.view(int(batch_size), int(self.hidC), int(self.pt), int(self.skip));
#             print(s.shape)
            s = s.permute(2,0,3,1).contiguous();
#             print(s.shape)
            s = s.view(self.pt, batch_size * self.skip, self.hidC);
            _, s = self.GRUskip(s);
#             print(s.shape)
            s = s.view(batch_size, self.skip * self.hidS);
#             print(s.shape)
            s = self.dropout(s);
            r = torch.cat((r,s),1);
#             print(r.shape)
        
        res = self.linear1(out);
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            z = z.permute(0,2,1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1,self.m);
            res = res + z;
            
#         if (self.output):
#             res = self.output(res);
        return res;