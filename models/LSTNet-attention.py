import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window;
        self.m = data.m
        self.hidR =args.hidRNN;
        self.hidC = args.hidCNN;
        self.hidS = args.hidSkip;
        
        self.attention_dim = 128
        #linear layer to convert conv and gru output and get attention map across conv
        
        self.gru_attention = nn.Linear()
        
        self.Ck = args.CNN_kernel;
        self.skip = 0;#int(args.skip);
#         self.pt = (self.P - self.Ck)/self.skip
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m));
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        
        self.LSTM_cell = nn.LSTMCell(self.hidC, self.hidR)
        
        
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
        print("Shape after Contiguous =",r.shape)
#         _, r = self.GRU1(r);

        
        print("Shape after GRU =",r.shape)
        r = self.dropout(torch.squeeze(r,0));

        
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
        res = self.linear1(r);
        
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            z = z.permute(0,2,1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1,self.m);
            res = res + z;
            
        if (self.output):
            res = self.output(res);
        return res;
    
        
        
        
