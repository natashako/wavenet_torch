import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math
import time
from time import strftime, localtime



class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot,self).__init__()
        self.depth = depth
        self.ones = torch.eye(depth,dtype=torch.float,device="cuda:0")
    def forward(self, X_in):
        X_in = X_in
        return Variable(self.ones.index_select(0,X_in.data))
    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


# ## Wavenet  class
# 
# The preprocess( ) function applies one-hot encoding. For 8-bit audio signals, the quantization size is 128. Then one-hot 128 features are combined to 32 new features/channels to feed the dilation layers.
# 
# The postprocess( ) function transform the dilation layer outputs twice, and convert them to softmax logits. Two 1×1 convolution layers are used here(, and a lot more are used in dilation layers). The purpose of the 1×1 convolution is to apply linear transformation, because dense layers are not convenient  to use here.
# 
# The residue_forward( ) function takes four convolution modules as inputs. The skip convolution increases the number of channels for final output. The residue convolution keeps the same number of channels as dilation layer inputs. At last, only skip connections are summed, and the residue output from the last layer is discarded.
# 
# Two generation functions are included. generate_slow( ) is easy to understand, but generate( ) is much faster.


#            |----------------------------------------|      *residual*
#            |                                        |
#            |    |-- conv -- tanh --|                |
# -> dilate -|----|                  * ----|-- 1x1 -- + -->  *input*
#                 |-- conv -- sigm --|     |
#                                         1x1
#                                          |
# ---------------------------------------> + ------------->  *skip*


class WaveNet(nn.Module):
    """
    
    Wavenet Model
    
    Args:
        mu:                        audio quantization size
        n_residue:                 residue channels
        n_skip:                    skip channels
        dilation_depth & n_repeat: dilation layer setup
        
    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """
    
    def __init__(self, 
                 mu=64,                #256,
                 n_residue=24,         #32, 
                 n_skip=128,           #512, 
                 dilation_depth=10,    #10, 
                 n_repeat=2):           #5
                         
        
        
        super(WaveNet, self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dilation_depth = dilation_depth
        
        dilations = self.dilations = [2**i for i in range(dilation_depth)] * n_repeat
        # 1, 2, 4, 8, 16, ..., 512 repeated 5 times
        
        self.one_hot = One_Hot(mu)  #one_hot encoding of mu
        
        self.from_input = nn.Conv1d(in_channels=mu, out_channels=n_residue, kernel_size=1)
        
        self.conv_sigmoid = nn.ModuleList([nn.Conv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=2, dilation=d)
                         for d in dilations])
        
        self.conv_tanh = nn.ModuleList([nn.Conv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=2, dilation=d)
                         for d in dilations])
        
        self.skip_scale = nn.ModuleList([nn.Conv1d(in_channels=n_residue, out_channels=n_skip, kernel_size=1)
                         for d in dilations])
        
        self.residue_scale = nn.ModuleList([nn.Conv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=1)
                         for d in dilations])
        
        self.conv_post_1 = nn.Conv1d(in_channels=n_skip, out_channels=n_skip, kernel_size=1)
        self.conv_post_2 = nn.Conv1d(in_channels=n_skip, out_channels=mu, kernel_size=1)
        
    def forward(self, input):
        output = self.preprocess(input)
        skip_connections = [] # save for generation purposes
        for s, t, skip_scale, residue_scale in zip(self.conv_sigmoid, self.conv_tanh, self.skip_scale, self.residue_scale):
            output, skip = self.residue_forward(output, s, t, skip_scale, residue_scale)
            skip_connections.append(skip)
        # sum up skip connections
        output = sum([s[:,:,-output.size(2):] for s in skip_connections])
        output = self.postprocess(output)
        return output
    
    def preprocess(self, input):
        output = self.one_hot(input).unsqueeze(0).transpose(1,2)
        output = self.from_input(output)
        return output
    
    def postprocess(self, input):
        output = nn.functional.elu(input)
        output = self.conv_post_1(output)
        output = nn.functional.elu(output)
        output = self.conv_post_2(output).squeeze(0).transpose(0,1)
        return output
    
    def residue_forward(self, input, conv_sigmoid, conv_tanh, skip_scale, residue_scale):
        output = input
        output_sigmoid, output_tanh = conv_sigmoid(output), conv_tanh(output)
        output = nn.functional.sigmoid(output_sigmoid) * nn.functional.tanh(output_tanh)
        skip = skip_scale(output)
        output = residue_scale(output)
        output = output + input[:,:,-output.size(2):]
        return output, skip
    
    def generate_slow(self, input, n=100):
        res = input.data.tolist()
        for _ in range(n):
            x = Variable(torch.LongTensor(res[-sum(self.dilations)-1:])).to(self.device)
            y = self.forward(x)
            _, i = y.max(dim=1)
            res.append(i.data.tolist()[-1])
        return res
    
    def generate(self, input=None, n=100, temperature=None, estimate_time=False):
        ## prepare output_buffer
        output = self.preprocess(input)
        output_buffer = []
        for s, t, skip_scale, residue_scale, d in zip(self.conv_sigmoid, self.conv_tanh, self.skip_scale, self.residue_scale, self.dilations):
            output, _ = self.residue_forward(output, s, t, skip_scale, residue_scale)
            sz = 1 if d==2**(self.dilation_depth-1) else d*2
            output_buffer.append(output[:,:,-sz-1:-1])
        ## generate new 
        res = input.data.tolist()
        for i in range(n):
            output = Variable(torch.LongTensor(res[-2:])).to(self.device)
            output = self.preprocess(output)
            output_buffer_next = []
            skip_connections = [] # save for generation purposes
            for s, t, skip_scale, residue_scale, b in zip(self.conv_sigmoid, self.conv_tanh, self.skip_scale, self.residue_scale, output_buffer):
                output, residue = self.residue_forward(output, s, t, skip_scale, residue_scale)
                output = torch.cat([b, output], dim=2)
                skip_connections.append(residue)
                if i%100==0:
                    output = output.clone()
                output_buffer_next.append(output[:,:,-b.size(2):])
            output_buffer = output_buffer_next
            output = output[:,:,-1:]
            # sum up skip connections
            output = sum(skip_connections)
            output = self.postprocess(output)
            if temperature is None:
                _, output = output.max(dim=1)
            else:
                output = output.div(temperature).exp().multinomial(1).squeeze()
            res.append(output.data[-1])
        return res
    
    def generate_gap(self, input, t, framerate, t_start, t_end):
        #framerate = 1000
        #t = np.linspace(0,5,framerate*5)
    
        data_cpu = input.cpu()
        data = data_cpu.numpy()
        print('input size: '+str(data.size))
    
        i_start = (np.abs(t-t_start)).argmin()   #t[t_start] start index
        print('i_start: ', i_start)
        i_end = (np.abs(t-t_end)).argmin()       #t[t_end] end index
        print('i_end: ', i_end)
    
        data_left = data[:i_start]             #select data at left of gap
        print('number of data points before gap: '+str(data_left.size))
        data_right = data[i_end:]
        print('number of data points after gap: '+str(data_right.size))
    
        n_gap = np.int_((t_end-t_start)*framerate)        #n of samples to generate
        print('gap size: '+str(n_gap))
        generated_samples = np.empty([0, n_gap])
    
        for _ in range(n_gap):
        
            x = Variable(torch.LongTensor(data_left[-sum(self.dilations)-1:]).to(self.device)) #select receptive field data
            y = self.forward(x)
            _, i = y.max(dim=1)                   #predict missing values
            i_cpu = i.cpu()
            generated_samples = np.append(generated_samples, i_cpu.numpy()[-1]) #save generated samples
            data_left = np.append(data_left, i_cpu.numpy()[-1])       #append to the left
            
        return np.append(data_left, data_right), generated_samples #append completed left to right
    

# ## mu-law encode and decode

def encode_mu_law(x, mu=256):
    mu = mu-1
    fx = np.sign(x)*np.log(1+mu*np.abs(x))/np.log(1+mu)
    return np.floor((fx+1)/2*mu+0.5).astype(np.long)

def decode_mu_law(y, mu=256):
    mu = mu-1
    fx = (y-0.5)/mu*2-1
    x = np.sign(fx)/mu*((1+mu)**np.abs(fx)-1)
    return x  


# ## [-1, 1] normalisation

def normalise(x):
    y = (2*(x - x.min())/(x.max() - x.min())) - 1
    return y


# ## Load / Save functions

def save_checkpoint(state, filename):
    torch.save(state, filename)
    print("Checkpoint saved: " + filename)
    
def load_checkpoint(filename):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = WaveNet().to(device)

    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    loss_save = checkpoint['loss_save']
    optimizer = optim.Adam(net.parameters(),lr=0.01)
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    print("Loaded checkpoint: " + filename)
    return net, epoch, loss_save, optimizer    
