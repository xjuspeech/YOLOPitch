import torch
import torch.nn.functional as F
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class Snake(nn.Module):
    def __init__(self, a=5):
        super(Snake, self).__init__()
        self.a = a
    def forward(self, x):
        return (x + (torch.sin(self.a * x) ** 2) / self.a)
    
class LSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=False):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden

def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale

def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference) 

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result  
    
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PNP(nn.Module):
    def __init__(self,chan=6,ker_size=4,std=4):
        super(PNP,self).__init__()
        
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels= chan ,kernel_size=(1,ker_size),stride=(1,std),padding=(0,0)),
            nn.ReLU(),
            nn.Conv2d(in_channels= chan ,out_channels= chan ,kernel_size=(1,1),stride=(1,1)),
            Snake(a=17)
        )

        self.np1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels= chan,kernel_size=(1,ker_size),stride=(1,std),padding=(0,0)),
            nn.ReLU(),
            nn.Conv2d(in_channels= chan,out_channels= chan,kernel_size=(1,1),stride=(1,1)),
            Snake(a=0.2)
        )

        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels= chan,out_channels= chan*2,kernel_size=(1,ker_size),stride=(1,std),padding=(0,0)),
            nn.ReLU(),
            nn.Conv2d(in_channels= chan*2,out_channels= chan*2,kernel_size=(1,1),stride=(1,1)),
            Snake(a=13)
        )

        self.np2 = nn.Sequential(
            nn.Conv2d(in_channels= chan,out_channels= chan*2,kernel_size=(1,ker_size),stride=(1,std),padding=(0,0)),
            nn.ReLU(),
            nn.Conv2d(in_channels= chan*2,out_channels= chan*2,kernel_size=(1,1),stride=(1,1)),
            Snake(a=0.2)
        )

        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels= chan *2,out_channels= chan*4,kernel_size=(1,ker_size*2),stride=(1,std),padding=(0,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels= chan*4,out_channels= chan*4,kernel_size=(1,1),stride=(1,1)),
            Snake(a=11)
        )

        self.p4 = nn.Sequential(
            nn.Conv2d(in_channels= chan*4,out_channels= chan*8,kernel_size=(1,ker_size*2),stride=(1,std),padding=(0,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels= chan*8,out_channels= chan*8,kernel_size=(1,1),stride=(1,1)),
            Snake(a=7)
        )

        self.p5 = nn.Sequential(
            nn.Conv2d(in_channels= chan*8,out_channels= chan*16,kernel_size=(1,ker_size*3),stride=(1,std),padding=(0,4)),
            nn.ReLU(),
            nn.Conv2d(in_channels= chan*16,out_channels= chan*16,kernel_size=(1,1),stride=(1,1)),
            Snake(a=5)
        )

        self.lstm = LSTM(6*16, bi=False)

    def forward(self, x):
        x1 = self.p1(x) + self.np1(x)
        x2 = self.p2(x1) + self.np2(x1)
        x3 = self.p3(x2)
        x4 = self.p4(x3)
        x5 = self.p5(x4)

        x6, _ = self.lstm(x5.squeeze(0).transpose(0,2))
        x6 = x6.transpose(0,2).unsqueeze(0)
        # x6, _ = self.lstm(x5.squeeze(0)).unsqueeze(0)

        # rescale = 0.1
        # if rescale:
        #     rescale_module(self, reference=rescale)

        return x2,x3,x4,x5,x6


class SeparableConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels=None, norm=True, activation=False, orig_swish=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        
        self.depthwise_conv = nn.Conv2d(in_channels,in_channels,kernel_size=5, stride=1,padding=2,bias=False,groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels,out_channels,kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
        
        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not orig_swish else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

class BiFPN(nn.Module):
    def __init__(self, num_channels, epsilon=1e-8, orig_swish=False):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        
        # Pre resize
        self.p1_upch = SeparableConvBlock(num_channels//4, num_channels)
        self.p2_upch = SeparableConvBlock(num_channels//2, num_channels)
        self.p4_dnch = SeparableConvBlock(num_channels*2, num_channels)
        self.p5_dnch = SeparableConvBlock(num_channels*2, num_channels)

        self.p4_upsample = nn.Upsample(scale_factor=(1,2), mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=(1,2), mode='nearest')

        # BiFPN conv layers
        self.conv6_up = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv5_up = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv4_up = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv3_up = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv4_down = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv5_down = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv6_down = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv7_down = SeparableConvBlock(num_channels, orig_swish=orig_swish)
    

        self.swish = MemoryEfficientSwish() if not orig_swish else Swish()

        # Weight
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_w1_relu = nn.ReLU()
        self.p1_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p1_w1_relu = nn.ReLU()

        self.p2_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p2_w2_relu = nn.ReLU()
        self.p3_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w2_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()

    def pre_resize(self, inputs):
        # [B,12,16020], [B,24,4004], [B,48,1000], [B,96,249], [B,96,249]
        p1, p2, p3, p4, p5 = inputs
        
        # Pre resizing
        p1 = self.p1_upch(F.avg_pool2d(p1, (1,32)))
        # f1: [B,48,500,1]
        p2 = self.p2_upch(F.avg_pool2d(p2, (1,8)))
        # f2: [B,48,500,1]
        p3 = F.avg_pool2d(p3, (1,2))
        # p4 = self.p4_dnch(self.p4_upsample(F.pad(p4,(0,0,2,1))))
        p4 = self.p4_dnch(self.p4_upsample(p4))
        # f4: [B,48,500,1]
        # p5 = self.p5_dnch(self.p5_upsample(F.pad(p5,(0,0,2,1))))
        p5 = self.p5_dnch(self.p4_upsample(p5))
        # f5: [B,48,500,1]
        return p1, p2, p3, p4, p5
        

    def forward(self, inputs):
        # The BiFPN illustration is an upside down form of the figure in the paper.
        """
        Illustration of a bifpn layer unit
            p5_in ---------------------------> p5_out -------->
               |---------------|                  ↑
                               ↓                  |
            p4_in ---------> p4_mid ---------> p4_out -------->
               |---------------|----------------↑ ↑
                               ↓                  |
            p3_in ---------> p3_mid ---------> p3_out -------->
               |---------------|----------------↑ ↑
                               ↓                  |
            p2_in ---------> p2_mid ---------> p2_out -------->
               |---------------|----------------↑ ↑
                               |------------ ---↓ |
            p1_in ---------------------------> p1_out -------->
        """
        
        p1_in, p2_in, p3_in, p4_in, p5_in = self.pre_resize(inputs)
        # [B,48,500,1], [B,48,500,1], [B,48,500,1], [B,48,500,1], [B,48,500,1]  [B,48,3,2]
        
        # BiFPN operation
        ## Top-bottom process
        # Weights for p4_in and p5_in to p4_mid
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for p4_in and p5_in to p4_mid respectively
        p4_mid = self.conv6_up(self.swish(weight[0] * p4_in + weight[1] * (p5_in)))

        # Weights for p3_in and p4_mid to p3_mid
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for p3_in and p4_mid to p3_mid respectively
        p3_mid = self.conv5_up(self.swish(weight[0] * p3_in + weight[1] * (p4_mid)))

        # Weights for p2_in and p3_mid to p2_mid
        p2_w1 = self.p2_w1_relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        # Connections for p2_in and p3_mid to p2_mid respectively
        p2_mid = self.conv4_up(self.swish(weight[0] * p2_in + weight[1] * (p3_mid)))

        # Weights for p1_in and p2_mid to p1_out
        p1_w1 = self.p1_w1_relu(self.p1_w1)
        weight = p1_w1 / (torch.sum(p1_w1, dim=0) + self.epsilon)
        # Connections for p1_in and p2_mid to p1_out respectively
        p1_out = self.conv3_up(self.swish(weight[0] * p1_in + weight[1] * (p2_mid)))

        ## Down-Up process
        # Weights for p2_in, p2_mid and p1_out to p2_out
        p2_w2 = self.p2_w2_relu(self.p2_w2)
        weight = p2_w2 / (torch.sum(p2_w2, dim=0) + self.epsilon)
        # Connections for p2_in, p2_mid and p1_out to p2_out respectively
        p2_out = self.conv4_down(self.swish(weight[0] * p2_in + weight[1] * p2_mid + weight[2] * (p1_out)))

        # Weights for p3_in, p3_mid and p2_out to p3_out
        p3_w2 = self.p3_w2_relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        # Connections for p3_in, p3_mid and p2_out to p3_out respectively
        p3_out = self.conv5_down(self.swish(weight[0] * p3_in + weight[1] * p3_mid + weight[2] * (p2_out)))

        # Weights for p4_in, p4_mid and p3_out to p4_out
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for p4_in, p4_mid and p3_out to p4_out respectively
        p4_out = self.conv6_down(self.swish(weight[0] * p4_in + weight[1] * p4_mid + weight[2] * (p3_out)))

        # Weights for p5_in and p4_out to p5_out
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for p5_in and p4_out to p5_out
        p5_out = self.conv7_down(self.swish(weight[0] * p5_in + weight[1] * (p4_out)))

        return p1_out, p2_out, p3_out, p4_out, p5_out
    

class Estimation_stage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_channels = 48
        self.a_stage = PNP()
        self.light_bifpn = BiFPN(num_channels=num_channels)
        self.ln = (nn.Linear(in_features=40, out_features=1))
        self.fc = nn.Linear(480,550)

    def forward(self, input):   
        input = input.unsqueeze(0)
        p1, p2, p3, p4, p5 = self.a_stage(input)
        # [B,12,16020], [B,24,4004], [B,48,1000], [B,96,249], [B,96,249]   ## *4+4
        
        # p1 = p1[:,:,:,None]
        # p2 = p2[:,:,:,None]
        # p3 = p3[:,:,:,None]
        # p4 = p4[:,:,:,None]
        # p5 = p5[:,:,:,None]

        features = (p1, p2, p3, p4, p5)
        f0_features = self.light_bifpn(features)
        # [B,48,500,1], [B,48,500,1], [B,48,500,1], [B,48,500,1], [B,48,500,1]
        " 500 = frame number while the input_length is 4sec & hop_lenghth is 128 & sampling rate is 16 kHz"

        f0_feature = torch.cat(f0_features, dim=1)
        # f0_feature: [B, 48*5=240, 500, 1]
        # f0_feature = f0_feature.permute(0,2,1,3).squeeze(3)
        f0_feature = f0_feature.permute(0,2,1,3)
        f0_feature = f0_feature.contiguous().view(f0_feature.shape[0],-1,f0_feature.shape[2]*f0_feature.shape[3])
        # f0_feature: [B, 500, 240]
        # f0_feature = F.avg_pool1d(f0_feature,6)
        # f0_feature: [B, 500, 40]
        f0out = self.fc(f0_feature).squeeze(0)
        # [B, 500]
        # f0out = torch.sigmoid(f0out)

        return f0out



if __name__ == '__main__':
    import librosa
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # path = '/ssdhome/lixuefei/data/PTDB/SPEECH_DATA_2/wav/mic_F01_sa1.wav'
    # a,sr = librosa.load(path,sr=16000)
    # # a = torch.rand(1,1,1024).to(device)
    # a = torch.from_numpy(a).reshape(1,1,-1).to(device)
    a = torch.rand(1,3,1024).to(device)
    # print(a.shape)
    net = Estimation_stage().to(device)
    # net = PNP().to(device)
    b = net(a)
    print(b.shape)
    num = sum(p.numel() for p in net.parameters())
    print(num)