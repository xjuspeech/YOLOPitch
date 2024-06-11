import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class Crepe(nn.Module):
    def __init__(self):
        super(Crepe, self).__init__()
        #a=torch.randn(1,1024)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=1024,kernel_size=(512,1),padding=(254,0),stride=(4,1)),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=(64,1), padding=(32,0),stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25)
        )
   
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(64,1), padding=(32,0), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25)
        )
  
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(64,1), padding=(32,0), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25)
        )
    
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(64,1), padding=(32,0), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.25)
        )
       
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(64,1), padding=(32,0), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(0.25)
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2048,551)
        

    def forward(self, x):
        
        x = torch.unsqueeze(x,0)
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.conv6(x)
        x = x.view(2048,-1).transpose(0,1)
        x = self.linear(x)

        return x

