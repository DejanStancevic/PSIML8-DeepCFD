# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 21:28:18 2022

@author: psiml8
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F


#%% Loading dataX file

A = pd.read_pickle(r'C:\Users\psiml8\Desktop\DeepCFD\dataX.pkl')
B = pd.read_pickle(r'C:\Users\psiml8\Desktop\DeepCFD\dataY.pkl')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


#%% Seeing input and output data

        
# =============================================================================
# for r in range(len(A[0][0])):
#     for c in range(len(A[0][0][0])):
#         if A[0][0][r][c] < -49.5:
#             A[0][0][r][c] = -0.5
# =============================================================================
        


sns.heatmap(B[0][0])
plt.show()

#%%
A = torch.tensor(A)

B = torch.tensor(B)


#%%
input_set = [0, 1, 2]

data_set = [(A[i, input_set], B[i, [0,1]]) for i in range(781)] #A.size(dim=0))]



#%%
#print((data_set[0][1]).size())

#%%

batch_size = 64
lr = 0.0001


trainloader = torch.utils.data.DataLoader(data_set, batch_size = batch_size, shuffle = True)

#%% MODEL
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        k = 5 #kernel size of convolutions
        
        self.conv1 = nn.Conv2d(3, 8, (k, 4))
        self.conv2 = nn.Conv2d(8, 8, k)
        self.conv3 = nn.Conv2d(8, 16, k)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)
        self.conv4 = nn.Conv2d(16, 16, k)
        self.conv5 = nn.Conv2d(16, 32, k)
        self.pool2 = nn.MaxPool2d(2, return_indices=True)
        self.conv6 = nn.Conv2d(32, 32, k)
        self.conv7 = nn.Conv2d(32, 32, k)
        #self.pool3 = nn.MaxPool2d(2)
        self.conv8 = nn.Conv2d(32, 32, k)
        
        self.deconv0 = nn.ConvTranspose2d(32, 32, k)
        #self.unpool3 = nn.MaxUnpool2d(2)
        self.deconv1 = nn.ConvTranspose2d(64, 32, k)
        self.deconv2 = nn.ConvTranspose2d(32, 32, k)
        self.unpool2 = nn.MaxUnpool2d(2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, k)
        self.deconv4 = nn.ConvTranspose2d(16, 16, k)
        self.unpool1 = nn.MaxUnpool2d(2)
        self.deconv5 = nn.ConvTranspose2d(32, 8, k)
        self.deconv6 = nn.ConvTranspose2d(8, 8, k)
        self.deconv7 = nn.ConvTranspose2d(16, 1, (k, 4))
        
        self.deconv0b = nn.ConvTranspose2d(32, 32, k)
        #self.unpool3b = nn.MaxUnpool2d(2)
        self.deconv1b = nn.ConvTranspose2d(64, 32, k)
        self.deconv2b = nn.ConvTranspose2d(32, 32, k)
        self.unpool2b = nn.MaxUnpool2d(2)
        self.deconv3b = nn.ConvTranspose2d(64, 16, k)
        self.deconv4b = nn.ConvTranspose2d(16, 16, k)
        self.unpool1b = nn.MaxUnpool2d(2)
        self.deconv5b = nn.ConvTranspose2d(32, 8, k)
        self.deconv6b = nn.ConvTranspose2d(8, 8, k)
        self.deconv7b = nn.ConvTranspose2d(16, 1, (k, 4))
        
        

    def forward(self, x):
        x_cat1 = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x_cat1)))
        x_cat2 = (F.relu(self.conv3(x)))
        x, indices1 = self.pool1(x_cat2)
        x = (F.relu(self.conv4(x)))
        x_cat3 = (F.relu(self.conv5(x)))
        x, indices2 = self.pool2(x_cat3)
        x = (F.relu(self.conv6(x)))
        x_cat4 = (F.relu(self.conv7(x)))
        #x_cat4 = self.pool3(x_cat4)
        x = (F.relu(self.conv8(x_cat4)))
        
        x1 = (F.relu(self.deconv0(x)))
        #x1 = self.unpool3(x1)
        x1 = torch.cat((x1, x_cat4), 1)
        x1 = (F.relu(self.deconv1(x1)))
        x1 = (F.relu(self.deconv2(x1)))
        x1 = self.unpool2(x1, indices2)
        x1 = torch.cat((x1, x_cat3), 1)
        x1 = (F.relu(self.deconv3(x1)))
        x1 = (F.relu(self.deconv4(x1)))
        x1 = self.unpool1(x1, indices1)
        x1 = torch.cat((x1, x_cat2), 1)
        x1 = (F.relu(self.deconv5(x1)))
        x1 = (F.relu(self.deconv6(x1)))
        x1 = torch.cat((x1, x_cat1), 1)
        x1 = ((self.deconv7(x1)))
        
        x2 = (F.relu(self.deconv0b(x)))
        #x2 = self.unpool3b(x2)
        x2 = torch.cat((x2, x_cat4), 1)
        x2 = (F.relu(self.deconv1b(x2)))
        x2 = (F.relu(self.deconv2b(x2)))
        x2 = self.unpool2b(x2, indices2)
        x2 = torch.cat((x2, x_cat3), 1)
        x2 = (F.relu(self.deconv3b(x2)))
        x2 = (F.relu(self.deconv4b(x2)))
        x2 = self.unpool1b(x2, indices1)
        x2 = torch.cat((x2, x_cat2), 1)
        x2 = (F.relu(self.deconv5b(x2)))
        x2 = (F.relu(self.deconv6b(x2)))
        x2 = torch.cat((x2, x_cat1), 1)
        x2 = ((self.deconv7b(x2)))
        

        return x1, x2
    
model = Net()

model.to(device)

#%% LOSS and OPTIMIZER
# =============================================================================
# def my_loss(output, target):
#     loss = 0
#     for i in range(len(output)):
#         loss += torch.mean((output[i] - target[:, i])**2)
#         #loss += torch.mean(abs(output[i] - target[:, i]))
#     return loss
# =============================================================================

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    


#%% TRAINING
LOSS = []

my_loss = nn.MSELoss()

for epoch in range(1000):  # loop over the dataset multiple times
    print(epoch)


    for i, (inputs, targets) in enumerate(trainloader):
        
        inputs = inputs.to(device)
        targets = targets.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        #loss = my_loss(outputs, targets)
        loss = my_loss(outputs[0][:, 0], targets[:, 0]) + my_loss(outputs[1][:, 0], targets[:, 1])
        loss.backward()
        optimizer.step()
        
        LOSS.append(loss)


print('Finished Training')


#%%
# =============================================================================
# print(model.parameters)
# for parm in model.parameters():
#     print(parm)
#     break
# =============================================================================

#%%
#print(LOSS[-1])

#%%
T = A[11, input_set]

T = T[None,:]

T = T.to(device)

M1 = model(T)[1]

#M1 = abs(M1 - B[0][1].to(device))

#M1 = targets

M1 = M1.cpu()
M = M1.detach().numpy()
sns.heatmap(M[0][0])
plt.show()

#%%

def my_validation_loss(output, target):
    loss = 0
    for i in range(len(output)):
        loss += torch.sum((output[i] - target[:, i])**2)
    return loss



validation_set = [(A[i, input_set], B[i, [0,1]]) for i in range(781, 881)]

validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = 20)

validation_loss = []

for i, (inputs, targets) in enumerate(validation_loader):
    
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    outputs = model(inputs)
    loss = my_validation_loss(outputs, targets)
    
    validation_loss.append(loss)
    
print(sum(validation_loss)/100)


#%%
print(batch_size, lr)




