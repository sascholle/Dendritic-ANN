#%% IMPORTS

import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets 
import torchvision.transforms as transforms
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% DATASET DOWNLOAD

train_data = datasets.MNIST(root = 'data', train = True, download = True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root = 'data', train = False, download = True, transform=transforms.ToTensor())

img, label = train_data[0]
plt.imshow(img.reshape(28,28), cmap='gray')

data_loader = DataLoader(train_data, batch_size=100, shuffle=True)
data_loader_test = DataLoader(test_data, batch_size=100, shuffle=False)

#%% MODEL

class Ann(nn.Module):
    """
    a simple ANN
    """

    def __init__(self, in_features =784, out_features=10):
        super(Ann, self).__init__()

        self.layer1 = nn.Linear(in_features,120)
        self.layer2 = nn.Linear(120,84)
        self.layer3 = nn.Linear(84, out_features)

    def forward(self, X):
        output1 = self.layer1(X)
        output1 = F.relu(output1)
        output2 = self.layer2(output1)
        output2 = F.relu(output2)
        output3 = self.layer3(output2)
        output3 = F.softmax(output3, dim = 1)

        return output3
        

#%% MODEL AND PARAMETERS
    
model = Ann()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

#%% TRAINING

epochs = 1
train_losses = []
test_losses = []
accuracy = []

for epoch in tqdm(range(epochs)):
    for batch, (img, label) in enumerate(data_loader):
        y_pred = model(img.view(100,-1))
        loss = criterion(y_pred,label)
        #_, prediction = torch.max(y_pred, dim=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch%200==0:
            acc=torch.sum(torch.argmax(y_pred, dim=1) == label).item()/len(label)
            accuracy.append(acc)
            print( f'Epoch:{epoch:2d} batch: {batch:2d} loss: {loss.item():4.4f} Accuracy: {acc:4.4f} %' )


# %% PLOTTING

print(accuracy[0:10])
        
plt.plot(accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.show()
