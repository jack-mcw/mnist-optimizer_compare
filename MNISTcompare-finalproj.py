#Jack McWeeney
#ECE 579 Final Project
#With reference from https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
#and https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt 

# Preparing for Data
print('==> Preparing data..')

#LeNet5 structure: CONV‐POOL‐CONV‐POOL‐FC‐FC

class LeNet(nn.Module):
    def __init__(self):  
        super().__init__()  
        self.conv1=nn.Conv2d(1,6,5,1)  
        self.conv2=nn.Conv2d(6,16,5,1)  
        self.bn = nn.BatchNorm2d(16) 
        self.fc1=nn.Linear(5*5*16,120)  
        self.dropout=nn.Dropout(0.5)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)  
    def forward(self,x):  
        x=F.relu(self.conv1(x))  
        x=F.max_pool2d(x,2,2)  
        x=F.relu(self.conv2(x))  
        #x=F.max_pool2d(x,2,2)
        x=F.max_pool2d(self.bn(x),2,2)  
        x=x.view(-1,5*5*16)  
        x=F.relu(self.fc1(x)) 
        #x=self.dropout(x)  
        x=F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

accuracy_list = [[],[],[],[]]

def train(model, device, train_loader, optimizer, epoch, loss):
    model.train()
    correct = 0
    loss_t = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        outputs=model(data)
        loss1=loss(outputs,targets)  
        optimizer.zero_grad()  
        loss1.backward()  
        optimizer.step()  
        _,preds=torch.max(outputs,1)  
        loss_t+=loss1.item()  
        correct+=torch.sum(preds==targets.data)  

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_t))

def test( model, device, test_loader, i):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() 
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    accuracy_list[i].append(100. * correct / len(test_loader.dataset))

def main():
    time0 = time.time()
    batch_size = 64
    epochs = 20
    lr = 1e-3
    save_model = False
    torch.manual_seed(100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset = torchvision.datasets.MNIST(root = './data', train = True,
                                            transform = transforms.Compose([
                                                    transforms.Resize((32,32)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
                                            download = True)

    testset = torchvision.datasets.MNIST(root = './data', train = False,
                                            transform = transforms.Compose([
                                                    transforms.Resize((32,32)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),
                                            download=True)

    train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = testset, batch_size = batch_size,shuffle = True)
    
    model = LeNet().to(device)
    loss=nn.CrossEntropyLoss()  
    optimizers = [optim.Adam(model.parameters(), lr), optim.SGD(model.parameters(), lr), optim.Adagrad(model.parameters(), lr), optim.RMSprop(model.parameters(), lr)]

    for i in range(len(optimizers)):
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizers[i], epoch, loss)
            test( model, device, test_loader, i)

    if (save_model):
        torch.save(model.state_dict(),"cifar_lenet.pt")
    time1 = time.time() 

    plt.figure(figsize=(12,5))
    plt.title("Testing accuracy/epoch")
    plt.plot(torch.arange(len(accuracy_list[0])), accuracy_list[0], alpha=0.6, color='blue')
    plt.plot(torch.arange(len(accuracy_list[1])), accuracy_list[1], alpha=0.6, color='black')
    plt.plot(torch.arange(len(accuracy_list[2])), accuracy_list[2], alpha=0.6, color='orange')
    plt.plot(torch.arange(len(accuracy_list[3])), accuracy_list[3], alpha=0.6, color='green')
    plt.savefig("finalproj.png")

    print ('Training and Testing total execution time is: %s seconds ' % (time1-time0))
    for i in range(len(accuracy_list)):
        print('Max accuracy is', max(accuracy_list[i]))
        print('Average accuracy is', sum(accuracy_list[i])/len(accuracy_list[i]))
   
if __name__ == '__main__':
    main()
