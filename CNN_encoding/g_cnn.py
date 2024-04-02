import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import logging

class g_net(nn.Module):
    def __init__(self):
        super(g_net, self).__init__()
        # Perform convolution while retaining dimensionality
        # (batch_size, 1, 1, 784) -> (bs, 16, 1, 784)
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=3, stride = 1, padding=1),  
            # nn.MaxPool2d(kernel_size=1, stride = 1)  
        )
        # Perform convolution while retaining dimensionality
        # (batch_size, 16, 1, 784) -> (bs, 32, 1, 784)  
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=3, stride = 1, padding=1),  
            # nn.MaxPool2d(kernel_size=1, stride = 1)               
        )
        # fully connected layer to map conv outputs to hidden dimension of rnn
        self.fc1 = nn.Sequential(
            nn.Linear(32*28*28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        # fully connected layer to map from latent space to output        
        self.fc2 = nn.Linear(512, 10)
        self.fc3 = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )
        # softmax for the class dimension
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # print(f"x shape before conv: {x.size()}")
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32*28*28)
        x = x.view(x.size(0), -1)   
        x = self.fc1(x)
        x = self.fc2(x)
        # print(f"x shape before sum: {x.size()}")
        # x = x.view(x.size(0), 32, 1, 784)
        # x = torch.sum(x, dim=-1).view(x.size(0), -1)
        # print(f"x shape: {x.size()}")
        # x = self.fc3(x)
        x = self.softmax(x)  
        return x

params = {
    'lr': 0.001,
    'epochs': 100,
    'batch_size': 64,
    'num_classes': 10
}

# set device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=16)
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=16)

num_classes = params['num_classes']
lr = params['lr']
num_epochs = params['epochs']

fn = "Mar27_g_cnn"

# configurate logging function
logging.basicConfig(filename = fn + '.log',
                    level = logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

# Initialize model
model = g_net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-6)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(params['epochs']):
    train_loss = []
    for i, (images, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        # Flatten the image to make pixels sequential
        # print(f"x shape before flatten: {images.size()}")
        flat_image = images.view(images.size(0), 1, 1, -1)
        flat_image = flat_image.to(torch.float32).to(device)
        labels = labels.to(device)
        # Train model
        # print(f"x shape before model: {flat_image.size()}")
        outputs = model(flat_image)
        loss = loss_fn(outputs, labels)
        # Back Propagation
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    test_loss = []
    test_accuracy = []

    for i, (data, labels) in enumerate(testloader):
        data = data.to(device)
        labels = labels.to(device)
        # Test current model
        outputs = model(data)
        # Calculate the predicted class based on largest probability value
        _, predicted = torch.max(outputs.data, 1)
        loss = loss_fn(outputs, labels)
        test_loss.append(loss.item())
        test_accuracy.append((predicted == labels).sum().item() / predicted.size(0))

    print('epoch: {}, train loss: {}, test loss: {}, test accuracy: {}'.format(epoch, np.mean(train_loss), np.mean(test_loss), np.mean(test_accuracy)))
    logging.info('epoch: {}, train loss: {}, test loss: {}, test accuracy: {}'.format(epoch, np.mean(train_loss), np.mean(test_loss), np.mean(test_accuracy)))

# save model
PATH_g = fn + ".pth"
torch.save(model.state_dict(), PATH_g)