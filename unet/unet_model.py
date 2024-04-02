""" Full assembly of the parts to form the complete network """
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from unet_parts import *


# load MNIST dataset, convert to binary pixel values
mnist_train = datasets.MNIST(root='./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Lambda(lambda x: torch.where(x > 0,1,0))
                             ]))
trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=False)
mnist_test = datasets.MNIST(root='./data', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Lambda(lambda x: torch.where(x > 0,1,0))
                             ]))
testloader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)
        self.fc = nn.Sequential(
            nn.Linear(64, 10),
            nn.BatchNorm1d(10), 
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.inc(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # x1 = self.inc(x)
        # print(f"x1: {x1.size()}")
        # x2 = self.down1(x1)
        # print(f"x2: {x2.size()}")
        # x3 = self.down2(x2)
        # print(f"x3: {x3.size()}")
        # x4 = self.down3(x3)
        # print(f"x4: {x4.size()}")
        # x5 = self.down4(x4)
        # print(f"x5: {x5.size()}")
        # x = self.up1(x5, x4)
        # print(f"x: {x.size()}")
        # x = self.up2(x, x3)
        # print(f"x: {x.size()}")
        # x = self.up3(x, x2)
        # print(f"x: {x.size()}")
        # x = self.up4(x, x1)
        # print(f"x: {x.size()}")
        # return
        # logits = self.outc(x)
        # x = self.outc(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.sum(x, dim=2).view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
def img_block_pos_expand(img_block, base):
    dim = img_block.shape
    values = img_block.view(dim[0], -1, 1)
    spacial_encoding_mat = torch.from_numpy(get_pos_matrix(values.shape[1], base_num=base)).T
    spacial_encoding_mat = spacial_encoding_mat.expand(dim[0], spacial_encoding_mat.shape[0], spacial_encoding_mat.shape[1])
    pc = torch.cat((spacial_encoding_mat, values), dim=2)
    return pc

def get_pos_matrix(n, base_num):
    # get the number of binary digits to represent n
    dim = int(np.ceil(np.log(n) / np.log(base_num)))

    mat = np.zeros((dim, base_num**dim), dtype=int)
    # basic pattern
    base = np.array([range(base_num)])

    for ii in range(dim):
        # number of same numbers in a row (repeat along row)
        unit = np.repeat(base, base_num**(ii), axis=1)
        # number of repeats (repeat along column)
        full = np.repeat(unit, base_num**(dim-ii-1), axis=0)
        mat[ii] = full.flatten()
    norm_mat = mat / base_num  # will give error from 0/0, but not important
    return norm_mat[:, 1:n+1]

fn = "Mar30_unet"
base_num = 2

path_point_net = fn + ".pth" 
# state_dict = torch.load(path_point_net)
model = UNet(n_channels=11, n_classes=10)
# model.load_state_dict(state_dict)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

logging.basicConfig(filename = fn + ".log",
                    level = logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

# training model
epochs=100
train_loss = []
test_loss = []
train_acc = []
test_acc = []
best_loss= np.inf

for epoch in range(epochs):
    epoch_train_loss = []
    epoch_train_acc = []

    # training loop
    for i, (images, labels) in enumerate(trainloader):
        batch_pc = []
        # print(type(images), images.shape)
        batch_pc = img_block_pos_expand(images, base_num)
        pc = batch_pc.to(torch.float32).to(device)
        # print(pc.size())
        pc = pc.view(images.size(0), 11, 28, 28)
        labels = labels.to(device)
        optimizer.zero_grad()
        model = model.train()
        output = model(pc)
        # print(f"---output size: {output.size()}---")

        # Loss
        loss = F.nll_loss(output, labels)
        epoch_train_loss.append(loss.cpu().item())
        loss.backward()
        optimizer.step()
        output = output.data.max(1)[1]
        corrects = output.eq(labels.data).cpu().sum()

        accuracy = corrects.item() / float(images.size(0))
        epoch_train_acc.append(accuracy)

    epoch_test_loss = []
    epoch_test_acc = []

    # validation loop
    for i, (val_images, val_labels) in enumerate(testloader):
        val_pc = []
        val_pc = img_block_pos_expand(val_images, base_num)
        val_pc = val_pc.to(torch.float32).to(device)
        val_pc = val_pc.view(val_images.size(0), 11, 28, 28)
        val_labels = val_labels.to(device)
        model = model.eval()
        output = model(val_pc)
        val_loss = F.nll_loss(output, val_labels)
        epoch_test_loss.append(val_loss.cpu().item())
        output = output.data.max(1)[1]
        corrects = output.eq(val_labels.data).cpu().sum()
        accuracy = corrects.item() / float(val_images.size(0))
        epoch_test_acc.append(accuracy)

    output_str = 'Epoch %s: train loss: %s, val loss: %f, train accuracy: %s,  val accuracy: %f'\
                % (epoch,
                round(np.mean(epoch_train_loss), 4),
                round(np.mean(epoch_test_loss), 4),
                round(np.mean(epoch_train_acc), 4),
                round(np.mean(epoch_test_acc), 4))
    print(output_str)
    logging.info(output_str)

    if np.mean(test_loss) < best_loss:
        state = {
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict()
        }
        # torch.save(state, os.path.join('checkpoints', '3Dmnist_checkpoint_%s.pth' % (number_of_points)))
        best_loss=np.mean(test_loss)

    train_loss.append(np.mean(epoch_train_loss))
    test_loss.append(np.mean(epoch_test_loss))
    train_acc.append(np.mean(epoch_train_acc))
    test_acc.append(np.mean(epoch_test_acc))

torch.save(model.state_dict(), path_point_net)

