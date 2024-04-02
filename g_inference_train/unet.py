import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from torchvision import models
from torch.nn.functional import relu

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
    def __init__(self, n_class):
        super().__init__()
        
        # Encoder
        self.e11 = nn.Conv2d(11, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = F.relu(self.e11(x))
        xe12 = F.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(self.e21(xp1))
        xe22 = F.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(self.e31(xp2))
        xe32 = F.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(self.e41(xp3))
        xe42 = F.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = F.relu(self.e51(xp4))
        xe52 = F.relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu1 = F.interpolate(xu1, size=xu1.size()[2:], mode='bilinear', align_corners=True)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(self.d11(xu11))
        xd12 = F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu2 = F.interpolate(xu2, size=xu2.size()[2:], mode='bilinear', align_corners=True)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.d21(xu22))
        xd22 = F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu3 = F.interpolate(xu3, size=xu3.size()[2:], mode='bilinear', align_corners=True)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(self.d31(xu33))
        xd32 = F.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu4 = F.interpolate(xu4, size=xu4.size()[2:], mode='bilinear', align_corners=True)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.relu(self.d41(xu44))
        xd42 = F.relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out


    
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
model = UNet(n_class=10)
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

        # Loss
        loss = F.nll_loss(preds, labels)
        epoch_train_loss.append(loss.cpu().item())
        loss.backward()
        optimizer.step()
        preds = preds.data.max(1)[1]
        corrects = preds.eq(labels.data).cpu().sum()

        accuracy = corrects.item() / float(images.size(0))
        epoch_train_acc.append(accuracy)

    epoch_test_loss = []
    epoch_test_acc = []

    # validation loop
    for i, (val_images, val_labels) in enumerate(testloader):
        val_pc = []
        val_pc = img_block_pos_expand(val_images, base_num)
        val_pc = val_pc.to(torch.float32).to(device)
        val_pc = val_pc.view(images.size(0), 11, 28, 28)
        val_labels = val_labels.to(device)
        model = model.eval()
        output = model(val_pc)
        val_loss = F.nll_loss(val_preds, val_labels)
        epoch_test_loss.append(val_loss.cpu().item())
        val_preds = val_preds.data.max(1)[1]
        corrects = val_preds.eq(val_labels.data).cpu().sum()
        accuracy = corrects.item() / float(val_images.size(0))
        epoch_test_acc.append(accuracy)

    output_str = 'Epoch %s: train loss: %s, val loss: %f, train accuracy: %s,  val accuracy: %f,  time: %.2f'\
                % (epoch,
                round(np.mean(epoch_train_loss), 4),
                round(np.mean(epoch_test_loss), 4),
                round(np.mean(epoch_train_acc), 4),
                round(np.mean(epoch_test_acc), 4),
                (time.time() - last_epoch_time))
    last_epoch_time = time.time()
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