# https://datascienceub.medium.com/pointnet-implementation-explained-visually-c7e300139698
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import logging

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

# tranform image to 3D (x, y, binary value)
def img_to_3d(img):
    # get coordinates of pixels
    coords_x, coords_y = torch.meshgrid(torch.arange(0, img.size(1)), torch.arange(0, img.size(2)))
    coords_x = coords_x.flatten().float().unsqueeze(1)
    coords_y = coords_y.flatten().float().unsqueeze(1)
    values = img.view(-1).unsqueeze(1)
    pc = torch.cat((coords_x, coords_y, values), dim=1)
    return pc
  
class TransformationNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(TransformationNet, self).__init__()
        self.output_dim = output_dim

        self.conv_1 = nn.Conv1d(input_dim, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 256, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(256)
        self.bn_4 = nn.BatchNorm1d(256)
        self.bn_5 = nn.BatchNorm1d(128)

        self.fc_1 = nn.Linear(256, 256)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_3 = nn.Linear(128, self.output_dim*self.output_dim)

    def forward(self, x):
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = nn.MaxPool1d(num_points)(x)
        x = x.view(-1, 256)

        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)

        identity_matrix = torch.eye(self.output_dim)
        if torch.cuda.is_available():
            identity_matrix = identity_matrix.cuda()
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x


class BasePointNet(nn.Module):

    def __init__(self, point_dimension):
        super(BasePointNet, self).__init__()
        self.input_transform = TransformationNet(input_dim=point_dimension, output_dim=point_dimension)
        self.feature_transform = TransformationNet(input_dim=64, output_dim=64)
        
        self.conv_1 = nn.Conv1d(point_dimension, 64, 1)
        self.conv_2 = nn.Conv1d(64, 64, 1)
        self.conv_3 = nn.Conv1d(64, 64, 1)
        self.conv_4 = nn.Conv1d(64, 128, 1)
        self.conv_5 = nn.Conv1d(128, 256, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(256)
        

    def forward(self, x, plot=False):
        num_points = x.shape[1]
        
        input_transform = self.input_transform(x) # T-Net tensor [batch, 3, 3]
        x = torch.bmm(x, input_transform) # Batch matrix-matrix product 
        x = x.transpose(2, 1) 
        tnet_out=x.cpu().detach().numpy()
        
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = x.transpose(2, 1)

        feature_transform = self.feature_transform(x) # T-Net tensor [batch, 64, 64]
        x = torch.bmm(x, feature_transform)
        x = x.transpose(2, 1)
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = F.relu(self.bn_5(self.conv_5(x)))
        x = torch.sum(x, dim=2)  # summation
        x = x.view(-1, 256)  # global feature vector 
        # print(x)

        return x, feature_transform, tnet_out


class ClassificationPointNet(nn.Module):

    def __init__(self, num_classes, dropout=0.3, point_dimension=3):
        super(ClassificationPointNet, self).__init__()
        self.base_pointnet = BasePointNet(point_dimension=point_dimension)

        self.fc_1 = nn.Linear(256, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, num_classes)

        self.bn_1 = nn.BatchNorm1d(128)
        self.bn_2 = nn.BatchNorm1d(64)

        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        x, feature_transform, tnet_out = self.base_pointnet(x)

        x = F.relu(self.bn_1(self.fc_1(x)))
        x = F.relu(self.bn_2(self.fc_2(x)))
        x = self.dropout_1(x)

        return F.log_softmax(self.fc_3(x), dim=1), feature_transform, tnet_out
    
model = ClassificationPointNet(num_classes=10,
                                   point_dimension=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

fn_d = "Mar28_point_net"
logging.basicConfig(filename = fn_d + ".log",
                    level = logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

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
        for img in images:
            batch_pc.append(img_to_3d(img))
        pc = torch.stack(batch_pc, dim=0)
        pc = pc.to(torch.float32).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        model = model.train()
        preds, feature_transform, tnet_out = model(pc)

        identity = torch.eye(feature_transform.shape[-1])
        identity = identity.to(device)
        regularization_loss = torch.norm(
            identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1)))
        # Loss
        loss = F.nll_loss(preds, labels) + 0.001 * regularization_loss
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
        val_batch_pc = []
        for img in val_images:
            val_batch_pc.append(img_to_3d(img))
        val_pc = torch.stack(val_batch_pc, dim=0)
        val_pc = val_pc.to(torch.float32).to(device)
        val_labels = val_labels.to(device)
        model = model.eval()
        val_preds, feature_transform, tnet_out = model(val_pc)
        val_loss = F.nll_loss(val_preds, val_labels)
        epoch_test_loss.append(val_loss.cpu().item())
        val_preds = val_preds.data.max(1)[1]
        corrects = val_preds.eq(val_labels.data).cpu().sum()
        accuracy = corrects.item() / float(val_images.size(0))
        epoch_test_acc.append(accuracy)

    print('Epoch %s: train loss: %s, val loss: %f, train accuracy: %s,  val accuracy: %f'
              % (epoch,
                round(np.mean(epoch_train_loss), 4),
                round(np.mean(epoch_test_loss), 4),
                round(np.mean(epoch_train_acc), 4),
                round(np.mean(epoch_test_acc), 4)))
    
    logging.info('Epoch %s: train loss: %s, val loss: %f, train accuracy: %s,  val accuracy: %f'
        % (epoch,
        round(np.mean(epoch_train_loss), 4),
        round(np.mean(epoch_test_loss), 4),
        round(np.mean(epoch_train_acc), 4),
        round(np.mean(epoch_test_acc), 4)))

    # if np.mean(test_loss) < best_loss:
    #     state = {
    #         'model':model.state_dict(),
    #         'optimizer':optimizer.state_dict()
    #     }
    #     torch.save(state, os.path.join('checkpoints', '3Dmnist_checkpoint_%s.pth' % (number_of_points)))
    #     best_loss=np.mean(test_loss)

    # train_loss.append(np.mean(epoch_train_loss))
    # test_loss.append(np.mean(epoch_test_loss))
    # train_acc.append(np.mean(epoch_train_acc))
    # test_acc.append(np.mean(epoch_test_acc))

path_point_net = fn_d + ".pth"
torch.save(model.state_dict(), path_point_net)