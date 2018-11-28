from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sgd_lr_adjust import SGDAdjustOptimizer

from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from collections import namedtuple
from sacred import Experiment
from sacred.stflow import LogFileWriter
import os
import shutil

ex = Experiment('torch_mnist_tutorial')
from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create(url='gpu-plx01.ef.technion.ac.il', db_name='torch_mnist_tutorial_db'))

def allocate_tensorboard_dir():
    BASE_DIR = '/home/shai/tensorflow/generated_data'
    used = [int(x) for x in shutil.os.listdir(BASE_DIR) if x.isdigit()]
    if len(used) == 0:
        return BASE_DIR + '/' + str(1)

    path = BASE_DIR + '/' + str(max(used) + 1)
    if not os.path.exists(path):
        os.mkdir(path)

    return path

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train(model, train_loader, optimizer, epoch, args, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.data[0]))

def test(model, test_loader, args, writer):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     accuracy))
    return test_loss, accuracy


class Config:
    def __init__(self, batch_size, test_batch_size, epochs, lr, momentum, cuda, seed, log_interval):
        pass


Config = namedtuple('Config',
                     'batch_size, test_batch_size, epochs, lr, momentum, cuda, seed, log_interval, tensorboard_dir')
@ex.config
def my_config():
    batch_size = 100
    test_batch_size = 1000
    epochs = 30
    lr = 0.1
    momentum = 0.5
    cuda = True
    seed = 1
    log_interval = 10
    tensorboard_dir = allocate_tensorboard_dir()

    #iters_per_adjust = 250
    #default to twise an epoch
    iters_per_adjust = int((60000/batch_size)/2)

    disable_lr_change = False
@ex.automain
@LogFileWriter(ex)
def my_main(batch_size, test_batch_size, epochs, lr, momentum, cuda, seed, log_interval, iters_per_adjust, disable_lr_change,
            tensorboard_dir):
    args = Config(batch_size, test_batch_size, epochs, lr, momentum, cuda, seed, log_interval, tensorboard_dir)

    writer = SummaryWriter(args.tensorboard_dir)

    torch.manual_seed(seed)
    if args.cuda == True:
        torch.cuda.manual_seed(args.seed)


    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net()
    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    optimizer = SGDAdjustOptimizer(base_optimizer=optimizer, iters_per_adjust=iters_per_adjust, writer=writer, disable_lr_change=disable_lr_change)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, epoch, args, writer)
        test_loss, test_accuracy = test(model, test_loader, args, writer)

        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_accuracy', test_accuracy, epoch)

    writer.close()

