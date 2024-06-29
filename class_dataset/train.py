import os
import sys
import json
import time
import torch
import torch.optim as optim
from torchvision import transforms, datasets
import torch.nn as nn
from model import resnet34
from utils import train_and_val,plot_acc,plot_loss
import numpy as np


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if not os.path.exists('./weight'):
        os.makedirs('./weight')

    BATCH_SIZE = 16

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # train_dataset = datasets.ImageFolder("F:/class_dataset/archive/training/training/", transform=data_transform["train"])  # train data

    train_dataset = datasets.ImageFolder("class_dataset/train/",
                                         transform=data_transform["train"])   # train data
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=4) 
    len_train = len(train_dataset)
    # val_dataset = datasets.ImageFolder("F:/class_dataset/archive/validation/validation/", transform=data_transform["val"])  #test data
    val_dataset = datasets.ImageFolder("class_dataset/validate/",
                                       transform=data_transform["val"])   #test data

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                             num_workers=1) 
    len_val = len(val_dataset)

    net = resnet34()
    loss_function = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(net.parameters(), lr=0.0001) 
    epoch = 60000


    history = train_and_val(epoch, net, train_loader, len_train,val_loader, len_val,loss_function, optimizer,device)

    plot_loss(np.arange(0,epoch), history)
    plot_acc(np.arange(0,epoch), history)


