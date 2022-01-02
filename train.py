# google colab地址：https://colab.research.google.com/drive/1JYB1hkz1Uu0Clqtl0EuPLZKClvjc4imu#scrollTo=8Zp_mxA1o-2b

import torchvision
import torch
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model import Cifar10Model

# GPU 版本
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = torchvision.datasets.CIFAR10("dataset", download=True, train=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("dataset", download=True, train=False,
                                         transform=torchvision.transforms.ToTensor())

train_data_len = len(train_data)
test_data_len = len(test_data)
print("训练数据集的长度为:{}".format(train_data_len))
print("测试数据集的长度为:{}".format(test_data_len))

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

model = Cifar10Model()
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

total_train_step = 0
total_test_step = 0
epoch = 10

writer = SummaryWriter("logs")

for i in range(epoch):
    print("-------第{}轮训练开始-------".format(i + 1))
    # 开始训练
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数:{}，Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # 开始测试
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuarcy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuarcy
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy / test_data_len))
    total_test_step = total_test_step + 1
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_len, total_test_step)

    torch.save(model, "cifar10-model{}.pth".format(i + 1))
    print("模型已保存")
writer.close()
