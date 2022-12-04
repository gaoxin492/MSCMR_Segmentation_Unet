from torch.nn import CrossEntropyLoss
from datasets import CardiacDataset
from loss import SoftDiceLoss
from metrics import diceCoeffv2
from model import UNet
from pathlib import Path
import torch
import imgaug.augmenters as iaa
import imgaug as ia
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class Segmentation():
    def __init__(self,train_loader,val_loader, epochs):
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            print('CUDA is not available. Training on CPU')
        else:
            print('CUDA is available. Training on GPU')
        self.device = torch.device("cuda:0" if train_on_gpu else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epoch = epochs
        self.model = UNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4,weight_decay=1e-5)
        self.loss_fn = SoftDiceLoss(4,activation="sigmoid")

    def train(self):
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        # 设置训练网络的一些参数
        # 记录训练的次数
        total_train_step = 0
        # 添加tensorboard
        writer = SummaryWriter("/home/gaoxin/projects/SegUnet/logs_train")
        # 防止tensorboard出问题
        train_loss = []
        train_dice = []
        val_dice = []

        for j in range(1,self.epoch+1):
            print("----------第{}轮训练开始了---------".format(j))
            self.model.train()
            total_train_loss = 0
            # 训练步骤开始
            for data in self.train_loader:
                images, targets,_ = data
                images = images.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets.long())
                total_train_loss += loss.item()
                total_train_step += 1
                # 优化器优化模型
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if total_train_step % 20 == 0:
                    print("训练次数:{},loss:{}".format(total_train_step, loss.item()))

            writer.add_scalar("train_loss", total_train_loss, j)
            train_loss.append(total_train_loss)

            # 测试步骤开始
            self.model.eval()
            with torch.no_grad():
                train_class_dices = np.zeros(3)
                for train_batch, data in enumerate(self.train_loader):
                    X1, Y1,_ = data
                    X1 = X1.to(self.device)
                    Y1_hat = self.model(X1)
                    Y1_hat = torch.sigmoid(Y1_hat)
                    Y1_hat[Y1_hat < 0.5] = 0
                    Y1_hat[Y1_hat > 0.5] = 1
                    Y1_hat = Y1_hat.cpu().detach()
                    train_class_dice = []
                    for i in range(1, 4):
                        train_class_dice.append(diceCoeffv2(Y1_hat[:, i:i + 1, :],Y1[:, i:i + 1, :]))
                    train_class_dices += np.array(train_class_dice)
                train_class_dices = train_class_dices / train_batch
                train_mean_dice = train_class_dices.sum() / train_class_dices.size

                print(train_mean_dice,train_class_dices[0], train_class_dices[1], train_class_dices[2])

                val_class_dices = np.zeros(3)
                for val_batch,data in enumerate(self.val_loader):
                    X2, Y2,_ = data
                    X2 = X2.to(self.device)
                    Y2_hat = self.model(X2)
                    Y2_hat = torch.sigmoid(Y2_hat)
                    Y2_hat[Y2_hat < 0.5] = 0
                    Y2_hat[Y2_hat > 0.5] = 1
                    Y2_hat = Y2_hat.cpu().detach()
                    val_class_dice = []
                    for i in range(1, 4):
                        val_class_dice.append(diceCoeffv2(Y2_hat[:, i:i + 1, :], Y2[:, i:i + 1, :]))
                    val_class_dices += np.array(val_class_dice)
                val_class_dices = val_class_dices / val_batch
                val_mean_dice = val_class_dices.sum() / val_class_dices.size

                print(val_mean_dice, val_class_dices[0], val_class_dices[1], val_class_dices[2])

            writer.add_scalar("train_dice",  train_mean_dice, j)
            writer.add_scalar("val_dice", val_mean_dice, j)
            train_dice.append(train_mean_dice)
            val_dice.append(val_mean_dice)

        dataframe = pd.DataFrame({'train_loss':train_loss,'train_dice':train_dice,'val_dice':val_dice})
        dataframe.to_csv('/home/gaoxin/projects/SegUnet/result2.csv')
        writer.close()
        return self.model

def dataset_(path):
    dataset_list = []
    for i in range(12):
        if i < 5 or i==10:
            seq = iaa.Sequential([
                iaa.Affine(scale=(0.9, 1.5),
                           rotate=(-45,45)),
                iaa.ElasticTransformation(alpha=(0,2.0),sigma=(1.0,2.0)),
                iaa.Resize({"height": 224, "width": 224}),
                iaa.size.Crop(percent=0.2, keep_size=True)
            ])
        else:
            seq = iaa.Sequential([
                iaa.Affine(scale=(0.8, 1.3),
                           rotate=(-45,45)),
                iaa.GammaContrast((0.5, 1.5)),
                iaa.Resize({"height": 224, "width": 224}),
                iaa.size.Crop(percent=0.25, keep_size=True)
            ])
        dataset_list.append(CardiacDataset(path, seq, i))
    dataset = dataset_list[0]
    for i in range(1,12):
        dataset = dataset + dataset_list[i]

    return dataset

if __name__ == '__main__':
    # Create the dataset objects
    train_path = Path("/home/gaoxin/projects/SegUnet/Preprocessed/train")
    val_path = Path("/home/gaoxin/projects/SegUnet/Preprocessed/val")

    train_dataset = dataset_(train_path)
    val_dataset = dataset_(val_path)

    print(f"There are {len(train_dataset)} train images, {len(val_dataset)} val images")

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Instanciate the model and set the random seed
    Seg = Segmentation(train_loader,val_loader,100)
    model = Seg.train()
    torch.save(model.state_dict(), "/home/gaoxin/projects/SegUnet/MyoPS_LGE2.pth")

