from train import Segmentation,dataset_
from model import UNet
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
from metrics import diceCoeffv2
import matplotlib.pyplot as plt

val_path = Path("/home/gaoxin/projects/SegUnet/Preprocessed/val")
val_dataset = dataset_(val_path)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)

model = UNet()
model.load_state_dict(torch.load("/home/gaoxin/projects/SegUnet/MyoPS_LGE2.pth"))

model.eval()

for data in val_loader:
    slice,_,mask  = data
    with torch.no_grad():
        pred = model(slice)
        pred = torch.sigmoid(pred)
        pred[pred < 0.5] = 0
        pred[pred > 0.5] = 1
        val_class_dice = []
        for i in range(1, 4):
            val_class_dice.append(diceCoeffv2(pred[:, i:i + 1, :], pred[:, i:i + 1, :]))
    val_mean_dice = sum(val_class_dice) / 3

    if val_mean_dice > 0.85:
        mask2 = np.argmax(pred.numpy().squeeze(),axis=0)
        print(mask2.shape)
        print(mask2)
        fig, axis = plt.subplots(1, 2, figsize=(13, 6))
        mask_ = np.ma.masked_where(mask[0] == 0, mask[0])
        mask2_ = np.ma.masked_where(mask2 == 0, mask2)
        print(slice.shape)
        axis[0].imshow(slice[0][0], cmap="bone")
        axis[0].imshow(mask_, alpha=0.6, cmap="autumn")
        axis[1].imshow(slice[0][0], cmap="bone")
        axis[1].imshow(mask2_, alpha=0.6, cmap="autumn")
        axis[0].axis("off")
        axis[1].axis("off")
        fig.suptitle("Ground Truth VS Prediction")
        plt.tight_layout()
        plt.show()
        break