from pathlib import Path
import torch
import numpy as np
import imgaug
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from metrics import mask_to_onehot

class CardiacDataset(torch.utils.data.Dataset):
    def __init__(self, root, augment_params,seed):
        self.seed = seed
        self.augment_params = augment_params
        self.all_files = self.clean(self.extract_files(root))

    @staticmethod
    def extract_files(root):
        """
        Extract the paths to all slices given the root path (ends with train or val)
        """
        files = []
        for file in root.glob("*"):  # Iterate over the subjects
            slice_path = file / "data"  # Get the slices for current subject
            for slice in slice_path.glob("*.npy"):
                files.append(slice)
        return files

    @staticmethod
    def change_img_to_label_path(path):
        """
        Replace data with mask to get the masks
        """
        parts = list(path.parts)
        parts[parts.index("data")] = "masks"
        return Path(*parts)

    def clean(self,all_files):
        all_files2 = []
        for i in range(len(all_files)):
            file_path = all_files[i]
            mask_path = self.change_img_to_label_path(file_path)
            slice = np.load(file_path).astype(np.float32)
            mask = np.load(mask_path)
            if self.augment_params:
                slice, mask = self.augment(slice, mask)
            if np.unique(mask).size == 4:
                all_files2.append(all_files[i])
        return all_files2

    def augment(self, slice, mask):
        """
        Augments slice and segmentation mask in the exact same way
        Note the manual seed initialization
        """
        # random_seed = torch.randint(0, 1000000, (1,)).item()
        # imgaug.seed(random_seed)
        imgaug.seed(self.seed)
        mask = SegmentationMapsOnImage(mask, mask.shape)
        slice_aug, mask_aug = self.augment_params(image=slice, segmentation_maps=mask)
        mask_aug = mask_aug.get_arr()
        return slice_aug, mask_aug

    def __len__(self):
        """
        Return the length of the dataset (length of all files)
        """
        return len(self.all_files)

    def __getitem__(self, idx):
        """
        Given an index return the (augmented) slice and corresponding mask
        Add another dimension for pytorch
        """

        file_path = self.all_files[idx]
        mask_path = self.change_img_to_label_path(file_path)
        slice = np.load(file_path).astype(np.float32)  # Convert to float for torch
        mask = np.load(mask_path)

        if self.augment_params:
            slice, mask = self.augment(slice, mask)
        # np.unique(mask) = [0,88,200,244]
        mask[mask==88] = 1
        mask[mask == 200] = 2
        mask[mask == 244] = 3
        mask1 = np.expand_dims(mask,-1)
        mask1 = mask_to_onehot(mask1,4)
        # Note that pytorch expects the input of shape BxCxHxW, where B corresponds to the batch size, C to the channels, H to the height and W to Width.
        # As our data is of shape (HxW) we need to manually add the C axis by using expand_dims.
        return np.expand_dims(slice, 0), mask1, mask

if __name__ == '__main__':
    seq = iaa.Sequential([
        iaa.Affine(scale=(0.9, 1.4),
                   rotate=(-45, 45),random_state=0),
        iaa.ElasticTransformation(random_state=0),
        iaa.Resize({"height": 256, "width": 256},random_state=0),
        iaa.size.Crop(percent=0.2,keep_size=True,random_state=0)
    ])
    seq1 = iaa.Sequential([
        iaa.Affine(scale=(0.8, 1.2),
                   rotate=(-90, 90), random_state=6),
        iaa.GammaContrast((0.5, 1.5), random_state=6),
        iaa.Resize({"height": 256, "width": 256}, random_state=6),
        iaa.size.Crop(percent=0.25, keep_size=True, random_state=6)
    ])

    # Create the dataset objects
    train_path = Path("/home/gaoxin/projects/SegUnet/Preprocessed/train")

    train_dataset1 = CardiacDataset(train_path, seq1)
    train_dataset2 = CardiacDataset(train_path, seq1)
    train_dataset3 = CardiacDataset(train_path, seq1)
    train_dataset4 = CardiacDataset(train_path, seq1)
    train_dataset5 = CardiacDataset(train_path, seq1)
    train_dataset = train_dataset1 + train_dataset2 + train_dataset3 + train_dataset4 + train_dataset5
    print(len(train_dataset))

    # for i in range(len(train_dataset)):
    #     _, _, mask = train_dataset[i]
    #     print(np.unique(mask,return_counts=True))
    #
    # for j in range(len(val_dataset)):
    #     _, _, mask = val_dataset[j]
    #     print(np.unique(mask,return_counts=True))

    fig, axis = plt.subplots(3, 3, figsize=(9, 9))

    for i in range(3):
        for j in range(3):
            slice, mask1, mask = train_dataset[4]
            mask_ = np.ma.masked_where(mask==0, mask)
            axis[i][j].imshow(slice[0], cmap="bone")
            axis[i][j].imshow(mask_, cmap="autumn")
            axis[i][j].axis("off")

    fig.suptitle("Sample augmentations")
    plt.tight_layout()
    plt.show()

    slice, mask1, mask = train_dataset[4]
    print(np.unique(mask,return_counts=True))
    print(slice.shape, mask1.shape, mask.shape)
