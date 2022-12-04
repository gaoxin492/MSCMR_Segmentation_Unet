import random
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

root = Path("/home/gaoxin/projects/SegUnet/datasets/images/DE")
label = Path("/home/gaoxin/projects/SegUnet/datasets/labels/DE")

def change_img_to_label_path(path):
    """
    Replaces imagesTr with labelsTr
    """
    parts = list(path.parts)  # get all directories within the path
    parts[parts.index("images")] = "labels"  # Replace imagesTr with labelsTr
    parts[-1] = parts[-1].replace(".nii","_manual.nii")
    return Path(*parts)  # Combine list back into a Path object

sample_path = list(root.glob("subject*"))[0]  # Choose a subject
print(sample_path)
sample_path_label = change_img_to_label_path(sample_path)

print(sample_path, sample_path_label)

data = nib.load(sample_path)
label = nib.load(sample_path_label)
mri = data.get_fdata()
mask = label.get_fdata().astype(np.uint8)  # Class labels should not be handled as float64

print(nib.aff2axcodes(data.affine))

print(np.unique(mask,return_counts=True))

fig = plt.figure()
plt.imshow(mri[:,:,0], cmap="bone")
mask_ = np.ma.masked_where(mask[:,:,0]==0, mask[:,:,0])
plt.imshow(mask_, alpha=0.5, cmap="autumn")
plt.show()

def equalize(img):
    img *= 255
    img = img.astype('uint8')
    img_eq = cv2.equalizeHist(img)/255
    return img_eq

# Helper functions for normalization and standardization
def normalize(full_volume):
    """
    Z-Normalization of the whole subject
    """
    mu = full_volume.mean()
    std = np.std(full_volume)
    normalized = (full_volume - mu) / std
    return normalized

def standardize(normalized_data):
    """
    Standardize the normalized data into the 0-1 range
    """
    standardized_data = (normalized_data - normalized_data.min()) / (normalized_data.max() - normalized_data.min())
    return standardized_data

all_files = list(root.glob("subject*"))  # Get all subjects
print(len(all_files))

random.seed(122)
random.shuffle(all_files)
save_root = Path("/home/gaoxin/projects/SegUnet/Preprocessed")
for counter, path_to_mri_data in enumerate(tqdm(all_files)):

    path_to_label = change_img_to_label_path(path_to_mri_data)

    mri = nib.load(path_to_mri_data)
    label = nib.load(path_to_label)

    mri_data = mri.get_fdata()
    label_data = label.get_fdata().astype(np.uint8)
    normalized_mri_data = normalize(mri_data)
    standardized_mri_data = standardize(normalized_mri_data)

    # 35 for train, 10 for val
    if counter <= 35:
        current_path = save_root / "train" / str(counter)
    else:
        current_path = save_root / "val" / str(counter)

    # Loop over the slices in the full volume and store the images and labels in the data/masks directory
    for i in range(standardized_mri_data.shape[-1]):
        slice = standardized_mri_data[:, :, i]
        mask = label_data[:, :, i]
        a,num = np.unique(mask,return_counts=True)
        if a.size == 4 and num[1]>1000:
            slice_path = current_path / "data"
            mask_path = current_path / "masks"
            slice_path.mkdir(parents=True, exist_ok=True)
            mask_path.mkdir(parents=True, exist_ok=True)
            np.save(slice_path / str(i), slice)
            np.save(mask_path / str(i), mask)

path = Path("/home/gaoxin/projects/SegUnet/Preprocessed/train/1/")  # Select a subject
# Choose a file and load slice + mask
file = "6.npy"
slice = np.load(path/"data"/file)
mask = np.load(path/"masks"/file)
a,num = np.unique(mask,return_counts=True)
print(num[1]<1000)

# Plot everything
plt.figure()
plt.imshow(slice, cmap="bone")
mask_ = np.ma.masked_where(mask==0, mask)
plt.imshow(mask_, cmap="autumn")
plt.show()

print(slice.min(), slice.max())