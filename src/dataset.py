import albumentations as A
import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def get_transforms(data, cfg, debug: bool = False):
    aug = A.Compose(cfg.train_aug_list) if data == 'train' else A.Compose(cfg.valid_aug_list)
    if debug:
        print(aug)
    return aug


class CustomDataset(Dataset):
    def __init__(self, images, cfg, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']

        return image, label


def read_image_mask(fragment_id, cfg):
    images = []

    mid = 65 // 2
    start = mid - cfg.in_chans // 2
    end = mid + cfg.in_chans // 2
    idxs = range(start, end)

    for i in tqdm(idxs):
        image = cv2.imread(cfg.comp_dataset_path + f"train/{fragment_id}/surface_volume/{i:02}.tif", 0)

        pad0 = (cfg.tile_size - image.shape[0] % cfg.tile_size)
        pad1 = (cfg.tile_size - image.shape[1] % cfg.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)

    mask = cv2.imread(cfg.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

    mask = mask.astype('float32')
    mask /= 255.0

    return images, mask


def get_train_valid_dataset(cfg):
    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []

    for fragment_id in range(1, 4):

        image, mask = read_image_mask(fragment_id, cfg)

        x1_list = list(range(0, image.shape[1] - cfg.tile_size + 1, cfg.stride))
        y1_list = list(range(0, image.shape[0] - cfg.tile_size + 1, cfg.stride))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + cfg.tile_size
                x2 = x1 + cfg.tile_size

                if fragment_id == cfg.valid_id:
                    valid_images.append(image[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2, None])

                    valid_xyxys.append([x1, y1, x2, y2])
                else:
                    train_images.append(image[y1:y2, x1:x2])
                    train_masks.append(mask[y1:y2, x1:x2, None])

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys
