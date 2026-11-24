import os
import random
import numpy as np
import cv2
from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img

def _hw(size):

    if isinstance(size, (list, tuple)):
        return int(size[0]), int(size[1])
    return int(size), int(size)

def augment(imgs, size=256, edge_decay=0., only_h_flip=False):

    H, W, _ = imgs[0].shape
    Hc, Wc = _hw(size)

    if H < Hc or W < Wc:
        scale = max(Hc / H, Wc / W) * 1.05
        new_h, new_w = int(np.ceil(H * scale)), int(np.ceil(W * scale))

        imgs = [cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC) for img in imgs]
        imgs = [np.clip(img, -1, 1) for img in imgs]

        H, W, _ = imgs[0].shape

    Hs = random.randint(0, H - Hc)
    Ws = random.randint(0, W - Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)

    if not only_h_flip:
        if random.randint(0, 1) == 1:
            rot_deg = random.randint(0, 3) if Hc == Wc else random.choice([0, 2])
            for i in range(len(imgs)):
                imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))

    return imgs

def align(imgs, size=256):

    H, W, _ = imgs[0].shape
    Hc, Wc = _hw(size)

    if H < Hc or W < Wc:
        scale = max(Hc / H, Wc / W)
        new_h, new_w = int(np.ceil(H * scale)), int(np.ceil(W * scale))
        imgs = [cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC) for img in imgs]
        imgs = [np.clip(img, -1, 1) for img in imgs]
        H, W, _ = imgs[0].shape

    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

    return imgs

class PairLoader(Dataset):
    def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip

        self.root_dir = os.path.join(data_dir, sub_dir)

        if os.path.exists(os.path.join(self.root_dir, 'hazy')):
            self.hazy_dir = os.path.join(self.root_dir, 'hazy')
            if os.path.exists(os.path.join(self.root_dir, 'GT')):
                self.gt_dir = os.path.join(self.root_dir, 'GT')
            elif os.path.exists(os.path.join(self.root_dir, 'clear')):
                self.gt_dir = os.path.join(self.root_dir, 'clear')
            else:
                self.gt_dir = os.path.join(self.root_dir, 'GT')
        else:
            self.hazy_dir = os.path.join(data_dir, 'hazy')
            self.gt_dir = os.path.join(data_dir, 'GT')

        self.hazy_names = []
        self.gt_names = []

        if not os.path.exists(self.hazy_dir):
            print(f"❌ Error: Cannot find Hazy folder: {self.hazy_dir}")
        else:
            all_hazy_files = sorted(os.listdir(self.hazy_dir))
            gt_set = set(os.listdir(self.gt_dir)) if os.path.isdir(self.gt_dir) else set()

            for h_name in all_hazy_files:
                if not h_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    continue
                if h_name in gt_set:
                    self.hazy_names.append(h_name)
                    self.gt_names.append(h_name)
                    continue
                if '_' in h_name:
                    short_id = h_name.split('_')[0]
                    for ext in ('.png', '.jpg'):
                        if short_id + ext in gt_set:
                            self.hazy_names.append(h_name)
                            self.gt_names.append(short_id + ext)
                            break

            print(f"✅ [{sub_dir}] Successfully loaded {len(self.hazy_names)} image pairs (Hazy: {self.hazy_dir})")

    def __len__(self):
        return len(self.hazy_names)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        hazy_name = self.hazy_names[idx]
        gt_name = self.gt_names[idx]

        hazy_path = os.path.join(self.hazy_dir, hazy_name)
        gt_path = os.path.join(self.gt_dir, gt_name)

        source_img = read_img(hazy_path)
        target_img = read_img(gt_path)

        if source_img is None:
            raise ValueError(f"Cannot read image: {hazy_path}")
        if target_img is None:
            raise ValueError(f"Cannot read image: {gt_path}")

        source_img = source_img * 2 - 1
        target_img = target_img * 2 - 1

        if self.mode == 'train':
            [source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

        if self.mode == 'valid':
            [source_img, target_img] = align([source_img, target_img], self.size)

        return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': hazy_name}

class SingleLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_names = sorted(os.listdir(self.root_dir))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img_name = self.img_names[idx]
        img_path = os.path.join(self.root_dir, img_name)

        img = read_img(img_path)
        if img is None:
             raise ValueError(f"Cannot read image: {img_path}")

        img = img * 2 - 1

        return {'img': hwc_to_chw(img), 'filename': img_name}