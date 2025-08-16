from omegaconf import DictConfig, ValueNode
import torch
from torch.utils.data import Dataset
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import csv
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10 as cifar10_data
from torch.nn import functional as F
import numpy as np
from glob2 import glob
from skimage import io
from src.pl_data import normalizations
from tqdm import tqdm
from cv2 import imread
import pandas as pd


DATADIR = "data/"


def load_image(directory):
    return Image.open(directory).convert('L')


def invert(img):

    if img.ndim < 3:
        raise TypeError("Input image tensor should have at least 3 dimensions, but found {}".format(img.ndim))

    bound = torch.tensor(1 if img.is_floating_point() else 255, dtype=img.dtype, device=img.device)
    return bound - img


def colour(img, ch=0, num_ch=3):

    colimg = [torch.zeros_like(img)] * num_ch
    # colimg[ch] = img
    # Use beta distribution to push the mixture to ch 1 or ch 2
    if ch == 0:
        rand = torch.distributions.beta.Beta(0.5, 1.)
    elif ch == 1:
        rand = torch.distributions.beta.Beta(1., 0.5) 
    else:
        raise NotImplementedError("Only 2 channel images supported now.")
    rand = rand.sample()
    colimg[0] = img * rand
    colimg[1] = img * (1 - rand)
    return torch.cat(colimg)


class CIFAR10(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform

        self.dataset = cifar10_data(root=DATADIR, download=True)
        self.data_len = len(self.dataset)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        img, label = self.dataset[index]
        img = np.asarray(img)
        label = np.asarray(label)
        # Transpose shape from H,W,C to C,H,W
        img = img.transpose(2, 0, 1).astype(np.float32)
        # img = F.to_tensor(img)
        # label = F.to_tensor(label)
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class COR14_MULTI(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 33000
        self.minval = 0
        self.denom = self.maxval - self.minval
        self.control = ["20CAG"]
        self.disease = ["72CAG"]
        self.KO = ["KO"]

        # List all the files
        print("Globbing files for COR14, this may take a while...")
        self.c1 = glob(os.path.join(self.path, "**", "20CAG", "*.tif"))
        self.c2 = glob(os.path.join(self.path, "**", "72CAG", "*.tif"))
        self.c3 = glob(os.path.join(self.path, "**", "KO", "*.tif"))
        min_files = min(len(self.c1), len(self.c2))
        print("Using {} files".format(min_files))
        self.files = self.c1[:min_files] + self.c2[:min_files] + self.c3[:min_files]  # Assuming KO has the fewest but who cares
        # self.files = glob(os.path.join(self.path, "**", "**", "*.tif"))
        self.files = np.asarray(self.files)
        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(self.files))
        self.files = self.files[shuffle_idx]
        self.data_len = len(self.files)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        img = io.imread(fn, plugin='pil')
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        img = torch.clip(img, 0, 1)
        img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel

        cell_line = fn.split(os.path.sep)[-2]
        if cell_line in self.control:
            label = 0
        elif cell_line in self.disease:
            label = 1
        elif cell_line in self.KO:
            label = 2
        else:
            raise RuntimeError("Found label={} but expecting labels in [1, 2].".format(label))
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class COR14(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 33000
        self.minval = 0
        self.denom = self.maxval - self.minval
        self.control = ["20CAG30"]
        self.disease = ["72CAG12"]

        # List all the files
        print("Globbing files for COR14, this may take a while...")
        self.c1 = glob(os.path.join(self.path, "**", "20CAG30", "*.tif"))
        self.c2 = glob(os.path.join(self.path, "**", "72CAG12", "*.tif"))
        min_files = min(len(self.c1), len(self.c2))
        print("Using {} files".format(min_files))
        self.files = self.c1[:min_files] + self.c2[:min_files]
        # self.files = glob(os.path.join(self.path, "**", "**", "*.tif"))
        self.files = np.asarray(self.files)
        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(self.files))
        self.files = self.files[shuffle_idx]
        self.data_len = len(self.files)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        img = io.imread(fn, plugin='pil')
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel

        cell_line = fn.split(os.path.sep)[-2]
        if cell_line in self.control:
            label = 0
        elif cell_line in self.disease:
            label = 1
        else:
            raise RuntimeError("Found label={} but expecting labels in [1, 2].".format(label))
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class DRGSCRN(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, balance=True, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 255  # 10000  # 33000
        self.minval = 0
        self.denom = self.maxval - self.minval
        self.hd_control = ["CTR"]
        self.hd_disease = ["Sporadic"]
        self.train = train

        # List all the files
        cached = "cached_files.npz"
        if 1:  # not os.path.exists(cached):
            c1, c2, c3 = [], [], []
            self.files, self.labs = [], []
            for p in tqdm(self.path, total=len(self.path), desc="Processing files"):
                fs = glob(os.path.join(p, "*.png"))
                ts = [int(f.split("_T")[-1].split("_")[0]) for f in fs]
                lbs = np.asarray(ts)
                lbs[lbs == 4] = 1
                assert len(np.unique(lbs)) == 2
                self.files.append(fs)
                self.labs.append(lbs)
        else:
            data = np.load(cached)
            c1 = data["c1"]
            c2 = data["c2"]
            c3 = data["c3"]
        self.files = np.concatenate(self.files)
        self.labs = np.concatenate(self.labs)
        self.data_len = len(self.files)
        self.lens = [(self.labs == 0).sum(), (self.labs == 1).sum()]

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        # img = io.imread(fn, plugin='pil')
        img = imread(fn, 0)
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        img = np.clip(img, 0, 1).astype(np.float32)

        if self.transform is not None:
            img = self.transform(img)
        img = img.repeat(3, 1, 1)  # Stupid but let's replicate 1->3 channel
        label = self.labs[index]
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class DRGSCRN_WELLS(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, balance=True, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 255  # 10000  # 33000
        self.minval = 0
        self.denom = self.maxval - self.minval
        self.hd_control = ["CTR"]
        self.hd_disease = ["Sporadic"]
        self.train = train

        # List all the files
        cached = "cached_files.npz"
        if 1:  # not os.path.exists(cached):
            c1, c2, c3 = [], [], []
            self.files, self.labs = [], []
            for p in tqdm(self.path, total=len(self.path), desc="Processing files"):
                fs = glob(os.path.join(p, "*.png"))
                ts = [int(f.split("_T")[-1].split("_")[0]) for f in fs]
                lbs = np.asarray(ts)
                lbs[lbs == 4] = 1
                self.files.append(fs)
                self.labs.append(lbs)
        else:
            data = np.load(cached)
            c1 = data["c1"]
            c2 = data["c2"]
            c3 = data["c3"]
        self.files = np.concatenate(self.files)
        self.labs = np.concatenate(self.labs)
        self.data_len = len(self.files)
        self.lens = [(self.labs == 0).sum(), (self.labs == 1).sum()]

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        # img = io.imread(fn, plugin='pil')
        img = imread(fn, 0)
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        img = np.clip(img, 0, 1).astype(np.float32)

        if self.transform is not None:
            img = self.transform(img)
        img = img.repeat(3, 1, 1)  # Stupid but let's replicate 1->3 channel
        label = self.labs[index]
        return img, label, fn

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class JULIA_CV_AALS_multi_class(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, balance=True, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 65536  # 33000
        self.minval = 0
        self.denom = self.maxval - self.minval
        self.hd_control = ["CTR"]
        self.hd_disease = ["Sporadic"]
        self.train = train

        # List all the files
        cached = "cached_files.npz"
        if 1:  # not os.path.exists(cached):
            c1, c2, c3 = [], [], []
            self.files, self.labs = [], []
            for p in tqdm(self.path, total=len(self.path), desc="Processing files"):
                fs = glob(os.path.join(p, "*.tif"))
                if len(p.split(os.path.sep)) == 2:
                    tp = p.split(os.path.sep)[-1]
                else:
                    tp = p.split(os.path.sep)[-2]
                if tp in self.hd_control:
                    lb = 0
                elif tp in self.hd_disease:
                    lb = 1
                else:
                    raise NotImplementedError(tp)
                lbs = np.zeros(len(fs)) + lb
                self.files.append(fs)
                self.labs.append(lbs)
        else:
            data = np.load(cached)
            c1 = data["c1"]
            c2 = data["c2"]
            c3 = data["c3"]
        self.files = np.concatenate(self.files)
        self.ids = np.unique(["_".join(x.split(os.path.sep)[2:4]) for x in self.files], return_inverse=True)[1]
        """
        # self.files  # split by wells
        wells = [x.split("_")[4] for x in self.files]
        wells = np.asarray(wells)
        uwells = np.unique(wells)
        utrain_wells = uwells[:len(uwells) // 2]
        utest_wells = uwells[len(uwells) // 2:]
        train_idx = np.in1d(wells, utrain_wells)
        test_idx = np.in1d(wells, utest_wells)
        self.labs = np.concatenate(self.labs)
        train_files = self.files[train_idx]
        test_files = self.files[test_idx]
        train_labs = self.labs[train_idx]
        test_labs = self.labs[test_idx]
        train_ids = self.ids[train_idx]
        test_ids = self.ids[test_idx]
        if self.train == "train":
            self.files = train_files
            self.labs = train_labs
            self.pids = train_ids
        else:
            self.files = test_files
            self.labs = test_labs
            self.pids = test_ids
        """
        self.labs = np.concatenate(self.labs).astype(int)
        self.data_len = len(self.files)
        self.lens = [(self.labs == 0).sum(), (self.labs == 1).sum()]

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        img = io.imread(fn, plugin='pil')
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        img = np.clip(img, 0, 1).astype(np.float32)
        # img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel
        # img = img[None]

        if self.transform is not None:
            img = self.transform(img)
        img = img.repeat(3, 1, 1)  # Stupid but let's replicate 1->3 channel

        cell_line = fn.split(os.path.sep)[-3]
        pid = self.ids[index]
        """
        if cell_line in self.hd_control:
            label = 0
        elif cell_line in self.hd_disease:
            label = 1
        else:
            raise RuntimeError("Found label={} but expecting labels in [0, 1].".format(cell_line))
        """
        label = self.labs[index]
        return img, [label, pid]

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class JULIA_CV_HD_multi_class(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, balance=True, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 65536  # 33000
        self.minval = 0
        self.denom = self.maxval - self.minval
        self.hd_control = ["CS2518n2"]
        self.hd_disease = ["HD53n5"]
        self.train = train

        # List all the files
        cached = "cached_files.npz"
        if 1:  # not os.path.exists(cached):
            c1, c2, c3 = [], [], []
            self.files, self.labs = [], []
            for p in tqdm(self.path, total=len(self.path), desc="Processing files"):
                fs = glob(os.path.join(p, "*.tif"))
                tp = p.split(os.path.sep)[-1]
                if tp in self.hd_control:
                    lb = 0
                elif tp in self.hd_disease:
                    lb = 1
                else:
                    raise NotImplementedError(tp)
                lbs = np.zeros(len(fs)) + lb
                self.files.append(fs)
                self.labs.append(lbs)
        else:
            data = np.load(cached)
            c1 = data["c1"]
            c2 = data["c2"]
            c3 = data["c3"]
        self.files = np.concatenate(self.files)
        # self.files  # split by wells
        wells = [x.split("_")[4] for x in self.files]
        wells = np.asarray(wells)
        uwells = np.unique(wells)
        utrain_wells = uwells[:len(uwells) // 2]
        utest_wells = uwells[len(uwells) // 2:]
        train_idx = np.in1d(wells, utrain_wells)
        test_idx = np.in1d(wells, utest_wells)
        self.labs = np.concatenate(self.labs)
        train_files = self.files[train_idx]
        test_files = self.files[test_idx]
        train_labs = self.labs[train_idx]
        test_labs = self.labs[test_idx]
        if self.train == "train":
            self.files = train_files
            self.labs = train_labs
        else:
            self.files = test_files
            self.labs = test_labs
        self.data_len = len(self.files)
        self.lens = [(self.labs == 0).sum(), (self.labs == 1).sum()]

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        img = io.imread(fn, plugin='pil')
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        img = np.clip(img, 0, 1).astype(np.float32)
        # img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel
        # img = img[None]

        if self.transform is not None:
            img = self.transform(img)
        img = img.repeat(3, 1, 1)  # Stupid but let's replicate 1->3 channel

        cell_line = fn.split(os.path.sep)[-2]
        if cell_line in self.hd_control:
            label = 0
        elif cell_line in self.hd_disease:
            label = 1
        else:
            raise RuntimeError("Found label={} but expecting labels in [0, 1].".format(cell_line))
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class HD_multi_class(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, balance=True, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 65536  # 33000
        self.minval = 0
        self.denom = self.maxval - self.minval
        self.als_control = ["NN0005319", "5319", "PGP", "PGP1-Normal"]
        self.als_disease = ["NN0005320", "5320", "M33V", "A382T", "Q331K", "Q33"]
        self.hd_control = ["20CAG30", "20CAG65", "COD7-20CAG65"]
        self.hd_disease = ["72CAG12", "72CAG2", "72CAG4", "72CAG9"]
        self.ko = ["KO8A"]
        self.train = train

        # List all the files
        cached = "cached_files.npz"
        if 1:  # not os.path.exists(cached):
            c1, c2, c3 = [], [], []
            self.files, self.labs = [], []
            for p in tqdm(self.path, total=len(self.path), desc="Processing files"):
                # fs = glob(os.path.join(p, "Soma", "*.tif"))
                fs = glob(os.path.join(p, "*.tif"))
                tp = p.split(os.path.sep)[-1]
                if tp in self.als_control:
                    lb = 0
                elif tp in self.als_disease:
                    lb = 1
                elif tp in self.hd_control:
                    lb = 0
                elif tp in self.hd_disease:
                    lb = 1
                elif tp in self.ko:
                    lb = 4
                else:
                    raise NotImplementedError(tp)
                lbs = np.zeros(len(fs)) + lb
                self.files.append(fs)
                self.labs.append(lbs)
        else:
            data = np.load(cached)
            c1 = data["c1"]
            c2 = data["c2"]
            c3 = data["c3"]
        self.files = np.concatenate(self.files)
        """
        # self.files  # split by wells
        wells = [x.split("_")[4] for x in self.files]
        wells = np.asarray(wells)
        uwells = np.unique(wells)
        utrain_wells = uwells[:len(uwells) // 2]
        utest_wells = uwells[len(uwells) // 2:]
        train_idx = np.in1d(wells, utrain_wells)
        test_idx = np.in1d(wells, utest_wells)
        self.labs = np.concatenate(self.labs)
        train_files = self.files[train_idx]
        test_files = self.files[test_idx]
        train_labs = self.labs[train_idx]
        test_labs = self.labs[test_idx]
        if self.train == "train":
            self.files = train_files
            self.labs = train_labs
        else:
            self.files = test_files
            self.labs = test_labs
        self.data_len = len(self.files)
        """
        self.labs = np.concatenate(self.labs).astype(int)
        self.lens = [(self.labs == 0).sum(), (self.labs == 1).sum()]
        self.data_len = len(self.labs)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        img = io.imread(fn, plugin='pil')
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        img = np.clip(img, 0, 1).astype(np.float32)
        # img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel
        # img = img[None]

        if self.transform is not None:
            img = self.transform(img)
        img = img.repeat(3, 1, 1)  # Stupid but let's replicate 1->3 channel

        """
        cell_line = fn.split(os.path.sep)[-3]
        if cell_line in self.hd_control:
            label = 0
        elif cell_line in self.hd_disease:
            label = 1
        else:
            raise RuntimeError("Found label={} but expecting labels in [0, 1, 2].".format(cell_line))
        """
        label = self.labs[index]
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class JAK_multi_class(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, balance=True, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 65536  # 33000
        self.minval = 0
        self.denom = self.maxval - self.minval
        self.als_control = ["NN0005319", "5319", "PGP", "PGP1-Normal"]
        self.als_disease = ["NN0005320", "5320", "M33V", "A382T", "Q331K", "Q33"]
        self.hd_control = ["20CAG30", "20CAG65"]
        self.hd_disease = ["72CAG12", "72CAG2"]
        self.ko = ["KO8A"]

        # List all the files
        cached = "cached_files.npz"
        if 1:  # not os.path.exists(cached):
            c1, c2, c3 = [], [], []
            self.files, self.labs = [], []
            for p in tqdm(self.path, total=len(self.path), desc="Processing files"):
                fs = glob(os.path.join(p, "Soma", "*.tif"))
                tp = p.split(os.path.sep)[-1]
                if tp in self.als_control:
                    lb = 0
                elif tp in self.als_disease:
                    lb = 1
                elif tp in self.hd_control:
                    lb = 2
                elif tp in self.hd_disease:
                    lb = 3
                elif tp in self.ko:
                    lb = 4
                else:
                    raise NotImplementedError(tp)
                lbs = np.zeros(len(fs)) + lb
                self.files.append(fs)
                self.labs.append(lbs)
        else:
            data = np.load(cached)
            c1 = data["c1"]
            c2 = data["c2"]
            c3 = data["c3"]
        self.files = np.concatenate(self.files)
        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(self.files))
        self.files = self.files[shuffle_idx]
        self.data_len = len(self.files)
        self.labs = np.concatenate(self.labs)  
        self.lens = [(self.labs == 0).sum(), (self.labs == 1).sum(), (self.labs == 2).sum(), (self.labs == 3).sum(), (self.labs == 4).sum()]

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        try:
            img = io.imread(fn, plugin='pil')
        except:
            print(fn)
            os._exit(1)
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        img = np.clip(img, 0, 1).astype(np.float32)
        # img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel
        # img = img[None]

        if self.transform is not None:
            img = self.transform(img)
        img = img.repeat(3, 1, 1)  # Stupid but let's replicate 1->3 channel

        cell_line = fn.split(os.path.sep)[-3]
        if cell_line in self.als_control:
            label = 0
        elif cell_line in self.als_disease:
            label = 1
        elif cell_line in self.hd_control:
            label = 2
        elif cell_line in self.hd_disease:
            label = 3
        elif cell_line in self.ko:
            label = 4
        else:
            raise RuntimeError("Found label={} but expecting labels in [0, 1, 2].".format(cell_line))
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


from joblib import Parallel, delayed


def check_file(f):
    fp = f.split(os.path.sep)[-1].replace(".tif", "_MASK.tif")
    p = os.path.join("Encoded_CellMask_Crops", fp)
    if os.path.exists(p):
        return p, True
    else:
        return p, False


class JAK_multi(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, filter_files=None, balance=True, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.train = train
        self.transform = transform
        # self.control = ["NN0005319", "PGP"]
        # self.disease = ["NN0005320", "M33V", "A382T", "Q331K"]
        self.control = ["5319", "PGP", "PGP1-Normal"]
        self.disease = ["5320", "M337V", "A382T", "Q331K"]

        if isinstance(path, str):
            file_dct = pd.read_csv(path)
            flt = pd.read_csv(filter_files).values.ravel()
            self.files = file_dct.file_path.values
            if filter_files is not None:
                flt = pd.read_csv(filter_files).values.ravel()
                self.files = np.asarray([x for x in self.files if x.split(os.path.sep)[-1] not in flt])
        else:
            file_dct = path
            self.files = file_dct.file_path.values

        """
        res = Parallel(n_jobs=-1)(delayed(check_file)(f) for idx, f in tqdm(enumerate(self.files), total=len(self.files)))
        masks, mask_idx = [], []
        for r in res:
            masks.append(r[0])
            mask_idx.append(r[1])
        masks, mask_idx = np.asarray(masks), np.asarray(mask_idx)
        """
        # self.files = self.files[mask_idx]
        # self.masks = masks[mask_idx]
        assert len(self.files)
        if 0:  # "label" in file_dct.keys():
            self.labs = file_dct.label.values[mask_idx]
        elif 1 in file_dct.keys():
            # self.labs = file_dct[1].values[mask_idx]
            self.labs = file_dct[1].values
        else:
            raise NotImplementedError
            labs = []
            for f in self.files:
                control_check = np.any([x in f for x in self.control])
                if control_check:
                    labs.append(0)
                else:
                    labs.append(1)
            self.labs = np.asarray(labs)
        if balance:
            class_0_indices = np.where(self.labs == 0)[0]
            class_1_indices = np.where(self.labs == 1)[0]
    
            # Determine the minimum number of samples per class
            min_samples = min(len(class_0_indices), len(class_1_indices))
    
            # Randomly select equal number of samples from each class
            np.random.seed(42)  # For reproducibility
            selected_class_0 = np.random.choice(class_0_indices, min_samples, replace=False)
            selected_class_1 = np.random.choice(class_1_indices, min_samples, replace=False)
    
            # Combine the selected indices
            balanced_indices = np.concatenate([selected_class_0, selected_class_1])
            np.random.shuffle(balanced_indices)  # Shuffle to avoid ordered batches
    
            # Update the files and labels
            self.files = self.files[balanced_indices]
            self.labs = self.labs[balanced_indices]
            # self.masks = self.masks[balanced_indices]
        self.lens = np.unique(self.labs, return_counts=True)[1]
        np.random.seed(42)
        # shuffle_idx = np.random.permutation(len(self.files))
        # self.files = self.files[shuffle_idx]
        # self.labs = self.labs[shuffle_idx]
        # self.masks = self.masks[shuffle_idx]
        self.data_len = len(self.files)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        # mn = self.masks[index]
        img = np.asarray(Image.open(fn))
        img = img.astype(np.float32)
        # mask = np.asarray(Image.open(mn))
        # mask = mask.astype(np.float32)

        # img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        # img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel
        # img = img[None]

        # # Filter images
        # combined = np.stack((img, mask), -1)
        # if self.transform is not None:
        #     # img = self.transform(img)
        #     combined = self.transform(combined)
        # img = combined[[0]]
        # mask = combined[[1]]
        # img = img.repeat(3, 1, 1)  # Stupid but let's replicate 1->3 channel
        # mask = (mask > 0).float()
        # img = img * mask
        if self.transform is not None:
            img = self.transform(img)
        img = img.repeat(3, 1, 1)  # Stupid but let's replicate 1->3 channel

        # cell_line = fn.split(os.path.sep)[-3]
        # if cell_line in self.control:
        #     label = 0
        # elif cell_line in self.disease:
        #     label = 1
        # else:
        #     raise RuntimeError("Found label={} but expecting labels in [0, 1].".format(cell_line))
        label = self.labs[index]
        # if len([x for x in self.control if x in cell_line]):
        #     label = 0
        # elif len([x for x in self.disease if x in cell_line]):
        #     label = 1
        # else:
        #     raise RuntimeError("Found label={} but expecting labels in [0, 1].".format(cell_line))
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class JAK_multi_new(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, balance=True, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 33000
        self.minval = 0
        self.denom = self.maxval - self.minval
        # self.control = ["NN0005319", "PGP"]
        # self.disease = ["NN0005320", "M33V", "A382T", "Q331K"]
        self.control = ["NN0005319", "PGP"]
        self.disease = ["NN0005320", "M33V", "A382T", "Q331K"]
        self.control = ["5319", "PGP"]
        self.disease = ["5320", "M33V", "A382T", "Q331K"]

        # List all the files
        print("Globbing files for JAK, this may take a while...")
        c1, c2 = [], []
        for p in self.path:
            c1.append(glob(os.path.join(p, "*", "Soma", "*.tif")))
        self.files = np.concatenate(c1)
        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(self.files))
        self.files = self.files[shuffle_idx]
        self.data_len = len(self.files)
        self.lens = self.data_len

    def __len__(self) -> int:
        return len(self.files)  # self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        img = io.imread(fn, plugin='pil')
        img = img.astype(np.float32)
        # img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        # img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel
        # img = img[None]

        if self.transform is not None:
            img = self.transform(img)
        img = img.repeat(3, 1, 1)  # Stupid but let's replicate 1->3 channel

        cell_line = fn.split(os.path.sep)[-3]
        # if cell_line in self.control:
        #     label = 0
        # elif cell_line in self.disease:
        #     label = 1
        # else:
        #     raise RuntimeError("Found label={} but expecting labels in [0, 1].".format(cell_line))
        if len([x for x in self.control if x in cell_line]):
            label = 0
        elif len([x for x in self.disease if x in cell_line]):
            label = 1
        else:
            raise RuntimeError("Found label={} but expecting labels in [0, 1]. File: {}".format(cell_line, fn))
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class JAK(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, balance=True, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 33000
        self.minval = 0
        self.denom = self.maxval - self.minval
        self.control = ["NN0005319", "PGP"]
        self.disease = ["NN0005320", "M33V", "A382T", "Q331K"]

        # List all the files
        print("Globbing files for JAK, this may take a while...")
        c1s = []
        for c in self.control:
            f = glob(os.path.join(self.path, "**", c, "Soma", "*.tif"))
            if len(f):
                c1s.append(f)
        c1s = np.asarray(c1s).ravel()
        print("{} controls".format(len(c1s)))
        c2s = []
        for c in self.disease:
            f = glob(os.path.join(self.path, "**", c, "Soma", "*.tif"))
            if len(f):
                c2s.append(f)
        c2s = np.asarray(c2s).ravel()
        print("{} disease".format(len(c2s)))
        # self.c1 = glob(os.path.join(self.path, "**", self.control, "Soma", "*.tif"))
        # self.c2 = glob(os.path.join(self.path, "**", self.disease, "Soma", "*.tif"))
        self.c1 = c1s
        self.c2 = c2s
        min_files = min(len(self.c1), len(self.c2))
        if balance:
            print("Using {} files".format(min_files))
            self.files = np.concatenate((self.c1[:min_files], self.c2[:min_files]))
        else:
            self.files = np.concatenate((self.c1, self.c2))
        self.files = np.asarray(self.files)
        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(self.files))
        self.files = self.files[shuffle_idx]
        self.data_len = len(self.files)
        self.lens = [len(self.c1), len(self.c2)]
        self.labs = np.concatenate((np.zeros(len(self.c1)), np.ones(len(self.c2))))

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        img = io.imread(fn, plugin='pil')
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        # img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel
        # img = img[None]

        if self.transform is not None:
            img = self.transform(img)
        img = img.repeat(3, 1, 1)  # Stupid but let's replicate 1->3 channel

        cell_line = fn.split(os.path.sep)[-3]
        if cell_line in self.control:
            label = 0
        elif cell_line in self.disease:
            label = 1
        else:
            raise RuntimeError("Found label={} but expecting labels in [0, 1].".format(cell_line))
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class JAK_fn(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, balance=True, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 33000
        self.minval = 0
        self.denom = self.maxval - self.minval

        # List all the files
        print("Globbing files for JAK, this may take a while...")
        c1s = glob("{}{}".format(self.path, ".tif"))
        c1s = np.asarray(c1s).ravel()
        self.c1 = c1s
        self.c2 = c1s
        min_files = min(len(self.c1), len(self.c2))
        if balance:
            print("Using {} files".format(min_files))
            self.files = np.concatenate((self.c1[:min_files], self.c2[:min_files]))
        else:
            self.files = np.concatenate((self.c1, self.c2))
        self.files = np.asarray(self.files)
        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(self.files))
        self.files = self.files[shuffle_idx]
        self.data_len = len(self.files)
        self.lens = [len(self.c1), len(self.c2)]
        self.labs = np.concatenate((np.zeros(len(self.c1)), np.ones(len(self.c2))))

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        img = io.imread(fn, plugin='pil')
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        # img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel
        # img = img[None]

        if self.transform is not None:
            img = self.transform(img)
        img = img.repeat(3, 1, 1)  # Stupid but let's replicate 1->3 channel

        cell_line = fn.split(os.path.sep)[-3]
        label = 1
        return img, label, fn

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class SIMCLR_COR14(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 33000
        self.minval = 0
        self.denom = self.maxval - self.minval
        self.control = ["20CAG30", "20CAG44", "20CAG65"]
        self.disease = ["72CAG2", "72CAG4", "72CAG9", "72CAG12"]

        self.mu = 1086.6762200888888 / self.maxval
        self.sd = 2019.9389348809887 / self.maxval
        self.mu *= 255
        self.sd *= 255

        # List all the files
        print("Globbing files for COR14, this may take a while...")
        self.files = glob(os.path.join(self.path, "**", "**", "*.tif"))
        self.files = np.asarray(self.files)
        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(self.files))
        self.files = self.files[shuffle_idx]
        self.data_len = len(self.files)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        img = io.imread(fn, plugin='pil')
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        # img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel

        transform = SimCLRTrainDataTransform(
            input_height=200,
            # normalize=normalizations.COR14_normalization,
            gaussian_blur=False)
        img = Image.fromarray((img * 255).astype(np.uint8))
        (xi, xj) = transform(img)
        xi = xi.tile(3, 1, 1)
        xj = xj.tile(3, 1, 1)
        xi = (xi - self.mu) / self.sd
        xj = (xj - self.mu) / self.sd
        cell_line = fn.split(os.path.sep)[-2]
        if cell_line in self.control:
            label = 0
        elif cell_line in self.disease:
            label = 1
        else:
            raise RuntimeError("Found label={} but expecting labels in [1, 2].".format(label))
        return (xi, xj), label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class SimCLRTrainDataTransform:
    """Transforms for SimCLR.
    Transform::
        RandomResizedCrop(size=self.input_height)
        RandomHorizontalFlip()
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        GaussianBlur(kernel_size=int(0.1 * self.input_height))
        transforms.ToTensor()
    Example::
        from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform
        transform = SimCLRTrainDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self, input_height: int = 224, gaussian_blur: bool = False, jitter_strength: float = 1.0, normalize=None
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        data_transforms = [
            transforms.RandomResizedCrop(size=self.input_height),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms.append(transforms.GaussianBlur(kernel_size=kernel_size))

        data_transforms = transforms.Compose(data_transforms)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.train_transform = transforms.Compose([data_transforms, self.final_transform])

        # # add online train transform of the size of global view
        # self.online_transform = transforms.Compose(
        #     [transforms.RandomResizedCrop(self.input_height), transforms.RandomHorizontalFlip(), self.final_transform]
        # )

    def __call__(self, sample):
        transform = self.train_transform

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj  # , self.online_transform(sample)


class JULIA_CV_AALS_REID(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, balance=True, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 65536  # 33000
        self.minval = 0
        self.denom = self.maxval - self.minval
        self.hd_control = ["CTR"]
        self.hd_disease = ["Sporadic"]
        self.train = train

        # List all the files
        cached = "cached_files.npz"
        if 1:  # not os.path.exists(cached):
            c1, c2, c3 = [], [], []
            self.files, self.labs = [], []
            for p in tqdm(self.path, total=len(self.path), desc="Processing files"):
                fs = glob(os.path.join(p, "*.tif"))
                if len(p.split(os.path.sep)) == 2:
                    tp = p.split(os.path.sep)[-1]
                else:
                    tp = p.split(os.path.sep)[-2]
                if tp in self.hd_control:
                    lb = 0
                elif tp in self.hd_disease:
                    lb = 1
                else:
                    raise NotImplementedError(tp)
                lbs = np.zeros(len(fs)) + lb
                self.files.append(fs)
                self.labs.append(lbs)
        else:
            data = np.load(cached)
            c1 = data["c1"]
            c2 = data["c2"]
            c3 = data["c3"]
        self.files = np.concatenate(self.files)
        pt_ids = np.asarray([int(x.split(os.path.sep)[3]) for x in self.files])
        upts, labels, counts = np.unique(pt_ids, return_inverse=True, return_counts=True)
        self.labs = labels

        self.ids = np.unique(["_".join(x.split(os.path.sep)[2:4]) for x in self.files], return_inverse=True)[1]
        # self.files  # split by wells
        wells = [x.split("_")[4] for x in self.files]
        wells = np.asarray(wells)
        uwells = np.unique(wells)
        utrain_wells = uwells[:len(uwells) // 2]
        utest_wells = uwells[len(uwells) // 2:]
        train_idx = np.in1d(wells, utrain_wells)
        test_idx = np.in1d(wells, utest_wells)
        # self.labs = np.concatenate(self.labs)
        train_files = self.files[train_idx]
        test_files = self.files[test_idx]
        train_labs = self.labs[train_idx]
        test_labs = self.labs[test_idx]
        train_ids = self.ids[train_idx]
        test_ids = self.ids[test_idx]
        if self.train == "train":
            self.files = train_files
            self.labs = train_labs
            self.pids = train_ids
        else:
            self.files = test_files
            self.labs = test_labs
            self.pids = test_ids
        self.data_len = len(self.files)
        self.lens = counts

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        label = self.labs[index]
        img = io.imread(fn, plugin='pil')
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        img = np.clip(img, 0, 1).astype(np.float32)

        if self.transform is not None:
            img = self.transform(img)
        img = img.repeat(3, 1, 1)  # Stupid but let's replicate 1->3 channel
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"

