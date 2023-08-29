import os
import timm
from src.pl_data import dataset
from accelerate import Accelerator
from tqdm import tqdm

import numpy as np
import torch
from torchvision import transforms
from src.pl_data import normalizations
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from timm.models._hub import get_cache_dir
from gradients import SmoothGrad
from matplotlib import pyplot as plt

nc = 2
epochs = 10
lr = 1e-4
bs = 16
tbs = 2
# avail_pretrained_models = timm.list_models(pretrained=True)
accelerator = Accelerator()

models = {
    "maxvit_large_tf_224.in1k": [84.9, 211.8],
}

train_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomRotation((90, -90)),
    normalizations.COR14_normalization(),
])
val_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(224),
    normalizations.COR14_normalization(),
])

train_path = [
   "genentech_data/GSGT1-JAK-ZF-UC-NN5319AND5320-GEDI8-29-22",  # /FIJI_All_T1",
   # "genentech_data/JAK-GSGT13-Plate1/FIJI_All_T1",
   "genentech_data/JAK-GSGT14"  # /FIJI_All_T1",
   # "genentech_data/JAK-GSGT13-Plate1/FIJI_All_T1"
]

val_path = "genentech_data/JAK-GSGT2-Plate1"  # /FIJI_All_T1"
train_data = dataset.JAK_multi(train_path, True, None, transform=train_trans, balance=False)
val_data = dataset.JAK(val_path, False, None, transform=val_trans, balance=False)
print(len(train_data))
print(len(val_data))

# Inverse weighting for sampling
class_sample_count = np.asarray(train_data.lens)
uni_c = np.arange(len(train_data.lens))
weight = 1. / class_sample_count
weight_dict = {k: v for k, v in zip(uni_c, weight)}
samples_weight = np.array([weight_dict[t] for t in train_data.labs])
samples_weight = torch.from_numpy(samples_weight)
sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

# Data loaders
train_loader = torch.utils.data.DataLoader(
    train_data,
    drop_last=True,
    batch_size=bs,
    sampler=sampler,
    pin_memory=True)  # Remove pin memory if using accelerate
val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=1,
    drop_last=True,
    shuffle=False,
    pin_memory=True)

# Prepare and run training
outputs = {}
best_eval_loss = 100000
best_ap = 0
train_data, val_data = accelerator.prepare(train_data, val_data)
# device = accelerator.device
device = "cuda"
torch.hub.set_dir(".")
model = timm.create_model("maxvit_large_tf_224.in1k", pretrained=True, num_classes=nc)
model.load_state_dict(torch.load("maxvit.pth"))

smooth_grad = SmoothGrad(
    pretrained_model=model,
    cuda=False,
    n_samples=10,
    magnitude=True)

# Now run some visualization
batches = 1
iterations = 20
grads, ims, labs, preds = [], [], [], []
model.eval()
maxx = 10
for idx, batch in tqdm(enumerate(val_loader), total=maxx, desc="Grads"):
    X, y = batch
    pred = model(X)
    pred = pred.squeeze().argmax()
    smooth_saliency = smooth_grad(X, index=1)  # y
    grads.append(smooth_saliency.mean(0))
    ims.append(X[0].mean(0))
    labs.append(y[0])
    preds.append(pred)
    if idx > maxx:
        break

os.makedirs("gradims", exist_ok=True)
maxs = np.max([x.max() for x in grads])
mins = np.min([x.min() for x in grads]) 
for idx, (i, g, l, p) in tqdm(enumerate(zip(ims, grads, labs, preds)), total=len(ims), desc="Saving"):
    f = plt.figure()
    plt.subplot(121)
    plt.imshow(i, cmap="Greys_r")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(g, cmap="Reds", vmin=mins, vmax=maxs)
    plt.axis("off")

    if l == 0:
        health = "control"
    else:
        health = "ALS"
    if p == 0:
        pred = "control"
    else:
        pred = "ALS"
    plt.suptitle("{} neuron predictd to be {}".format(health, pred))
    plt.savefig(os.path.join("gradims", "{}.png".format(idx)))
    plt.close(f)

