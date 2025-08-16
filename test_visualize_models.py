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
from skimage import io

from timm.models._hub import get_cache_dir
from gradients import SmoothGrad
from matplotlib import pyplot as plt

from glob import glob

from horama import maco, fourier, plot_maco

from src.pl_data.normalizations import COR14_normalization
from scipy.stats import spearmanr

from skimage.transform import rescale, resize, downscale_local_mean


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
# val_path = "Nuclear_pore/GSGT27/CuratedCrops/GM130"
# val_path = "Nuclear_pore/GSGT27/MontagedImages/FITC"

stain = "GM130"
# stain = "Calnexin"
# stain = "LaminB1"
# stain = "SUN2"
if stain == "GM130":
    val_path = "Nuclear_pore/GSGT27/CuratedCrops/GM130/PID20231222_GSGT27-GEDI-ICC-12222023_T1_0-1_*FITC*"
elif stain == "Calnexin":
    val_path = "Nuclear_pore/GSGT27/CuratedCrops/Calnexin/PID20231222_GSGT27-GEDI-ICC-12222023_T1_0-1_*FITC*"
elif stain == "LaminB1":
    val_path = "Nuclear_pore/GSGT27/CuratedCrops/LaminB1/PID20231222_GSGT27-GEDI-ICC-12222023_T1_0-1_*FITC*"
elif stain == "SUN2":
    val_path = "Nuclear_pore/GSGT27/CuratedCrops/SUN2/PID20231222_GSGT27-GEDI-ICC-12222023_T1_0-1_*FITC*"
else:
    raise NotImplementedError(stain)


# val_path = "GSGT27/CuratedImages"
train_data = dataset.JAK_multi(train_path, True, None, transform=train_trans, balance=False)
val_data = dataset.JAK_fn(val_path, False, None, transform=val_trans, balance=False)
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
model = timm.create_model("maxvit_large_tf_224.in1k", pretrained=True, num_classes=nc).eval()
model.load_state_dict(torch.load("maxvit.pth"))

smooth_grad = SmoothGrad(
    pretrained_model=model,
    cuda=False,
    n_samples=10,
    magnitude=True)

# Now run some gradient visualization
batches = 1
iterations = 10
grads, ims, labs, preds = [], [], [], []
model.eval()
maxx = 20
maxval = 33000
minval = 0
denom = maxval - minval

corrs = []
for idx, batch in tqdm(enumerate(val_loader), total=maxx, desc="Grads"):
    X, y, fn = batch
    wc = fn[0].replace("FITC", stain).replace("BGs", "BGsw")
    # wc = glob("{}*".format(os.path.join(stains[0], fn[0].split("_0_")[0].split(os.path.sep)[-1])))[0]

    # Nuclear_pore/GSGT27/CuratedCrops/GM130
    pred = model(X)

    try:
        stain_img = io.imread(wc)
    except:
        continue
    stain_img = resize(stain_img.astype(np.float32), [224, 224])
    # stain_img = (stain_img - minval) / denom  # Normalize to [0, 1]
    # stain_img = COR14_normalization(stain_img + 1e-6)

    pred = pred.squeeze().argmax()
    print(pred)
    smooth_saliency = smooth_grad(X, index=pred.item())  # y
    smooth_saliency = np.abs(smooth_saliency).max(0)
    mask = smooth_saliency > smooth_saliency.mean()
    # mask = np.repeat(mask[None], 3, axis=0)
    stain_img_res = stain_img[mask]
    smooth_saliency_mask = smooth_saliency[mask]
    corrs.append(spearmanr(stain_img_res, smooth_saliency_mask)[0])
    plt.subplot(131)
    plt.imshow(resize(io.imread(fn[0]), [224, 224]) * mask.astype(np.float32))
    plt.axis("off")
    plt.subplot(132)
    plt.imshow(resize(stain_img, [224, 224]) * mask.astype(np.float32))
    plt.axis("off")
    plt.subplot(133)
    plt.imshow(resize(smooth_saliency, [224, 224]) * mask.astype(np.float32))
    plt.axis("off")
    plt.show()

    if idx > maxx:
        break
print(np.nanmean(corrs))
np.save(stain, corrs)
os._exit(1)

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
    plt.show()
    os._exit(1)
    plt.savefig(os.path.join("gradims", "{}.png".format(idx)))
    plt.close(f)


# Now run some MACO visualization
model = timm.create_model("maxvit_large_tf_224.in1k", pretrained=True, num_classes=nc).cuda().eval()
model.load_state_dict(torch.load("maxvit.pth"))

batches = 1
iterations = 20
grads, ims, labs, preds = [], [], [], []
maxx = 10
for i in range(2):
    objective = lambda image: model(image)[:, i].mean()
    image1, alpha1 = maco(objective, values_range=(-2.5, 2.5), total_steps=2000, box_size=(0.1, 0.25), image_size=3000)
    plot_maco(image1, alpha1)
    plt.show()
os._exit(1)

