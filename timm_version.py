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


nc = 2
epochs = 10
lr = 1e-4
bs = 16
tbs = 2
# avail_pretrained_models = timm.list_models(pretrained=True)
accelerator = Accelerator()

models = {
    "maxvit_large_tf_224.in1k": [84.9, 211.8],
    "maxvit_base_tf_224.in1k": [84.9, 119.5],
    "vit_large_patch14_clip_224.openai_ft_in12k_in1k": [88.2, 304.2],
    "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k": [86.2, 86.6],
    # "eva_giant_patch14_224.clip_ft_in1k": [89.1, 1012.6],
    "resnet50d": [80.53, 25.6],
    "resnet34": [75.11, 22],
    "resnet18d": [72.26, 11.7],
    "vgg16.tv_in1k": [71.59, 138],
    "vgg13.tv_in1k": [69.93, 133],
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
    batch_size=tbs,
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
for m in tqdm(models, total=len(models), desc="Model training"):
    # get_cache_dir()
    model = timm.create_model(m, pretrained=True, num_classes=nc)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=1e-6,  # Default
        lr=lr)  # ,
    model, optimizer = accelerator.prepare(model, optimizer)

    train_losses, eval_losses = [], []
    it_best_eval_loss = np.copy(best_eval_loss)
    it_best_ap = np.copy(best_ap)
    for epoch in range(epochs):
        model.train()
        mean_loss = []
        for batch in train_loader:
            optimizer.zero_grad()
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            loss = F.cross_entropy(output, y)
            accelerator.backward(loss)
            optimizer.step()
            mean_loss.append(loss.item())
        mean_loss = np.mean(mean_loss)
        train_losses.append(mean_loss)

        model.eval()
        mean_loss = []
        preds, labs = [], []
        with torch.no_grad():
            for batch in val_loader:
                X, y = batch
                X = X.to(device)
                y = y.to(device)
                output = model(X)
                loss = F.cross_entropy(output, y)
                mean_loss.append(loss.item())
                preds.append(output.cpu().numpy())
                labs.append(y.cpu().numpy())
        cat_preds = np.concatenate(preds, 0)
        cat_gts = np.concatenate(labs, 0)
        ap = average_precision_score(cat_gts, cat_preds[:, 1], average="micro")
        mean_loss = np.mean(mean_loss)
        eval_losses.append(mean_loss)
        if mean_loss < it_best_eval_loss:
            it_best_eval_loss = mean_loss
            it_best_ap = ap
            pass
            # Save weights here
    outputs[m] = {"train": train_losses, "val": eval_losses, "acc": ap}
    print("{}: Loss: {}, AP: {}".format(m, it_best_eval_loss, it_best_ap))

# Save results
np.savez("assay_results", outputs=outputs, models=models)

