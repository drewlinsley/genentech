import timm
from src.pl_data import dataset
from accelerate import Accelerator
from tqdm import tqdm

import numpy as np
import torch
from torchvision.transforms import v2 as transforms
from src.pl_data import normalizations
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from timm.models._hub import get_cache_dir
from timm.data import resolve_data_config


def norm(x, mu, sd, maxval=65535):
    # Normalize X by the max/min
    # x = x / maxval
    # x = torch.clamp(x, min=0, max=1)

    # Then normalize it to imagenet 
    x = (x - mu) / sd

    return x


def evaluate(model, val_loader, mu, sd, denom):
    # Evaluation phase
    model.eval()
    preds, labs = [], []
    losses = []
        
    eval_pbar = tqdm(val_loader,
                        desc=f'Epoch {epoch+1}/{epochs} [Val]',
                        leave=False)
        
    with torch.no_grad():
        for batch in eval_pbar:
            X, y = batch
            X = X / denom
            X = torch.clip(X, 0, 1)
            X = norm(X, mu=mu, sd=sd)
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            loss = F.cross_entropy(output, y)
            preds.append(output.cpu().numpy())
            labs.append(y.cpu().numpy())
            losses.append(loss.item())
                
            # Update progress bar with current loss
            eval_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    return np.mean(losses)


nc = 2
epochs = 100
lr = 1e-4
bs = 256
tbs = 256
num_eval_steps = 1000
denom = 2 ** 14
balance = True
# avail_pretrained_models = timm.list_models(pretrained=True)
accelerator = Accelerator()

models = {
    # "maxvit_large_tf_224.in1k": [84.9, 211.8],
    # "maxvit_base_tf_224.in1k": [84.9, 119.5],
    # "vit_large_patch14_clip_224.openai_ft_in12k_in1k": [88.2, 304.2],
    # "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k": [86.2, 86.6],
    # "eva_giant_patch14_224.clip_ft_in1k": [89.1, 1012.6],
    "resnet50": [80.53, 25.6],
    # "resnet34": [75.11, 22],
    # "resnet18d": [72.26, 11.7],
    # "vgg16.tv_in1k": [71.59, 138],
    # "vgg13.tv_in1k": [69.93, 133],
}
model_name = "resnet50"
train_trans = transforms.Compose([
    transforms.ToTensor(),
    # transforms.ToImage(),
    # transforms.ToDtype(torch.float32, scale=True),
    # transforms.RandomCrop(224),
    # transforms.RandomResizedCrop(224),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 7)),
    # transforms.GaussianNoise(clip=False, sigma=0.01),
    # transforms.RandomRotation((90, -90)),
    # normalizations.COR14_normalization(),
])
val_trans = transforms.Compose([
    transforms.ToTensor(),
    # transforms.CenterCrop(224),
    transforms.Resize(224)
    # normalizations.COR14_normalization(),
])
train_path = "proc_filelists/train_new_files.csv"
# val_path = "genentech_data/JAK-GSGT2-Plate1"  # /FIJI_All_T1"
val_path = "proc_filelists/test_new_files.csv"
filter_files = "discard_edge_texture.csv"
with accelerator.main_process_first():
    train_data = dataset.JAK_multi(train_path, True, None, transform=train_trans, balance=balance, filter_files=filter_files)
    val_data = dataset.JAK_multi(val_path, False, None, transform=val_trans, balance=balance, filter_files=filter_files)
    print(len(train_data))
    print(len(val_data))

"""
# Inverse weighting for sampling
class_sample_count = np.asarray(train_data.lens)
uni_c = np.arange(len(train_data.lens))
weight = 1. / class_sample_count
weight_dict = {k: v for k, v in zip(uni_c, weight)}
samples_weight = np.array([weight_dict[t] for t in train_data.labs])
samples_weight = torch.from_numpy(samples_weight)
sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

class_sample_count = np.asarray(val_data.lens)
uni_c = np.arange(len(val_data.lens))
weight = 1. / class_sample_count
weight_dict = {k: v for k, v in zip(uni_c, weight)}
val_samples_weight = np.array([weight_dict[t] for t in val_data.labs])
val_samples_weight = torch.from_numpy(val_samples_weight)
val_sampler = torch.utils.data.WeightedRandomSampler(val_samples_weight, len(val_samples_weight))
"""

# Data loaders
num_train_workers = 0
num_val_workers = 0
train_loader = torch.utils.data.DataLoader(
    train_data,
    drop_last=True,
    batch_size=bs,
    # sampler=sampler,
    shuffle=True,
    num_workers=num_train_workers,
    pin_memory=False)  # Remove pin memory if using accelerate
val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=tbs,
    drop_last=True,
    # sampler=val_sampler,
    num_workers=num_val_workers,
    pin_memory=False)

# Prepare and run training
outputs = {}
best_eval_loss = 100000
best_ap = 0
device = accelerator.device
torch.hub.set_dir(".")

model = timm.create_model(model_name, pretrained=True, num_classes=nc)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
TIMM = resolve_data_config({}, model=model_name)
mu = torch.from_numpy(np.asarray(TIMM["mean"][0]).astype(np.float32))
sd = torch.from_numpy(np.asarray(TIMM["std"][0]).astype(np.float32))

model, optimizer, train_data, val_data = accelerator.prepare(model, optimizer, train_data, val_data)
best_val_loss = 1.
val_loss = 1.
train_losses, eval_losses = [], []
for epoch in range(epochs):
    model.train()
    train_batch_losses = []
    train_pbar = tqdm(train_loader,
                         desc=f'Epoch {epoch+1}/{epochs} [Train]',
                         leave=False)
    for idx, batch in enumerate(train_pbar):
        optimizer.zero_grad()
        X, y = batch
        X = X / denom
        X = torch.clip(X, 0, 1)
        X = norm(X, mu=mu, sd=sd)
        X = X.to(device)
        y = y.to(device)
        output = model(X)
        loss = F.cross_entropy(output, y)
        accelerator.backward(loss)
        optimizer.step()
        train_batch_losses.append(loss.item())

        # Update progress bar with current loss
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'best_val_loss': f'{best_val_loss:.4f}'})

    val_loss = evaluate(model, val_loader, mu, sd, denom)
    eval_losses.append(val_loss)
    outputs[model_name] = {"train": train_losses, "val": eval_losses}
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        if accelerator.is_main_process:
            torch.save(model.state_dict(), "weights.pth")

    epoch_train_loss = np.mean(train_batch_losses)
    train_losses.append(epoch_train_loss)

    print(f"Val Loss: {val_loss} Best Loss: {best_val_loss:.4f}\n")  # , Best AP: {it_best_ap:.4f}")

# Save results
np.savez("assay_results", outputs=outputs, models=models)

