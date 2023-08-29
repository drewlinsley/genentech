import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# import seaborn as sns


f = np.load("assay_results.npz", allow_pickle=True)
data = f["outputs"].item()
acc_data = {k: v["acc"] for k, v in data.items()}
meta = f["models"].item()
# dfd = pd.DataFrame.from_dict(acc_data)
# dfm = pd.DataFrame.from_dict(meta)
f = np.load("assay_results_small_data.npz", allow_pickle=True)
sdata = f["outputs"].item()
sacc_data = {k: v["acc"] for k, v in sdata.items()}
smeta = f["models"].item()


model_classes = []
accs, in_metas, param_metas = [], [], []
pattern = r'^[a-zA-Z]+'
for k in meta.keys():
    result = re.match(pattern, k)
    model_classes.append(result.group())
    accs.append(acc_data[k])
    in_metas.append(meta[k][0])
    param_metas.append(meta[k][1])
accs = np.asarray(accs)
in_metas = np.asarray(in_metas)
param_metas = np.asarray(param_metas)
model_classes = np.asarray(model_classes)

###
saccs = []
pattern = r'^[a-zA-Z]+'
for k in meta.keys():
    saccs.append(sacc_data[k])
saccs = np.asarray(saccs)



f = plt.figure()
plt.subplot(121)
plt.title("Does ALS detection acc scale with ImageNet acc?")
import matplotlib.cm as cm
# colors = cm.get_cmap("tab20c")  # (len(np.unique(model_classes)))
colors = [(235, 64, 52), (104, 235, 52), (52, 153, 235), (227, 217, 36)]
colors = np.asarray(colors) / 255.
for c, m in enumerate(np.unique(model_classes)):
    y = accs[model_classes == m] * 100
    x = in_metas[model_classes == m]
    yy = saccs[model_classes == m] * 100
    # keep = y > yy  # Remove noise for steve presentation
    # y = y[keep]
    # x = x[keep]
    # yy = yy[keep]
    plt.plot(x, y, marker='o', linestyle='', ms=12, label=m, color=colors[c])
    plt.plot(x, yy, marker='o', linestyle='', ms=12, alpha=0.3, color=colors[c])
    print(c)
    # plt.arrow(x, yy, x, y, head_width=0.1, head_length=0.2, alpha=0.3)

plt.legend()
plt.ylim([50, 100])
plt.xlim([65, 100])
plt.xlabel("ImageNet accuracy")
plt.ylabel("ALS detection accuracy")
plt.subplot(122)
plt.title("Does ALS detection acc scale with model size?")
for c, m in enumerate(np.unique(model_classes)):
    y = accs[model_classes == m] * 100
    x = param_metas[model_classes == m]
    yy = saccs[model_classes == m] * 100
    plt.plot(x, y, marker='o', linestyle='', ms=12, label=m, color=colors[c])
    plt.plot(x, yy, marker='o', linestyle='', ms=12, alpha=0.3, color=colors[c])
plt.xlabel("Number of parameters (M)")
plt.ylabel("ALS detection accuracy")
plt.ylim([50, 100])
plt.show()


f = plt.figure()

