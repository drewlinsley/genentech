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

f = np.load("assay_2.0_results.npz", allow_pickle=True)
sdata2 = f["outputs"].item()
sacc_data2 = {k: v["acc"] for k, v in sdata.items()}
smeta2 = f["models"].item()

model_classes = []
accs, in_metas, param_metas = [], [], []
pattern = r'^[a-zA-Z]+'
mk = {}
for k in meta.keys():
    result = re.match(pattern, k)
    model_classes.append(result.group())
    mk[result.group()] = k
    accs.append(acc_data[k])
    in_metas.append(meta[k][0])
    param_metas.append(meta[k][1])
accs = np.asarray(accs)
in_metas = np.asarray(in_metas)
param_metas = np.asarray(param_metas)
model_classes = np.asarray(model_classes)

###
saccs = []
for k in smeta.keys():
    saccs.append(sacc_data[k])
saccs = np.asarray(saccs)

saccs2 = []
s2models = []
for k in smeta2.keys():
    saccs2.append(sacc_data2[k])
    s2models.append(k)
saccs2 = np.asarray(saccs2)
s2models = np.asarray(s2models)


res_0 = {k: data[k]["acc"] for k in data.keys()}
res_1 = {k: sdata[k]["acc"] for k in sdata.keys()}
res_2 = {k: sdata2[k]["acc"] for k in sdata2.keys()}

f = plt.figure()

plt.subplot(131)
plt.title("Does ALS detection acc scale with ImageNet acc?")
import matplotlib.cm as cm
# colors = cm.get_cmap("tab20c")  # (len(np.unique(model_classes)))
colors = [(235, 64, 52), (104, 235, 52), (52, 153, 235), (227, 217, 36)]
colors = np.asarray(colors) / 255.
for c, m in enumerate(np.unique(model_classes)):
    y = accs[model_classes == m] * 100
    x = in_metas[model_classes == m][0]
    yy = saccs[model_classes == m] * 100
    
    y = res_0[mk[m]] * 100
    yy = res_1[mk[m]] * 100

    # keep = y > yy  # Remove noise for steve presentation
    # y = y[keep]
    # x = x[keep]
    # yy = yy[keep]
    plt.plot(x, y, marker='o', linestyle='', ms=12, label=m, alpha=0.6, color=colors[c])
    plt.plot(x, yy, marker='o', linestyle='', ms=12, alpha=0.2, color=colors[c])
    if mk[m] in res_2:
        y2 = res_2[mk[m]] * 100
        plt.plot(x, y2, marker='o', linestyle='', ms=12, color=colors[c])

    print(c)
    # plt.arrow(x, yy, x, y, head_width=0.1, head_length=0.2, alpha=0.3)

plt.legend()
plt.ylim([50, 100])
plt.xlim([65, 100])
plt.xlabel("ImageNet accuracy")
plt.ylabel("ALS detection accuracy")
plt.subplot(132)
plt.title("Does ALS detection acc scale with model size?")
for c, m in enumerate(np.unique(model_classes)):
    y = accs[model_classes == m] * 100
    x = param_metas[model_classes == m][0]
    yy = saccs[model_classes == m] * 100

    y = res_0[mk[m]] * 100
    yy = res_1[mk[m]] * 100

    plt.plot(x, y, marker='o', linestyle='', ms=12, label=m, alpha=0.6, color=colors[c])
    plt.plot(x, yy, marker='o', linestyle='', ms=12, alpha=0.2, color=colors[c])
    if mk[m] in res_2:
        # y2 = saccs2[s2models == m] * 100
        y2 = res_2[mk[m]] * 100
        plt.plot(x, y2, marker='o', linestyle='', ms=12, color=colors[c])

plt.xlabel("Number of parameters (M)")
plt.ylabel("ALS detection accuracy")
plt.ylim([50, 100])



# Lastly, plot scaling law for resnet
ax = plt.subplot(133)
plt.title("Resnet 18 performance vs. data")

perfs = [res_1["resnet18d"], res_0["resnet18d"], res_2["resnet18d"]]
perfs = np.asarray(perfs) * 100
lens = [2000, 16000, 160000]

from statsmodels.regression.linear_model import OLS
X = np.stack((lens, np.ones_like(lens)), 1)
y = np.asarray(perfs).reshape(-1, 1)
model = OLS(y, X).fit()
params = model.params

startp = [0, params[1]]
xs = [0, lens[-1]]
ys = [params[1], params[0] * lens[-1] + params[1]]
# plt.plot(xs, ys, "-", color=np.asarray((104, 235, 52)) / 255.)
plt.plot(lens, perfs, color=np.asarray((104, 235, 52)) / 255., marker='o', linestyle='', ms=12)

ax.set_xscale('log')
# plt.text(lens[0], perfs[-1], "y = {} * Images + {}".format(round(params[0], 4), round(params[1], 4)), fontsize=8)
print(params)
plt.xlabel("Number of Images")
plt.ylabel("ALS detection accuracy")
plt.ylim([50, 100])
plt.show()


