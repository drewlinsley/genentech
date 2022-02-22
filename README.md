# Models to decode cell phenotypes during live imaging experiments
Built using Python 3.7.12, Pytorch, and Pytorchlightning.
Visualization with [Weights&Biases](https://wandb.ai/) 

---

## Supervised classification

*Train models to learn a mapping from cell image to a label (e.g., TDP-43 or control).*

CIFAR10 example: `python run.py --config-name=cifar10_resnet18.yaml`
[W&B Progress](https://wandb.ai/drewlinsley/genentech/runs/3w4tgc75?workspace=user-drewlinsley)
[IPython notebook](https://colab.research.google.com/drive/1z_oPHNqNw_e7DCIL6SfvADRCgUA8RrMR)

COR14 example: `python run.py --config-name=cor14_resnet18.yaml`
[W&B Progress](https://wandb.ai/drewlinsley/genentech/runs/1cganw4q)
[IPython notebook](https://colab.research.google.com/drive/1z_oPHNqNw_e7DCIL6SfvADRCgUA8RrMR)

---

## Self-supervised learning

*Train models with SimCLR to learn good visual features on live-cell imaging datasets.*


## Self-supervised testing

*Transfer trained self-sup models to a new task, such as classifying disease in a cell.*