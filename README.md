# Models to decode cell phenotypes during live imaging experiments
Built using Python 3.7.12

---

## Supervised classification

*Train models to learn a mapping from cell image to a label (e.g., TDP-43 or control).*

CIFAR10 example: `python run.py --config-name=cifar10_resnet18.yaml`


COR14 example: `python run.py --config-name=cor14_resnet18.yaml`

[IPython notebook](https://colab.research.google.com/drive/1z_oPHNqNw_e7DCIL6SfvADRCgUA8RrMR)

---

## Self-supervised learning

*Train models with SimCLR to learn good visual features on live-cell imaging datasets.*


## Self-supervised testing

*Transfer trained self-sup models to a new task, such as classifying disease in a cell.*