# Self-Supervised Heterogeneous Graph Neural Network Based on Deep and Broad Neighborhood Encoding
## Dataset
Most of the used original heterogeneous graph datasets can be downloaded [here](https://1drv.ms/f/c/1b2f69874f634cd8/ElArKe6mhI1HjCZuYh8SG80Bx-PI3CePKx5kBdRtCsLBSQ?e=zbzZBO). 
Please download them and put them in ```dataset`` folder. 

## Requirements

The following libraries are required:

- [PyTorch 1.12.0](https://pytorch.org/)
- [NumPy](https://github.com/numpy/numpy)
- [Pandas](https://github.com/pandas-dev/pandas)
- [DGL 1.0.0](https://www.dgl.ai/)
- [SciPy](https://scipy.org/)

## Quick Start

### Scripts for  Node Classification
* Example of training *HGNN-DB* on *ACM* dataset:
```{bash}
python train_node_classification.py --dataset acm --alpha 0.5 --gamma 0.8 --beta 3 --k 6
```
### Scripts for  Link Prediction
```{bash}
python train_link_prediction.py --dataset acm --alpha 0.5 --gamma 0.8 --beta 3 --k 6 --strategy rand
```
