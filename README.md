# Self-Supervised Heterogeneous Graph Neural Network Based on Deep and Broad Neighborhood Encoding

Most of the used original heterogeneous graph datasets can be downloaded [here](https://1drv.ms/f/c/1b2f69874f634cd8/ElArKe6mhI1HjCZuYh8SG80Bx-PI3CePKx5kBdRtCsLBSQ?e=zbzZBO). 
Please download them and put them in ```dataset`` folder. 

## Environments

[PyTorch 1.12.0](https://pytorch.org/),
[numpy](https://github.com/numpy/numpy),
[pandas](https://github.com/pandas-dev/pandas),
[dgl 1.0.0](https://www.dgl.ai/)
[scipy](https://scipy.org/)

## Model Training
* Example of training *HGNN-DB* on *ACM* dataset:
```{bash}
python main.py --dataset acm --alpha 0.5 --gamma 0.8 --beta 3 --k 6
```