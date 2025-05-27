# pyg_SRC
Sparse representation-based classification of graph signals implemented in the pytorch-geometric framework

---

## scripts
To initiate training:
```bash
python3 run_example.py --[FLAGS]
```
Default flags are set in './config/DEFAULT/...' To initiate evaluation with the eval model specified in './config/DEFAULT/DEF_config.json':

```bash
python3 eval_example.py
```

## features
1. Batch-wise fast iterative least shrinkage algorithm (FISTA) (differentiable with torchopt decorator)
2. Graph Fourier inversion of polynomial kernels
3. Diagonalisation of graphs from pytorch geometric datasets
