# **NMTFcoclust**  
### **NMTFcocluster** (Non-negative Matrix Tri-Factorization for coclustering) is a library that implements decomposition on a data matrix $\mathbf{X}$ (document-word matrix and so on) with finding three  matrices $\mathbf{F}$ (roles membership rows), $\mathbf{G}$ (roles membership columns) and $\mathbf{S}$ (roles summary matrix) based optimazed $\alpha$-divergence.

 The low-rank approximation of $\mathbf{X}$ by
     $$\mathbf{X} \approx \mathbf{FSG}^{T}$$
where $n$, $m$, $g\leqslant n$ and $s\leqslant m$ are the number of rows, columns, row clusters and column clusters, respectively.

[!NMTF](https://github.com/Saeidhoseinipour/NMTFcoclust.git/Doc/Image/nmtf3.png)


### Brief description 
TensorClus library provides multiple functionalities:
- Several datasets 
- Tensor co-clustering with various data type
- Tensor decomposition and clustering
- Visualization

### Requirements
```python
numpy==1.18.3
pandas==1.0.3
scipy==1.4.1
matplotlib==3.0.3
scikit-learn==0.22.2.post1
coclust==0.2.1
tensorD==0.1
tensorflow==2.3.0
tensorflow-gpu==2.3.0
tensorflow-estimator==2.3.0
tensorly==0.4.5
```
### Installing TensorClus

### License

### Examples

### Datasets

### References
