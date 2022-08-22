# **NMTFcoclust**  
### **NMTFcocluster** (Non-negative Matrix Tri-Factorization for coclustering) is a library that implements decomposition on a data matrix $\mathbf{X}$ (document-word matrix and so on) with finding three  matrices $\mathbf{F}$ (roles membership rows), $\mathbf{G}$ (roles membership columns) and $\mathbf{S}$ (roles summary matrix) based optimazed $\alpha$-divergence.

 The low-rank approximation of $\mathbf{X}$ by
     $$\mathbf{X} \approx \mathbf{FSG}^{T}+ \mu D_{\alpha}(\mathbf{I} ||\mathbf{FSG^{T}})+  D_{\alpha}(\mathbf{I} ||\mathbf{FSG}^{T}) $$
where $n$, $m$, $g\leqslant n$ and $s\leqslant m$ are the number of rows, columns, row clusters and column clusters, respectively.


![NMTF](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Doc/Image/nmtf3.png?raw=true)

### Brief description 
NMTFcoclust library implements three proposed algorithms and other orthogonal NMTF:
- $OPNMTF_{\alpha}$ 
 $$D_{\alpha}(\mathbf{X}||\mathbf{FSG}^{T})+ \mu D_{\alpha}(\mathbf{I} ||\mathbf{FSG^{T}})+  D_{\alpha}(\mathbf{I} ||\mathbf{FSG}^{T})$$
- $ONMTF_{\alpha}$
- $NMTF_{\alpha}$
- $NBVD$
- $ONM3T$
- $ODNMTF$

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
### Installing NMTFcoclust

### License

### Examples

### Datasets

- [CSTR](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/cstr.mat)
- [Classic3](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/classic3.mat)
- [RCV1_4Class](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/RCV1_4Class.mat)
- [20Newsgroups](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/20Newsgroups.mat)
- [NG20](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/NG20..mat)
- [TDT2](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/TDT2..mat)
- [WebACE](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/WebACE..mat)
- [Reviews](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/reviews..mat)
- [Sports](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/sports..mat)
- [Reuters21578](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/Reuters21578..mat)
- [RCV1_ori](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/RCV1_ori..mat)


### References

[1] L. Labiod, M. Nadif, Co-clustering under nonnegative matrix tri-factorization.
