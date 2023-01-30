# **NMTFcoclust**  
### **NMTFcocluster** (Non-negative Matrix Tri-Factorization for coclustering) is a package that implements decomposition on a data matrix $\mathbf{X}$ (document-word matrix and so on) with finding three  matrices $\mathbf{F}$ (roles membership rows), $\mathbf{G}$ (roles membership columns) and $\mathbf{S}$ (roles summary matrix) based optimazed $\alpha$-divergence.

 The low-rank approximation of $\mathbf{X}$ by
     $$\mathbf{X} \approx \mathbf{FSG}^{T} $$
where $n$, $m$, $g \leqslant n$ and $s \leqslant m$ are the number of rows, columns, row clusters and column clusters, respectively.


![NMTF](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Doc/Image/nmtf3.png?raw=true)


### Brief description 
NMTFcoclust implements three proposed algorithms and other orthogonal NMTF:
- $OPNMTF_{\alpha}$ 
  
  $$D_{\alpha}(\mathbf{X}||\mathbf{FSG}^{T})+ 
      \lambda D_{\alpha}(\mathbf{I}_{g}||\mathbf{F}^{T}\mathbf{F})+\mu D_{\alpha}(\mathbf{I}_{s}||\mathbf{G}^{T}\mathbf{G}) $$  
- $ONMTF_{\alpha}$
   $$D_{\alpha}(\mathbf{X}||\mathbf{FSG}^{T})+ \delta Tr(\mathbf{F}\Psi_{g}\mathbf{F}^{T}) +	\beta Tr(\mathbf{G} \Psi_{s}\mathbf{G}^{T})$$
- $NMTF_{\alpha}$
 $$D_{\alpha}(\mathbf{X}||\mathbf{FSG}^{T})$$ 
- $NBVD$
 $$||\mathbf{X}-\mathbf{FSG}^{T}||^{2}$$
- $ONM3T$
 $$||\mathbf{X}-\mathbf{FSG}^{T}||^{2}$$
- $ODNMTF$
 $$||\mathbf{X}-\mathbf{FF^{T}XGG}^{T}||^{2}$$

### Requirements
```python
numpy==1.18.3
pandas==1.0.3
scipy==1.4.1
matplotlib==3.0.3
scikit-learn==0.22.2.post1
coclust==0.2.1

```
### Installing NMTFcoclust

### License

### Examples


![WC](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Doc/Image/WC_1_5_bold_31_32_11_22_33_v2.png?raw=true)

### Datasets

| Datasets | Documents | Words | Sporsity | Number of clusters |
| -- | ----------- | -- | -- | -- |
| [CSTR](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/cstr.mat) | 475 | 1000 | 96% | 4 |
| [WebACE](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/WebACE..mat) |2340  |1000  | 91.83% |20  |
| [Classic3](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/classic3.mat) |3891  |4303  |98%  |3  |
| [Sports](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/sports..mat) |8580  |14870  | 99.99% |7  |
| [Reviews](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/reviews..mat) |4069  |18483  | 99.99% |5  |
| [RCV1_4Class](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/RCV1_4Class.mat) |9625  |29992  | 99.75% |4  |
| [NG20](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/NG20..mat) |19949  | 43586 | 99.99% |20  |


- [20Newsgroups](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/20Newsgroups.mat)
- [TDT2](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/TDT2..mat)
- [Reuters21578](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/Reuters21578..mat)
- [RCV1_ori](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/RCV1_ori..mat)


### References

