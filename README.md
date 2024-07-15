[![License](https://img.shields.io/badge/license-Apache%202-blue)](LICENSE)
[![](https://badgen.net/badge/DOI/10.1016j.eswa.2023.120680/blue?icon=instgrame)](https://www.sciencedirect.com/getaccess/pii/S095741742301182X/purchase)
[![https://github.com/Saeidhoseinipour/NMTFcoclust](https://badgen.net/badge/NMTF/Coclust/pink?icon=instgrame)](https://github.com/Saeidhoseinipour/NMTFcoclust/tree/master/Models)
[![https://github.com/Saeidhoseinipour/NMTFcoclust](https://badgen.net/badge/Original/Paper/yellow?icon=instgrame)](https://www.sciencedirect.com/science/article/abs/pii/S095741742301182X?via%3Dihub)
[![Supplementary material](https://badgen.net/badge/Supplementary/material/orange?icon=instgrame)](https://ars.els-cdn.com/content/image/1-s2.0-S095741742301182X-mmc1.pdf)
[![https://github.com/Saeidhoseinipour/NMTFcoclust](https://badgen.net/badge/Purchase/PDF/brown?icon=instgrame)](https://www.sciencedirect.com/getaccess/pii/S095741742301182X/purchase)
[![https://github.com/Saeidhoseinipour/NMTFcoclust](https://badgen.net/badge/Presentation/Video/green?icon=instgrame)](https://www.sciencedirect.com/getaccess/pii/S095741742301182X/purchase)
[![https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Models/NMTFcoclust_OPNMTF_alpha.py](https://badgen.net/badge/OPNMTF/Coclust/pink?icon=instgrame)](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Models/NMTFcoclust_OPNMTF_alpha.py)
[![https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Models/NMTFcoclust_OPNMTF_alpha.py](https://badgen.net/badge/ACM/Digital_Library/black?icon=instgrame)](https://dl.acm.org/doi/10.1016/j.eswa.2023.120680)
[![https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Models/NMTFcoclust_OPNMTF_alpha.py](https://badgen.net/badge/SSRN/Prereview/blue?icon=instgrame)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416222)
[![https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Models/NMTFcoclust_OPNMTF_alpha.py](https://badgen.net/badge/X/Mol/blue?icon=instgrame)](https://www.x-mol.net/paper/article/1665507162148552704)
[![https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Models/NMTFcoclust_OPNMTF_alpha.py](https://badgen.net/badge/PlumX/Metrics/green?icon=instgrame)](https://plu.mx/plum/a/?doi=10.1016/j.eswa.2023.120680&theme=plum-sciencedirect-theme&hideUsage=true)
 [![https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Models/NMTFcoclust_OPNMTF_alpha.py](https://badgen.net/badge/Colab/WS/red?icon=instgrame)](https://colab.ws/articles/10.1016/j.eswa.2023.120680)
 [![https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Models/NMTFcoclust_OPNMTF_alpha.py](https://badgen.net/badge/Prereview/PDF/pink?icon=instgrame)](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/SSRN-id4416222_Prereview_OPNMTF.pdf)
[![](https://badgen.net/badge/Devpost/OPNMTF/blue?icon=instgrame)](https://devpost.com/software/nmtfcoclust)
[![](https://badgen.net/badge/Semanticscholar/259071441/red?icon=instgrame)](https://www.semanticscholar.org/paper/Orthogonal-parametric-non-negative-matrix-with-for-Hoseinipour-Aminghafari/f2aa345a9a729171331f7d2aee7235772c7ac73a)











## Table of Contents
1. [Non-negative Matrix Tri-Factorization for Co-clustering](#Non-negative-Matrix-Tri-Factorization-for-Co-clustering)
2. [Brief Description of Models](#brief-description-of-models)
3. [Requirements](#requirements)
4. [Datasets](#datasets)
5. [Model Implementation](#model-implementation)
6. [Cite](#cite)
7. [Supplementary Material](#supplementary-material)
8. [Presentation Video](#presentation-video)
9. [References](#references)



## [`NMTFcoclust` (Non-negative Matrix Tri-Factorization for Co-clustering)](https://www.sciencedirect.com/science/article/abs/pii/S095741742301182X)
NMTFcoclust  implements decomposition on a data matrix 𝐗 (document-word counts, movie-viewer ratings, and product-customer purchases matrices) with  finding three matrices:


- **𝐅** (roles membership rows)
- **𝐆** (roles membership columns)
- **𝐒** (roles summary matrix)








 The low-rank approximation of \mathbf{X}\mathbf{X} by
     \mathbf{X} \approx \mathbf{FSG}^{\top}\mathbf{X} \approx \mathbf{FSG}^{\top}
     
<img alt="NMTF, Saeid Hoseinipour, text mining, Co-clustering" src="https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Doc/Image/NMTF2.png?raw=true" width="100%">




## Brief description of models
`NMTFcoclust` implements the proposed algorithm (**OPNMTF**) and some NMTF according to the objective functions below:
- [**OPNMTF**](https://www.sciencedirect.com/science/article/abs/pii/S095741742301182X) 
```math
D_{\alpha}(\mathbf{X}||\mathbf{FSG}^{\top})+
  \lambda \; D_{\alpha}(\mathbf{I}_{g}||\mathbf{F}^{\top}\mathbf{F})+
  \mu \; D_{\alpha}(\mathbf{I}_{s}||\mathbf{G}^{\top}\mathbf{G})
```
- [PNMTF](https://www.sciencedirect.com/science/article/abs/pii/S0957417417300283)
```math
 0.5||\mathbf{X}-\mathbf{F}\mathbf{S}\mathbf{G}^{\top}||^{2}+0.5 \tau \; Tr(\mathbf{F} \Psi_{g}\mathbf{F}^{\top})+0.5 \eta \; Tr(\mathbf{G} \Psi_{s}\mathbf{G}^{\top})+ 0.5 \gamma \; Tr(\mathbf{S}^{\top}\mathbf{S})
```
- [ONMTF](https://www.sciencedirect.com/science/article/abs/pii/S0306457310000038)
```math
	0.5 ||\mathbf{X}-\mathbf{F}\mathbf{S}\mathbf{G}^{\top}||^{2}
```
- [NBVD](https://dl.acm.org/doi/abs/10.1145/1081870.1081949)
 $$||\mathbf{X}-\mathbf{FSG}^{\top}||^{2}$$
- [ONM3T](https://dl.acm.org/doi/abs/10.1145/1150402.1150420)
```math
	||\mathbf{X}-\mathbf{F}\mathbf{S}\mathbf{G}^{\top}||^{2}+ 
 Tr(\Lambda (\mathbf{F}^{\top}\mathbf{F}-\mathbf{I}_{s}))+ 
 Tr(\Gamma (\mathbf{G}^{\top}\mathbf{G}-\mathbf{I}_{g}))
```
- [ODNMTF](https://link.springer.com/chapter/10.1007/978-3-642-24958-7_82)
```math
 ||\mathbf{X}-\mathbf{FF^{\top}XGG}^{\top}||^{2}+ Tr(\Lambda \mathbf{F}^{\top})+ Tr( \Gamma \mathbf{G}^{\top})
```
- [DNMTF](https://link.springer.com/chapter/10.1007/978-3-642-24958-7_82)
```math
 ||\mathbf{X}-\mathbf{FF^{\top}XGG}^{\top}||^{2}
```

## Requirements
```python
numpy==1.18.3
pandas==1.0.3
scipy==1.4.1
matplotlib==3.0.3
scikit-learn==0.22.2.post1
coclust==0.2.1

```


## [Datasets](https://github.com/Saeidhoseinipour/NMTFcoclust/tree/master/Datasets)

| Datasets | \#Documents | \#Words | Sporsity(%0) | Number of clusters |
| -- | ----------- | -- | -- | -- |
| [CSTR](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/cstr.mat) | 475 | 1000 | 96% | 4 |
| [WebACE](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/WebACE..mat) |2340  |1000  | 91.83% |20  |
| [Classic3](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/classic3.mat) |3891  |4303  |98%  |3  |
| [Sports](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/sports..mat) |8580  |14870  | 99.99% |7  |
| [Reviews](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/reviews..mat) |4069  |18483  | 99.99% |5  |
| [RCV1_4Class](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/RCV1_4Class.mat) |9625  |29992  | 99.75% |4  |
| [NG20](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/NG20..mat) |19949  | 43586 | 99.99% |20  |
| [20Newsgroups](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/20Newsgroups.mat) | 18846 | 26214 | 96.96% | 20 |
|[TDT2](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/TDT2..mat)|9394|36771|99.64%|30|
|[RCV1_ori](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Datasets/RCV1_ori..mat)|9625|29992|96.62%|4|

```python
import pandas as pd 
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix 



                                                                   # Read Datasets ------->  Classic3

file_name=r"NMTFcoclust\Dataset\Classic3\classic3.mat"
mydata = loadmat(file_name)

                                                                    # Data matrix 
X_Classic3 = mydata['A'].toarray()
X_Classic3_sum_1 = X_Classic3/X_Classic3.sum()
                                                                   
true_labels = mydata['labels'].flatten().tolist()                   # True labels list [0,0,0,..,1,1,1,..,2,2,2]  n_row_cluster = 3
true_labels = [x+1 for x in true_labels]                            # True labels list [1,1,1,..,2,2,2,..,3,3,3]  n_row_cluster = 3
print(confusion_matrix(true_labels, true_labels))



        Medical:        [[1033    0    0]
 Information Retrieval: [   0 1460    0]
 Aeronautical Systems:  [   0    0 1398]]

```

## Model

```python
from NMTFcoclust.Models.NMTFcoclust_OPNMTF_alpha_2 import OPNMTF
from NMTFcoclust.Evaluation.EV import Process_EV

OPNMTF_alpha = OPNMTF(n_row_clusters = 3, n_col_clusters = 3, landa = 0.3,  mu = 0.3,  alpha = 0.4, max_iter=1)
OPNMTF_alpha.fit(X_Classic3_sum_1)
Process_Ev = Process_EV( true_labels ,X_Classic3_sum_1, OPNMTF_alpha) 



Accuracy (Acc):0.9100488306347982
Normalized Mutual Info (NMI):0.7703948803438703
Adjusted Rand Index (ARI):0.7641161476685447

Confusion Matrix (CM):
				[[1033    0    0]
				 [ 276 1184    0]
				 [   0   74 1324]]
Total Time:  26.558243700000276
```

![DC](?raw=true)

<a href="">
  <img src="https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Doc/Image/WC_classic3.png" alt="OPNMTF, Text mining, Matrix factorization, Co-clustering, Saeid Hoseinipour, divergence" style="width: 70%;">
</a>

- [Download full-size image available in ESWA](https://ars.els-cdn.com/content/image/1-s2.0-S095741742301182X-gr4_lrg.jpg)

## Cite
Please cite the following paper in your publication if you are using [`NMTFcoclust`]() in your research:

```bibtex
 @article{NMTFcoclust, 
    title={Orthogonal Parametric Non-negative Matrix Tri-Factorization with $\alpha$-Divergence for Co-clustering}, 
    DOI={10.1016/j.eswa.2023.120680},
  volume  = {231}, 
   number   = {120680},
    journal={Expert Systems with Applications}, 
    authors={Saeid Hoseinipour, Mina Aminghafari, Adel Mohammadpour}, 
    year={2023}
} 
```

## Supplementary material
**OPNMTF** implements on synthetic datasets such as Bernoulli, Poisson, and Truncated Gaussian.
- [Available from GitHub](https://github.com/Saeidhoseinipour/NMTFcoclust/tree/master/Supplementary%20material "Saeid Hoseinipour, orthogonal parameteric, matrix factorization, co-clustering, text mining, word cloud")
- [Available from ESWA](https://ars.els-cdn.com/content/image/1-s2.0-S095741742301182X-mmc1.pdf "co-clustering, OPNMTF, NMTF, matrix factorization")
- [Pre-review version](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/SSRN-id4416222_Prereview_OPNMTF.pdf  "Saeid Hoseinipour, orthogonal parameteric, matrix factorization, co-clustering, text mining, word cloud")
- [Personalized URL providing 50 days' free access to the orginal article](https://authors.elsevier.com/c/1hFjU_LnESYZ-~ "Saeid Hoseinipour, orthogonal parameteric, matrix factorization, co-clustering, text mining, word cloud")
- [Industry Relations and Applications](https://github.com/Saeidhoseinipour/NMTFcoclust/tree/master/Applications "Saeid Hoseinipour, orthogonal parameteric, matrix factorization, co-clustering, text mining, word cloud")






## Highlights
- Our algorithm works by multiplicative update rules and it is convergence.
- Adding two penalties  for controlling the orthogonality of row and column clusters.
- Unifying a class of algorithms for co-clustering based on $\alpha$-divergence.
- All datasets and algorithm codes are available on GitHub as [`NMTFcoclust`]() repository.



##  Presentation video

<!--

[![Presentation video for OPNMTF, Text mining, Matrix factorization, Co-clustering](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Doc/Image/OPNMTF_video.png)](https://www.youtube.com/watch?v=LCamkfTYGyM&t=5s)


<p align="center">
  <a href="https://www.youtube.com/watch?v=LCamkfTYGyM&t=5s">
    <img src="https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Doc/Image/OPNMTF_video.png" alt="Presentation video for OPNMTF, Text mining, Matrix factorization, Co-clustering" style="width:60%; transform: perspective(1000px) rotateY(-70deg);">
  </a>
</p>


<p align="center">
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C14.09 3.81 15.76 3 17.5 3 20.58 3 23 5.42 23 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
  </svg>
</p>

-->

<a href="">
  <img src="https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Doc/Image/Cover_OPNMTF.png" alt="Presentation video for OPNMTF, Text mining, Matrix factorization, Co-clustering, Saeid Hoseinipour" style="width: 70%;">
</a>

## References


[1] [Wang et al, Penalized nonnegative matrix tri-factorization for co-clustering (2017), Expert Systems with Applications.](https://www.sciencedirect.com/science/article/abs/pii/S0957417417300283)

[2] [Yoo et al, Orthogonal nonnegative matrix tri-factorization for co-clustering: Multiplicative updates on Stiefel manifolds (2010), Information Processing and Management.](https://www.sciencedirect.com/science/article/abs/pii/S0306457310000038)
	
[3] [Ding et al, Orthogonal nonnegative matrix tri-factorizations for clustering (2008), Proceedings of the 12th ACM SIGKDD International Conference on Knowledge 	Discovery and Data Mining.](https://dl.acm.org/doi/abs/10.1145/1150402.1150420)

[4] [Long et al, Co-clustering by block value decomposition (2005), Proceedings of the Eleventh ACM SIGKDD International Conference on Knowledge Discovery in Data 	Mining.](https://dl.acm.org/doi/abs/10.1145/1081870.1081949)

[5] [Labiod et al, Co-clustering under nonnegative matrix tri-factorization (2011), International Conference on Neural Information Processing.](https://link.springer.com/chapter/10.1007/978-3-642-24958-7_82)

[6] [Li et al, Nonnegative Matrix Factorization on Orthogonal Subspace (2010), Pattern Recognition Letters.](sciencedirect.com/science/article/abs/pii/S0167865509003651)

[7] [Li et al, Nonnegative Matrix Factorizations for Clustering: A Survey (2019), Data Clustering.](https://www.taylorfrancis.com/chapters/edit/10.1201/9781315373515-7/nonnegative-matrix-factorizations-clustering-survey-tao-li-cha-charis-ding)

[8] [Cichocki et al, Non-negative matrix factorization with $\alpha$-divergence (2008), Pattern Recognition Letters.](https://www.sciencedirect.com/science/article/abs/pii/S0167865508000767)

[9] [Saeid, Hoseinipour et al, Orthogonal parametric non-negative matrix tri-factorization with $\alpha$-Divergence for co-clustering, Expert Systems with Applications (2023).](https://doi.org/10.1016/j.eswa.2023.120680)
