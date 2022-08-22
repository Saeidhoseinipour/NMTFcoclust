Co-clustering ,Non-negative Matrix Tri-Factorization
,$\alpha$-divergance ,Orthogonality.

Introduction
============

Notation
--------

                                          Notation                                                                            Usage
  ----------------------------------------------------------------------------------------- --------------------------------------------------------------------------
                                        $i = 1,...,n$                                                                 $n$ is number of rows
                                        $j = 1,...,m$                                                                $m$ is number of columns
                                        $k = 1,...,g$                                                             $g$ is number of row clusters
                                        $h = 1,...,s$                                                            $s$ is number of column clusters
                   $\mathbf{X}=($X$_{ij}) \in \mathbb{R}_{+}^{n \times m}$                                                 Data matrix
                   $\mathbf{F}=($F$_{ik})\in \mathbb{R}_{+}^{n \times g}$                                               Row cluster matrix
                   $\mathbf{S}=($S$_{kh})\in \mathbb{R}_{+}^{g \times s}$                                           Summary co-cluster matrix
                   $\mathbf{G}=($G$_{jh})\in \mathbb{R}_{+}^{m \times s}$                                             Column cluster matrix
         $\mathbf{FSG}^{T}=([\mathbf{FSG}^{T}]_{ij})\in \mathbb{R}_{+}^{n \times m}$                                   Approximation matrix
                      $\mathbf{E}_{nm}\in \mathbb{R}_{1}^{n \times m}$                                       A matrix of ones with size $n \times m$
                      $\mathbf{E}_{gg}\in \mathbb{R}_{1}^{g \times g}$                                       A matrix of ones with size $g \times g$
                      $\mathbf{E}_{ss}\in \mathbb{R}_{1}^{s \times s}$                                       A matrix of ones with size $s \times s$
               $\mathbf{I}_{g}=[I_{kk}]\in  \mathbb{R}_{\{0,1\}}^{g \times g}$                                Identity matrix with size $g \times g$
               $\mathbf{I}_{s}=[I_{hh}]\in  \mathbb{R}_{\{0,1\}}^{s \times s}$                               Identity matrix with size $s \times s$.
    $\bm{\Psi}_{g}= \mathbf{E}_{gg} -\mathbf{I}_{g}\in \mathbb{R}_{\{0,1\}}^{g \times g}$                        Penalized matrix of $\mathbf{F}$
   $\bm{\Psi}_{s}== \mathbf{E}_{ss} -\mathbf{I}_{s} \in \mathbb{R}_{\{0,1\}}^{s \times s}$                       Penalized matrix of $\mathbf{G}$
                       $\mathbf{D_{F}}=diag(\mathbf{1}^{T}\mathbf{F})$                                    Diagonal matrix normalization for $\mathbf{F}$
                       $\mathbf{D_{G}}=diag(\mathbf{1}^{T}\mathbf{G})$                                    Diagonal matrix normalization for $\mathbf{G}$
                         $D_{\alpha}(\mathbf{X}|| \mathbf{FSG}^{T})$                             $\alpha$-divergance between $\mathbf{X}$ and $\mathbf{FSG}^{T}$
                                $\xi_{\text{NMTF}_{\alpha}}$                                            Object function for NMTF with $\alpha$-divergence
                                $\xi_{\text{ONMTF}_{\alpha}}$                                           Object function for ONMTF with $\alpha$-divergence
                               $\xi_{\text{OPNMTF}_{\alpha}}$                                          Object function for OPNMTF with $\alpha$-divergence
                                     $\bm{\mathcal{F}}$                                                          The sets of possible row labels
                                     $\bm{\mathcal{G}}$                                                         The sets of possible column labels
                                      $\mathbf{X_{kh}}$                                                                  Co-cluster $kh$
                               $f(\mathbf{X};\mathbf{\Theta})$                                                  Total probablity density function
                                $\varphi(X_{ij};\delta_{kh})$                                              Probability density function co-cluster $kh$
                              $\bm{\pi}=(\pi_{1},...,\pi_{g})$                                                      Proportion of row clusters
                              $\bm{\rho}=(\rho_1,...,\rho_{s})$                                                   Proportion of column clusters
                         $\bm{\delta}=(\delta_{11},...,\delta_{gs})$                                                Parameters co-cluster $kh$
                       $\bm{\theta}=(\bm{\pi},\bm{\rho},\bm{\delta})$                                                 Total parameters model
                   $\bm{\alpha}=(\alpha_{kh})\in \mathbb{R}^{g \times s}$                           The effect of co-cluster matrix in $\textbf{Poisson LBM}$
                          $\bm{\omega}=(\omega_{1},...,\omega_{g})$                                    The effect of row clusters in $\textbf{Poisson LBM}$
                              $\bm{\nu}=(\nu_{1},...,\nu_{s})$                                       The effect of column clusters in $\textbf{Poisson LBM}$
                      $\bm{\mu}=(\mu_{kh}) \in \mathbb{R}^{g \times s}$                         The mean of co-cluster matrix in $\textbf{Truncated Gussian LBM}$
               $\bm{\sigma}^{2}=(\sigma^{2}_{kh}) \in \mathbb{R}^{g \times s}$                The variance of co-cluster matrix in $\textbf{Truncated Gussian LBM}$
                   $\bm{\gamma}=(\gamma_{kh})\in \mathbb{R}^{g \times s}$                    The probability of success co-cluster matrix in $\textbf{Bernoulli LBM}$

        Notation                                      Usage
  -------------------- -------------------------------------------------------------------
         $\phi$                           Standard gussian distribution
         $\Phi$                         Cumulative distribution function
         $\psi$                                Projection function
          $a$                               Minimum value of $X_{ij}$
          $b$                               Maximum value of $X_{ij}$
       $\lambda$              Parameter orthogonaly row in $\text{OPNMTF}_{\alpha}$
         $\mu$              Parameter orthogonaly column in $\text{OPNMTF}_{\alpha}$
        $\delta$        Parameter orthogonaly for $\mathbf{F}$ in $\text{ONMTF}_{\alpha}$
        $\beta$         Parameter orthogonaly for $\mathbf{G}$ in $\text{ONMTF}_{\alpha}$
        $\alpha$                          Parameter $\alpha$-divergence
     $\varepsilon$                           A small positive number
          $t$                                    Iteration of t
        $\odot$                            Element-wise multiplication
       $\oslash$                              Element-wise division
   $[P]_{+}=max(P,0)$         Applying the non-negativity condition for matrix $P$
     $\text{Tr}(P)$                            Trace of matrix $P$
    $\text{det}(P)$                      The determinant of a matrix $P$

Related works
-------------

Dyadic data
-----------

Dyadic data matrices, such as word-document counts, movie-viewer
ratings, product-customer purchases matrices, refer to the duality
between rows and columns that frequently arise in various essential
applications---for example, collaborative filtering
[@shan2010generalized], text mining, and gene expression data analysis.
[@cheng2000biclustering], [@hofmann1999learning]. A fundamental problem
in dyadic data analysis is discovering the hidden block structure of the
data matrix. [@Long2005], [@DelBuono2015]. Seek sub-matrices in a dyadic
data matrix is a simple idea called Co-clustering.

Co-clustering
-------------

Traditional one-side clustering cannot seek a relation between sub-sets
of rows (words) and columns (documents). However, simulations clustering
on rows and columns are very useful for discovering these patterns
[@hartigan1972direct], [@govaert2013co]. All most co-clustering
techniques are optimization problems with a certain objective function.
These algorithms are different based on the minimuming or maximuming the
objective function. Statistical algorithms are based on maximization
because the objective function is the complete log-likelihood function
[@govaert2013co]. Matrix factorization algorithms are based on
minimization because the objective function is divergence. This duality
relates to information theory and the measure of similarity or
dissimilarity. Also, there is a clear relation between these two
approaches [@dhillon2003information], [@e16073552].

![A graphical example of co-clustering with a word-document matrix
$\mathbf{X} \in \mathbb{R}_{+}^{6 \times 8}$ is shown in the case of
$g=3$ and $s=2$. The larger square denotes the larger value of a
corresponding element in the matrix.](nmtf3){#fig:nmtf3
width="0.7\\linewidth"}

Nonnegative matrix factorization (NMF)
--------------------------------------

Nonnegative matrix factorization (NMF) in its modern form has become a
standard tool in the analysis of high-dimensional data sets
[@Gillis2020] such as text data. Matrix factorization methods have been
widely used in dyadic data analysis
[@Junior2020; @ding2006orthogonal; @Long2005; @DelBuono2015a; @Yoo2010].
Non-negative Matrix Tri-Factorization (NMTF) [@lee1999learning] was
presented for co-clustering dyadic data (on both sets of rows and
columns) whose interest is well established. Non-negative matrix
tri-factorization methods have been widely applied in dyadic data
analysis. [@lee1999learning]. Orthogonal Non-negative Matrix
Factorization was first introduced in [@ding2006orthogonal]. The
nonnegative matrix tri-factorization (NMTF) algorithm, which aims to
decompose an objective matrix into three low-dimensional matrices, is an
important tool to achieve co-clustering [@deng2021tri].

width=,center

          Ref                  Algorithm                                                                                                                       Object function                                                                                                                                                                             Constraints                                                                                                                                                                                                                             Multiplicative Update Rules
  -------------------- -------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ ---------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        $\text{OPNMTF}_{\alpha}$                         $D_{\alpha}(\mathbf{X}|| \mathbf{F}\mathbf{S}\mathbf{G}^{T})+\lambda D_{\alpha}(\mathbf{I} \parallel \mathbf{F}^{T}\mathbf{F})+\mu D_{\alpha}(\mathbf{I}\parallel \mathbf{G}^{T}\mathbf{G)}$                                                                    $\mathbf{F}\geqslant 0$, $\mathbf{G}\geqslant 0$                                                                            $\mathbf{F}\odot \left[ \left(\dfrac{(\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{GS}^{T}+2\lambda \mathbf{F} (\mathbf{I}_{g}\oslash \mathbf{F}^{T}\mathbf{F})^{\alpha}}{\mathbf{E}_{nm}\mathbf{G}\mathbf{S}^{T}+2\lambda \mathbf{F}\mathbf{E}_{gg}}\right)^{\dfrac{1}{\alpha}}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \right]_{+} ,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   \mathbf{G}\odot \left[ \left(\dfrac{((\mathbf{X}\oslash \mathbf{FSG}^{T})^{T})^{\alpha} \mathbf{FS}+2\mu \mathbf{G}(\mathbf{I}_{s}\oslash \mathbf{G}^{T}\mathbf{G})^{\alpha} }{\mathbf{E}_{nm}^{T}\mathbf{FS}+2\mu \mathbf{G}\mathbf{E}_{ss}}\right)^{\dfrac{1}{\alpha}}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \right]_{+} , 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \mathbf{S}\odot \left[ \left(\dfrac{\mathbf{F}^{T} (\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{G}}{\mathbf{F}^{T}\mathbf{E}_{nm}\mathbf{G}}\right)^{\dfrac{1}{\alpha}}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \right] _{+}$
                        $\text{ONMTF}_{\alpha}$                                                                                                   $D_{\alpha}(\mathbf{X}|| \mathbf{FSG}^{T})                                                                                                                                             $\mathbf{F}\geqslant 0$, $\mathbf{G}\geqslant 0$                                                                                                          $\mathbf{F}\odot \left[ \left(\dfrac{(\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{GS}^{T}}{\mathbf{E}_{nm}\mathbf{G}\mathbf{S}^{T}+2 \delta \alpha \mathbf{F}\bm{\Psi}_{g}}\right)^{\frac{1}{\alpha}}\right]_{+} ,
                                                                                                                                                                            +                                                                                                                                                                                                                                                                                                                                 \mathbf{G}\odot \left[\left( \dfrac{((\mathbf{X}\oslash \mathbf{FSG}^{T})^{T})^{\alpha} \mathbf{FS}}{\mathbf{E}_{nm}^{T}\mathbf{FS}+2\beta \alpha \mathbf{G}\bm{\Psi}_{s}}\right)^{\frac{1}{\alpha}}\right]_{+} , 
                                                                                                                                                      \delta \text{Tr}(\mathbf{F}\Psi\mathbf{F}^{T})                                                                                                                                                                                                                                                                                                                     \mathbf{S}\odot \left[\left( \dfrac{\mathbf{F}^{T} (\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{G}}{\mathbf{F}^{T}\mathbf{E}_{nm}\mathbf{G}}\right)^{\frac{1}{\alpha}}\right] _{+}$
                                                                                                                                                                            +                                                                                                                                                                                                                                                         
                                                                                                                                                     \beta \text{Tr}(\mathbf{G}\Omega\mathbf{G}^{T})$                                                                                                                                                                                                                                 
                         $\text{NMTF}_{\alpha}$                                                                                                  $D_{\alpha}(\mathbf{X}|| \mathbf{FSG}^{T})$                                                                                                                                             $\mathbf{F}\geqslant 0$, $\mathbf{G}\geqslant 0$                                                                                                                                  $\mathbf{F}\odot \left[ \left(\dfrac{( \mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{GS}^{T}}{\mathbf{E}_{nm}\mathbf{GS}^{T}}\right)^{\dfrac{1}{\alpha}}\right] _{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \mathbf{G}\odot \left[ \left(\dfrac{((\mathbf{X}\oslash \mathbf{FSG}^{T})^{T}) ^{\alpha} \mathbf{FS}}{\mathbf{E}_{nm}^{T}\mathbf{FS}}\right)^{\dfrac{1}{\alpha}}\right]_{+} ,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         \mathbf{S}\odot \left[\left(\dfrac{\mathbf{F}^{T} (\mathbf{X}\oslash \mathbf{FSG}^{T}) ^{\alpha} \mathbf{G}}{\mathbf{F}^{T}\mathbf{E}_{nm}\mathbf{G}}\right)^{\dfrac{1}{\alpha}}\right]_{+}$
       @Long2005a            $\text{NBVD}$                                                                                                 $\|\mathbf{X}-\mathbf{F}\mathbf{S}\mathbf{G}^{T}\|^{2}$                                                                                                                                       $\mathbf{F}\geqslant 0$, $\mathbf{G}\geqslant 0$                                                                                                                                                                         $\mathbf{F} \odot \left[\dfrac{\mathbf{XGS}^{T}}{\mathbf{FSG}^{T}\mathbf{GS}^{T}}\right]_{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \mathbf{G} \odot \left[\dfrac{\mathbf{X}^{T}\mathbf{FS}}{\mathbf{GS}^{T}\mathbf{F}^{T}\mathbf{FS}}\right]_{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \mathbf{S} \odot \left[\dfrac{\mathbf{F}^{T}\mathbf{XG}}{\mathbf{F}^{T}\mathbf{FSG}^{T}\mathbf{G}}\right]_{+}$
       @Ding2006c            $\text{ONM3F}$                                                                                                $\|\mathbf{X}-\mathbf{F}\mathbf{S}\mathbf{G}^{T}\|^{2}$                                                                                            $\mathbf{F}\geqslant 0$, $\mathbf{G}\geqslant 0$, $\mathbf{F}^{T}\mathbf{F}=\mathbf{I}_{g}$, $\mathbf{G}^{T}\mathbf{G}=\mathbf{I}_{s}$                                                                                                                    $\mathbf{F} \odot \left[\left(\dfrac{\mathbf{XGS}^{T}}{\mathbf{FF}^{T}\mathbf{XGS}^{T}}\right)^{-1/2}\right]_{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \mathbf{G} \odot \left[\left(\dfrac{\mathbf{X}^{T}\mathbf{FS}}{\mathbf{GG}^{T}\mathbf{X}^{T}\mathbf{FS}}\right)^{-1/2}\right]_{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \mathbf{S} \odot \left[\left(\dfrac{\mathbf{F}^{T}\mathbf{XG}}{\mathbf{F}^{T}\mathbf{FSG}^{T}\mathbf{G}}\right)^{-1/2}\right]_{+}$
        @Yoo2010             $\text{ONMTF}$                                                                                          $\dfrac{1}{2}\|\mathbf{X}-\mathbf{F}\mathbf{S}\mathbf{G}^{T}\|^{2}$                                                                                      $\mathbf{F}\geqslant 0$, $\mathbf{G}\geqslant 0$, $\mathbf{F}^{T}\mathbf{F}=\mathbf{I}_{g}$, $\mathbf{G}^{T}\mathbf{G}=\mathbf{I}_{s}$   $\mathbf{F}$ $\odot$ $\left[\dfrac{\mathbf{XGS}^{T}}{\mathbf{FSG}^{T}\mathbf{X}^{T}\mathbf{F}}\right]_{+}$, $\mathbf{G}$ $\odot$ $\left[\dfrac{\mathbf{X}^{T}\mathbf{FS}}{\mathbf{GS}^{T}\mathbf{F}^{T}\mathbf{XG}}\right]_{+}$, $\mathbf{S}$ $\odot$ $\left[\dfrac{\mathbf{F}^{T}\mathbf{XG}}{\mathbf{F}^{T}\mathbf{FSG}^{T}\mathbf{G}}\right]_{+}$
      @Labiod2011a           $\text{ODNMF}$                                                                                    $\|\mathbf{X}-\mathbf{F}\mathbf{F}^{T}\mathbf{X}\mathbf{G}\mathbf{G}^{T}\|^{2}$                                                                                $\mathbf{F}\geqslant 0$, $\mathbf{G}\geqslant 0$, $\mathbf{F}^{T}\mathbf{F}=\mathbf{I}_{g}$, $\mathbf{G}^{T}\mathbf{G}=\mathbf{I}_{s}$                                                                                                      $\mathbf{F} \odot \left[\dfrac{\mathbf{XGG}^{T}\mathbf{X}^{T}\mathbf{F}}{\mathbf{FF}^{T}\mathbf{XGG}^{T}\mathbf{X}^{T}\mathbf{F}}\right]_{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 \mathbf{G} \odot \left[\dfrac{\mathbf{X}^{T}\mathbf{FF}^{T}\mathbf{XG}}{\mathbf{GG}^{T}\mathbf{X}^{T}\mathbf{FF}^{T}\mathbf{XG}}\right]_{+}$
      @Labiod2011a           $\text{DNMF}$                                                  $\|\mathbf{X}-\mathbf{F}\mathbf{F}^{T}\mathbf{X}\mathbf{G}\mathbf{G}^{T}\|^{2}+\text{Tr}(\varLambda \mathbf{F}^{T})+\text{Tr}( \Gamma \mathbf{G}^{T})$                                                                                       $\mathbf{G}\geqslant 0$, $\mathbf{G}\geqslant 0$                                                                          $\mathbf{F} \odot \left[\dfrac{2\mathbf{X} \mathbf{G}\mathbf{G}^{T}\mathbf{X}^{T}\mathbf{F}}{\mathbf{FF}^{T}\mathbf{XG}\mathbf{G}^{T}\mathbf{G}\mathbf{G}^{T}\mathbf{X}^{T}\mathbf{F}+\mathbf{XG}\mathbf{G}^{T}\mathbf{G}\mathbf{G}^{T}\mathbf{X}^{T}\mathbf{FF}^{T}\mathbf{F}}\right]_{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                     \mathbf{G} \odot \left[\dfrac{2\mathbf{X}^{T}\mathbf{F}\mathbf{F}^{T}\mathbf{XG}}{\mathbf{GG}^{T}\mathbf{F}\mathbf{F}^{T}\mathbf{X} \mathbf{X}^{T}\mathbf{F}\mathbf{F}^{T}\mathbf{G}+\mathbf{F}\mathbf{F}^{T}\mathbf{X}\mathbf{X}^{T}\mathbf{F}\mathbf{F}^{T}\mathbf{GG}^{T}\mathbf{G}}\right]_{+}$
   @wang2017penalized        $\text{PNMTF}$        $\frac{1}{2} \|\mathbf{X}-\mathbf{F}\mathbf{S}\mathbf{G}^{T}\|^{2}+\frac{\tau}{2}\text{Tr}( \mathbf{F}\Psi\mathbf{F}^{T})+\frac{\eta}{2}\text{Tr}(  \mathbf{G}\Omega\mathbf{G}^{T})+\frac{\gamma}{2}\text{Tr}(\mathbf{S}^{T}\mathbf{S})$                                              $\mathbf{F}\geqslant 0$, $\mathbf{G}\geqslant 0$                                                                                                                               $\mathbf{F} \odot \left[\left( \dfrac{\mathbf{X}\mathbf{G}\mathbf{S}^{T}}{\mathbf{F}\mathbf{S}\mathbf{G}^{T}\mathbf{G}\mathbf{S}^{T}+\tau \mathbf{F}\Psi}\right)^{1/2}\right]_{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             \mathbf{G} \odot \left[\left( \dfrac{\mathbf{X}^{T}\mathbf{F}\mathbf{S}}{\mathbf{G}\mathbf{S}^{T}\mathbf{F}^{T}\mathbf{F}\mathbf{S}+\eta \mathbf{G}\Omega}\right)^{1/2} \right]_{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \mathbf{S} \odot \left[\left( \dfrac{\mathbf{F}^{T}\mathbf{X}\mathbf{G}}{\mathbf{F}^{T}\mathbf{F}\mathbf{S}\mathbf{G}^{T}\mathbf{G}+\gamma \mathbf{S}}\right)^{1/2} \right]_{+}$

  : Multiplicative updating algorithms

$\alpha$-divergence
-------------------

The generalized co-clustering framework with Bregman divergence as
objective function was introduced in [@banerjee2007generalized].

@Cichocki2008a,

@DelBuono2015,

@Yoo2010,

@Junior2020,

@Labiod2011a,

In the context of NMF, various error measures such as Amari's
a-divergences, Csiszár's f-divergences and Bregman divergences have been
considered @cichocki2006extended, @cichocki2006csiszar @cichocki2006new,
@dhillon2005generalized.

Multiplicative Update Rules algorithms were proposed in
@cichocki2006extended
[@cichocki2006csiszar; @cichocki2006new],@dhillon2005generalized, based
on  $\alpha$-divergence @amari2012differential, @ahn2004multiple as a
particular case of Csiszár's f-divergence @ali1966general, @Csiszar1974.

$\alpha$-divergence based NMF multiplicative algorithm was proposed in
[@Cichocki2008a] for image denoising and EEG classification.

@amari2012differential @Keribin2017 @lee1999learning

The $\alpha$-divergence is defined by $$\begin{aligned}
\label{D}
        D_{\alpha}(\mathbf{X}|| \mathbf{FSG}^{T})&=\dfrac{1}{\alpha (1-\alpha)} \sum\limits_{i,j} \left(  \alpha X_{ij} +(1-\alpha) [\mathbf{FSG}^{T}]_{ij} -X^{\alpha}_{ij} [\mathbf{FSG}^{T}]^{1-\alpha}_{ij}  \right)  \nonumber\\
        &=\sum\limits_{i,j}X_{ij} f\left( \dfrac{\sum\limits_{k,h}F_{ik}S_{kh}G_{hj}^{T}}{X_{ij}}\right),
    \end{aligned}$$ where $f(.)$ as a convex function for positive
values $\alpha$ in Figure $~\ref{td}$ is as follows $$\begin{aligned}
\label{convex}
        f(y) 
        &=
        \dfrac
        {1}
        {\alpha (1-\alpha)} 
        \left( 
        \alpha     +
         (1- \alpha)y   - 
         y^{1- \alpha} 
         \right),\\
         \dfrac
         {\partial   f(y)}
         {\partial y}
         &=
         \dfrac{1}{\alpha}
         \left( 
         1-y^{-\alpha}
         \right).   
    \end{aligned}$$ The $\alpha$-divergence includes Kullback-Leibler
(KL) divergence, Hellinger divergence, and $\chi^{2}$-divergence.\

![](alpha){#fig:alpha_f width="0.7\\linewidth"}

::: {#td}
   Divergence   $\alpha$                                         $D_{\alpha}(\mathbf{X}|| \mathbf{FSG}^{T})$
  ------------ ---------- --------------------------------------------------------------------------------------------------------------------------
       KL         $1$      $\sum\limits_{i,j} \left( X_{ij} \log( \dfrac{X_{ij}}{[\mathbf{FSG}^{T}]_{ij}})-X_{ij}+[\mathbf{FSG}^{T}]_{ij}  \right)$
   Hellinger     $0.5$                                                       $2\sum\limits_{i,j} 
                                                                                        \left(
                                                                   \sqrt{X_{ij}} -\sqrt{[\mathbf{FSG}^{T}]_{ij}}  
                                                                                     \right)^{2}$
   $\chi^{2}$      2                                                    $\frac{1}{2}\sum\limits_{i,j} 
                                                                                       \dfrac{
                                                                                          \left(
                                                                            X_{ij} -[\mathbf{FSG}^{T}]_{ij}  
                                                                                       \right)^{2}
                                                                             }{[\mathbf{FSG}^{T}]_{ij}}$

  :  Types of $\alpha$-divergence
:::

Our goals and outline paper
---------------------------

In this paper, we derive an $\alpha$-divergence based on NMTF
multiplicative algorithm in a different way as well as in a rigorous
manner, with proving the monotonic local convergence of the algorithm
using auxiliary functions. We also show that the same algorithm can be
derived using Karush--Kuhn--Tucker (KKT) conditions as well as the
projected gradient and detNMTF. Our contribution is primarily in the
derivation of a generic multiplicative algorithm, its monotonic
convergence, and alternative views when a-divergence is used as a
discrepancy measure in the context of NMF.

Non-negative matrix tri-factorization {#S:2}
=====================================

Given data matrix
$\mathbf{X}=($X$_{ij}) \in \mathbb{R}_{+}^{n \times m}$ and
approximation matrix
$\mathbf{FSG}^{T}=([\mathbf{FSG}^{T}]_{ij})\in \mathbb{R}_{+}^{n \times m}$.
Non-negative matrix tri-factorization (NMTF) amis to find three matrices
$\mathbf{F}=($F$_{ik})\in \mathbb{R}_{+}^{n \times g}$,
$\mathbf{S}=($S$_{kh})\in \mathbb{R}_{+}^{g \times s}$ and
$\mathbf{G}=($G$_{jh})\in \mathbb{R}_{+}^{m \times s}$ with non-negative
elements. The low-rank approximation of $\mathbf{X}$ by
$$\mathbf{X} \approx \mathbf{FSG}^{T},$$ where $n$, $m$, $g\leqslant n$
and $s\leqslant m$ are the number of rows, columns, row clusters and
column clusters, respectively. Also $\mathbf{F}$, $\mathbf{G}$ and
$\mathbf{S}$ play roles membership in row clustered, column clustered
and summarization matrix, respectively. This triple decomposition
provides a good framework for the simultaneous clustering of the rows
and columns of $\mathbf{X}$. We have fuzzy (overlapping, soft)
co-clustering if $\mathbf{F}^{T}\mathbf{F}=\mathbf{I}_{g}$,
$\mathbf{G}^{T}\mathbf{G}=\mathbf{I}_{s}$ and $\mathbf{F}$, $\mathbf{G}$
represented probability membership of rows and columns. When
$\mathbf{F}$ and $\mathbf{G}$ binarized so-called hard co-clustering. We
are attempting to solve the following optimization problem
$$\min\limits_{\mathbf{F},\mathbf{G},\mathbf{S} \geq 0} D_{\alpha}(\mathbf{X}\parallel \mathbf{FSG}^{T})$$
As in [@NIPS2000_f9d11525], we introduce an auxiliary function that is
used in convergence analysis as well as in algorithm derivation. The
minimization of the objective functions described above, should be done
with non-negativity constraints for both A and S. Multiplicative
updating is an efficient way in such a case, since it can easily
preserve non-negativity constraints at each iteration. Multiplicative
updating algorithms for NMF associated with these two objective
functions are given as follows:

$A(\mathbf{F},\mathbf{F}^{(t)})$ is called be an auxiliary function for
$D_{\alpha}(\mathbf{X}|| \mathbf{FSG}^{T})$ as the function of
$\mathbf{F}$, if the following two conditions are satisfied:
$$\begin{aligned}
\label{01}
        A(\mathbf{F},\mathbf{F})&=D_{\alpha}(\mathbf{X}||\mathbf{FSG}^{T}),\\
                A(\mathbf{F},\mathbf{F}^{(t)})&\geq D_{\alpha}(\mathbf{X}|| \mathbf{FSG}^{T}),\label{02}\end{aligned}$$
where $\mathbf{F}^{(t)}$ denotes the iteration of $t$ that resulted in
the updating of $\mathbf{F}$.

[\[lem0\]]{#lem0 label="lem0"} If
$A(\mathbf{F}^{(t+1)},\mathbf{F}^{(t)})$ is an auxiliray function for
$D_{\alpha}(\mathbf{X}||\mathbf{F}^{(t+1)}\mathbf{S}\mathbf{G}^{T})$,
then
$D_{\alpha}(\mathbf{X}||\mathbf{F}^{(t+1)}\mathbf{S}\mathbf{G}^{T})$ is
non-increasing function of $\mathbf{F}^{(t+1)}$ with respect to the
updating rule: $$\label{0+}
         \mathbf{F}^{(t+1)}=\arg\min\limits_{\mathbf{F}} A(\mathbf{F}, \mathbf{F}^{(t)}).$$

[\[A\_lem0\]](#A_lem0){reference-type="eqref" reference="A_lem0"}

[\[lem1\]]{#lem1 label="lem1"} If $\mathbf{S}$ and $\mathbf{G}$ are
fixed, the function $$\label{L1}
    A(\mathbf{F},\mathbf{F}^{(t)})= \sum\limits_{i,j,k,h} X_{ij} Q^{\mathbf{F}^{(t)}}_{ijkh} 
    f
    \left(  
    \dfrac{F_{ik}S_{kh}G_{hj}^{T}}{X_{ij}Q^{\mathbf{F}^{(t)}}_{ijkh}} 
    \right),$$ with $Q^{\mathbf{F}^{(t)}}_{ijkh}
    =
    \dfrac{F^{(t)}_{ik}S_{kh}G_{hj}^{T}}
    {\sum\limits_{k^{\prime},h^\prime} F^{(t)}_{i k^\prime}S_{k^\prime h^\prime }G_{h^\prime j}^{T}},$
is an auxiliary function for
$D_{\alpha}(\mathbf{X}|| \mathbf{FSG}^{T}).$

[\[A\_lem1\]](#A_lem1){reference-type="eqref" reference="A_lem1"}

If $\mathbf{F}$ and $\mathbf{S}$ are fixed, the function $$\label{lem3}
        A
        (\mathbf{G},\mathbf{G}^{(t)})=
        \sum\limits_{i,j,k,h}
         X_{ij} Q^{\mathbf{G}^{(t)}}_{ijkh} 
         f
         \left(  
         \dfrac{F_{ik}S_{kh}G_{hj}^{T}}{X_{ij}Q^{\mathbf{G}^{(t)}}_{ijkh}} 
         \right),$$ with
$Q^{\mathbf{G}^{(t)}}_{ijkh}=\dfrac{F_{ik}S_{kh} G^{(t)T}_{hj}}{\sum\limits_{k^{\prime},h^\prime} F_{i k^\prime}S_{k^\prime h^\prime } G^{(t)T}_{h^\prime j}}$
is an auxiliary function for
$D_{\alpha}(\mathbf{X}|| \mathbf{FSG}^{T}).$

Can be proved in the same way Lemma $\eqref{lem1}$.

If $\mathbf{F}$ and $\mathbf{G}$ are fixed, the function $$\label{lem2}
    A(\mathbf{S},\mathbf{S^{(t)}})=
    \sum\limits_{i,j,k,h} 
    X_{ij} Q^{\mathbf{S}^{(t)}}_{ijkh} 
    f
    \left(  
    \dfrac{F_{ik}S_{kh}G_{hj}^{T}}{X_{ij}Q^{\mathbf{S}^{(t)}}_{ijkh}} 
    \right),$$ with $Q^{\mathbf{S}^{(t)}}_{ijkh}=
    \dfrac
    {F_{ik}S^{(t)}_{kh}G_{hj}^{T}}
    {\sum\limits_{k^{\prime},h^\prime} F_{i k^\prime} S^{(t)}_{k^\prime h^\prime } G_{h^\prime j}^{T}}$
is an auxiliary function for
$D_{\alpha}(\mathbf{X}|| \mathbf{FSG}^{T}).$

Can be proved in the same way Lemma $\eqref{lem1}$.

[\[T1\]]{#T1 label="T1"} The divergence
$D_{\alpha}(\mathbf{X}|| \mathbf{FSG}^{T})$ is non-increasing based on
multiplicative update rules $$\begin{aligned}
\label{t0}
    &\mathbf{F}^{(t+1)}\leftarrow \mathbf{F}^{(t)}\odot \left[ \left(\dfrac{( \mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{GS}^{T}}{\mathbf{E}_{nm}\mathbf{GS}^{T}}\right)^{\dfrac{1}{\alpha}}\right] _{+},\\\label{t1}
    &\mathbf{G}^{(t+1)}\leftarrow \mathbf{G}^{(t)}\odot \left[ \left(\dfrac{((\mathbf{X}\oslash \mathbf{FSG}^{T})^{T}) ^{\alpha} \mathbf{FS}}{\mathbf{E}_{nm}^{T}\mathbf{FS}}\right)^{\dfrac{1}{\alpha}}\right]_{+} , \\\label{t2}
    &\mathbf{S}^{(t+1)}\leftarrow \mathbf{S}^{(t)}\odot \left[\left(\dfrac{\mathbf{F}^{T} (\mathbf{X}\oslash \mathbf{FSG}^{T}) ^{\alpha} \mathbf{G}}{\mathbf{F}^{T}\mathbf{E}_{nm}\mathbf{G}}\right)^{\dfrac{1}{\alpha}}\right]_{+} ,
    \end{aligned}$$ where $[P]_{+}=max(P,0)$, $\odot$ and $\oslash$ are
element-wise multiplication and element-wise division, respectively.
$\mathbf{E}_{nm}$ is matrix of ones with size $n \times m$. To avoid
divide-by-zero problems, a small positive $\varepsilon$ number is added
to the denominator of each iteration $t$.

[\[A\_T1\]](#A_T1){reference-type="eqref" reference="A_T1"}

Orthogonal Nonnegative Matrix Tri-Factorization (ONMTF)
=======================================================

@wang2017penalized proposed new penalty term based on penalized matrices
similar to $$\bm{\Psi}_{g} = \left(\begin{array}{cccc}
    0 & 1 & \ldots & 1 \\
    1 & 0 & \ldots & 1 \\
    \vdots & \vdots & & \vdots \\
    1 & 1 & \ldots & 0
\end{array}\right)_{g \times g},\quad
\bm{\Psi}_{s} = \left(\begin{array}{cccc}
    0 & 1 & \ldots & 1 \\
    1 & 0 & \ldots & 1 \\
    \vdots & \vdots & & \vdots \\
    1 & 1 & \ldots & 0
\end{array}\right)_{s \times s}$$

that observed
$$\text{Tr}(\mathbf{F}\bm{\Psi}_{g}\mathbf{F}^{T})=\text{Tr}(\mathbf{F}^{T}\mathbf{F}\bm{\Psi}_{g})= \sum\limits_{i\neq i^\prime}F^{T}_{ki^\prime}F_{ik},$$
$$\text{Tr}(\mathbf{G}\bm{\Psi}_{s}\mathbf{G}^{T})=\text{Tr}(\mathbf{G}^{T}\mathbf{G}\bm{\Psi}_{s})= \sum\limits_{j\neq j^\prime}G^{T}_{hj^\prime}G_{jh},$$
where $\text{Tr}(.)$ is trace of a matrix. When $\mathbf{F}$ and
$\mathbf{G}$ are orthogonal, these terms are minimised to zero. We use
this technique for $\alpha$-divergence and writing object function as
$$\label{xi_OPNMTF}
    \xi_{\text{ONMTF}_{\alpha}}=D_{\alpha}(\mathbf{X}|| \mathbf{FSG}^{T})
    +
    \delta \text{Tr}(\mathbf{F}\bm{\Psi}_{g}\mathbf{F}^{T})
    +
    \beta \text{Tr}(\mathbf{G}\bm{\Psi}_{s}\mathbf{G}^{T})$$ where
$\bm{\Psi}_{g} \in \mathbb{R}_{\{0,1\}}^{g \times g}$ and
$\bm{\Psi}_{s} \in \mathbb{R}_{\{0,1\}}^{s \times s}$ are two penalized
matrices to guarantee the orthogonality of $\mathbf{F}$ and
$\mathbf{G}$, respectively. Also $\delta$ and $\beta$ are used to
balance the two terms. By taking the derivatives of
$\xi_{\text{ONMTF}_{\alpha}}$ with respect to $\mathbf{F}$,
$\mathbf{S}$, and $\mathbf{G}$ are as follows:

$$\begin{aligned}
    \dfrac{\partial \xi_{\text{ONMTF}_{\alpha}}}{\partial\mathbf{F}}&=\dfrac{1}{\alpha} \left(\mathbf{E}_{nm}\mathbf{GS}^{T}\right) - \dfrac{1}{\alpha} \left( (\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{GS}^{T} \right)+2 \delta \mathbf{F}\bm{\Psi}_{g},\\
        \dfrac{\partial \xi_{\text{ONMTF}_{\alpha}} }{\partial \mathbf{G}}&=\dfrac{1}{\alpha} \left(\mathbf{E}_{nm}^{T}\mathbf{FS}\right) -  \dfrac{1}{\alpha} \left( \left((\mathbf{X}\oslash \mathbf{FSG}^{T})^{T}\right)^{\alpha}  \mathbf{FS} \right)+2\beta \mathbf{G}\bm{\Psi}_{s}, \\
    \dfrac{\partial \xi_{\text{ONMTF}_{\alpha}} }{\partial \mathbf{S}}&=\mathbf{F}^{T}\mathbf{E}_{nm}\mathbf{G}-\mathbf{F}^{T} (\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{G}.\end{aligned}$$
Using the KKT conditions and

$$\label{11--}
        \mathbf{F}\leftarrow \mathbf{F}\odot \left[ \left(\dfrac{(\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{GS}^{T}}{\mathbf{E}_{nm}\mathbf{G}\mathbf{S}^{T}+2 \delta \alpha \mathbf{F}\bm{\Psi}_{g}}\right)^{\frac{1}{\alpha}}\right]_{+} ,$$

$$\label{11---}
        \mathbf{G}\leftarrow \mathbf{G}\odot \left[\left( \dfrac{((\mathbf{X}\oslash \mathbf{FSG}^{T})^{T})^{\alpha} \mathbf{FS}}{\mathbf{E}_{nm}^{T}\mathbf{FS}+2\beta \alpha \mathbf{G}\bm{\Psi}_{s}}\right)^{\frac{1}{\alpha}}\right]_{+} ,$$

$$\label{11----}
        \mathbf{S}\leftarrow \mathbf{S}\odot \left[\left( \dfrac{\mathbf{F}^{T} (\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{G}}{\mathbf{F}^{T}\mathbf{E}_{nm}\mathbf{G}}\right)^{\frac{1}{\alpha}}\right] _{+},$$

Initialize $\mathbf{F}^{(0)}$ and $\mathbf{G}^{(0)}$ using Double
k-means $\cite{Labiod2011a}$ on $\mathbf{X}.$

Compute
$\mathbf{S}^{(0)}=\left(\mathbf{G}^{T} \mathbf{G}\right)^{-1} \mathbf{G}^{T} \mathbf{X} \mathbf{F}\left(\mathbf{F}^{T} \mathbf{F}\right)^{-1}.$

While not convergent and $1 \leqslant t  \leqslant n$ do:

Fixing $\mathbf{G}$ and $\mathbf{S}$ and Update $\mathbf{F}^{(t)}$ from
$\eqref{11--},$

Fixing $\mathbf{F}$ and $\mathbf{G}$ and Update $\mathbf{S}^{(t)}$ from
$\eqref{11---},$

Fixing $\mathbf{F}$ and $\mathbf{S}$ and Update $\mathbf{G}^{(t)}$ from
$\eqref{11----},$

Set $t= t+1$.

Normalize $\mathbf{F}$, $\mathbf{S}$ and $\mathbf{G}$ with probabilistic
interpretation [@Yoo2010]:

$\qquad$ $\mathbf{F}$ $\longleftarrow$ $\mathbf{F} \mathbf{D^{-1}_{F}},$

$\qquad$ $\mathbf{S}$ $\longleftarrow$
$\mathbf{D_{F}}\mathbf{S} \mathbf{D^{-1}_{G}},$

$\qquad$ $\mathbf{G}$ $\longleftarrow$ $\mathbf{G} \mathbf{D_{G}},$

where $\mathbf{D_{F}}=diag(\mathbf{1}^{T}\mathbf{F})$ and
$\mathbf{D_{G}}=diag(\mathbf{1}^{T}\mathbf{G})$.

Assign row $i$ and column $j$ to row and column cluster $k^{*}$ and
$h^{*}$ if

$\qquad k^{*}=\arg\max\limits_{k} F_{ik}$,
$\qquad h^{*}= \arg\max\limits_{h} G_{hj}.$

Orthogonal Parametric Nonnegative Matrix Tri-Factorization (OPNMTF)
===================================================================

This section introduces two penalty terms to replace with the two
orthogonality constraints $\mathbf{F}^{T}\mathbf{F}=\mathbf{I}_{g}$,
$\mathbf{G}^{T}\mathbf{G}=\mathbf{I}_{s}$.

Orthogonality $\mathbf{F}$ and $\mathbf{G}$ can be achieved without
considering restrictions. This goal is applied in OPNMTF. The object
function in OPNMTF has a two-term penalty based on $\alpha$-divergence.
The first term for the row space with control parameter $\lambda$. The
second term for the column space with control parameter $\mu$. We
introduce our objective function for orthogonal $\mathbf{F}$ and
$\mathbf{G}$ as follows: $$\label{12-}
\xi_{\text{OPNMTF}_{\alpha}}=D_{\alpha}(\mathbf{X}|| \mathbf{FSG}^{T})+\lambda D_{\alpha}(\mathbf{I}_{g} \parallel \mathbf{F}^{T}\mathbf{F})+\mu D_{\alpha}(\mathbf{I}_{s}\parallel \mathbf{G}^{T}\mathbf{G)}$$

On the other hand, a multiplicative method developed in
@NIPS2000_f9d11525 provides a simple algorithm for (10).We give a
slightly different approach from @NIPS2000_f9d11525 to derive the same
multiplicative algorithm. Suppose that the gradient of an error function
has a decomposition that is of the form

[\[OPNMTF\_1\]]{#OPNMTF_1 label="OPNMTF_1"} If $\mathbf{G}$ and
$\mathbf{S}$ are fixed, the function $$\label{lem6}
        A^{*}
        (\mathbf{F},\mathbf{F}^{(t)})=
        \sum\limits_{i,j,k,h}
        X_{ij} Q^{\mathbf{F}^{(t)}}_{ijkh} 
        f
        \left(  
        \dfrac{F_{ik}S_{kh}G_{hj}^{T}}{X_{ij}Q^{\mathbf{F}^{(t)}}_{ijkh}} 
        \right)
        +
        \lambda \sum\limits_{i,k}
         Q^{\mathbf{F}^{(t)}}_{ik} 
        f
        \left(  
        \dfrac{F^{T}_{ki}F_{ik}}{Q^{\mathbf{F}^{(t)}}_{ik}} 
        \right)$$ where
$Q^{\mathbf{F}^{(t)}}_{ijkh}=\dfrac{F^{(t)}_{ik}S_{kh} G^{T}_{hj}}{\sum\limits_{k^{\prime},h^\prime} F^{(t)}_{i k^\prime}S_{k^\prime h^\prime } G^{T}_{h^\prime j}}$
and
$Q^{\mathbf{F}^{(t)}}_{ik}=\dfrac{F^{(t)T}_{ki} F^{(t)}_{ik}}{\sum\limits_{i^{\prime}} F^{(t)T}_{ki^{\prime}}F^{(t)}_{i^{\prime}k}}$
is an auxiliary function for $\xi_{\text{OPNMTF}_{\alpha}}.$

[\[A\_OPNMTF\_1\]](#A_OPNMTF_1){reference-type="eqref"
reference="A_OPNMTF_1"}

[\[OPNMTF\_2\]]{#OPNMTF_2 label="OPNMTF_2"} If $\mathbf{F}$ and
$\mathbf{S}$ are fixed, the function $$\label{lem7}
        A^{*}
        (\mathbf{G},\mathbf{G}^{(t)})=
        \sum\limits_{i,j,k,h}
        X_{ij} Q^{\mathbf{G}^{(t)}}_{ijkh} 
        f
        \left(  
        \dfrac{F_{ik}S_{kh}G_{hj}^{T}}{X_{ij}Q^{\mathbf{G}^{(t)}}_{ijkh}} 
        \right)
        +
        \mu \sum\limits_{j,h}
        Q^{\mathbf{G}^{(t)}}_{jh} 
        f
        \left(  
        \dfrac{G^{T}_{hj}G_{jh}}{Q^{\mathbf{G}^{(t)}}_{jh}} 
        \right)$$ where
$Q^{\mathbf{G}^{(t)}}_{ijkh}=\dfrac{F_{ik}S_{kh} G^{(t)T}_{hj}}{\sum\limits_{k^{\prime},h^\prime} F_{i k^\prime}S_{k^\prime h^\prime } G^{(t)T}_{h^\prime j}}$
and
$Q^{\mathbf{G}^{(t)}}_{jh}=\dfrac{G^{(t)T}_{hj} G^{(t)}_{jh}}{\sum\limits_{j^{\prime}} G^{(t)T}_{hj^{\prime}}G^{(t)}_{j^{\prime}h}}$
is an auxiliary function for $\xi_{\text{OPNMTF}_{\alpha}}.$

Can be proved in the same way Lemma $\eqref{OPNMTF_1}$.

[\[T3\]]{#T3 label="T3"} The multiplicative update rule for
[\[12-\]](#12-){reference-type="eqref" reference="12-"} is given as
follows:

$$\label{12--}
        \mathbf{F}\leftarrow \mathbf{F}\odot \left[ \left(\dfrac{(\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{GS}^{T}+2\lambda \mathbf{F} (\mathbf{I}_{g}\oslash \mathbf{F}^{T}\mathbf{F})^{\alpha}}{\mathbf{E}_{nm}\mathbf{G}\mathbf{S}^{T}+2\lambda \mathbf{F}\mathbf{E}_{gg}}\right)^{\dfrac{1}{\alpha}}
        \right]_{+} ,$$

$$\label{12---}
        \mathbf{G}\leftarrow \mathbf{G}\odot \left[ \left(\dfrac{((\mathbf{X}\oslash \mathbf{FSG}^{T})^{T})^{\alpha} \mathbf{FS}+2\mu \mathbf{G}(\mathbf{I}_{s}\oslash \mathbf{G}^{T}\mathbf{G})^{\alpha} }{\mathbf{E}_{nm}^{T}\mathbf{FS}+2\mu \mathbf{G}\mathbf{E}_{ss}}\right)^{\dfrac{1}{\alpha}}
        \right]_{+} ,$$

$$\label{12----}
        \mathbf{S}\leftarrow \mathbf{S}\odot \left[ \left(\dfrac{\mathbf{F}^{T} (\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{G}}{\mathbf{F}^{T}\mathbf{E}_{nm}\mathbf{G}}\right)^{\dfrac{1}{\alpha}}
        \right] _{+},$$

where $[P]_{+}=max(P,0)$ and $\mathbf{E}_{nm}$, $\mathbf{E}_{gg}$ and
$\mathbf{E}_{ss}$ are matrices of ones with size $n \times m$,
$g\times g$ and $s \times s$, respectively. Also $\mathbf{I}_{g}$ and
$\mathbf{I}_{s}$ are identity matrix with size $g \times g$ and
$s \times s$. The parameters $\lambda$ and $\mu$ control the
orthogonality of the vectors in $\mathbf{F}$ and $\mathbf{G}$.

[\[A\_T3\]](#A_T3){reference-type="eqref" reference="A_T3"}

Initialize $\mathbf{F}^{(0)}$ and $\mathbf{G}^{(0)}$ using Double
k-means $\cite{Labiod2011a}$ on $\mathbf{X}.$

Compute
$\mathbf{S}^{(0)}=\left(\mathbf{G}^{T} \mathbf{G}\right)^{-1} \mathbf{G}^{T} \mathbf{X} \mathbf{F}\left(\mathbf{F}^{T} \mathbf{F}\right)^{-1}.$

While not convergent and $1 \leqslant t  \leqslant n$ do:

Update $\mathbf{F}^{(t)}$ from $\eqref{12--},$

Update $\mathbf{S}^{(t)}$ from $\eqref{12---},$

Update $\mathbf{G}^{(t)}$ from $\eqref{12----},$

Set $t= t+1$.

Normalize $\mathbf{F}$, $\mathbf{S}$ and $\mathbf{G}$ with probabilistic
interpretation [@Yoo2010]:

$\qquad$ $\mathbf{F}$ $\longleftarrow$ $\mathbf{F} \mathbf{D^{-1}_{F}},$

$\qquad$ $\mathbf{S}$ $\longleftarrow$
$\mathbf{D_{F}}\mathbf{S} \mathbf{D^{-1}_{G}},$

$\qquad$ $\mathbf{G}$ $\longleftarrow$ $\mathbf{G} \mathbf{D_{G}},$

where $\mathbf{D_{F}}=diag(\mathbf{1}^{T}\mathbf{F})$ and
$\mathbf{D_{G}}=diag(\mathbf{1}^{T}\mathbf{G})$.

Assign row $i$ and column $j$ to row and column cluster $k^{*}$ and
$h^{*}$ if

$\qquad k^{*}=\arg\max\limits_{k} F_{ik}$,
$\qquad h^{*}= \arg\max\limits_{h} G_{hj}.$

Alternatives
============

Projected gradient
------------------

A comparable NMF technique has been described in somewhat different
versions with over-relaxation and regularisation terms 2. We
provide  for this class of NMTF algorithms with our rigorous
mathematical derivation and convergence analysis.  In this section, we
demonstrate how to derive updating rules (22) and (23) using the
projected gradient (Cichocki et al., 2006a). The partial derivatives of
$\eqref{12-}$ in $\mathbf{F}$, $\mathbf{S}$ and $\mathbf{G}$ are as
follows Taking derivatives of $\xi_{\text{OPNMTF}_{\alpha}}$ with
respect to $\mathbf{F}$, $\mathbf{G}$ and $\mathbf{S}$, we have
$$\begin{aligned}
    \dfrac{\partial \xi_{\text{OPNMTF}_{\alpha}} }{\partial\mathbf{F}}&=\dfrac{1}{\alpha} \left(\mathbf{E}_{nm}\mathbf{GS}^{T}+2\lambda \mathbf{F} \mathbf{E}_{gg}\right) - \dfrac{1}{\alpha} \left( (\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{GS}^{T}+2\lambda \mathbf{F}(\mathbf{I}_{g}\oslash \mathbf{F}^{T}\mathbf{F} )^{\alpha} \right) \label{p1} \\ 
    \dfrac{\partial  \xi_{\text{OPNMTF}_{\alpha}} }{\partial \mathbf{G}}&=\dfrac{1}{\alpha} \left(\mathbf{E}_{nm}^{T}\mathbf{FS}+2\mu \mathbf{G} \mathbf{E}_{ss}\right) -  \dfrac{1}{\alpha} \left( \left((\mathbf{X}\oslash \mathbf{FSG}^{T})^{T}\right)^{\alpha}  \mathbf{FS}+2\mu \mathbf{G} (\mathbf{I}_{s}\oslash \mathbf{G}^{T}\mathbf{G} )^{\alpha} \right)  \label{p2} \\
    \dfrac{\partial  \xi_{\text{OPNMTF}_{\alpha}} }{\partial \mathbf{S}}&=\mathbf{F}^{T}\mathbf{E}_{nm}\mathbf{G}-\mathbf{F}^{T} (\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{G}\end{aligned}$$
The projected gradient method updates transformed parameters using the
gradient information, which is of the form

$$\begin{aligned}
    &\psi\left(\mathbf{F}\right) \leftarrow \psi\left(\mathbf{F}\right)-\bm{\eta}   \dfrac{\partial \xi_{\text{OPNMTF}_{\alpha}} }{\partial\mathbf{F}}, \\
    &\psi\left(\mathbf{G}\right) \leftarrow \psi\left(\mathbf{G}\right)-\bm{\eta}   \dfrac{\partial \xi_{\text{OPNMTF}_{\alpha}} }{\partial\mathbf{G}},\end{aligned}$$
where $\bm{\eta}=(\eta_{i j}) \in \mathbb{R}$$\psi(.)$ is a suitably
chosen function. Note that the exponentiated gradient emerges for
$\psi(\theta)=\log(\theta).$ Therefore we have $$\begin{aligned}
    &\mathbf{F} \leftarrow \psi^{-1}\left(\psi\left(\mathbf{F}\right)-\eta_{i j}    \dfrac{\partial  \xi_{\text{OPNMTF}_{\alpha}} }{\partial \mathbf{F}}\right) , \\
    &\mathbf{G} \leftarrow \psi^{-1}\left(\psi\left(\mathbf{G}\right)-\eta_{i j}    \dfrac{\partial  \xi_{\text{OPNMTF}_{\alpha}} }{\partial \mathbf{G}}\right) ,\end{aligned}$$
choosing $\psi(\theta)=\theta^{\alpha}$ and incorporating with
$\eqref{p1}$ and $\eqref{p2}$, leads to $\eqref{12--},$ $\eqref{12---},$
and $\eqref{12----}$.

KKT conditions
--------------

We show that the algorithms $\eqref{12--},$ $\eqref{12---}$ and
$\eqref{12--},$ can be also derived using the KKT conditions. The
minimization of $\xi_{\text{OPNMTF}_{\alpha}}$ in $\eqref{12-}$ with
non-negativity constraints, $F_{ik}\geqslant 0$, $S_{kh}\geqslant 0$ and
$G_{jh}\geqslant 0$, can be formulated as a constrained minimization
problem with inequality constraints. Denote by
$\varLambda_{ik}\geqslant 0$, $\Delta_{kh} \geqslant 0$ and
$\Omega_{jh}\geqslant 0$ Lagrangian multipliers associated with
constraints,$F_{ik}\geqslant 0$, $S_{kh}\geqslant 0$ and
$G_{jh}\geqslant 0$, respectively. The Karush--Kuhn--Tucker (KKT)
conditions define the first-order constraint that must be met when
solving non-linear optimization problems with inequality constraints.
Optimality conditions in KKT is $$\begin{aligned}
    \dfrac{\partial \xi_{\text{OPNMTF}_{\alpha}} }{\partial\mathbf{F}}\odot \mathbf{F}&=\mathbf{0},\\
    \dfrac{\partial \xi_{\text{OPNMTF}_{\alpha}} }{\partial\mathbf{G}}\odot \mathbf{G}&=\mathbf{0},\\
    \dfrac{\partial \xi_{\text{OPNMTF}_{\alpha}} }{\partial\mathbf{S}}\odot \mathbf{S}&=\mathbf{0},\end{aligned}$$
and complementary slackness conditions, implying that $$\begin{aligned}
    \dfrac{\partial \xi_{\text{OPNMTF}_{\alpha}} }{\partial\mathbf{F}}\odot \mathbf{F}^{\alpha}&=\mathbf{0},\\
    \dfrac{\partial \xi_{\text{OPNMTF}_{\alpha}} }{\partial\mathbf{G}}\odot \mathbf{G}^{\alpha}&=\mathbf{0},\\
    \dfrac{\partial \xi_{\text{OPNMTF}_{\alpha}} }{\partial\mathbf{S}}\odot \mathbf{S}^{\alpha}&=\mathbf{0},\end{aligned}$$
$$\dfrac{1}{\alpha} \left(\mathbf{E}_{nm}\mathbf{GS}^{T}+2\lambda \mathbf{F} \mathbf{E}_{gg}\right) - \dfrac{1}{\alpha} \left( (\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{GS}^{T}+2\lambda \mathbf{F}(\mathbf{I}_{g}\oslash \mathbf{F}^{T}\mathbf{F} )^{\alpha} \right) \mathbf{F}^{\alpha}=\mathbf{0}$$
$$\dfrac{1}{\alpha} \left(\mathbf{E}_{nm}^{T}\mathbf{FS}+2\mu \mathbf{G} \mathbf{E}_{ss}\right) -  \dfrac{1}{\alpha} \left( \left((\mathbf{X}\oslash \mathbf{FSG}^{T})^{T}\right)^{\alpha}  \mathbf{FS}+2\mu \mathbf{G} (\mathbf{I}_{s}\oslash \mathbf{G}^{T}\mathbf{G} )^{\alpha} \right) \mathbf{G}^{\alpha}=\mathbf{0}$$
$$\left( \mathbf{F}^{T}\mathbf{E}_{nm}\mathbf{G}-\mathbf{F}^{T} (\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{G}\right) \mathbf{S}^{\alpha}=\mathbf{0}$$

Therefore, the multiplicative update rule for
[\[12-\]](#12-){reference-type="eqref" reference="12-"} is given as
follows equations [\[12\--\]](#12--){reference-type="eqref"
reference="12--"}, [\[12\-\--\]](#12---){reference-type="eqref"
reference="12---"} and [\[12\-\-\--\]](#12----){reference-type="eqref"
reference="12----"}.

detNMTF
-------

In [@schachtner2011towards], [@schachtner2009minimum] suggested an extra
minimal determinant constraint and developed the detNMF algorithm
[@naik2016non].

In [@schachtner2011towards], [@schachtner2009minimum] we introduced a
new NMF algorithm named detNMF which directly imple- ments a determinant
criterion. As an objective function, we chose the regularized least
squares cost function

$$D_{\text{det}}(\mathbf{X}, \mathbf{W H})=\sum_{n=1}^{m} \sum_{m=1}^{n}\left(X_{n m}-[\mathbf{W H}]_{n m}\right)^{2}+\lambda \text{det}\left(\mathbf{H} \mathbf{H}^{T}\right)$$

where $\lambda$\> 0 is a, usually small, positive regularization
parameter. Optimization of the constrained cost function (1.67) can be
achieved via gradient descent techniques according to

Numerical Experiments
=====================

Synthetic Data
--------------

We propose to use a Latent Block Model (LBM) [@Keribin2017a] for
generate non-negative data matrix. This generative model considering
$\bm{\mathcal{F}}$ and $\bm{\mathcal{G}}$ the sets of possible labels
$\mathbf{F}$ (rows) and $\mathbf{G}$ (columns), the values $x_{ij}$ for
each co-cluster $\mathbf{X_{kh}}$ are distributed according to the
following probability density function:
$$f(\mathbf{X};\bm{\theta})=\sum\limits_{(\mathbf{F},\mathbf{G})\in \bm{\mathcal{F}} \times \bm{\mathcal{G}}} \prod\limits_{i,k} \pi_{k}^{F_{ik}} \prod\limits_{j,h} \rho_{h}^{G_{jh}} \prod\limits_{i,j,k,h} \varphi(X_{ij};\delta_{kh}),$$
where $\bm{\theta}=(\bm{\pi},\bm{\rho},\bm{\delta})$,
$\bm{\pi}=(\pi_{1},...,\pi_{g})$ and $\bm{\rho}=(\rho_1,...,\rho_{s})$,
also $\bm{\delta}=(\delta_{11},...,\delta_{gs})$ are all parameters
co-clusters that defined according to probability density function.

-   **Bernoulli LBM**: the values $x_{ij}$ are distributed according to
    a Bernoulli distribution $\mathcal{B}$($\gamma_{kh}$) with
    $\gamma_{kh} \in [0,1]$, also $$\label{b}
        \varphi(x_{ij};\gamma_{kh})= \gamma_{kh}^{x_{ij}} (1-\gamma_{kh})^{1-x_{ij}}.$$

-   **Poisson LBM**: the values $x_{ij}$ are distributed according to a
    Poisson distribution $\mathcal{P}(\omega_{i} \nu_{j} \alpha_{kh})$
    where $\omega_{i}$ and $\nu_{j}$ are the effects of row $i$ and
    column $j$, respectively. Also, $\alpha_{kh}$ is the effect of
    co-cluster $kh$ for $x_{ij}$ that belong to the $k$-th row cluster
    and $h$-th column cluster and $$\label{p}
        \varphi(x_{ij};\alpha_{kh})= \dfrac{e^{-\omega_{i} \nu_{j} \alpha_{kh}} (\omega_{i} \nu_{j} \alpha_{kh})^{x_{ij}}}{x_{ij}!}.$$

-   **Truncated Gussian LBM**: the values $x_{ij}$ beloge to co-cluster
    $kh$ are distributed according to a Truncated Gussian distribution
    with $a \leq x_{ij} \leq b$, mean $\mu_{kh}$ and variance
    $\sigma_{kh}$, also $$\label{n}
    \varphi(x_{ij} ; \mu_{kh}, \sigma_{kh}, a, b)=\frac{1}{\sigma_{kh}} \frac{\phi\left(\frac{x_{ij}-\mu_{kh}}{\sigma_{kh}}\right)}{\Phi\left(\frac{b-\mu_{kh}}{\sigma_{kh}}\right)-\Phi\left(\frac{a-\mu_{kh}}{\sigma_{kh}}\right)},$$
    where $\phi$ and $\Phi$ are standard gussian distribution and
    cumulative distribution function, respectively.

Simulate $\mathbf{F}$ according to a Multinomial distribution with
parameters $\bm{\pi}=(1,\pi_1,...,\pi_g)$.

Simulate $\mathbf{G}$ according to a Multinomial distribution with
parameters $\bm{\rho}=(1,\rho_1,...,\rho_s)$.

Simulate each co-cluster $\mathbf{X_{kh}}$ according to Bernoulli
density $\eqref{b}$, Poisson density $\eqref{p}$ and Truncated Normal
$\eqref{n}$.

Permutation all $\mathbf{X_{kh}}$.

Setting
-------

In setting A the distributions $\bm{\pi}$ and $\bm{\rho}$ are uniform,
but in setting B they are not uniform. Table $\eqref{setting}$ shows
further parameters that we need to generate a non-negative data matrix.

Performance evaluation(Evaluation measures) 
-------------------------------------------

To measure the clustering performance of the proposed algorithms we use
the commonly adopted external metrics, the accuracy, the Normalize
Mutual Information [@strehl2002cluster] and the Adjusted Rand Index
[@hubert1985comparing]. We focus only on the quality of row clustering.
Clustering accuracy (noted Acc) is one of the most widely used
evaluation criteria and is defined as:
$$\text{Acc}=\dfrac{1}{n} \max\left[\sum\limits_{\mathcal{C}_{k}, \mathcal{L}_{h}} T(\mathcal{C}_{k}, \mathcal{L}_{h})  \right],$$
where $\mathcal{C}_{k}$ is the $k$-th cluster in the final results, and
$\mathcal{L}_{h}$ is the true $h$-th class.
$T(\mathcal{C}_{k}, \mathcal{L}_{h})$ is the proportion of objects that
were correctly recovered by the clustering algorithm, i.e.,
$T(\mathcal{C}_{k}, \mathcal{L}_{h}) = \mathcal{C}_{k} \cap \mathcal{L}_{h}$.
Accuracy computes the maximum sum of
$T(\mathcal{C}_{k}, \mathcal{L}_{h})$ for all pairs of clusters and
classes.

The second measure used is the Normalized Mutual Information (NMI). It
is calculated as follows: $$\text{NMI} = \dfrac{
        \sum_{k,h} 
        \frac{n_{kh}}{n} 
        \log \frac{n_{kh}}{n_{k}n_{h}}
    }
    {
        \sqrt{
        \left( \sum_{k} \frac{n_{k}}{n} \log \frac{n_{k}}{n}\right)
        \left( \sum_{h} \frac{n_{h}}{n} \log \frac{n_{h}}{n}\right)
    }   
},$$ where$n_k$ denotes the number of data contained in the cluster
$\mathcal{C}_{k}(1 \leqslant k \leqslant g)$, $n_h$ is the number of
data belonging to the class
$\mathcal{L}_{h} (1 \leqslant h \leqslant s)$,and $n_{kh}$ denotes the
number of data that are in the intersection between the cluster
$\mathcal{C}_{k}$ and the class $\mathcal{L}_{h}$ The last evaluation
criterion Adjusted Rand Index (noted ARI) measures the similarity
between two clustering partitions. From a mathematical standpoint, the
Rand index is related to the accuracy. The adjusted form of the Rand
Index is defined as: $$\text{ARI} = \dfrac{
\sum_{k,h} \tbinom{n_{kh}}{2}   
-
\left[ \sum_{k}  \tbinom{n_{k}}{2}   \sum_{h}  \tbinom{n_{h}}{2}  \right]
/\tbinom{n}{2}
}
{
\frac{1}{2} \left[ \sum_{k}  \tbinom{n_{k}}{2}  + \sum_{h}  \tbinom{n_{h}}{2}  \right]
-
\left[ \sum_{k}  \tbinom{n_{k}}{2}   \sum_{h}  \tbinom{n_{h}}{2}  \right]/\tbinom{n}{2}
},$$ The$\text{ ARI}$ is related to the clustering accuracy and measures
the degree of agreement between an estimated clustering and a reference
clustering. Both $\text{NMI}$ and $\text{ARI}$ are equal to 1 if the
resulting clustering is identical to the true one.

![(Left) Data matrix **Truncated Gussian LBM** on setting B, (Middle)
Clustering data matrix with OPNMTF algorithm on setting B, (Right)
Co-clustering reorganization data matrix according to the output of the
OPNMTF algorithm on setting
B.](../../Application/NMTFcoclust/Result/OSNMTF_GA){#fig:osnmtfga
width="0.7\\linewidth"}

![(Left) Data matrix Binary on setting B, (Middle) Clustering data
matrix with OPNMTF algorithm on setting B, (Right) Co-clustering
reorganization data matrix according to the output of the OPNMTF
algorithm on setting
B.](../../Application/NMTFcoclust/Result/OSNMTF_BA){#fig:osnmtfba
width="0.7\\linewidth"}

![(Left) Data matrix Poisson on setting B, (Middle) Clustering data
matrix with OPNMTF algorithm on setting B, (Right) Co-clustering
reorganization data matrix according to the output of the OPNMTF
algorithm on setting
B.](../../Application/NMTFcoclust/Result/OSNMTF_PA){#fig:osnmtfpa
width="0.7\\linewidth"}

![OPNMTF compared to three types of data matrix based on four different
measures on setting
B.](../../Application/NMTFcoclust/Result/Measure2){#fig:measure2
width="0.7\\linewidth"}

![(Left) Data matrix truncated normal on setting A, (Middle) Clustering
data matrix with OPNMTF algorithm on setting A, (Right) Co-clustering
reorganization data matrix according to the output of the OPNMTF
algorithm on setting
A.](../../Application/NMTFcoclust/Result/OSNMTF_G){#fig:osnmtfg
width="0.7\\linewidth"}

![(Left) Data matrix Binary on setting A, (Middle) Clustering data
matrix with OPNMTF algorithm on setting A, (Right) Co-clustering
reorganization data matrix according to the output of the OPNMTF
algorithm on setting
A.](../../Application/NMTFcoclust/Result/OSNMTF_B){#fig:osnmtfb
width="0.7\\linewidth"}

![(Left) Data matrix Poisson on setting A, (Middle) Clustering data
matrix with OPNMTF algorithm on setting A, (Right) Co-clustering
reorganization data matrix according to the output of the OPNMTF
algorithm on setting
A.](../../Application/NMTFcoclust/Result/OSNMTF_P){#fig:osnmtfp
width="0.7\\linewidth"}

![OPNMTF compared to three types of data matrix based on four different
measures on setting
A.](../../Application/NMTFcoclust/Result/MeasureB){#fig:measureb
width="0.7\\linewidth"}

### Algorithms

Parammeters $\lambda$ and $\mu$ in our algorithm.

width=,center

          Ref                  Algorithm                                                                                                                       Object function                                                                                                                                                                             Constraints                                                                                                                                                                                                                             Multiplicative Update Rules
  -------------------- -------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ ---------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        $\text{OPNMTF}_{\alpha}$                         $D_{\alpha}(\mathbf{X}|| \mathbf{F}\mathbf{S}\mathbf{G}^{T})+\lambda D_{\alpha}(\mathbf{I} \parallel \mathbf{F}^{T}\mathbf{F})+\mu D_{\alpha}(\mathbf{I}\parallel \mathbf{G}^{T}\mathbf{G)}$                                                                    $\mathbf{F}\geqslant 0$, $\mathbf{G}\geqslant 0$                                                                            $\mathbf{F}\odot \left[ \left(\dfrac{(\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{GS}^{T}+2\lambda \mathbf{F} (\mathbf{I}_{g}\oslash \mathbf{F}^{T}\mathbf{F})^{\alpha}}{\mathbf{E}_{nm}\mathbf{G}\mathbf{S}^{T}+2\lambda \mathbf{F}\mathbf{E}_{gg}}\right)^{\dfrac{1}{\alpha}}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \right]_{+} ,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   \mathbf{G}\odot \left[ \left(\dfrac{((\mathbf{X}\oslash \mathbf{FSG}^{T})^{T})^{\alpha} \mathbf{FS}+2\mu \mathbf{G}(\mathbf{I}_{s}\oslash \mathbf{G}^{T}\mathbf{G})^{\alpha} }{\mathbf{E}_{nm}^{T}\mathbf{FS}+2\mu \mathbf{G}\mathbf{E}_{ss}}\right)^{\dfrac{1}{\alpha}}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \right]_{+} , 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \mathbf{S}\odot \left[ \left(\dfrac{\mathbf{F}^{T} (\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{G}}{\mathbf{F}^{T}\mathbf{E}_{nm}\mathbf{G}}\right)^{\dfrac{1}{\alpha}}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \right] _{+}$
                        $\text{ONMTF}_{\alpha}$                                                                                                   $D_{\alpha}(\mathbf{X}|| \mathbf{FSG}^{T})                                                                                                                                             $\mathbf{F}\geqslant 0$, $\mathbf{G}\geqslant 0$                                                                                                          $\mathbf{F}\odot \left[ \left(\dfrac{(\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{GS}^{T}}{\mathbf{E}_{nm}\mathbf{G}\mathbf{S}^{T}+2 \delta \alpha \mathbf{F}\bm{\Psi}_{g}}\right)^{\frac{1}{\alpha}}\right]_{+} ,
                                                                                                                                                                            +                                                                                                                                                                                                                                                                                                                                 \mathbf{G}\odot \left[\left( \dfrac{((\mathbf{X}\oslash \mathbf{FSG}^{T})^{T})^{\alpha} \mathbf{FS}}{\mathbf{E}_{nm}^{T}\mathbf{FS}+2\beta \alpha \mathbf{G}\bm{\Psi}_{s}}\right)^{\frac{1}{\alpha}}\right]_{+} , 
                                                                                                                                                      \delta \text{Tr}(\mathbf{F}\Psi\mathbf{F}^{T})                                                                                                                                                                                                                                                                                                                     \mathbf{S}\odot \left[\left( \dfrac{\mathbf{F}^{T} (\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{G}}{\mathbf{F}^{T}\mathbf{E}_{nm}\mathbf{G}}\right)^{\frac{1}{\alpha}}\right] _{+}$
                                                                                                                                                                            +                                                                                                                                                                                                                                                         
                                                                                                                                                     \beta \text{Tr}(\mathbf{G}\Omega\mathbf{G}^{T})$                                                                                                                                                                                                                                 
                         $\text{NMTF}_{\alpha}$                                                                                                  $D_{\alpha}(\mathbf{X}|| \mathbf{FSG}^{T})$                                                                                                                                             $\mathbf{F}\geqslant 0$, $\mathbf{G}\geqslant 0$                                                                                                                                  $\mathbf{F}\odot \left[ \left(\dfrac{( \mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{GS}^{T}}{\mathbf{E}_{nm}\mathbf{GS}^{T}}\right)^{\dfrac{1}{\alpha}}\right] _{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \mathbf{G}\odot \left[ \left(\dfrac{((\mathbf{X}\oslash \mathbf{FSG}^{T})^{T}) ^{\alpha} \mathbf{FS}}{\mathbf{E}_{nm}^{T}\mathbf{FS}}\right)^{\dfrac{1}{\alpha}}\right]_{+} ,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         \mathbf{S}\odot \left[\left(\dfrac{\mathbf{F}^{T} (\mathbf{X}\oslash \mathbf{FSG}^{T}) ^{\alpha} \mathbf{G}}{\mathbf{F}^{T}\mathbf{E}_{nm}\mathbf{G}}\right)^{\dfrac{1}{\alpha}}\right]_{+}$
       @Long2005a            $\text{NBVD}$                                                                                                 $\|\mathbf{X}-\mathbf{F}\mathbf{S}\mathbf{G}^{T}\|^{2}$                                                                                                                                       $\mathbf{F}\geqslant 0$, $\mathbf{G}\geqslant 0$                                                                                                                                                                         $\mathbf{F} \odot \left[\dfrac{\mathbf{XGS}^{T}}{\mathbf{FSG}^{T}\mathbf{GS}^{T}}\right]_{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \mathbf{G} \odot \left[\dfrac{\mathbf{X}^{T}\mathbf{FS}}{\mathbf{GS}^{T}\mathbf{F}^{T}\mathbf{FS}}\right]_{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \mathbf{S} \odot \left[\dfrac{\mathbf{F}^{T}\mathbf{XG}}{\mathbf{F}^{T}\mathbf{FSG}^{T}\mathbf{G}}\right]_{+}$
       @Ding2006c            $\text{ONM3F}$                                                                                                $\|\mathbf{X}-\mathbf{F}\mathbf{S}\mathbf{G}^{T}\|^{2}$                                                                                            $\mathbf{F}\geqslant 0$, $\mathbf{G}\geqslant 0$, $\mathbf{F}^{T}\mathbf{F}=\mathbf{I}_{g}$, $\mathbf{G}^{T}\mathbf{G}=\mathbf{I}_{s}$                                                                                                                    $\mathbf{F} \odot \left[\left(\dfrac{\mathbf{XGS}^{T}}{\mathbf{FF}^{T}\mathbf{XGS}^{T}}\right)^{-1/2}\right]_{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \mathbf{G} \odot \left[\left(\dfrac{\mathbf{X}^{T}\mathbf{FS}}{\mathbf{GG}^{T}\mathbf{X}^{T}\mathbf{FS}}\right)^{-1/2}\right]_{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \mathbf{S} \odot \left[\left(\dfrac{\mathbf{F}^{T}\mathbf{XG}}{\mathbf{F}^{T}\mathbf{FSG}^{T}\mathbf{G}}\right)^{-1/2}\right]_{+}$
        @Yoo2010             $\text{ONMTF}$                                                                                          $\dfrac{1}{2}\|\mathbf{X}-\mathbf{F}\mathbf{S}\mathbf{G}^{T}\|^{2}$                                                                                      $\mathbf{F}\geqslant 0$, $\mathbf{G}\geqslant 0$, $\mathbf{F}^{T}\mathbf{F}=\mathbf{I}_{g}$, $\mathbf{G}^{T}\mathbf{G}=\mathbf{I}_{s}$   $\mathbf{F}$ $\odot$ $\left[\dfrac{\mathbf{XGS}^{T}}{\mathbf{FSG}^{T}\mathbf{X}^{T}\mathbf{F}}\right]_{+}$, $\mathbf{G}$ $\odot$ $\left[\dfrac{\mathbf{X}^{T}\mathbf{FS}}{\mathbf{GS}^{T}\mathbf{F}^{T}\mathbf{XG}}\right]_{+}$, $\mathbf{S}$ $\odot$ $\left[\dfrac{\mathbf{F}^{T}\mathbf{XG}}{\mathbf{F}^{T}\mathbf{FSG}^{T}\mathbf{G}}\right]_{+}$
      @Labiod2011a           $\text{ODNMF}$                                                                                    $\|\mathbf{X}-\mathbf{F}\mathbf{F}^{T}\mathbf{X}\mathbf{G}\mathbf{G}^{T}\|^{2}$                                                                                $\mathbf{F}\geqslant 0$, $\mathbf{G}\geqslant 0$, $\mathbf{F}^{T}\mathbf{F}=\mathbf{I}_{g}$, $\mathbf{G}^{T}\mathbf{G}=\mathbf{I}_{s}$                                                                                                      $\mathbf{F} \odot \left[\dfrac{\mathbf{XGG}^{T}\mathbf{X}^{T}\mathbf{F}}{\mathbf{FF}^{T}\mathbf{XGG}^{T}\mathbf{X}^{T}\mathbf{F}}\right]_{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 \mathbf{G} \odot \left[\dfrac{\mathbf{X}^{T}\mathbf{FF}^{T}\mathbf{XG}}{\mathbf{GG}^{T}\mathbf{X}^{T}\mathbf{FF}^{T}\mathbf{XG}}\right]_{+}$
      @Labiod2011a           $\text{DNMF}$                                                  $\|\mathbf{X}-\mathbf{F}\mathbf{F}^{T}\mathbf{X}\mathbf{G}\mathbf{G}^{T}\|^{2}+\text{Tr}(\varLambda \mathbf{F}^{T})+\text{Tr}( \Gamma \mathbf{G}^{T})$                                                                                       $\mathbf{G}\geqslant 0$, $\mathbf{G}\geqslant 0$                                                                          $\mathbf{F} \odot \left[\dfrac{2\mathbf{X} \mathbf{G}\mathbf{G}^{T}\mathbf{X}^{T}\mathbf{F}}{\mathbf{FF}^{T}\mathbf{XG}\mathbf{G}^{T}\mathbf{G}\mathbf{G}^{T}\mathbf{X}^{T}\mathbf{F}+\mathbf{XG}\mathbf{G}^{T}\mathbf{G}\mathbf{G}^{T}\mathbf{X}^{T}\mathbf{FF}^{T}\mathbf{F}}\right]_{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                     \mathbf{G} \odot \left[\dfrac{2\mathbf{X}^{T}\mathbf{F}\mathbf{F}^{T}\mathbf{XG}}{\mathbf{GG}^{T}\mathbf{F}\mathbf{F}^{T}\mathbf{X} \mathbf{X}^{T}\mathbf{F}\mathbf{F}^{T}\mathbf{G}+\mathbf{F}\mathbf{F}^{T}\mathbf{X}\mathbf{X}^{T}\mathbf{F}\mathbf{F}^{T}\mathbf{GG}^{T}\mathbf{G}}\right]_{+}$
   @wang2017penalized        $\text{PNMTF}$        $\frac{1}{2} \|\mathbf{X}-\mathbf{F}\mathbf{S}\mathbf{G}^{T}\|^{2}+\frac{\tau}{2}\text{Tr}( \mathbf{F}\Psi\mathbf{F}^{T})+\frac{\eta}{2}\text{Tr}(  \mathbf{G}\Omega\mathbf{G}^{T})+\frac{\gamma}{2}\text{Tr}(\mathbf{S}^{T}\mathbf{S})$                                              $\mathbf{F}\geqslant 0$, $\mathbf{G}\geqslant 0$                                                                                                                               $\mathbf{F} \odot \left[\left( \dfrac{\mathbf{X}\mathbf{G}\mathbf{S}^{T}}{\mathbf{F}\mathbf{S}\mathbf{G}^{T}\mathbf{G}\mathbf{S}^{T}+\tau \mathbf{F}\Psi}\right)^{1/2}\right]_{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             \mathbf{G} \odot \left[\left( \dfrac{\mathbf{X}^{T}\mathbf{F}\mathbf{S}}{\mathbf{G}\mathbf{S}^{T}\mathbf{F}^{T}\mathbf{F}\mathbf{S}+\eta \mathbf{G}\Omega}\right)^{1/2} \right]_{+},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \mathbf{S} \odot \left[\left( \dfrac{\mathbf{F}^{T}\mathbf{X}\mathbf{G}}{\mathbf{F}^{T}\mathbf{F}\mathbf{S}\mathbf{G}^{T}\mathbf{G}+\gamma \mathbf{S}}\right)^{1/2} \right]_{+}$

  : Multiplicative updating algorithms

Real Datasets
-------------

The experiments were performed using some benchmark Document-term data
sets from the clustering literature. Table 1 summarizes the
characteristics of these data sets.

width=,center

  ------------------------ ------------ ------------------- -------------------------- -- ------------------- -------------------------- -- ------------------- -------------------------
                                                                                                                                                                
                                                                                                                                                                
  (lr)3-10 **Data sets**    **Metric**    $\mu=\lambda=0.3$   $0.8=\mu\neq\lambda=0.3$      $\mu=\lambda=0.3$   $0.6=\mu\neq\lambda=0.4$      $\mu=\lambda=0.3$   $1=\mu\neq\lambda=0.03$
                               Acc                 0.858947                   0.901053               0.907368               **0.936842**               0.833684                  0.785263
                               NMI                 0.798237                   0.813591               0.822953               **0.859959**               0.734286                  0.732291
                               ARI                 0.796805                   0.822318               0.790153               **0.837830**               0.659590                  0.636824
                             Runtime                      0                          0                      0                          0                      0                         0
                               Acc                 0.319234                   0.288035               0.265384                   0.280736               0.245295              **0.332055**
                               NMI                 0.763067                   0.752598               0.773733                   0.781544               0.777383              **0.794198**
                               ARI                 0.453325                   0.446059               0.467637               **0.494658**               0.481518                  0.489289
                             Runtime                  0.08s                 32s 10iter                  0.08s                      0.06s                  0.03s                     0.04s
                               Acc                       43                         34                     65                         41                     33                        46
                               NMI                       88                         94                     93                         89                     91                        90
                               ARI                       88                         94                     93                         89                     91                        90
                             Runtime                      0                          0                      0                          0                      0                         0
                               Acc                 0.828155                   0.889662           **0.902129**                   0.872765               0.870649                  0.888415
                               NMI                 0.730024                   0.772439           **0.812577**                   0.754727               0.758880                  0.788006
                               ARI                 0.640643                   0.740786           **0.785946**                   0.713090               0.714084                  0.755304
                             Runtime                      0                          0                      0                          0                      0                         0
                               Acc                 0.526419               **0.650282**               0.417547                   0.628164               0.514868                  0.584664
                               NMI                 0.642857                   0.664136               0.625317                   0.662820               0.645261              **0.678232**
                               ARI                 0.491184                   0.537940               0.457508                   0.527926               0.502736              **0.559712**
                             Runtime                      0                          0                      0                          0                      0                         0
                               Acc             **0.413053**                   0.349184               0.275407                   0.384265               0.360256                  0.326456
                               NMI                 0.661158                   0.651339               0.623253                   0.635293           **0.698431**                  0.650793
                               ARI                 0.457032                   0.415098               0.379111                   0.399471           **0.477211**                  0.421108
                             Runtime                      0                          0                      0                          0                      0                         0
                               Acc                 0.766383                   0.903109               0.884098                   0.906193               0.912361              **0.932665**
                               NMI                 0.528799                   0.758672               0,741787                   0.757267               0.767456              **0.804639**
                               ARI                 0.528553                   0.748051               0.713872                   0.751282               0.765602              **0.815059**
                             Runtime                  1h35m                          0                      0                          0                      0                         0
                               Acc                       43                         34                     65                         41                     33                        46
                               NMI                       88                         94                     93                         89                     91                        90
                               ARI                       88                         94                     93                         89                     91                        90
                             Runtime                      0                          0                      0                          0                      0                         0
  ------------------------ ------------ ------------------- -------------------------- -- ------------------- -------------------------- -- ------------------- -------------------------

  : : Description of Word-Document Data sets

width=,center

  ------------------------ ------------ -------------------- --------------------------- -- -------------------- --------------------------- -- -------------------- ---------------------------
                                                                                                                                                                     
                                                                                                                                                                     
  (lr)3-10 **Data sets**    **Metric**   $\delta=\beta=0.3$   $0.8=\delta\neq\beta=0.3$      $\delta=\beta=0.3$   $0.6=\delta\neq\beta=0.4$      $\delta=\beta=0.3$   $0.8=\delta\neq\beta=0.3$
                               Acc            0.747364                0.757894                    0.661025                0.572631                  **0.854736**              0.816842
                               NMI            0.693849                0.684339                    0.620053                0.643864                  **0.748504**              0.724820
                               ARI            0.551499                0.565756                    0.468056                0.464099                  **0.659154**              0.655453
                               Acc            0.283786              **0.305912**                  0.082045                0.246585                    0.280434                0.247050
                               NMI            0.767898              **0.774398**                  0.758709                0.720886                    0.754212                0.758991
                               ARI            0.452925                0.476223                  **0.494814**              0.463905                    0.420485                0.427344
                               Acc               43                      34                          65                      41                          33                      46
                               NMI               88                      94                          93                      89                          91                      90
                               ARI               88                      94                          93                      89                          91                      90
                               Acc            0.753454              **0.884883**                  0.718545                0.721246                    0.873870                0.832987
                               NMI            0.646241              **0.771035**                  0.650519                0.753134                  **0.774617**              0.725284
                               ARI            0.530603              **0.737627**                  0.508191                0.655501                  **0.731630**              0.640416
                               Acc            0.324649                0.453674                  **0.681494**              0.414352                    0.608257                0.421725
                               NMI            0.652888                0.639136                  **0.706367**              0.416003                    0.640075                0.636733
                               ARI            0.459249                0.491521                  **0.601088**              0.225034                    0.502489                0.492484
                               Acc            0.359790                0.357808                  **0.565034**              0.150815                    0.385664                0.428787
                               NMI            0.685959                0.669217                    0.625279                0.468596                  **0.720757**              0.680489
                               ARI            0.461935                0.439680                    0.401014                0.200984                  **0.497901**              0.465421
                               Acc            0.863531                0.726034                  **0.912361**              0.856335                    0.762271                0.852223
                               NMI            0.710859                0.575768                  **0.808769**              0.681263                    0.573944                0.666496
                               ARI            0.710722                0.575565                  **0.766619**              0.681110                    0.573742                0.623207
                               Acc               43                      34                          65                      41                          33                      46
                               NMI               88                      94                          93                      89                          91                      90
                               ARI               88                      94                          93                      89                          91                      90
  ------------------------ ------------ -------------------- --------------------------- -- -------------------- --------------------------- -- -------------------- ---------------------------

  : : Description of Word-Document Data sets

width=,center

  ----------------------- ------------ -------------- -- -------------- -- --------------
                                                                           
  (lr)3-7 **Data sets**    **Metric**   $\alpha=0.1$      $\alpha=0.5$       $\alpha=2$
                              Acc         0.787368        **0.897657**        0.713684
                              NMI         0.691912          0.726669        **0.740537**
                              ARI         0.567874        **0.646372**        0.63174
                              Acc         0.204770        **0.343758**        0.294877
                              NMI         0.706976          0.749267        **0.777887**
                              ARI         0.366356          0.418849        **0.475319**
                              Acc            0                 0                 0
                              NMI            0                 0                 0
                              ARI            0                 00                0
                              Acc         0.704207          0.789922        **0.803636**
                              NMI         0.640633        **0.682576**      **0.688710**
                              ARI         0.522414          0.568680        **0.614275**
                              Acc         0.474072          0.513885        **0.587367**
                              NMI         0.468994          0.647106        **0.705751**
                              ARI         0.241415          0.507584        **0.595286**
                              Acc         0.252331        **0.393822**        0.303613
                              NMI         0.665291          0.669766        **0.701870**
                              ARI       **0.632909**        0.468107          0.484793
                              Acc         0.774351        **0.951426**        0.698827
                              NMI         0.617198        **0.838344**        0.743932
                              ARI         0.526575        **0.857365**        0.698827
                              Acc            0                 0                 0
                              NMI            0                 0                 0
                              ARI            0                 00                0
  ----------------------- ------------ -------------- -- -------------- -- --------------

  : : Description of Word-Document Data sets

width=,center

  --------------- ------------ --------------- -- ---------------- -- ---------------- -- ----------------- -- ---------------- -- ---------------- -- --------------------------
  **Data sets**    **Metric**   $\text{NBVD}$      $\text{ONM3F}$      $\text{ONMTF}$      $\text{ODNMTF}$      $\text{DNMTF}$      $\text{PNMTF}$      $\text{OPNMTF}_{\alpha}$
                      Acc         0.787368          **0.897657**          0.713684                                                                            **0.936842**
                      NMI         0.691912            0.726669          **0.740537**                                                                          **0.859959**
                      ARI         0.567874          **0.646372**          0.63174                                                                             **0.837830**
                      Acc         0.204770          **0.343758**          0.294877                                                                            **0.332055**
                      NMI         0.706976            0.749267          **0.777887**                                                                          **0.794198**
                      ARI         0.366356            0.418849          **0.475319**                                                                          **0.494658**
                      Acc             0                  0                   0                                                                         
                      NMI             0                  0                   0                                                                         
                      ARI             0                  00                  0                                                                         
                      Acc         0.704207            0.789922          **0.803636**                                                                            0.902129
                      NMI         0.640633          **0.682576**        **0.688710**                                                                            0.812577
                      ARI         0.522414            0.568680          **0.614275**                                                                            0.785946
                      Acc         0.474072            0.513885          **0.587367**                                                                            0.650282
                      NMI         0.468994            0.647106          **0.705751**                                                                            0.678232
                      ARI         0.241415            0.507584          **0.595286**                                                                            0.559712
                      Acc         0.252331          **0.393822**          0.303613                                                                              0.413053
                      NMI         0.665291            0.669766          **0.701870**                                                                            0.698431
                      ARI       **0.632909**          0.468107            0.484793                                                                              0.477211
                      Acc         0.774351          **0.951426**          0.698827                                                                              0.932665
                      NMI         0.617198          **0.838344**          0.743932                                                                              0.804639
                      ARI         0.526575          **0.857365**          0.698827                                                                              0.815059
                      Acc             0                  0                   0                                                                         
                      NMI             0                  0                   0                                                                         
                      ARI             0                  00                  0                                                                         
  --------------- ------------ --------------- -- ---------------- -- ---------------- -- ----------------- -- ---------------- -- ---------------- -- --------------------------

  : : Description of Word-Document Data sets

lS\[table-format=2.2\]\*6S

&\
(lr)2-5 Data
sets&$\#\text{Document}$&$\#\text{Words}$&$\#\text{Clusters}$&$\text{Sparsity}(\%)$
(lr)1-8**CSTR**&475&1000&4&96.60 **WebACE**&2340&1000&20&91.83
**NG20**&19949&43586&20&99.99 **RCV1**&9625&29992&4&99.75
**Reviews**&4069&18483&5&99.99 **Sports**&8580&14870&7&99.99
**Classic3**&3891&4303&3&98.00 **Classic4**&7095&5896&4&99.41

Note that, for all the used word-document data sets, we apply the TF-IDF
transformation on all the word-document frequency matrices. We used the
TF-IDF weighting scheme proposed in scikit-learn [@pedregosa2011scikit]
which is defined by $w_{ij} = tf_{ij}(1 +
 \log( \frac{1+n}{1+d_{j}}))$, where $w_{ij}$ is the weight of term $i$
in document $j$, $tf_{ij}$ is the frequency of word $i$ in document $j$,
$n$ is the total number of documents and $d_{j}$ is the number of
documents containing term $j$.

To study the performance of our algorithm against other algorithms, we
selected eight different real-world text datasets exhibiting various
challenging situations, namely balanced and unbalanced cluster sizes,
overlapping clusters, different number of clusters, etc. The balance
coefficient is defined as the ratio of the number of documents in the
smallest class to the number of documents in the largest class
[@zhong2005generative]. Each matrix $\mathbf{x}$ is in its original
form; i.e. each entry $x_{ij}$ of $\mathbf{x}$ is the number of
occurrences of word $j$ in document $i$ and can be viewed as a
contingency table. The characteristics of each dataset are reported in
table 1.

The **CSTR** dataset was previously used in [@li2005general] and
includes the abstracts of technical reports published in the Department
of Computer Science of Rochester University. These abstracts were
divided into 4 research fields: Natural Language Processing (NLP),
Robotics/Vision, Systems and Theory.

**CLASSIC3** and **CLASSIC42** consist respectively of 3 different
document collections: CISI, CRANFIELD, and MEDLINE and 4 different
document collections: CACM, CISI, CRANFIELD, and MEDLINE.

**K1B** contains 6 different document categories where each document
corresponds to a web page listed in the sub- ject hierarchy of Yahoo.

**SPORTS** contains documents about 7 different sports including
baseball, basketball, bicycling, boxing, football, golfing and hockey.

**TDT24** is a document $\times$ term matrix obtained from a subset of
the Nist Topic Detection and Tracking corpus derived from 6 sources
including 2 newswires (APW, NYT), 2 radio programs (VOA, PRI) and 2
television programs (CNN, ABC).

The **Reuters30** and **Reuters40** datasets contain, respec- tively,
the 30 and 40 largest classes of the ModApte version of the standard
Reuters-215785 dataset.

Result
------

![This Graph shows iterations of $\text{OPNMTF}_{\alpha=0.4}$ vs Object
function $\xi_{\text{OPNMTF}_{\alpha}}$ that convergence is
clear.](CSTR_iter_object){#fig:CSTR_it width="0.7\\linewidth"}

![Orthogonality of matrix F vs
iterations](CSTR_iter_orthogonal_F){#fig:CSTR_it_ortho_F
width="0.7\\linewidth"}

![(Left) Data matrix $\textbf{CSTR}$, (Middle) Clustering data matrix
with $\text{OPNMTF}_{\alpha=0.4}$ algorithm with $\mu=0.6$ and
$\lambda=0.4$ on $\textbf{CSTR}$, (Right) Co-clustering reorganization
data matrix on
$\textbf{CSTR}$.](CSTR_data_cluster_cocluster){#fig:CSTR_data_cluster_cocluster
width="1\\linewidth"}

![CSTR data set](CSTR_data){#fig:cstrdata width="0.7\\linewidth"}

![(Left) Data matrix $\textbf{Classic3}$, (Middle) Clustering data
matrix with $\text{OPNMTF}_{\alpha=0.5}$ algorithm with $\mu=0.8$ and
$\lambda=0.3$ on $\textbf{Classic3}$, (Right) Co-clustering
reorganization data matrix on
$\textbf{Classic3}$.](Classic3_data_cluster_cocluster){#fig:Classic3_data_cluster_cocluster
width="1\\linewidth"}

![](WC_Classic3){#fig:WC1 width="1\\linewidth"}

![This Graph shows iterations of $\text{OPNMTF}_{\alpha=0.5}$ vs Object
function $\xi_{\text{OPNMTF}_{\alpha}}$ that convergence is
clear.](Classic3_iter_object){#fig:Classic3_it width="0.7\\linewidth"}

![Orthogonality of matrix F vs
iterations](Classic3_iter_orthogonal_F){#fig:Classic3_it_ortho_F
width="0.7\\linewidth"}

![Orthogonality of matrix G vs
iterations](Classic3_iter_orthogonal_G){#fig:Classic3_it_ortho_G
width="0.7\\linewidth"}

Conclusion
==========

Proof $\eqref{lem0}$ {#A_lem0}
====================

$$D_{\alpha}(
        \mathbf{X}||\mathbf{F}^{(t+1)}\mathbf{S}\mathbf{G}^{T}
        )
        \overset{\eqref{02}}{\leqslant}
        A(
        \mathbf{F}^{(t+1)}, \mathbf{F}^{(t)}
        ) 
        \overset{\eqref{0+}}{\leqslant}
        A(
        \mathbf{F}^{(t)}, \mathbf{F}^{(t)}
        )
        \overset{\eqref{01}}{=}
        D_{\alpha}(\mathbf{X}||\mathbf{F}^{(t)}\mathbf{S}\mathbf{G}^{T})$$
By repeating the updating rule in [\[0+\]](#0+){reference-type="eqref"
reference="0+"}, we can identify the sequence of estimates that will
lead to a local minimum of the cost function. 

$$D_{\alpha}(
        \mathbf{X}||\mathbf{F}^{(t_{min})}\mathbf{S}\mathbf{G}^{T}
        )
        \leqslant
        ...
        \leqslant
        D_{\alpha}(\mathbf{X}||\mathbf{F}^{(t)}\mathbf{S}\mathbf{G}^{T})
        \leqslant
        ...
        \leqslant
        D_{\alpha}(\mathbf{X}||\mathbf{F}^{(0)}\mathbf{S}\mathbf{G}^{T})$$

proof $\eqref{lem1}$ {#A_lem1}
====================

We need to show that the auxiliary function
$A(\mathbf{F},\mathbf{F}^{(t)})$ in $\eqref{L1}$ satisfies the following
two conditions:

1.  $A(\mathbf{F},\mathbf{F})\overset{(a)}{=}D_{\alpha}(\mathbf{X}||\mathbf{FSG}^{T})$,

2.  $A(\mathbf{F},\mathbf{F}^{(t)}) \overset{\eqref{convex},\eqref{L1}}{=}
            \sum\limits_{i,j,k,h} X_{ij}Q^{\mathbf{F}^{(t)}}_{ijkh} 
            f
            \left(  
            \dfrac{F_{ik}S_{kh}G_{hj}^{T}}{X_{ij}Q^{\mathbf{F}^{(t)}}_{ijkh}} 
            \right)
            \overset{(a),(b)}{\geq} \sum\limits_{i,j}X_{ij} 
            f
            \left(
            \dfrac
            {\sum\limits_{k,h}F_{ik}S_{kh}G_{hj}^{T}}
            {X_{ij}}
            \right)
            \overset{\eqref{D}}{=}D_{\alpha}(\mathbf{X}|| \mathbf{FSG}^{T})$,
    these two conditions are proved by

    1.  $\sum\limits_{k,h}Q^{\mathbf{F}^{(t)}}_{ijkh}=
                    \dfrac
                    {\sum\limits_{k,h}F_{ik}S_{kh}G_{hj}^{T}}
                    {\sum\limits_{k^{\prime},h^\prime} F_{i k^\prime}S_{k^\prime h^\prime }G_{h^\prime j}^{T}}
                    =\dfrac
                    {[\mathbf{FSG}^{T}]_{ij}}
                    {[\mathbf{FSG}^{T}]_{ij}}=1$

    2.  Because of the convexity of $f$ in
        [\[convex\]](#convex){reference-type="eqref" reference="convex"}
        and the use of Jensen's inequality.

Proof $\eqref{T1}$ {#A_T1}
==================

The minimum of $\eqref{L1}$ is resolved by gradient equals zero using
$\eqref{convex}$:
$$\dfrac{\partial A(\mathbf{F},\mathbf{F}^{(t)}) }{\partial F_{ik}}=
        \dfrac{1}{\alpha} \sum\limits_{h,j} S_{kh} G_{hj}^{T} 
        \left( 
        1- 
        (
        \dfrac{F_{ik}S_{kh}G_{hj}^{T}}{X_{ij}Q^{\mathbf{F}^{(t)}}_{ijkh}}  
        )^{-\alpha} 
        \right) 
        =0$$ which gives rise to $$\left(
        \dfrac{F_{ik}}{F^{(t)}_{ik}} 
        \right)^{\alpha}=
        \dfrac
        {
            \sum\limits_{h,j} S_{kh} G_{hj}^{T}
            \left(
            \dfrac{X_{ij}}{\sum\limits_{k^{\prime},h^\prime} F^{(t)}_{i k^\prime}S_{k^\prime h^\prime }G_{h^\prime j}^{T}}
            \right)
            ^{\alpha} 
        }
        {\sum\limits_{h,j} S_{kh} G_{hj}^{T}}$$ which suggest the
element-wise updating rule for $F_{ik}:$
$$F^{(t+1)}_{ik}\leftarrow F^{(t)}_{ik} \left( \dfrac{\sum\limits_{h,j} S_{kh} G_{hj}^{T} \left(\dfrac{X_{ij}}{[\mathbf{FSG}^{T}]_{ij}} \right)^{\alpha} }{\sum\limits_{j} [\mathbf{SG}^{T}]_{kj}}   \right)^{\dfrac{1}{\alpha}}$$
and suggest the matrix-wise updating rule for $\mathbf{F}$ $$\label{F1}
        \mathbf{F}^{(t+1)}\leftarrow \mathbf{F}^{(t)}\odot \left[ \left(\dfrac{(\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{GS}^{T}}{\mathbf{EGS}^{T}}\right)^{\dfrac{1}{\alpha}}\right]_{+}$$
that is identical to $\eqref{t0}$.

In a similar manner, the updating rule $\eqref{t1}$ and $\eqref{t2}$ are
determined by solving
$\dfrac{\partial A(\mathbf{G},\mathbf{G}^{(t)}) }{\partial G_{jh}}$ and
$\dfrac{\partial A(\mathbf{S},\mathbf{S}^{(t)}) }{\partial S_{kh}}$
where $A(\mathbf{G},\mathbf{G}^{(t)})$ and
$A(\mathbf{S},\mathbf{S}^{(t)})$ are given in $\eqref{lem3}$ and
$\eqref{lem2}$.

[ The divergence $D_{\alpha = 0}(\mathbf{X}|| \mathbf{FSG}^{T})$ is
non-increasing based on multiplicative update rules (special case
$\alpha = 0$) $$\begin{aligned}
        &F\leftarrow F\circ \dfrac{(X\oslash FSG^{T}) GS^{T}}{EGS^{T}+FF^{T}(X\oslash FSG^{T})GS^{T} - FF^{T}EGS^{T}}
        \\
        \\
        &G\leftarrow G\circ \dfrac{(X\oslash FSG^{T})^{T} FS}{E^{T}FS+GG^{T}(X\oslash FSG^{T})^{T}FS - GG^{T}E^{T}FS} 
        \\
        \\
        &S\leftarrow S\circ \dfrac{F^{T} (X\oslash FSG^{T}) G}{F^{T}EG} 
    \end{aligned}$$ ]{style="color: violet"}

Proof $\eqref{OPNMTF_1}$ {#A_OPNMTF_1}
========================

We need to show that the auxiliary function
$A^{*}(\mathbf{F},\mathbf{F}^{(t)})$ in $\eqref{lem6}$ satisfies the
following two conditions:

1.  $A^{*}(\mathbf{F},\mathbf{F})\overset{(a)}{=} \xi_{\text{OPNMTF}_{\alpha}}$,

2.  $A^{*}(\mathbf{F},\mathbf{F}^{(t)}) \overset{\eqref{convex},\eqref{L1}}{=}
            \sum\limits_{i,j,k,h} X_{ij}Q^{\mathbf{F}^{(t)}}_{ijkh} 
            f
            \left(  
            \dfrac{F_{ik}S_{kh}G_{hj}^{T}}{X_{ij}Q^{\mathbf{F}^{(t)}}_{ijkh}} 
            \right)
                    +
            \sum\limits_{i,k}
            Q^{\mathbf{F}^{(t)}}_{ik} 
            f
            \left(  
            \dfrac{F^{T}_{ki}F_{ik}}{Q^{\mathbf{F}^{(t)}}_{ik}} 
            \right)
            \overset{(a),(b),(c)}{\geq} \sum\limits_{i,j}X_{ij} 
            f
            \left(
            \dfrac
            {\sum\limits_{k,h}F_{ik}S_{kh}G_{hj}^{T}}
            {X_{ij}}
            \right)     + 
            f
            \left(  
        \dfrac{\sum\limits_{i} F^{(t)T}_{ki}F^{(t)}_{ik}}{I_{kk}}
            \right)
            \overset{\eqref{D},\eqref{12-}}{\geq}\xi_{\text{OPNMTF}_{\alpha}}$.

    These two conditions are proved by

    1.  $\sum\limits_{k,h}Q^{\mathbf{F}^{(t)}}_{ijkh}=
                    \dfrac
                    {\sum\limits_{k,h}F^{(t)}_{ik}S_{kh}G_{hj}^{T}}
                    {\sum\limits_{k^{\prime},h^\prime} F^{(t)}_{i k^\prime}S_{k^\prime h^\prime }G_{h^\prime j}^{T}}
                    =\dfrac
                    {[\mathbf{F}^{(t)}\mathbf{SG}^{T}]_{ij}}
                    {[\mathbf{F}^{(t)}\mathbf{SG}^{T}]_{ij}}=1$,

    2.  $\sum\limits_{i}Q^{\mathbf{F}^{(t)}}_{ik}=
                \dfrac{\sum\limits_{i} F^{(t)T}_{ki} F^{(t)}_{ik}}{\sum\limits_{i^{\prime}} F^{(t)T}_{ki^{\prime}}F^{(t)}_{i^{\prime}k}}=\dfrac
                {[\mathbf{F}^{(t)^T}\mathbf{F}^{(t)}]_{kk}}
                {[\mathbf{F}^{(t)^T}\mathbf{F}^{(t)}]_{kk}}=1$,

    3.  Because of the convexity of $f$ in
        [\[convex\]](#convex){reference-type="eqref" reference="convex"}
        and the use of Jensen's inequality.

Proof $\eqref{T3}$ {#A_T3}
==================

The minimum of $\eqref{lem6}$ is resolved by gradient equals zero using
$\eqref{convex}$:
$$\dfrac{\partial A^{*}(\mathbf{F},\mathbf{F}^{(t)}) }{\partial F_{ik}}=
        \dfrac{1}{\alpha} \sum\limits_{h,j} S_{kh} G_{hj}^{T} 
        \left( 
        1- 
        (
        \dfrac{F_{ik}S_{kh}G_{hj}^{T}}{X_{ij}Q^{\mathbf{F}^{(t)}}_{ijkh}}  
        )^{-\alpha} 
        \right) 
        +
            \dfrac{1}{\alpha} 2 \lambda
        \sum\limits_{k} F_{ik}
            \left( 
        1- 
        (
        \dfrac{F^{T}_{ki}F_{ik}}{Q^{\mathbf{F}^{(t)}}_{ik}}  
        )^{-\alpha} 
        \right) 
        =0$$ which gives rise to $$\left(
        \dfrac{F_{ik}}{F^{(t)}_{ik}} 
        \right)^{\alpha}=
        \dfrac
        {
            \sum\limits_{h,j} S_{kh} G_{hj}^{T}
            \left(
            \dfrac{X_{ij}}{\sum\limits_{k^{\prime},h^\prime} F^{(t)}_{i k^\prime}S_{k^\prime h^\prime }G_{h^\prime j}^{T}}
            \right)
            ^{\alpha} 
            +
            2 \lambda \sum\limits_{k} F_{ik}
            \left(
            \dfrac{I_{kk}}{[\mathbf{F}^{T} \mathbf{F}]_{kk}}
            \right)^{\alpha}
        }
        {
            \sum\limits_{h,j} S_{kh} G_{hj}^{T}
        +
        2 \lambda \sum\limits_{k} F_{ik}
    }$$ which suggest the element-wise updating rule for $F_{ik}:$
$$F^{(t+1)}_{ik}\leftarrow F^{(t)}_{ik}
        \left( 
        \dfrac{
            \sum\limits_{h,j} S_{kh} G_{hj}^{T} \left(\dfrac{X_{ij}}{[\mathbf{FSG}^{T}]_{ij}} \right)^{\alpha}
            +
            2 \lambda \sum\limits_{k} F_{ik}
            \left(
            \dfrac{I_{kk}}{[\mathbf{F}^{T} \mathbf{F}]_{kk}}
            \right)^{\alpha}
        }
    {
        \sum\limits_{j} [\mathbf{SG}^{T}]_{kj}
        +
    2 \lambda \sum\limits_{k} F_{ik}
}   \right)^{\dfrac{1}{\alpha}}$$ and suggest the matrix-wise updating
rule for $\mathbf{F}$
$$\mathbf{F}^{(t+1)}\leftarrow \mathbf{F}^{(t)}\odot \left[ \left(\dfrac{(\mathbf{X}\oslash \mathbf{FSG}^{T})^{\alpha} \mathbf{GS}^{T}+2\lambda \mathbf{F} (\mathbf{I}_{g}\oslash \mathbf{F}^{T}\mathbf{F})^{\alpha}}{\mathbf{E}_{nm}\mathbf{G}\mathbf{S}^{T}+2\lambda \mathbf{F}\mathbf{E}_{gg}}\right)^{\dfrac{1}{\alpha}}
        \right]_{+} ,$$ that is identical to $\eqref{12--}$.

In a similar manner, the updating rule $\eqref{12---}$ and
$\eqref{12----}$ are determined by solving
$\dfrac{\partial A^{*}(\mathbf{G},\mathbf{G}^{(t)}) }{\partial G_{jh}}$
and $\dfrac{\partial A(\mathbf{S},\mathbf{S}^{(t)}) }{\partial S_{kh}}$
where $A^{*}(\mathbf{G},\mathbf{G}^{(t)})$ and
$A(\mathbf{S},\mathbf{S}^{(t)})$ are given in $\eqref{lem7}$ and
$\eqref{lem2}$.
