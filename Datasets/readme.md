
# Datasets

This directory contains benchmark datasets commonly used for clustering, co-clustering, and text mining experiments. Most datasets are provided in `.mat` format for easy use in MATLAB and Python (via `scipy.io.loadmat`).

## Available Datasets

### Text Clustering Datasets

- **20Newsgroups.mat**  
  A popular text dataset containing documents from 20 different newsgroups. Widely used for benchmarking text classification and clustering algorithms.

- **NG20.mat**  
  A processed or alternative version of the 20 Newsgroups dataset, typically preprocessed for clustering tasks.

- **Reuters21578.mat**  
  A well-known dataset of newswire articles labeled with multiple categories. Frequently used for text categorization and clustering.

- **WebACE.mat**  
  A dataset derived from web documents, often used in clustering and topic modeling research.

- **reviews.mat**  
  Contains review text data (e.g., product or service reviews), useful for sentiment analysis and clustering.

- **sports.mat**  
  A dataset focused on sports-related documents, suitable for domain-specific clustering tasks.

---

### Document-Term / Co-clustering Datasets

- **RCV1_4Class.mat**  
  A subset of the Reuters Corpus Volume 1 (RCV1) with four selected classes. Commonly used in classification and clustering experiments.

- **RCV1_ori.mat**  
  The original RCV1 dataset representation, typically larger and more detailed.

- **TDT2.mat**  
  Topic Detection and Tracking dataset used for event detection and clustering in news streams.

- **classic3.mat**  
  A combination of three classic datasets (CACM, CISI, MED), often used for benchmarking clustering algorithms.

- **cstr.mat**  
  A dataset frequently used in co-clustering literature, containing structured document-term relationships.

---

## File Format

All datasets are stored in MATLAB `.mat` format and typically include:

- `X`: Document-term matrix  
- `labels` or `gnd`: Ground truth labels  
- Additional metadata depending on the dataset

You can load them in Python using:

```python  
from scipy.io import loadmat  

data = loadmat('dataset_name.mat')  
