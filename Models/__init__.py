#!/usr/bin/env python
# coding: utf-8

# In[ ]:



"""
Module `NMTFcoclust.Models` module gathers implementations of co-clustering
algorithms with Nonnegative Matrix Tri Factorization.
"""

from .NMTFcoclust_OPNMTF_alpha import OPNMTF
from .NMTFcoclust_ONMTF_alpha import ONMTF
from .NMTFcoclust_NMTF_alpha import NMTF
from .NMTFcoclust_NBVD import NBVD
from .NMTFcoclust_DNMF import DNMF
#from .NMTFcoclus_ODNMF import ODNMF
from .NMTFcoclust_ONM3F import ONM3F
from .NMTFcoclust_ONM3F_2 import ONM3F_2


__all__ = ['OPNMTF',
		'NBVD',
		'DNMF',
		'ONM3F',
		'ONM3F_2',
		'ONMTF',
		'NMTF']

