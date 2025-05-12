import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import scipy.stats as stats

# To apply Harmony correction if needed:
def apply_harmony_correction(adata, batch_key='condition', theta=1.0):
    """Apply Harmony batch correction"""
    print(f"Applying Harmony batch correction (theta={theta})...")
    
    # Run PCA if not already calculated
    if 'X_pca' not in adata.obsm:
        sc.pp.pca(adata)
    
    # Apply Harmony
    sc.external.pp.harmony_integrate(adata, batch_key, basis='X_pca', adjusted_basis='X_pca_harmony', theta=theta)
    
    # Recompute neighbors and UMAP using the batch-corrected PCs
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, use_rep='X_pca_harmony')
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)
    
    # Generate visualization
    sc.pl.umap(adata, color=[batch_key, 'leiden'], wspace=0.5, frameon=False)
    
    return adata

# Example usage:
# adata_harmony = apply_harmony_correction(adata, batch_key='condition', theta=1.0)
