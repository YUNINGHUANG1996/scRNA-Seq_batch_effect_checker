import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import scipy.stats as stats

# Assuming 'adata' is your already-processed combined AnnData object
# with condition information in adata.obs['condition']

def evaluate_batch_effects(adata, batch_key='condition', n_pcs=30):
    """
    Evaluate batch effects in scRNA-seq data to determine if batch correction
    is needed.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix with batch information in obs[batch_key]
    batch_key : str
        Column name in adata.obs containing batch/condition information
    n_pcs : int
        Number of principal components to use
    
    Returns:
    --------
    Need for batch correction (0-1 scale, higher means more likely needed)
    """
    results = {}
    
    print("Evaluating batch effects...")
    
    # 1. Check condition distribution across UMAP
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(adata, color=batch_key, ax=ax, show=False, title=f"UMAP by {batch_key}")
    plt.tight_layout()
    plt.savefig(f"umap_by_{batch_key}.png")
    plt.close()
    
    # 2. Visualize condition distribution in PCA space
    if 'X_pca' not in adata.obsm:
        print("Running PCA...")
        sc.pp.pca(adata)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.pca(adata, color=batch_key, ax=ax, show=False)
    plt.tight_layout()
    plt.savefig(f"pca_by_{batch_key}.png")
    plt.close()
    
    # 3. Calculate proportion of each condition in each cluster
    if 'leiden' not in adata.obs:
        print("Running clustering...")
        sc.tl.leiden(adata, resolution=0.5)
    
    print("\n3. Cluster composition by condition:")
    cluster_condition_counts = pd.crosstab(adata.obs['leiden'], adata.obs[batch_key])
    cluster_condition_props = cluster_condition_counts.div(cluster_condition_counts.sum(axis=1), axis=0)
    
    # Calculate entropy of condition distribution within each cluster
    def entropy(row):
        """Calculate Shannon entropy of a distribution"""
        # Filter out zeros to avoid log(0)
        p = row[row > 0]
        return -np.sum(p * np.log2(p))
    
    # Max entropy depends on number of conditions
    max_entropy = np.log2(len(adata.obs[batch_key].unique()))
    cluster_entropies = cluster_condition_props.apply(entropy, axis=1) / max_entropy
    
    print(f"Average normalized entropy across clusters: {cluster_entropies.mean():.3f}")
    print("(Values close to 1 indicate good mixing of conditions within clusters)")
    
    # Visualize as a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cluster_condition_props, cmap='viridis', annot=True, fmt='.2f')
    plt.title('Proportion of conditions in each cluster')
    plt.tight_layout()
    plt.savefig('condition_cluster_proportions.png')
    plt.close()
    
    results['mean_cluster_entropy'] = cluster_entropies.mean()
    
    # 4. PC variance explained by batch (R-squared of condition ~ PC)
    # Convert condition to numeric for regression
    condition_numeric = pd.Categorical(adata.obs[batch_key]).codes
    
    pc_rsquared = []
    for i in range(min(n_pcs, adata.obsm['X_pca'].shape[1])):
        pc_values = adata.obsm['X_pca'][:, i]
        
        # Calculate ANOVA 
        categories = np.unique(condition_numeric)
        cat_data = [pc_values[condition_numeric == c] for c in categories]
        f_val, p_val = stats.f_oneway(*cat_data)
        
        # Calculate R-squared (proportion of variance explained by condition)
        cat_means = [np.mean(d) for d in cat_data]
        cat_counts = [len(d) for d in cat_data]
        
        grand_mean = np.mean(pc_values)
        ss_between = sum([count * (mean - grand_mean)**2 for count, mean in zip(cat_counts, cat_means)])
        ss_total = sum((pc_values - grand_mean)**2)
        
        r_squared = ss_between / ss_total if ss_total > 0 else 0
        pc_rsquared.append(r_squared)
    
    # Plot variance explained by batch for each PC
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(pc_rsquared) + 1), pc_rsquared)
    plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.7)  # Threshold line
    plt.xlabel('Principal Component')
    plt.ylabel('R² (Variance explained by condition)')
    plt.title('Condition effect on principal components')
    plt.tight_layout()
    plt.savefig('pc_condition_effect.png')
    plt.close()
    
    results['max_pc_rsquared'] = max(pc_rsquared)
    
    # 5. Calculate silhouette score to measure separation by batch
    # Lower is better (less separation by batch)
    pca_data = adata.obsm['X_pca'][:, :min(n_pcs, adata.obsm['X_pca'].shape[1])]
    try:
        batch_silhouette = silhouette_score(pca_data, condition_numeric)
        print(f"\n5. Silhouette score by condition: {batch_silhouette:.3f}")
        print("(Values close to 1 indicate strong separation by condition - may need correction)")
        results['batch_silhouette'] = batch_silhouette
    except:
        print("Could not calculate silhouette score (possibly only one condition or too few cells)")
        results['batch_silhouette'] = np.nan
    
    # 6. ANOVA test on PC coordinates by condition
    # Higher F-statistic means stronger batch effect
    significant_pcs = 0
    for i in range(min(n_pcs, adata.obsm['X_pca'].shape[1])):
        pc_values = adata.obsm['X_pca'][:, i]
        f_val, p_val = stats.f_oneway(*[pc_values[condition_numeric == c] for c in categories])
        if p_val < 0.01:  # Significance threshold
            significant_pcs += 1
    
    results['significant_pcs_percent'] = significant_pcs / min(n_pcs, adata.obsm['X_pca'].shape[1])
    print(f"\n6. Percent of PCs significantly affected by condition: {results['significant_pcs_percent']:.1%}")
    
    # 7. Create condition split visualization
    conditions = adata.obs[batch_key].unique()
    n_conditions = len(conditions)
    
    # Create a grid layout
    n_cols = min(4, n_conditions)
    n_rows = (n_conditions + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
    axs = axs.flatten()
    
    for i, condition in enumerate(conditions):
        if i < len(axs):
            subset = adata[adata.obs[batch_key] == condition]
            sc.pl.umap(subset, color='leiden', size=50, ax=axs[i], show=False, title=condition)
    
    # Remove empty subplots if any
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.savefig("umap_split_by_condition.png")
    plt.close()
    
    # Calculate overall score for batch effect severity (0-1 scale)
    batch_effect_indicators = [
        results['max_pc_rsquared'] if 'max_pc_rsquared' in results else 0,
        results['batch_silhouette'] if 'batch_silhouette' in results and not np.isnan(results['batch_silhouette']) else 0,
        1 - results['mean_cluster_entropy'] if 'mean_cluster_entropy' in results else 0,
        results['significant_pcs_percent'] if 'significant_pcs_percent' in results else 0
    ]
    
    batch_effect_score = np.mean(batch_effect_indicators)
    
    # Provide recommendation
    print("\n====== BATCH EFFECT ANALYSIS SUMMARY ======")
    print(f"Overall batch effect score: {batch_effect_score:.3f} (0-1 scale, higher means stronger batch effect)")
    
    if batch_effect_score < 0.3:
        print("\nRECOMMENDATION: Batch correction likely NOT needed")
        print("- Your data shows minimal batch effects")
        print("- Proceed without Harmony or other batch correction methods")
    elif batch_effect_score < 0.6:
        print("\nRECOMMENDATION: Moderate batch effects detected")
        print("- Try analysis both with and without batch correction")
        print("- Consider using Harmony with default parameters")
    else:
        print("\nRECOMMENDATION: Strong batch effects detected")
        print("- Batch correction is highly recommended")
        print("- Use Harmony or another batch correction method")
        print("- Consider higher theta parameter in Harmony (e.g., 2.0) for stronger correction")
    
    print("\nSee generated plots for visual assessment of batch effects")
    
    return batch_effect_score

# Call the function on your data
# Example usage:
# batch_effect_score = evaluate_batch_effects(adata, batch_key='condition')
