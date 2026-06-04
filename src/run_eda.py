import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    processed_csv = os.path.join('data', 'processed', 'train_wide.csv')
    if not os.path.exists(processed_csv):
        print(f"Error: {processed_csv} does not exist.")
        return
        
    df = pd.read_csv(processed_csv)
    targets = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # 1. Target Distributions
    plt.figure(figsize=(12, 6))
    df_melted = df.melt(value_vars=targets, var_name='Target', value_name='Biomass (g)')
    sns.violinplot(x='Target', y='Biomass (g)', data=df_melted, palette='muted')
    plt.title('Distribution of Biomass Targets')
    plt.xticks(rotation=15)
    plt.tight_layout()
    os.makedirs(os.path.join('reports', 'figures'), exist_ok=True)
    dist_path = os.path.join('reports', 'figures', 'target_distributions.png')
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"Saved target distributions plot to: {dist_path}")
    
    # Target summary statistics
    summary = df[targets].describe()
    print("\n--- Target Summary Statistics ---")
    print(summary.to_string())
    
    # Skewness
    skew = df[targets].skew()
    print("\n--- Target Skewness ---")
    print(skew)
    
    # 2. Pairwise Target Correlations
    plt.figure(figsize=(8, 6))
    corr = df[targets].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1)
    plt.title('Target Correlation Matrix')
    plt.tight_layout()
    corr_path = os.path.join('reports', 'figures', 'target_correlations.png')
    plt.savefig(corr_path, dpi=150)
    plt.close()
    print(f"Saved target correlation matrix to: {corr_path}")
    
    # 3. Additive Check Violation
    calc_total = df['Dry_Clover_g'] + df['Dry_Dead_g'] + df['Dry_Green_g']
    diff = df['Dry_Total_g'] - calc_total
    
    plt.figure(figsize=(8, 5))
    sns.histplot(diff, bins=50, kde=True)
    plt.title('Additive Rule Violation (Dry_Total_g - (Clover + Dead + Green))')
    plt.xlabel('Difference (g)')
    plt.ylabel('Count')
    plt.tight_layout()
    add_path = os.path.join('reports', 'figures', 'additivity_error.png')
    plt.savefig(add_path, dpi=150)
    plt.close()
    print(f"Saved additivity error histogram to: {add_path}")
    
    # 4. NDVI and Height relations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.scatterplot(data=df, x='Pre_GSHH_NDVI', y='Dry_Green_g', hue='State', ax=axes[0, 0], alpha=0.7)
    sns.regplot(data=df, x='Pre_GSHH_NDVI', y='Dry_Green_g', scatter=False, ax=axes[0, 0], color='black')
    axes[0, 0].set_title('NDVI vs Dry Green Biomass')
    
    sns.scatterplot(data=df, x='Pre_GSHH_NDVI', y='Dry_Total_g', hue='State', ax=axes[0, 1], alpha=0.7)
    sns.regplot(data=df, x='Pre_GSHH_NDVI', y='Dry_Total_g', scatter=False, ax=axes[0, 1], color='black')
    axes[0, 1].set_title('NDVI vs Dry Total Biomass')
    
    sns.scatterplot(data=df, x='Height_Ave_cm', y='Dry_Total_g', hue='State', ax=axes[1, 0], alpha=0.7)
    sns.regplot(data=df, x='Height_Ave_cm', y='Dry_Total_g', scatter=False, ax=axes[1, 0], color='black')
    axes[1, 0].set_title('Height vs Dry Total Biomass')
    
    sns.scatterplot(data=df, x='Height_Ave_cm', y='GDM_g', hue='State', ax=axes[1, 1], alpha=0.7)
    sns.regplot(data=df, x='Height_Ave_cm', y='GDM_g', scatter=False, ax=axes[1, 1], color='black')
    axes[1, 1].set_title('Height vs GDM')
    
    plt.tight_layout()
    rel_path = os.path.join('reports', 'figures', 'ndvi_height_vs_targets.png')
    plt.savefig(rel_path, dpi=150)
    plt.close()
    print(f"Saved relations plot to: {rel_path}")
    
    # 5. Species and State bar charts
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.countplot(data=df, y='Species', ax=axes[0], order=df['Species'].value_counts().index, palette='viridis')
    axes[0].set_title('Species Distribution')
    
    sns.countplot(data=df, x='State', ax=axes[1], order=df['State'].value_counts().index, palette='viridis')
    axes[1].set_title('State Distribution')
    
    plt.tight_layout()
    cat_path = os.path.join('reports', 'figures', 'categorical_distributions.png')
    plt.savefig(cat_path, dpi=150)
    plt.close()
    print(f"Saved categorical distributions plot to: {cat_path}")
    
if __name__ == '__main__':
    main()
