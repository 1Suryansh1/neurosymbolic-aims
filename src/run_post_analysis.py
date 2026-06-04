import os
import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from src.config import Config
from src.dataset import get_dataloaders
from src.models.multimodal_model import MultimodalModel

def run_tsne_analysis(device):
    print("Running t-SNE latent representation projection...")
    
    # Load fold 0 validation loader (using use_image=True)
    _, val_loader, meta_dim = get_dataloaders(fold=0, use_image=True)
    
    # Define models to extract latents from
    models_dict = {
        "Neural-only": "multimodal_neural_fold0",
        "Multimodal Neuro-symbolic (Naive)": "neuro_symbolic_v1_fold0",
        "Neuro-symbolic V3 (LTN)": "multimodal_neuro_symbolic_v3_fold0",
        "Neuro-symbolic V3 (High Lambda 5.0)": "neuro_symbolic_v3_lambda5_fold0",
        "NODE Tabular Baseline": "meta_only_node_fold0",
        "SOTA (NODE + CNN + Naive)": "multimodal_node_neuro_symbolic_v1_fold0",
        "SOTA (NODE + CNN + LTN)": "multimodal_node_neuro_symbolic_v3_fold0"
    }
    
    for name, cp_name in models_dict.items():
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"{cp_name}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Skipping t-SNE for {name} (checkpoint not found)")
            continue
            
        use_node = ("node" in cp_name)
        use_image = not ("meta_only" in cp_name)
        model = MultimodalModel(
            meta_input_dim=meta_dim,
            use_image=use_image,
            use_meta=True,
            use_node=use_node
        ).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        all_latents = []
        all_species = []
        all_states = []
        all_totals = []
        
        # Load raw metadata from validation dataset to map labels
        wide_csv = pd.read_csv(Config.TRAIN_WIDE_CSV)
        val_df = wide_csv[wide_csv['fold'] == 0].reset_index(drop=True)
        
        with torch.no_grad():
            for i, (images, meta, targets) in enumerate(val_loader):
                images = images.to(device)
                meta = meta.to(device)
                _, shared_rep = model(images, meta)
                all_latents.append(shared_rep.cpu().numpy())
                
        all_latents = np.concatenate(all_latents, axis=0) # [N, 128]
        
        # Fit t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=15)
        tsne_results = tsne.fit_transform(all_latents)
        
        val_df['tsne_x'] = tsne_results[:, 0]
        val_df['tsne_y'] = tsne_results[:, 1]
        
        # Plot coloring by Species
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=val_df, x='tsne_x', y='tsne_y', 
            hue='Species', palette='tab20', alpha=0.8
        )
        plt.title(f't-SNE Projection of Fused Latent Space ({name}) - Coloured by Species')
        plt.tight_layout()
        tsne_species_path = os.path.join(Config.REPORT_DIR, 'figures', f'tsne_{cp_name}_species.png')
        plt.savefig(tsne_species_path, dpi=150)
        plt.close()
        
        # Plot coloring by Dry_Total_g
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=val_df, x='tsne_x', y='tsne_y', 
            hue='Dry_Total_g', palette='viridis', alpha=0.8
        )
        plt.title(f't-SNE Projection of Fused Latent Space ({name}) - Coloured by Dry Total Biomass')
        plt.tight_layout()
        tsne_total_path = os.path.join(Config.REPORT_DIR, 'figures', f'tsne_{cp_name}_biomass.png')
        plt.savefig(tsne_total_path, dpi=150)
        plt.close()
        
        print(f"Saved t-SNE plots for {name} to reports/figures/")

def run_constraint_comparison():
    print("Comparing constraint violations across models...")
    
    neural_csv = os.path.join(Config.CHECKPOINT_DIR, "multimodal_neural_fold0_predictions.csv")
    ns_v1_csv = os.path.join(Config.CHECKPOINT_DIR, "neuro_symbolic_v1_fold0_predictions.csv")
    ns_v3_csv = os.path.join(Config.CHECKPOINT_DIR, "multimodal_neuro_symbolic_v3_fold0_predictions.csv")
    
    if not os.path.exists(neural_csv) or not os.path.exists(ns_v1_csv):
        print("Skipping constraint violation comparison (csv files not found)")
        return
        
    df_neural = pd.read_csv(neural_csv)
    df_ns_v1 = pd.read_csv(ns_v1_csv)
    
    # Calculate additive violation
    calc_total_neural = df_neural['pred_Dry_Clover_g'] + df_neural['pred_Dry_Dead_g'] + df_neural['pred_Dry_Green_g']
    viol_neural = (df_neural['pred_Dry_Total_g'] - calc_total_neural).abs()
    
    calc_total_ns = df_ns_v1['pred_Dry_Clover_g'] + df_ns_v1['pred_Dry_Dead_g'] + df_ns_v1['pred_Dry_Green_g']
    viol_ns = (df_ns_v1['pred_Dry_Total_g'] - calc_total_ns).abs()

    plt.figure(figsize=(10, 5))
    sns.histplot(viol_neural, color='red', label='Neural-only (Mean Viol: {:.3f}g)'.format(viol_neural.mean()), kde=True, alpha=0.5, bins=30)
    sns.histplot(viol_ns, color='orange', label='Neuro-symbolic Naive (Mean Viol: {:.3f}g)'.format(viol_ns.mean()), kde=True, alpha=0.5, bins=30)
    
    if os.path.exists(ns_v3_csv):
        df_ns_v3 = pd.read_csv(ns_v3_csv)
        calc_total_v3 = df_ns_v3['pred_Dry_Clover_g'] + df_ns_v3['pred_Dry_Dead_g'] + df_ns_v3['pred_Dry_Green_g']
        viol_v3 = (df_ns_v3['pred_Dry_Total_g'] - calc_total_v3).abs()
        sns.histplot(viol_v3, color='blue', label='Neuro-symbolic LTN (Mean Viol: {:.3f}g)'.format(viol_v3.mean()), kde=True, alpha=0.5, bins=30)
        
    sota_csv = os.path.join(Config.CHECKPOINT_DIR, "multimodal_node_neuro_symbolic_v3_fold0_predictions.csv")
    if os.path.exists(sota_csv):
        df_sota = pd.read_csv(sota_csv)
        calc_total_sota = df_sota['pred_Dry_Clover_g'] + df_sota['pred_Dry_Dead_g'] + df_sota['pred_Dry_Green_g']
        viol_sota = (df_sota['pred_Dry_Total_g'] - calc_total_sota).abs()
        sns.histplot(viol_sota, color='green', label='SOTA NODE+LTN (Mean Viol: {:.3f}g)'.format(viol_sota.mean()), kde=True, alpha=0.5, bins=30)
        
    plt.title('Additive Rule Violation Distribution Comparison')
    plt.xlabel('Absolute Violation |Total - (Clover + Dead + Green)| in grams')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    viol_compare_path = os.path.join(Config.REPORT_DIR, 'figures', 'constraint_violation_comparison.png')
    plt.savefig(viol_compare_path, dpi=150)
    plt.close()
    print(f"Saved constraint violation comparison plot to: {viol_compare_path}")

def plot_residuals(csv_path, model_name, out_name):
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    
    # Compute residuals for GDM and Total
    df['residual_Total'] = df['true_Dry_Total_g'] - df['pred_Dry_Total_g']
    
    # Plot residuals vs ground truth
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(data=df, x='true_Dry_Total_g', y='residual_Total', hue='State', ax=axes[0], alpha=0.8)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title(f'Residuals vs True Dry Total ({model_name})')
    axes[0].set_ylabel('Residual (True - Pred) (g)')
    
    sns.scatterplot(data=df, x='Pre_GSHH_NDVI', y='residual_Total', hue='State', ax=axes[1], alpha=0.8)
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_title('Residuals vs NDVI')
    axes[1].set_ylabel('Residual (True - Pred) (g)')
    
    plt.tight_layout()
    residuals_path = os.path.join(Config.REPORT_DIR, 'figures', out_name)
    plt.savefig(residuals_path, dpi=150)
    plt.close()
    print(f"Saved residuals analysis plot to: {residuals_path}")

def run_error_slicing():
    print("Performing error slicing and residual analysis...")
    
    ns_v1_csv = os.path.join(Config.CHECKPOINT_DIR, "neuro_symbolic_v1_fold0_predictions.csv")
    ns_v3_csv = os.path.join(Config.CHECKPOINT_DIR, "multimodal_neuro_symbolic_v3_fold0_predictions.csv")
    ns_v3_lambda5_csv = os.path.join(Config.CHECKPOINT_DIR, "neuro_symbolic_v3_lambda5_fold0_predictions.csv")
    node_baseline_csv = os.path.join(Config.CHECKPOINT_DIR, "meta_only_node_fold0_predictions.csv")
    sota_naive_csv = os.path.join(Config.CHECKPOINT_DIR, "multimodal_node_neuro_symbolic_v1_fold0_predictions.csv")
    sota_csv = os.path.join(Config.CHECKPOINT_DIR, "multimodal_node_neuro_symbolic_v3_fold0_predictions.csv")
    
    plot_residuals(ns_v1_csv, "Neuro-symbolic Naive", "residuals_analysis_v1.png")
    plot_residuals(ns_v3_csv, "Neuro-symbolic LTN", "residuals_analysis_v3.png")
    plot_residuals(ns_v3_lambda5_csv, "High Lambda LTN", "residuals_analysis_lambda5.png")
    plot_residuals(node_baseline_csv, "NODE Tabular Baseline", "residuals_analysis_node_baseline.png")
    plot_residuals(sota_naive_csv, "SOTA NODE+Naive", "residuals_analysis_sota_naive.png")
    plot_residuals(sota_csv, "SOTA NODE+LTN", "residuals_analysis_sota.png")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.join(Config.REPORT_DIR, 'figures'), exist_ok=True)
    
    run_tsne_analysis(device)
    run_constraint_comparison()
    run_error_slicing()

if __name__ == '__main__':
    main()
