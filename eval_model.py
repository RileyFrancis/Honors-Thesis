"""
Model Evaluation and Analysis Script
Provides tools for analyzing trained GNN models
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from torch_geometric.loader import DataLoader
from train_brain_gnn import BrainGNN, BrainConnectivityDataset, GNNTrainer
import torch.nn.functional as F


def load_model(checkpoint_path, device='cuda'):
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = BrainGNN(
        num_node_features=checkpoint['num_node_features'],
        hidden_channels=checkpoint['hidden_channels'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout'],
        gnn_type=checkpoint['gnn_type']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Test Accuracy: {checkpoint.get('test_acc', 'N/A'):.4f}")
    print(f"Test F1: {checkpoint.get('test_f1', 'N/A'):.4f}")
    print(f"Test AUC: {checkpoint.get('test_auc', 'N/A'):.4f}")
    
    return model, checkpoint


def get_predictions(model, data_loader, device='cuda'):
    """Get predictions and probabilities for a dataset"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_subjects = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data)
            
            pred = out.argmax(dim=1)
            probs = F.softmax(out, dim=1)[:, 1]
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_subjects.extend(data.subject_id)
    
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs),
        'subjects': all_subjects
    }


def plot_roc_curve(labels, probs, save_path='roc_curve.png'):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curve to {save_path}")


def plot_precision_recall_curve(labels, probs, save_path='pr_curve.png'):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved PR curve to {save_path}")


def analyze_predictions(results, save_path='prediction_analysis.csv'):
    """Analyze predictions and save to CSV"""
    df = pd.DataFrame({
        'subject_id': results['subjects'],
        'true_label': results['labels'],
        'predicted_label': results['predictions'],
        'probability': results['probabilities'],
        'correct': results['labels'] == results['predictions']
    })
    
    df['confidence'] = np.abs(df['probability'] - 0.5)
    
    # Sort by confidence
    df = df.sort_values('confidence', ascending=False)
    
    df.to_csv(save_path, index=False)
    print(f"Saved prediction analysis to {save_path}")
    
    # Print summary
    print("\n=== Prediction Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Correct predictions: {df['correct'].sum()} ({df['correct'].mean()*100:.2f}%)")
    print(f"\nMost confident correct predictions:")
    print(df[df['correct']].head(5)[['subject_id', 'true_label', 'probability', 'confidence']])
    print(f"\nMost confident incorrect predictions:")
    print(df[~df['correct']].head(5)[['subject_id', 'true_label', 'predicted_label', 'probability', 'confidence']])
    
    return df


def compare_models(checkpoint_paths, test_loader, device='cuda'):
    """Compare multiple trained models"""
    results = []
    
    for i, path in enumerate(checkpoint_paths):
        print(f"\nEvaluating model {i+1}/{len(checkpoint_paths)}: {path}")
        model, checkpoint = load_model(path, device)
        
        preds = get_predictions(model, test_loader, device)
        
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        results.append({
            'model': path,
            'accuracy': accuracy_score(preds['labels'], preds['predictions']),
            'f1': f1_score(preds['labels'], preds['predictions']),
            'auc': roc_auc_score(preds['labels'], preds['probabilities']),
            'hidden_channels': checkpoint['hidden_channels'],
            'num_layers': checkpoint['num_layers'],
            'dropout': checkpoint['dropout'],
            'gnn_type': checkpoint['gnn_type']
        })
    
    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison ===")
    print(results_df)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, metric in enumerate(['accuracy', 'f1', 'auc']):
        axes[i].bar(range(len(results_df)), results_df[metric])
        axes[i].set_xticks(range(len(results_df)))
        axes[i].set_xticklabels([f"Model {j+1}" for j in range(len(results_df))], rotation=45)
        axes[i].set_ylabel(metric.upper())
        axes[i].set_title(f'{metric.upper()} Comparison')
        axes[i].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved model comparison to model_comparison.png")
    
    return results_df


def visualize_graph_embeddings(model, data_loader, device='cuda', save_path='embeddings.png'):
    """Visualize graph embeddings using t-SNE"""
    from sklearn.manifold import TSNE
    
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            
            # Get embeddings before final classification layer
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            # Forward through GNN layers
            x = model.conv1(x, edge_index)
            x = model.bns[0](x)
            x = F.relu(x)
            
            for i, conv in enumerate(model.convs):
                x = conv(x, edge_index)
                x = model.bns[i + 1](x)
                x = F.relu(x)
            
            # Pool to graph level
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)
            
            embeddings.append(x.cpu().numpy())
            labels.extend(data.y.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    # Apply t-SNE
    print("Computing t-SNE embeddings...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='coolwarm', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.title('Graph Embeddings Visualization (t-SNE)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved embeddings visualization to {save_path}")


def main():
    """Example usage of evaluation tools"""
    
    # Paths
    MODEL_PATH = 'brain_gnn_reward_best.pt'
    DATA_DIR = '/shared/healthinfolab/datasets/ABCD/Irritability/WholeBrainTaskfMRI/UConn/data'
    LABELS_CSV = '/shared/healthinfolab/datasets/ABCD/Irritability/WholeBrainTaskfMRI/UConn/subject_lists/CBCL_PLPH_m0m24_MID_n3705.csv'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading model...")
    model, checkpoint = load_model(MODEL_PATH, device)
    
    # Load test dataset
    print("\nLoading test dataset...")
    dataset = BrainConnectivityDataset(
        data_dir=DATA_DIR,
        labels_csv=LABELS_CSV,
        contrast_type='reward',
        threshold=0.0
    )
    
    # Create test loader (use your actual test indices)
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        range(len(dataset)), 
        test_size=0.2,
        stratify=[dataset.get(i).y.item() for i in range(len(dataset))],
        random_state=42
    )
    
    test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=32, shuffle=False)
    
    # Get predictions
    print("\nGenerating predictions...")
    results = get_predictions(model, test_loader, device)
    
    # Analyze predictions
    print("\nAnalyzing predictions...")
    df = analyze_predictions(results, save_path='prediction_analysis.csv')
    
    # Plot ROC curve
    print("\nPlotting ROC curve...")
    plot_roc_curve(results['labels'], results['probabilities'], save_path='roc_curve.png')
    
    # Plot PR curve
    print("\nPlotting Precision-Recall curve...")
    plot_precision_recall_curve(results['labels'], results['probabilities'], save_path='pr_curve.png')
    
    # Visualize embeddings
    print("\nVisualizing embeddings...")
    visualize_graph_embeddings(model, test_loader, device, save_path='embeddings.png')
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()