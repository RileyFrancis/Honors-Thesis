"""
GNN Classifier for Brain Connectivity Data
Trains on reward/loss contrasts from fMRI correlation matrices
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, SAGEConv
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class BrainConnectivityDataset(Dataset):
    """
    Dataset for brain connectivity graphs with contrast computation
    """
    def __init__(self, data_dir, labels_csv, contrast_type='reward', transform=None, 
                 threshold=0.0, use_absolute=False):
        """
        Parameters:
        -----------
        data_dir : str
            Base directory containing subdirectories with .npy files
        labels_csv : str
            Path to CSV file with subject_id and Irr_PL_PH columns
        contrast_type : str, default='reward'
            Either 'reward' or 'loss' - determines which contrast to compute
        transform : callable, optional
            Optional transform to apply to data
        threshold : float, default=0.0
            Threshold for edge weights (edges below this are removed)
        use_absolute : bool, default=False
            Whether to use absolute values of correlations
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.contrast_type = contrast_type
        self.threshold = threshold
        self.use_absolute = use_absolute
        self.transform = transform
        
        # Load labels
        self.labels_df = pd.read_csv(labels_csv)
        print(f"Loaded {len(self.labels_df)} subjects from labels file")
        
        # Define subdirectories
        self.neutral_dir = os.path.join(data_dir, 'MID_Cue_Neu_full_corr')
        if contrast_type == 'reward':
            self.contrast_dir = os.path.join(data_dir, 'MID_Cue_Rew_full_corr')
        elif contrast_type == 'loss':
            self.contrast_dir = os.path.join(data_dir, 'MID_Cue_Los_full_corr')
        else:
            raise ValueError("contrast_type must be 'reward' or 'loss'")
        
        # Find matching subjects
        self.valid_subjects = self._find_valid_subjects()
        print(f"Found {len(self.valid_subjects)} subjects with complete data")
        
        # Class distribution
        label_counts = self.labels_df[self.labels_df['subject_id'].isin(self.valid_subjects)]['Irr_PL_PH'].value_counts()
        print(f"Class distribution: {dict(label_counts)}")
    
    def _find_valid_subjects(self):
        """Find subjects that have both neutral and contrast files"""
        valid_subjects = []
        
        for _, row in self.labels_df.iterrows():
            subject_id = row['subject_id']
            
            # Check if files exist
            neutral_file = os.path.join(self.neutral_dir, f"sub-{subject_id}.npy")
            contrast_file = os.path.join(self.contrast_dir, f"sub-{subject_id}.npy")

            # print(neutral_file, contrast_file)
            
            if os.path.exists(neutral_file) and os.path.exists(contrast_file):
                valid_subjects.append(subject_id)
        
        return valid_subjects
    
    def len(self):
        return len(self.valid_subjects)
    
    def get(self, idx):
        """Load and process a single graph"""
        subject_id = self.valid_subjects[idx]
        
        # Load correlation matrices
        neutral_corr = np.load(os.path.join(self.neutral_dir, f"sub-{subject_id}.npy"))
        contrast_corr = np.load(os.path.join(self.contrast_dir, f"sub-{subject_id}.npy"))
        
        # Compute contrast (task - neutral)
        contrast_matrix = contrast_corr - neutral_corr
        
        # Handle NaN and Inf values
        contrast_matrix = np.nan_to_num(contrast_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply absolute value if requested
        if self.use_absolute:
            contrast_matrix = np.abs(contrast_matrix)
        
        # Remove diagonal (self-connections)
        np.fill_diagonal(contrast_matrix, 0)
        
        # Apply threshold and create edge index and edge attributes
        edge_index, edge_attr = self._matrix_to_edge_index(contrast_matrix)
        
        # Node features: use the row-wise statistics of the contrast matrix
        node_features = self._compute_node_features(contrast_matrix)
        
        # Get label
        label = self.labels_df[self.labels_df['subject_id'] == subject_id]['Irr_PL_PH'].values[0]
        
        # Create PyG Data object
        data = Data(
            x=torch.FloatTensor(node_features),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr).view(-1, 1),
            y=torch.LongTensor([label]),
            subject_id=subject_id
        )
        
        if self.transform:
            data = self.transform(data)
        
        return data
    
    def _matrix_to_edge_index(self, matrix):
        """Convert correlation matrix to edge_index and edge_attr"""
        # Apply threshold
        mask = np.abs(matrix) > self.threshold
        
        # Get edges
        edge_index = np.array(np.where(mask))
        edge_attr = matrix[mask]
        
        return edge_index, edge_attr
    
    def _compute_node_features(self, matrix):
        """Compute node features from connectivity matrix"""
        # Multiple features per node
        features = []
        
        # Mean connectivity
        features.append(np.mean(matrix, axis=1))
        
        # Std of connectivity
        features.append(np.std(matrix, axis=1))
        
        # Max connectivity
        features.append(np.max(matrix, axis=1))
        
        # Min connectivity
        features.append(np.min(matrix, axis=1))
        
        # Positive and negative connectivity
        features.append(np.sum(matrix > 0, axis=1))
        features.append(np.sum(matrix < 0, axis=1))
        
        return np.column_stack(features)


class BrainGNN(nn.Module):
    """
    Graph Neural Network for brain connectivity classification
    """
    def __init__(self, num_node_features, hidden_channels=64, num_layers=3, 
                 dropout=0.5, gnn_type='GCN', pooling='mean'):
        """
        Parameters:
        -----------
        num_node_features : int
            Number of input features per node
        hidden_channels : int, default=64
            Number of hidden channels
        num_layers : int, default=3
            Number of GNN layers
        dropout : float, default=0.5
            Dropout rate
        gnn_type : str, default='GCN'
            Type of GNN layer ('GCN', 'GAT', 'SAGE')
        pooling : str, default='mean'
            Graph pooling method
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        
        # Choose GNN layer type
        if gnn_type == 'GCN':
            GNNLayer = GCNConv
        elif gnn_type == 'GAT':
            GNNLayer = lambda in_ch, out_ch: GATConv(in_ch, out_ch, heads=4, concat=False)
        elif gnn_type == 'SAGE':
            GNNLayer = SAGEConv
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Input layer
        self.conv1 = GNNLayer(num_node_features, hidden_channels)
        
        # Hidden layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GNNLayer(hidden_channels, hidden_channels))
        
        # Batch normalization layers
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # MLP classifier
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 2)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First layer
        x = self.conv1(x, edge_index)
        x = self.bns[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i + 1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph pooling
        x = global_mean_pool(x, batch)
        
        # MLP classifier
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return x


class GNNTrainer:
    """
    Trainer class for GNN models
    """
    def __init__(self, model, device='cuda', learning_rate=0.001, weight_decay=5e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Track training history
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
    
    def train_epoch(self, loader, epoch=None, epochs=None):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(
            loader,
            desc=f"Epoch [{epoch}/{epochs}] Train" if epoch is not None else "Train",
            leave=False
        )

        for data in pbar:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(data)
            loss = self.criterion(out, data.y)

            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss * data.num_graphs

            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

            # Update progress bar
            pbar.set_postfix(loss=f"{batch_loss:.4f}")

        avg_loss = total_loss / len(loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    
    @torch.no_grad()
    def evaluate(self, loader, epoch=None, epochs=None):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(
            loader,
            desc=f"Epoch [{epoch}/{epochs}] Val" if epoch is not None else "Val",
            leave=False
        )

        for data in pbar:
            data = data.to(self.device)
            out = self.model(data)
            loss = self.criterion(out, data.y)

            batch_loss = loss.item()
            total_loss += batch_loss * data.num_graphs

            pred = out.argmax(dim=1)
            probs = F.softmax(out, dim=1)[:, 1]

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            pbar.set_postfix(loss=f"{batch_loss:.4f}")

        avg_loss = total_loss / len(loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0

        return avg_loss, accuracy, f1, auc, all_preds, all_labels

    
    def fit(self, train_loader, val_loader, epochs=100, patience=20, verbose=True):
        """Train the model with early stopping"""
        best_val_acc = 0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, val_f1, val_auc, _, _ = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        
        return best_val_acc


def plot_training_history(trainer, save_path='training_history.png'):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(trainer.train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(trainer.val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(trainer.train_accs, label='Train Accuracy', linewidth=2)
    axes[1].plot(trainer.val_accs, label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training history to {save_path}")


def plot_confusion_matrix(labels, preds, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {save_path}")


def main():
    """Main training pipeline"""
    
    # Configuration
    DATA_DIR = '/shared/healthinfolab/datasets/ABCD/Irritability/WholeBrainTaskfMRI/UConn/data'
    LABELS_CSV = '/shared/healthinfolab/datasets/ABCD/Irritability/WholeBrainTaskfMRI/UConn/subject_lists/CBCL_PLPH_m0m24_MID_n3705.csv'
    
    # Hyperparameters
    CONTRAST_TYPE = 'reward'  # 'reward' or 'loss'
    BATCH_SIZE = 32
    HIDDEN_CHANNELS = 64
    NUM_LAYERS = 3
    DROPOUT = 0.5
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 5e-4
    EPOCHS = 200
    PATIENCE = 30
    THRESHOLD = 0.0
    GNN_TYPE = 'GCN'  # 'GCN', 'GAT', or 'SAGE'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("\n" + "="*50)
    print(f"Loading {CONTRAST_TYPE} contrast dataset...")
    print("="*50)
    dataset = BrainConnectivityDataset(
        data_dir=DATA_DIR,
        labels_csv=LABELS_CSV,
        contrast_type=CONTRAST_TYPE,
        threshold=THRESHOLD,
        use_absolute=False
    )
    
    if len(dataset) == 0:
        print("ERROR: No valid subjects found! Check your data paths.")
        return
    
    # Get a sample to determine feature dimensions
    sample = dataset[0]
    num_node_features = sample.x.shape[1]
    print(f"\nNumber of ROIs (nodes): {sample.x.shape[0]}")
    print(f"Number of node features: {num_node_features}")
    print(f"Number of edges: {sample.edge_index.shape[1]}")
    
    # Split dataset
    # Precompute labels once (FAST)
    subject_to_label = dict(
        zip(dataset.labels_df['subject_id'], dataset.labels_df['Irr_PL_PH'])
    )

    labels = np.array([subject_to_label[sid] for sid in dataset.valid_subjects])
    indices = np.arange(len(labels))

    # Train / test split
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # Train / val split
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.2,
        stratify=labels[train_idx],
        random_state=42
    )

    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")
    
    # Create data loaders
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    print(f"\nInitializing {GNN_TYPE} model...")
    model = BrainGNN(
        num_node_features=num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        gnn_type=GNN_TYPE
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n" + "="*50)
    print("Training model...")
    print("="*50)

    trainer = GNNTrainer(model, device=device, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_val_acc = trainer.fit(train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE)
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50)
    test_loss, test_acc, test_f1, test_auc, test_preds, test_labels = trainer.evaluate(test_loader)
    
    print(f"\nTest Results:")
    print(f"  Loss:     {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  AUC:      {test_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Class 0', 'Class 1']))
    
    # Plot results
    plot_training_history(trainer, save_path=f'training_history_{CONTRAST_TYPE}.png')
    plot_confusion_matrix(test_labels, test_preds, save_path=f'confusion_matrix_{CONTRAST_TYPE}.png')
    
    # Save model
    model_path = f'brain_gnn_{CONTRAST_TYPE}_best.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_node_features': num_node_features,
        'hidden_channels': HIDDEN_CHANNELS,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'gnn_type': GNN_TYPE,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_auc': test_auc
    }, model_path)
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()