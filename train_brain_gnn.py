"""
GNN Classifier with Class Imbalance Handling
Multiple strategies to handle imbalanced datasets
Configuration loaded from YAML file
Includes comprehensive logging to run.log in the run directory
"""

import os
import numpy as np
import pandas as pd
import time
import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, SAGEConv
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml
import argparse
import warnings
warnings.filterwarnings('ignore')


class BrainConnectivityDataset(Dataset):
    """
    Dataset for brain connectivity graphs with contrast computation
    """
    def __init__(self, data_dir, labels_csv, contrast_type='reward', transform=None, 
                 threshold=0.0, use_absolute=False, logger=None):
        super().__init__()
        
        self.data_dir = data_dir
        self.contrast_type = contrast_type
        self.threshold = threshold
        self.use_absolute = use_absolute
        self.transform = transform
        self.logger = logger or logging.getLogger(__name__)
        
        # Load labels
        self.labels_df = pd.read_csv(labels_csv)
        self.logger.info(f"Loaded {len(self.labels_df)} subjects from labels file")
        
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
        self.logger.info(f"Found {len(self.valid_subjects)} subjects with complete data")
        
        # Class distribution
        label_counts = self.labels_df[self.labels_df['subject_id'].isin(self.valid_subjects)]['Irr_PL_PH'].value_counts()
        self.logger.info(f"Class distribution:")
        self.logger.info(f"  Class 0: {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(self.valid_subjects)*100:.1f}%)")
        self.logger.info(f"  Class 1: {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(self.valid_subjects)*100:.1f}%)")
        self.logger.info(f"  Imbalance ratio: {label_counts.get(0, 0) / max(label_counts.get(1, 1), 1):.2f}:1")
    
    def _find_valid_subjects(self):
        """Find subjects that have both neutral and contrast files"""
        valid_subjects = []
        
        for _, row in self.labels_df.iterrows():
            subject_id = row['subject_id']
            
            # Check if files exist
            neutral_file = os.path.join(self.neutral_dir, f"sub-{subject_id}.npy")
            contrast_file = os.path.join(self.contrast_dir, f"sub-{subject_id}.npy")
            
            if os.path.exists(neutral_file) and os.path.exists(contrast_file):
                valid_subjects.append(subject_id)
        
        return valid_subjects
    
    def len(self):
        return len(self.valid_subjects)
    
    def get(self, idx):
        """Load and process a single graph"""
        subject_id = self.valid_subjects[idx]
        
        try:
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
            
        except Exception as e:
            self.logger.exception(f"Error loading data for subject {subject_id}: {str(e)}")
            raise
    
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
    
    def get_class_weights(self):
        """Compute class weights for balanced training"""
        labels = [self.labels_df[self.labels_df['subject_id'] == sid]['Irr_PL_PH'].values[0] 
                  for sid in self.valid_subjects]
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        return torch.FloatTensor(class_weights)


class BrainGNN(nn.Module):
    """
    Graph Neural Network for brain connectivity classification
    """
    def __init__(self, num_node_features, hidden_channels=64, num_layers=3, 
                 dropout=0.5, gnn_type='GCN', pooling='mean'):
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


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GNNTrainer:
    """
    Trainer class for GNN models with class imbalance handling
    """
    def __init__(self, model, device='cuda', learning_rate=0.001, weight_decay=5e-4,
                 class_weights=None, use_focal_loss=False, focal_gamma=2.0, logger=None):
        self.model = model.to(device)
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Choose loss function
        if use_focal_loss:
            self.logger.info(f"Using Focal Loss with gamma={focal_gamma}")
            if class_weights is not None:
                class_weights = class_weights.to(device)
            self.criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            if class_weights is not None:
                self.logger.info(f"Using weighted CrossEntropy with weights: {class_weights.cpu().numpy()}")
                class_weights = class_weights.to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Track training history
        self.train_losses = []
        self.train_accs = []
        self.train_balanced_accs = []
        self.val_losses = []
        self.val_accs = []
        self.val_balanced_accs = []
    
    def train_epoch(self, loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        try:
            for data in loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                
                out = self.model(data)
                loss = self.criterion(out, data.y)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * data.num_graphs
                
                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
            
            avg_loss = total_loss / len(loader.dataset)
            accuracy = accuracy_score(all_labels, all_preds)
            balanced_acc = balanced_accuracy_score(all_labels, all_preds)
            
            return avg_loss, accuracy, balanced_acc
            
        except Exception as e:
            self.logger.exception(f"Error during training epoch: {str(e)}")
            raise
    
    @torch.no_grad()
    def evaluate(self, loader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        try:
            for data in loader:
                data = data.to(self.device)
                out = self.model(data)
                loss = self.criterion(out, data.y)
                
                total_loss += loss.item() * data.num_graphs
                
                pred = out.argmax(dim=1)
                probs = F.softmax(out, dim=1)[:, 1]
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
            
            avg_loss = total_loss / len(loader.dataset)
            accuracy = accuracy_score(all_labels, all_preds)
            balanced_acc = balanced_accuracy_score(all_labels, all_preds)
            
            # Additional metrics
            f1 = f1_score(all_labels, all_preds)
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except:
                auc = 0.0
            
            return avg_loss, accuracy, balanced_acc, f1, auc, all_preds, all_labels
            
        except Exception as e:
            self.logger.exception(f"Error during evaluation: {str(e)}")
            raise
    
    def fit(self, train_loader, val_loader, epochs=100, patience=20, verbose=True, 
            monitor='balanced_acc'):
        """Train the model with early stopping"""
        best_val_metric = 0
        best_model_state = None
        patience_counter = 0
        
        self.logger.info(f"Starting training for up to {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            try:
                # Train
                train_loss, train_acc, train_balanced_acc = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)
                self.train_balanced_accs.append(train_balanced_acc)
                
                # Validate
                val_loss, val_acc, val_balanced_acc, val_f1, val_auc, _, _ = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                self.val_balanced_accs.append(val_balanced_acc)
                
                # Choose metric to monitor
                if monitor == 'balanced_acc':
                    val_metric = val_balanced_acc
                elif monitor == 'f1':
                    val_metric = val_f1
                elif monitor == 'auc':
                    val_metric = val_auc
                else:
                    val_metric = val_acc
                
                if verbose and epoch % 10 == 0:
                    self.logger.info(
                        f"Epoch {epoch:03d}: "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Balanced Acc: {train_balanced_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Balanced Acc: {val_balanced_acc:.4f}, "
                        f"Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}"
                    )
                
                # Early stopping
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                    self.logger.info(f"Epoch {epoch}: New best {monitor}: {val_metric:.4f}")
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
            except Exception as e:
                self.logger.exception(f"Error in epoch {epoch}: {str(e)}")
                raise
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        self.logger.info(f"Training completed. Best {monitor}: {best_val_metric:.4f}")
        
        return best_val_metric


class ImbalancedSampler(torch.utils.data.Sampler):
    """
    Sampler that oversamples minority class
    """
    def __init__(self, dataset, indices=None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        
        # Get labels
        self.labels = [dataset.get(i).y.item() for i in self.indices]
        
        # Count samples per class
        label_counts = np.bincount(self.labels)
        
        # Compute weights (inverse of class frequency)
        weights = 1. / label_counts[self.labels]
        self.weights = torch.DoubleTensor(weights)
        
        # Number of samples to draw
        self.num_samples = len(self.indices)
    
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
    
    def __len__(self):
        return self.num_samples


def plot_training_history(trainer, save_path='training_history.png', logger=None):
    """Plot training and validation curves"""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
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
        
        # Balanced Accuracy plot
        axes[2].plot(trainer.train_balanced_accs, label='Train Balanced Acc', linewidth=2)
        axes[2].plot(trainer.val_balanced_accs, label='Val Balanced Acc', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Balanced Accuracy', fontsize=12)
        axes[2].set_title('Training and Validation Balanced Accuracy', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history to {save_path}")
        plt.close()
        
    except Exception as e:
        logger.exception(f"Error plotting training history: {str(e)}")
        raise


def plot_confusion_matrix(labels, preds, save_path='confusion_matrix.png', logger=None):
    """Plot confusion matrix"""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    try:
        cm = confusion_matrix(labels, preds)
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'], ax=axes[0])
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        
        # Normalized
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', cbar=True,
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'], ax=axes[1])
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
        plt.close()
        
    except Exception as e:
        logger.exception(f"Error plotting confusion matrix: {str(e)}")
        raise


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(run_dir):
    """Setup logging to file and console"""
    log_file = os.path.join(run_dir, 'run.log')
    
    # Create logger
    logger = logging.getLogger('BrainGNN')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def main(config_path='configs/config.yaml'):
    """Main training pipeline with class imbalance handling"""
    
    # Create run directory first (needed for logging)
    ts = time.time()
    run_name = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.expanduser(f"~/honors_thesis/runs/{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(run_dir)
    
    try:
        # Load configuration
        logger.info("="*50)
        logger.info(f"Loading configuration from {config_path}")
        logger.info("="*50)
        config = load_config(config_path)
        logger.info(f"Configuration loaded successfully")
        
        # Extract configuration sections
        data_config = config['data']
        model_config = config['model']
        train_config = config['training']
        imbalance_config = config['imbalance']
        output_config = config['output']
        device_config = config['device']
        
        # Unpack data config
        DATA_DIR = data_config['data_dir']
        LABELS_CSV = data_config['labels_csv']
        CONTRAST_TYPE = data_config['contrast_type']
        THRESHOLD = data_config['threshold']
        USE_ABSOLUTE = data_config['use_absolute']
        
        # Unpack model config
        GNN_TYPE = model_config['gnn_type']
        HIDDEN_CHANNELS = model_config['hidden_channels']
        NUM_LAYERS = model_config['num_layers']
        DROPOUT = model_config['dropout']
        
        # Unpack training config
        BATCH_SIZE = train_config['batch_size']
        LEARNING_RATE = train_config['learning_rate']
        WEIGHT_DECAY = train_config['weight_decay']
        EPOCHS = train_config['epochs']
        PATIENCE = train_config['patience']
        TEST_SIZE = train_config['test_size']
        VAL_SIZE = train_config['val_size']
        RANDOM_SEED = train_config['random_seed']
        
        # Unpack imbalance config
        USE_CLASS_WEIGHTS = imbalance_config['use_class_weights']
        USE_FOCAL_LOSS = imbalance_config['use_focal_loss']
        FOCAL_GAMMA = imbalance_config['focal_gamma']
        USE_OVERSAMPLING = imbalance_config['use_oversampling']
        MONITOR_METRIC = imbalance_config['monitor_metric']
        
        # Unpack output config
        SAVE_MODEL = output_config['save_model']
        MODEL_FILENAME = output_config['model_filename'].format(contrast_type=CONTRAST_TYPE)
        TRAINING_HISTORY_PLOT = output_config['training_history_plot'].format(contrast_type=CONTRAST_TYPE)
        CONFUSION_MATRIX_PLOT = output_config['confusion_matrix_plot'].format(contrast_type=CONTRAST_TYPE)
        VERBOSE = output_config['verbose']
        
        # Device setup
        if device_config['use_cuda'] and torch.cuda.is_available():
            device = torch.device(f"cuda:{device_config['cuda_device']}")
        else:
            device = torch.device('cpu')
        logger.info(f"Using device: {device}")
        
        # Print configuration summary
        logger.info("\n" + "="*50)
        logger.info("Configuration Summary")
        logger.info("="*50)
        logger.info(f"Run Directory: {run_dir}")
        logger.info(f"Data:")
        logger.info(f"  Data Directory: {DATA_DIR}")
        logger.info(f"  Labels CSV: {LABELS_CSV}")
        logger.info(f"  Contrast Type: {CONTRAST_TYPE}")
        logger.info(f"  Edge Threshold: {THRESHOLD}")
        logger.info(f"  Use Absolute Values: {USE_ABSOLUTE}")
        logger.info(f"\nModel:")
        logger.info(f"  GNN Type: {GNN_TYPE}")
        logger.info(f"  Hidden Channels: {HIDDEN_CHANNELS}")
        logger.info(f"  Num Layers: {NUM_LAYERS}")
        logger.info(f"  Dropout: {DROPOUT}")
        logger.info(f"\nTraining:")
        logger.info(f"  Batch Size: {BATCH_SIZE}")
        logger.info(f"  Learning Rate: {LEARNING_RATE}")
        logger.info(f"  Weight Decay: {WEIGHT_DECAY}")
        logger.info(f"  Epochs: {EPOCHS}")
        logger.info(f"  Patience: {PATIENCE}")
        logger.info(f"  Random Seed: {RANDOM_SEED}")
        logger.info(f"\nImbalance Handling:")
        logger.info(f"  Class Weights: {USE_CLASS_WEIGHTS}")
        logger.info(f"  Focal Loss: {USE_FOCAL_LOSS}")
        if USE_FOCAL_LOSS:
            logger.info(f"  Focal Gamma: {FOCAL_GAMMA}")
        logger.info(f"  Oversampling: {USE_OVERSAMPLING}")
        logger.info(f"  Monitor Metric: {MONITOR_METRIC}")
        
        # Load dataset
        logger.info("\n" + "="*50)
        logger.info(f"Loading {CONTRAST_TYPE} contrast dataset...")
        logger.info("="*50)
        
        dataset = BrainConnectivityDataset(
            data_dir=DATA_DIR,
            labels_csv=LABELS_CSV,
            contrast_type=CONTRAST_TYPE,
            threshold=THRESHOLD,
            use_absolute=USE_ABSOLUTE,
            logger=logger
        )
        
        if len(dataset) == 0:
            logger.error("No valid subjects found! Check your data paths.")
            logger.error(f"Data directory: {DATA_DIR}")
            logger.error(f"Labels CSV: {LABELS_CSV}")
            return
        
        logger.info(f"Dataset loaded successfully with {len(dataset)} subjects")
        
        # Get class weights
        class_weights = dataset.get_class_weights() if USE_CLASS_WEIGHTS or USE_FOCAL_LOSS else None
        if class_weights is not None:
            logger.info(f"Class weights: {class_weights.numpy()}")
        
        # Get a sample to determine feature dimensions
        sample = dataset[0]
        num_node_features = sample.x.shape[1]
        logger.info(f"Number of ROIs (nodes): {sample.x.shape[0]}")
        logger.info(f"Number of node features: {num_node_features}")
        logger.info(f"Number of edges: {sample.edge_index.shape[1]}")
        
        # Split dataset
        logger.info("Splitting dataset into train/val/test...")
        train_idx, test_idx = train_test_split(
            range(len(dataset)), 
            test_size=TEST_SIZE, 
            stratify=[dataset.get(i).y.item() for i in range(len(dataset))],
            random_state=RANDOM_SEED
        )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=VAL_SIZE,
            stratify=[dataset.get(i).y.item() for i in train_idx],
            random_state=RANDOM_SEED
        )
        
        logger.info(f"Dataset split:")
        logger.info(f"  Train: {len(train_idx)} samples")
        logger.info(f"  Val:   {len(val_idx)} samples")
        logger.info(f"  Test:  {len(test_idx)} samples")
        
        # Print class distribution in each split
        for name, indices in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
            labels = [dataset.get(i).y.item() for i in indices]
            counts = np.bincount(labels)
            logger.info(f"  {name} class dist: Class 0: {counts[0]}, Class 1: {counts[1]} "
                  f"({counts[1]/(counts[0]+counts[1])*100:.1f}% minority)")
        
        # Create data loaders
        if USE_OVERSAMPLING:
            logger.info("Using oversampling for training set")
            train_sampler = ImbalancedSampler(dataset, train_idx)
            # Important: Don't subset the dataset again when using a sampler
            train_loader = DataLoader(
                dataset,  # Use full dataset
                batch_size=BATCH_SIZE,
                sampler=train_sampler  # Sampler handles the subset
            )
        else:
            train_loader = DataLoader([dataset[i] for i in train_idx], 
                                       batch_size=BATCH_SIZE, shuffle=True)
        
        val_loader = DataLoader([dataset[i] for i in val_idx], batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=BATCH_SIZE, shuffle=False)
        
        # Create model
        logger.info(f"\nInitializing {GNN_TYPE} model...")
        model = BrainGNN(
            num_node_features=num_node_features,
            hidden_channels=HIDDEN_CHANNELS,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            gnn_type=GNN_TYPE
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params:,}")
        
        # Train model
        logger.info("\n" + "="*50)
        logger.info("Training model...")
        logger.info(f"Monitoring: {MONITOR_METRIC}")
        logger.info("="*50)
        
        trainer = GNNTrainer(
            model, 
            device=device, 
            learning_rate=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY,
            class_weights=class_weights,
            use_focal_loss=USE_FOCAL_LOSS,
            focal_gamma=FOCAL_GAMMA,
            logger=logger
        )
        
        best_val_metric = trainer.fit(
            train_loader, val_loader, 
            epochs=EPOCHS, 
            patience=PATIENCE, 
            monitor=MONITOR_METRIC,
            verbose=VERBOSE
        )
        
        logger.info(f"Best validation {MONITOR_METRIC}: {best_val_metric:.4f}")
        
        # Evaluate on test set
        logger.info("\n" + "="*50)
        logger.info("Evaluating on test set...")
        logger.info("="*50)
        
        test_loss, test_acc, test_balanced_acc, test_f1, test_auc, test_preds, test_labels = trainer.evaluate(test_loader)
        
        logger.info(f"\nTest Results:")
        logger.info(f"  Loss:              {test_loss:.4f}")
        logger.info(f"  Accuracy:          {test_acc:.4f}")
        logger.info(f"  Balanced Accuracy: {test_balanced_acc:.4f}")
        logger.info(f"  F1 Score:          {test_f1:.4f}")
        logger.info(f"  AUC:               {test_auc:.4f}")
        
        logger.info("\nClassification Report:")
        report = classification_report(test_labels, test_preds, target_names=['Class 0', 'Class 1'])
        logger.info("\n" + report)
        
        logger.info("\nPer-class metrics:")
        cm = confusion_matrix(test_labels, test_preds)
        logger.info(f"  Class 0 - Precision: {cm[0,0]/(cm[0,0]+cm[1,0]):.4f}, Recall: {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}")
        logger.info(f"  Class 1 - Precision: {cm[1,1]/(cm[1,1]+cm[0,1]):.4f}, Recall: {cm[1,1]/(cm[1,1]+cm[1,0]):.4f}")
        
        # Define file paths
        model_path = os.path.join(run_dir, MODEL_FILENAME)
        train_history_path = os.path.join(run_dir, TRAINING_HISTORY_PLOT)
        confusion_matrix_path = os.path.join(run_dir, CONFUSION_MATRIX_PLOT)
        results_filename = os.path.join(run_dir, f'results_{CONTRAST_TYPE}.yaml')
        
        # Plot results
        logger.info("\nSaving training visualizations...")
        plot_training_history(trainer, save_path=train_history_path, logger=logger)
        plot_confusion_matrix(test_labels, test_preds, save_path=confusion_matrix_path, logger=logger)
        
        # Save model
        if SAVE_MODEL:
            logger.info(f"Saving model to {model_path}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'num_node_features': num_node_features,
                'test_acc': test_acc,
                'test_balanced_acc': test_balanced_acc,
                'test_f1': test_f1,
                'test_auc': test_auc,
                'class_weights': class_weights.cpu() if class_weights is not None else None,
            }, model_path)
            logger.info(f"Model saved successfully")
        
        # Save results summary
        logger.info(f"Saving results summary to {results_filename}")
        results_summary = {
            'run_info': {
                'run_name': run_name,
                'run_dir': run_dir,
                'timestamp': ts,
            },
            'config': config,
            'results': {
                'test_loss': float(test_loss),
                'test_accuracy': float(test_acc),
                'test_balanced_accuracy': float(test_balanced_acc),
                'test_f1': float(test_f1),
                'test_auc': float(test_auc),
            },
            'confusion_matrix': cm.tolist(),
        }
        
        with open(results_filename, 'w') as f:
            yaml.dump(results_summary, f, default_flow_style=False)
        logger.info(f"Results summary saved successfully")
        
        logger.info("\n" + "="*50)
        logger.info("Training completed successfully!")
        logger.info(f"All outputs saved to: {run_dir}")
        logger.info("="*50)
        
    except Exception as e:
        logger.exception(f"An error occurred during training: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Brain GNN Classifier')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration YAML file (default: configs/config.yaml)')
    args = parser.parse_args()
    
    main(config_path=args.config)