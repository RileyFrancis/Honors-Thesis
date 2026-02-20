"""
Enhanced BrainConnectivityDataset with rich node features
This version computes many more features per node to give the model more signal
"""

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
import os
from sklearn.utils.class_weight import compute_class_weight
import logging


class EnhancedBrainConnectivityDataset(Dataset):
    """
    Dataset with enhanced node features for brain connectivity graphs
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
            
            # ENHANCED: Compute rich node features
            node_features = self._compute_enhanced_node_features(
                contrast_matrix, neutral_corr, contrast_corr
            )
            
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
    
    def _compute_enhanced_node_features(self, contrast_matrix, neutral_matrix, task_matrix):
        """
        Compute enhanced node features from connectivity matrices
        
        Returns features with ~20+ dimensions per node instead of just 6
        """
        features = []
        n_nodes = contrast_matrix.shape[0]
        
        # === CONTRAST MATRIX FEATURES (what changed) ===
        
        # Basic statistics
        features.append(np.mean(contrast_matrix, axis=1))      # Mean change
        features.append(np.std(contrast_matrix, axis=1))       # Variability of change
        features.append(np.median(contrast_matrix, axis=1))    # Median change
        features.append(np.max(contrast_matrix, axis=1))       # Max increase
        features.append(np.min(contrast_matrix, axis=1))       # Max decrease
        
        # Percentiles (robust to outliers)
        features.append(np.percentile(contrast_matrix, 25, axis=1))  # 25th percentile
        features.append(np.percentile(contrast_matrix, 75, axis=1))  # 75th percentile
        
        # Positive vs negative changes
        features.append(np.sum(contrast_matrix > 0, axis=1))   # Count of increases
        features.append(np.sum(contrast_matrix < 0, axis=1))   # Count of decreases


        pos_mask = contrast_matrix > 0
        neg_mask = contrast_matrix < 0

        pos_sum = np.sum(contrast_matrix * pos_mask, axis=1)
        pos_cnt = np.sum(pos_mask, axis=1)

        neg_sum = np.sum(np.abs(contrast_matrix) * neg_mask, axis=1)
        neg_cnt = np.sum(neg_mask, axis=1)

        mean_pos = np.divide(pos_sum, pos_cnt, out=np.zeros_like(pos_sum, dtype=float), where=pos_cnt > 0)
        mean_neg = np.divide(neg_sum, neg_cnt, out=np.zeros_like(neg_sum, dtype=float), where=neg_cnt > 0)

        features.append(mean_pos)  # Mean of positive changes per node
        features.append(mean_neg)  # Mean magnitude of negative changes per node

        
        # Magnitude features
        features.append(np.mean(np.abs(contrast_matrix), axis=1))  # Mean absolute change
        features.append(np.max(np.abs(contrast_matrix), axis=1))   # Max absolute change
        
        # === NEUTRAL (BASELINE) FEATURES ===
        
        # Baseline connectivity strength
        features.append(np.mean(neutral_matrix, axis=1))
        features.append(np.std(neutral_matrix, axis=1))
        features.append(np.mean(np.abs(neutral_matrix), axis=1))
        
        # === TASK (REWARD/LOSS) FEATURES ===
        
        # Task connectivity strength
        features.append(np.mean(task_matrix, axis=1))
        features.append(np.std(task_matrix, axis=1))
        features.append(np.mean(np.abs(task_matrix), axis=1))
        
        # === RATIO FEATURES (task vs neutral) ===
        
        # How much did connectivity change proportionally?
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.abs(task_matrix) / (np.abs(neutral_matrix) + 1e-8)
            features.append(np.mean(ratio, axis=1))
            features.append(np.std(ratio, axis=1))
        
        # === GRAPH-THEORETIC FEATURES ===
        
        # Degree (number of strong connections)
        strong_connections = np.sum(np.abs(contrast_matrix) > 0.1, axis=1)
        features.append(strong_connections)
        
        # "Hub-ness" - sum of absolute edge weights
        features.append(np.sum(np.abs(contrast_matrix), axis=1))
        
        # Variance in connection strength
        features.append(np.var(contrast_matrix, axis=1))
        
        # === ADDITIONAL STATISTICAL FEATURES ===
        
        # Skewness (asymmetry of distribution)
        from scipy.stats import skew
        features.append(skew(contrast_matrix, axis=1))
        
        # Kurtosis (tail heaviness)
        from scipy.stats import kurtosis
        features.append(kurtosis(contrast_matrix, axis=1))
        
        # Stack all features
        feature_matrix = np.column_stack(features)
        
        # Replace any NaN or Inf with 0
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)
        
        return feature_matrix
    
    def get_class_weights(self):
        """Compute class weights for balanced training"""
        labels = [self.labels_df[self.labels_df['subject_id'] == sid]['Irr_PL_PH'].values[0] 
                  for sid in self.valid_subjects]
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        return torch.FloatTensor(class_weights)