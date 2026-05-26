import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import glob

# Professional Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("HARTransformer")

class HARDataset(Dataset):
    """
    Custom Dataset for Human Activity Recognition.
    Handles windowed time-series data.
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based architecture for activity classification.
    Uses an embedding layer followed by a Transformer Encoder.
    """
    def __init__(self, input_dim: int, model_dim: int = 64, num_heads: int = 4, 
                 num_layers: int = 2, num_classes: int = 7):
        super(TimeSeriesTransformer, self).__init__()
        
        # Linear embedding to map input features to model dimension
        self.embedding = nn.Linear(input_dim, model_dim)
        
        # Transformer Encoder Layer
        # batch_first=True ensures input shape is (batch, seq, feature)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Head
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, input_dim)
        x = self.embedding(x)             # (batch, seq, model_dim)
        x = self.transformer_encoder(x) # (batch, seq, model_dim)
        
        # Global Average Pooling across the sequence dimension
        x = x.mean(dim=1)               # (batch, model_dim)
        return self.classifier(x)       # (batch, num_classes)

class HARPipeline:
    """
    End-to-end pipeline for data loading, preprocessing, training, and evaluation.
    """
    def __init__(self, window_size: int = 100, step: int = 50):
        self.window_size = window_size
        self.step = step
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def load_and_preprocess(self, data_path: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Loads multiple CSVs, segments them into windows, and returns tensors.
        """
        logger.info(f"Loading data from {data_path}...")
        all_files = glob.glob(os.path.join(data_path, "*.csv"))
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")

        df_list = [pd.read_csv(f) for f in all_files]
        df = pd.concat(df_list, ignore_index=True).dropna()

        # Feature extraction
        cols_to_drop = ['label', 'timestamp'] if 'timestamp' in df.columns else ['label']
        features = df.drop(cols_to_drop, axis=1).values
        labels = self.label_encoder.fit_transform(df['label'].values)
        
        # Standardize
        features = self.scaler.fit_transform(features)
        
        # Create sliding windows
        segments, segment_labels = self._create_segments(features, labels)
        
        X = torch.tensor(segments, dtype=torch.float32)
        y = torch.tensor(segment_labels, dtype=torch.long)
        
        return X, y, list(self.label_encoder.classes_)

    def _create_segments(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        segments, segment_labels = [], []
        for i in range(0, len(data) - self.window_size, self.step):
            segments.append(data[i : i + self.window_size])
            segment_labels.append(labels[i + self.window_size - 1])
        return np.array(segments), np.array(segment_labels)

    def train(self, model: nn.Module, train_loader: DataLoader, epochs: int = 10, lr: float = 0.001):
        """Training loop with basic logging."""
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    def evaluate(self, model: nn.Module, test_loader: DataLoader, class_names: List[str]):
        """Evaluates the model and prints a professional classification report."""
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        logger.info(f"Test Accuracy: {acc:.4f}")
        print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))
        print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
