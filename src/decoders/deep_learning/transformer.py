"""
Transformer decoder for neural signals.

Implements Transformer architecture with self-attention for capturing
long-range temporal dependencies in neural-kinematic relationships.

Reference:
    Ye & Bhagavatula (2021) "Neural Decoding with Transformer Networks"
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.decoders.base import OnlineDecoder

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


# Only define PyTorch classes when torch is available
if TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for sequence position information."""

        def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            # Create positional encoding matrix
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)

            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Add positional encoding to input."""
            x = x + self.pe[:, : x.size(1), :]
            return self.dropout(x)


class TransformerDecoder(OnlineDecoder):
    """
    Transformer decoder for continuous neural decoding.

    Uses self-attention mechanism to capture long-range temporal
    dependencies in neural activity patterns.

    Advantages over LSTM:
    - Parallel computation (faster training)
    - Better long-range dependency modeling
    - No vanishing gradient problem
    """

    def __init__(
        self,
        name: str = "Transformer",
        d_model: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 0.0001,
        batch_size: int = 32,
        n_epochs: int = 100,
        sequence_length: int = 20,
        device: Optional[str] = None,
        early_stopping_patience: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize Transformer decoder.

        Args:
            name: Decoder name.
            d_model: Dimension of the model (embedding size).
            n_heads: Number of attention heads.
            n_encoder_layers: Number of transformer encoder layers.
            dim_feedforward: Dimension of feedforward network.
            dropout: Dropout rate.
            learning_rate: Learning rate for Adam optimizer.
            batch_size: Training batch size.
            n_epochs: Maximum training epochs.
            sequence_length: Length of input sequences.
            device: PyTorch device ('cuda', 'cpu', or None for auto).
            early_stopping_patience: Epochs without improvement before stopping.
            verbose: Print training progress.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for TransformerDecoder. Install with: pip install torch"
            )

        super().__init__(name=name, learning_rate=learning_rate)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.sequence_length = sequence_length
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model = None
        self._optimizer = None
        self._train_losses: List[float] = []
        self._val_losses: List[float] = []

        # Normalization parameters
        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None
        self._y_mean: Optional[np.ndarray] = None
        self._y_std: Optional[np.ndarray] = None

        # Buffer for online prediction
        self._history_buffer: Optional[np.ndarray] = None

    def _build_model(self) -> nn.Module:
        """Build the Transformer network."""

        class TransformerNetwork(nn.Module):
            def __init__(
                inner_self,
                input_size: int,
                output_size: int,
                d_model: int,
                n_heads: int,
                n_encoder_layers: int,
                dim_feedforward: int,
                dropout: float,
                sequence_length: int,
            ):
                super().__init__()

                # Input projection
                inner_self.input_projection = nn.Linear(input_size, d_model)

                # Positional encoding
                inner_self.pos_encoder = PositionalEncoding(
                    d_model, max_len=sequence_length * 2, dropout=dropout
                )

                # Transformer encoder layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                )
                inner_self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=n_encoder_layers
                )

                # Output projection
                inner_self.output_projection = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, output_size),
                )

                inner_self.d_model = d_model

            def forward(inner_self, x: torch.Tensor) -> torch.Tensor:
                # x shape: (batch, seq_len, input_size)

                # Project input to d_model dimensions
                x = inner_self.input_projection(x) * math.sqrt(inner_self.d_model)

                # Add positional encoding
                x = inner_self.pos_encoder(x)

                # Create causal mask for autoregressive decoding
                seq_len = x.size(1)
                mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

                # Transformer encoding
                x = inner_self.transformer_encoder(x, mask=mask)

                # Output projection (for each time step)
                output = inner_self.output_projection(x)

                return output

        return TransformerNetwork(
            input_size=self.n_features,
            output_size=self.n_outputs,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_encoder_layers=self.n_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            sequence_length=self.sequence_length,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TransformerDecoder":
        """
        Fit Transformer decoder on training data.

        Args:
            X: Neural features of shape (n_samples, n_features).
            y: Kinematics of shape (n_samples, n_outputs).

        Returns:
            self: Fitted decoder.
        """
        self._validate_input(X, y)

        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1]

        # Normalize data
        self._X_mean = np.mean(X, axis=0)
        self._X_std = np.std(X, axis=0)
        self._X_std[self._X_std == 0] = 1.0

        self._y_mean = np.mean(y, axis=0)
        self._y_std = np.std(y, axis=0)
        self._y_std[self._y_std == 0] = 1.0

        X_norm = (X - self._X_mean) / self._X_std
        y_norm = (y - self._y_mean) / self._y_std

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_norm, y_norm)

        # Split for validation
        val_split = int(0.9 * len(X_seq))
        X_train, X_val = X_seq[:val_split], X_seq[val_split:]
        y_train, y_val = y_seq[:val_split], y_seq[val_split:]

        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Initialize model
        self._model = self._build_model().to(self.device)

        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=self.n_epochs)

        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.n_epochs):
            # Training
            self._model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self._optimizer.zero_grad()
                output = self._model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                self._optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self._train_losses.append(train_loss)

            # Validation
            self._model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    output = self._model(X_batch)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            self._val_losses.append(val_loss)

            scheduler.step()

            if self.verbose:
                print(
                    f"Epoch {epoch + 1}/{self.n_epochs}: "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Restore best model
        if best_state is not None:
            self._model.load_state_dict(best_state)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Decode kinematics from neural features.

        Args:
            X: Neural features of shape (n_samples, n_features).

        Returns:
            Decoded kinematics of shape (n_samples - seq_len + 1, n_outputs).
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before prediction.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Normalize
        X_norm = (X - self._X_mean) / self._X_std

        # Create sequences
        X_seq = self._create_sequences_predict(X_norm)

        # Predict
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            output = self._model(X_tensor)
            # Take the last time step prediction from each sequence
            predictions = output[:, -1, :].cpu().numpy()

        # Denormalize
        predictions = predictions * self._y_std + self._y_mean

        return predictions

    def predict_single(self, x: np.ndarray) -> np.ndarray:
        """
        Predict single time step for real-time use.

        Maintains a history buffer for context.

        Args:
            x: Neural features for single time step (n_features,).

        Returns:
            Predicted kinematics (n_outputs,).
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before prediction.")

        # Initialize buffer if needed
        if self._history_buffer is None:
            self._history_buffer = np.zeros((self.sequence_length, self.n_features))

        # Shift buffer and add new sample
        self._history_buffer = np.roll(self._history_buffer, -1, axis=0)
        self._history_buffer[-1] = x

        # Normalize
        X_norm = (self._history_buffer - self._X_mean) / self._X_std

        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_norm).unsqueeze(0).to(self.device)
            output = self._model(X_tensor)
            prediction = output[0, -1, :].cpu().numpy()

        # Denormalize
        prediction = prediction * self._y_std + self._y_mean

        return prediction

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Online update with new data.

        Args:
            X: New neural features.
            y: New kinematics.
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before update.")

        # Normalize
        X_norm = (X - self._X_mean) / self._X_std
        y_norm = (y - self._y_mean) / self._y_std

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_norm, y_norm)

        if len(X_seq) == 0:
            return

        # Quick update
        self._model.train()
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        # Few gradient steps
        for _ in range(3):
            self._optimizer.zero_grad()
            output = self._model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optimizer.step()

        self._update_count += 1

    def reset_history(self) -> None:
        """Reset history buffer (for new trial/session)."""
        self._history_buffer = None

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training."""
        n_samples = X.shape[0]
        n_sequences = n_samples - self.sequence_length + 1

        if n_sequences <= 0:
            return np.array([]), np.array([])

        X_seq = np.zeros((n_sequences, self.sequence_length, self.n_features))
        y_seq = np.zeros((n_sequences, self.sequence_length, self.n_outputs))

        for i in range(n_sequences):
            X_seq[i] = X[i : i + self.sequence_length]
            y_seq[i] = y[i : i + self.sequence_length]

        return X_seq, y_seq

    def _create_sequences_predict(self, X: np.ndarray) -> np.ndarray:
        """Create sequences for prediction."""
        n_samples = X.shape[0]
        n_sequences = n_samples - self.sequence_length + 1

        if n_sequences <= 0:
            # If not enough samples, pad with zeros
            X_padded = np.zeros((self.sequence_length, self.n_features))
            X_padded[-n_samples:] = X
            return X_padded.reshape(1, self.sequence_length, -1)

        X_seq = np.zeros((n_sequences, self.sequence_length, self.n_features))

        for i in range(n_sequences):
            X_seq[i] = X[i : i + self.sequence_length]

        return X_seq

    def get_params(self) -> Dict[str, Any]:
        """Get decoder parameters."""
        params = super().get_params()
        params.update(
            {
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_encoder_layers": self.n_encoder_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
                "sequence_length": self.sequence_length,
                "device": str(self.device) if self.device else None,
                "train_losses": self._train_losses[-10:] if self._train_losses else [],
            }
        )
        return params
