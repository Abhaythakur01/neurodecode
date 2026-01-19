"""
LSTM decoder for neural signals.

Implements Long Short-Term Memory networks for capturing temporal
dynamics in neural-kinematic relationships.

Reference:
    Sussillo et al. (2012) "A recurrent neural network for closed-loop
    intracortical brain-machine interface decoders" J Neural Eng
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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

if TYPE_CHECKING:
    import torch.nn as nn


class LSTMDecoder(OnlineDecoder):
    """
    LSTM decoder for continuous neural decoding.

    Uses a multi-layer LSTM network to capture temporal dependencies
    in neural activity for predicting kinematics.
    """

    def __init__(
        self,
        name: str = "LSTM",
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        n_epochs: int = 100,
        sequence_length: int = 20,
        bidirectional: bool = False,
        device: Optional[str] = None,
        early_stopping_patience: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize LSTM decoder.

        Args:
            name: Decoder name.
            hidden_size: Number of LSTM hidden units.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout rate between LSTM layers.
            learning_rate: Learning rate for Adam optimizer.
            batch_size: Training batch size.
            n_epochs: Maximum training epochs.
            sequence_length: Length of input sequences.
            bidirectional: Use bidirectional LSTM (offline only).
            device: PyTorch device ('cuda', 'cpu', or None for auto).
            early_stopping_patience: Epochs without improvement before stopping.
            verbose: Print training progress.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for LSTMDecoder. Install with: pip install torch"
            )

        super().__init__(name=name, learning_rate=learning_rate)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.sequence_length = sequence_length
        self.bidirectional = bidirectional
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model = None
        self._optimizer = None
        self._hidden = None
        self._train_losses: List[float] = []
        self._val_losses: List[float] = []

        # Normalization parameters
        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None
        self._y_mean: Optional[np.ndarray] = None
        self._y_std: Optional[np.ndarray] = None

    def _build_model(self) -> nn.Module:
        """Build the LSTM network."""
        num_directions = 2 if self.bidirectional else 1

        class LSTMNetwork(nn.Module):
            def __init__(
                inner_self,
                input_size: int,
                hidden_size: int,
                output_size: int,
                num_layers: int,
                dropout: float,
                bidirectional: bool,
            ):
                super().__init__()
                inner_self.hidden_size = hidden_size
                inner_self.num_layers = num_layers
                inner_self.num_directions = 2 if bidirectional else 1

                inner_self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional,
                )

                lstm_output_size = hidden_size * inner_self.num_directions
                inner_self.fc = nn.Linear(lstm_output_size, output_size)

            def forward(inner_self, x, hidden=None):
                lstm_out, hidden = inner_self.lstm(x, hidden)
                output = inner_self.fc(lstm_out)
                return output, hidden

            def init_hidden(inner_self, batch_size: int, device):
                h0 = torch.zeros(
                    inner_self.num_layers * inner_self.num_directions,
                    batch_size,
                    inner_self.hidden_size,
                    device=device,
                )
                c0 = torch.zeros(
                    inner_self.num_layers * inner_self.num_directions,
                    batch_size,
                    inner_self.hidden_size,
                    device=device,
                )
                return (h0, c0)

        return LSTMNetwork(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            output_size=self.n_outputs,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMDecoder":
        """
        Fit LSTM decoder on training data.

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
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Initialize model
        self._model = self._build_model().to(self.device)

        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.learning_rate
        )
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
                output, _ = self._model(X_batch)
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

                    output, _ = self._model(X_batch)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            self._val_losses.append(val_loss)

            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.n_epochs}: "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

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
            output, _ = self._model(X_tensor)
            # Take the last time step prediction from each sequence
            predictions = output[:, -1, :].cpu().numpy()

        # Denormalize
        predictions = predictions * self._y_std + self._y_mean

        return predictions

    def predict_single(self, x: np.ndarray) -> np.ndarray:
        """
        Predict single time step for real-time use.

        Maintains hidden state across calls.

        Args:
            x: Neural features for single time step (n_features,).

        Returns:
            Predicted kinematics (n_outputs,).
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before prediction.")

        # Normalize
        x_norm = (x - self._X_mean) / self._X_std

        self._model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_norm).view(1, 1, -1).to(self.device)

            # Initialize hidden state if needed
            if self._hidden is None:
                self._hidden = self._model.init_hidden(1, self.device)

            output, self._hidden = self._model(x_tensor, self._hidden)
            prediction = output[0, 0, :].cpu().numpy()

        # Denormalize
        prediction = prediction * self._y_std + self._y_mean

        return prediction

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Online update with new data.

        Performs a few gradient updates on the new data.

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
            output, _ = self._model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optimizer.step()

        self._update_count += 1

    def reset_hidden(self) -> None:
        """Reset hidden state (for new trial/session)."""
        self._hidden = None

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        params.update({
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "sequence_length": self.sequence_length,
            "bidirectional": self.bidirectional,
            "device": str(self.device) if self.device else None,
            "train_losses": self._train_losses[-10:] if self._train_losses else [],
        })
        return params


class BidirectionalLSTMDecoder(LSTMDecoder):
    """
    Bidirectional LSTM decoder for offline analysis.

    Uses both past and future context for improved accuracy.
    Not suitable for real-time applications.
    """

    def __init__(
        self,
        name: str = "BiLSTM",
        hidden_size: int = 64,
        num_layers: int = 2,
        **kwargs,
    ):
        super().__init__(
            name=name,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            **kwargs,
        )

    def predict_single(self, x: np.ndarray) -> np.ndarray:
        """Not supported for bidirectional LSTM."""
        raise NotImplementedError(
            "Bidirectional LSTM requires future context and cannot be used for "
            "single-step real-time prediction. Use standard LSTMDecoder instead."
        )
