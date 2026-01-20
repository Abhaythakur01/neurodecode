"""
Temporal Convolutional Network (TCN) decoder for neural signals.

Implements TCN with causal dilated convolutions for capturing temporal
dependencies without recurrence, offering faster training than RNNs.

Reference:
    Bai et al. (2018) "An Empirical Evaluation of Generic Convolutional
    and Recurrent Networks for Sequence Modeling"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.decoders.base import OnlineDecoder

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils import weight_norm
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    weight_norm = None
    DataLoader = None
    TensorDataset = None


# Only define PyTorch classes when torch is available
if TORCH_AVAILABLE:

    class Chomp1d(nn.Module):
        """Remove trailing padding to maintain causal convolution."""

        def __init__(self, chomp_size: int):
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :, : -self.chomp_size].contiguous()

    class TemporalBlock(nn.Module):
        """
        Single temporal block with dilated causal convolution.

        Contains two causal convolutions with residual connection.
        """

        def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            kernel_size: int,
            stride: int,
            dilation: int,
            padding: int,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.conv1 = weight_norm(
                nn.Conv1d(
                    n_inputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            )
            self.chomp1 = Chomp1d(padding)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)

            self.conv2 = weight_norm(
                nn.Conv1d(
                    n_outputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            )
            self.chomp2 = Chomp1d(padding)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)

            self.net = nn.Sequential(
                self.conv1,
                self.chomp1,
                self.relu1,
                self.dropout1,
                self.conv2,
                self.chomp2,
                self.relu2,
                self.dropout2,
            )

            # Residual connection (1x1 conv if dimensions differ)
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            self.relu = nn.ReLU()

            self._init_weights()

        def _init_weights(self) -> None:
            """Initialize weights with normal distribution."""
            self.conv1.weight.data.normal_(0, 0.01)
            self.conv2.weight.data.normal_(0, 0.01)
            if self.downsample is not None:
                self.downsample.weight.data.normal_(0, 0.01)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)

    class TemporalConvNet(nn.Module):
        """
        Temporal Convolutional Network.

        Stack of temporal blocks with exponentially increasing dilation.
        """

        def __init__(
            self,
            num_inputs: int,
            num_channels: List[int],
            kernel_size: int = 3,
            dropout: float = 0.2,
        ):
            super().__init__()
            layers = []
            num_levels = len(num_channels)

            for i in range(num_levels):
                dilation_size = 2**i
                in_channels = num_inputs if i == 0 else num_channels[i - 1]
                out_channels = num_channels[i]
                padding = (kernel_size - 1) * dilation_size

                layers.append(
                    TemporalBlock(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=1,
                        dilation=dilation_size,
                        padding=padding,
                        dropout=dropout,
                    )
                )

            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)


class TCNDecoder(OnlineDecoder):
    """
    Temporal Convolutional Network decoder for neural decoding.

    Uses causal dilated convolutions to capture temporal dependencies
    efficiently without recurrence.

    Advantages over LSTM:
    - Faster training (parallelizable)
    - Flexible receptive field through dilation
    - More stable gradients
    - Lower memory footprint
    """

    def __init__(
        self,
        name: str = "TCN",
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        n_epochs: int = 100,
        sequence_length: int = 20,
        device: Optional[str] = None,
        early_stopping_patience: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize TCN decoder.

        Args:
            name: Decoder name.
            num_channels: List of channel sizes for each temporal block.
                         Default: [64, 64, 64, 64] (4 blocks).
            kernel_size: Convolution kernel size.
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
            raise ImportError("PyTorch is required for TCNDecoder. Install with: pip install torch")

        super().__init__(name=name, learning_rate=learning_rate)

        self.num_channels = num_channels or [64, 64, 64, 64]
        self.kernel_size = kernel_size
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
        """Build the TCN network."""

        class TCNNetwork(nn.Module):
            def __init__(
                inner_self,
                input_size: int,
                output_size: int,
                num_channels: List[int],
                kernel_size: int,
                dropout: float,
            ):
                super().__init__()

                # TCN backbone
                inner_self.tcn = TemporalConvNet(
                    num_inputs=input_size,
                    num_channels=num_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )

                # Output projection
                inner_self.output_fc = nn.Sequential(
                    nn.Linear(num_channels[-1], num_channels[-1] // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(num_channels[-1] // 2, output_size),
                )

            def forward(inner_self, x: torch.Tensor) -> torch.Tensor:
                # x shape: (batch, seq_len, input_size)
                # TCN expects: (batch, channels, seq_len)
                x = x.transpose(1, 2)

                # Apply TCN
                tcn_out = inner_self.tcn(x)

                # Back to (batch, seq_len, channels)
                tcn_out = tcn_out.transpose(1, 2)

                # Project to output
                output = inner_self.output_fc(tcn_out)

                return output

        return TCNNetwork(
            input_size=self.n_features,
            output_size=self.n_outputs,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TCNDecoder":
        """
        Fit TCN decoder on training data.

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

        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, mode="min", factor=0.5, patience=5
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

            scheduler.step(val_loss)

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

    def get_receptive_field(self) -> int:
        """
        Calculate the receptive field size of the TCN.

        Returns:
            Receptive field size (number of time steps the model can see).
        """
        num_levels = len(self.num_channels)
        receptive_field = 1 + 2 * (self.kernel_size - 1) * (2**num_levels - 1)
        return receptive_field

    def get_params(self) -> Dict[str, Any]:
        """Get decoder parameters."""
        params = super().get_params()
        params.update(
            {
                "num_channels": self.num_channels,
                "kernel_size": self.kernel_size,
                "dropout": self.dropout,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
                "sequence_length": self.sequence_length,
                "device": str(self.device) if self.device else None,
                "receptive_field": (self.get_receptive_field() if self.is_fitted else None),
                "train_losses": self._train_losses[-10:] if self._train_losses else [],
            }
        )
        return params
