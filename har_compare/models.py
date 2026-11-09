"""Small, approximately parameter-matched classifiers for UCI HAR sequences."""

from torch import Tensor, nn


class CNNClassifier(nn.Module):
    """Three convolutional blocks; input shape is (batch, time, channels)."""

    def __init__(self, in_channels: int = 9, num_classes: int = 6,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x.transpose(1, 2)).squeeze(-1))


class RecurrentClassifier(nn.Module):
    """One recurrent layer, temporal mean pooling and a linear classifier."""

    def __init__(self, kind: str, hidden_size: int, in_channels: int = 9,
                 num_classes: int = 6, dropout: float = 0.2) -> None:
        super().__init__()
        if kind not in {"rnn", "lstm"}:
            raise ValueError(f"Unsupported recurrent model: {kind}")
        layer = nn.RNN if kind == "rnn" else nn.LSTM
        self.recurrent = layer(in_channels, hidden_size, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))

    def forward(self, x: Tensor) -> Tensor:
        sequence, _ = self.recurrent(x)
        return self.classifier(sequence.mean(dim=1))


def build_model(name: str, in_channels: int = 9, num_classes: int = 6,
                dropout: float = 0.2) -> nn.Module:
    """Return a CNN (32,710), RNN (33,974) or LSTM (29,606 parameters)."""
    name = name.lower()
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if name == "cnn":
        return CNNClassifier(in_channels, num_classes, dropout)
    if name in {"rnn", "lstm"}:
        hidden_size = {"rnn": 176, "lstm": 80}[name]
        return RecurrentClassifier(name, hidden_size, in_channels, num_classes, dropout)
    raise ValueError(f"Unknown model {name!r}; choose cnn, rnn or lstm")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
