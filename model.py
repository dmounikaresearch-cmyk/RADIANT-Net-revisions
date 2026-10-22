import torch
from torch import nn


class MultimodalFeatureEncoder(nn.Module):
    """Encode heterogeneous environmental and disaster features."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.BatchNorm1d(hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)


class NeuralTwinMemory(nn.Module):
    """
    Simplified neural-digital-twin module.

    It learns an internal disaster-state representation and combines it
    with the encoded sensor observation.
    """

    def __init__(
        self,
        hidden_dimension: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self.state_update = nn.GRUCell(
            hidden_dimension,
            hidden_dimension,
        )

        self.refinement = nn.Sequential(
            nn.Linear(hidden_dimension * 2, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dimension, hidden_dimension),
        )

    def forward(self, encoded_features: torch.Tensor) -> torch.Tensor:
        initial_state = torch.zeros_like(encoded_features)

        twin_state = self.state_update(
            encoded_features,
            initial_state,
        )

        combined_state = torch.cat(
            [encoded_features, twin_state],
            dim=1,
        )

        refined_state = self.refinement(combined_state)

        return refined_state + encoded_features


class AgenticDecisionLayer(nn.Module):
    """Generate disaster classification and confidence scores."""

    def __init__(
        self,
        hidden_dimension: int,
        number_of_classes: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self.decision_network = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dimension // 2, number_of_classes),
        )

    def forward(self, twin_state: torch.Tensor) -> torch.Tensor:
        return self.decision_network(twin_state)


class RadiantNet(nn.Module):
    """
    Beginner prototype of RADIANT-Net.

    Modules:
    1. Multimodal feature encoder
    2. Neural-twin memory
    3. Agentic decision layer
    """

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        number_of_classes: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self.multimodal_encoder = MultimodalFeatureEncoder(
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
            dropout_rate=dropout_rate,
        )

        self.neural_twin = NeuralTwinMemory(
            hidden_dimension=hidden_dimension,
            dropout_rate=dropout_rate,
        )

        self.agentic_decision_layer = AgenticDecisionLayer(
            hidden_dimension=hidden_dimension,
            number_of_classes=number_of_classes,
            dropout_rate=dropout_rate,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded_features = self.multimodal_encoder(features)
        twin_state = self.neural_twin(encoded_features)
        predictions = self.agentic_decision_layer(twin_state)

        return predictions