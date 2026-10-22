import argparse
import pickle
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import (
    ALL_ENCODER_FILE,
    ALL_MODEL_FILE,
    ALL_MULTIMODAL_DATA_FILE,
    ALL_SCALER_FILE,
    ALL_TRAINING_PLOT,
    BATCH_SIZE,
    COMPLETE_ENCODER_FILE,
    COMPLETE_MODEL_FILE,
    COMPLETE_MULTIMODAL_DATA_FILE,
    COMPLETE_SCALER_FILE,
    COMPLETE_TRAINING_PLOT,
    DATA_FILE,
    DROPOUT_RATE,
    ENCODER_FILE,
    HIDDEN_DIMENSION,
    IMERG_NOAA_DATA_FILE,
    IMERG_NOAA_ENCODER_FILE,
    IMERG_NOAA_MODEL_FILE,
    IMERG_NOAA_SCALER_FILE,
    IMERG_NOAA_TRAINING_PLOT,
    INPUT_FEATURES,
    LEARNING_RATE,
    MODEL_DIR,
    MODEL_FILE,
    NUMBER_OF_CLASSES,
    NUMBER_OF_EPOCHS,
    OUTPUT_DIR,
    RANDOM_SEED,
    REAL_DATA_FILE,
    REAL_ENCODER_FILE,
    REAL_MODEL_FILE,
    REAL_SCALER_FILE,
    REAL_TRAINING_PLOT,
    SCALER_FILE,
    TEST_SIZE,
    TRAINING_PLOT,
)
from src.dataset import DisasterDataset
from src.model import RadiantNet


FEATURE_COLUMNS = [
    "rainfall_mm",
    "temperature_c",
    "humidity_percent",
    "wind_speed_kmh",
    "atmospheric_pressure_hpa",
    "river_level_m",
    "soil_moisture_percent",
    "slope_degrees",
    "vegetation_dryness",
    "seismic_activity",
    "ground_displacement_mm",
    "satellite_water_index",
    "population_density",
    "road_accessibility",
    "historical_risk_score",
]

ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "sentinel1_vv",
    "sentinel1_vh",
    "sentinel2_ndvi",
    "sentinel2_ndwi",
]

COMPLETE_FEATURE_COLUMNS = ALL_FEATURE_COLUMNS + [
    "firms_hotspot_score",
    "osm_road_density",
    "worldpop_density",
]


def resolve_paths(data_mode: str) -> tuple:
    """Select dataset/checkpoint paths for the chosen training mode."""

    if data_mode == "real":
        return (
            REAL_DATA_FILE,
            REAL_MODEL_FILE,
            REAL_SCALER_FILE,
            REAL_ENCODER_FILE,
            REAL_TRAINING_PLOT,
        )

    if data_mode == "imerg-noaa":
        return (
            IMERG_NOAA_DATA_FILE,
            IMERG_NOAA_MODEL_FILE,
            IMERG_NOAA_SCALER_FILE,
            IMERG_NOAA_ENCODER_FILE,
            IMERG_NOAA_TRAINING_PLOT,
        )

    if data_mode == "all":
        return (
            ALL_MULTIMODAL_DATA_FILE,
            ALL_MODEL_FILE,
            ALL_SCALER_FILE,
            ALL_ENCODER_FILE,
            ALL_TRAINING_PLOT,
        )

    if data_mode == "complete":
        return (
            COMPLETE_MULTIMODAL_DATA_FILE,
            COMPLETE_MODEL_FILE,
            COMPLETE_SCALER_FILE,
            COMPLETE_ENCODER_FILE,
            COMPLETE_TRAINING_PLOT,
        )

    return (
        DATA_FILE,
        MODEL_FILE,
        SCALER_FILE,
        ENCODER_FILE,
        TRAINING_PLOT,
    )


def set_random_seeds(seed: int) -> None:
    """Make the experiment more reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Calculate classification accuracy."""

    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            predictions = torch.argmax(logits, dim=1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    return accuracy_score(true_labels, predicted_labels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RADIANT-Net on synthetic or real multimodal data.",
    )
    parser.add_argument(
        "--data",
        choices=["synthetic", "real", "imerg-noaa", "all", "complete"],
        default="synthetic",
        help=(
            "Training data mode. Use 'complete' for Sentinel-1/2 + IMERG + NOAA "
            "+ FIRMS + OSM + WorldPop."
        ),
    )
    arguments = parser.parse_args()

    set_random_seeds(RANDOM_SEED)

    (
        data_file,
        model_file,
        scaler_file,
        encoder_file,
        training_plot,
    ) = resolve_paths(arguments.data)

    if not data_file.exists():
        if arguments.data == "complete":
            raise FileNotFoundError(
                "Complete dataset not found. Run: python fetch_complete_multimodal_data.py"
            )
        if arguments.data == "all":
            raise FileNotFoundError(
                "All-source dataset not found. Run: python fetch_all_multimodal_data.py"
            )
        if arguments.data == "imerg-noaa":
            raise FileNotFoundError(
                "IMERG+NOAA dataset not found. Run: python fetch_imerg_noaa_data.py"
            )
        if arguments.data == "real":
            raise FileNotFoundError(
                "Real dataset not found. Run: python fetch_real_data.py"
            )
        raise FileNotFoundError(
            "Dataset was not found. Run: python generate_data.py"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(data_file)

    if arguments.data == "complete":
        selected_feature_columns = COMPLETE_FEATURE_COLUMNS
    elif arguments.data == "all":
        selected_feature_columns = ALL_FEATURE_COLUMNS
    else:
        selected_feature_columns = FEATURE_COLUMNS

    feature_columns = [
        column
        for column in selected_feature_columns
        if column in dataframe.columns
    ]
    expected_features = len(selected_feature_columns)
    if len(feature_columns) != expected_features:
        raise ValueError(
            f"Expected {expected_features} features, found {len(feature_columns)}."
        )

    print(f"Training mode: {arguments.data}")
    print(f"Dataset: {data_file}")
    print(f"Input features: {len(feature_columns)}")
    print("Class distribution:")
    print(dataframe["disaster_type"].value_counts().to_string())

    features = dataframe[feature_columns].values
    disaster_labels = dataframe["disaster_type"].values

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(disaster_labels)

    (
        training_features,
        testing_features,
        training_labels,
        testing_labels,
    ) = train_test_split(
        features,
        encoded_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=encoded_labels,
    )

    scaler = StandardScaler()
    training_features = scaler.fit_transform(training_features)
    testing_features = scaler.transform(testing_features)

    with open(scaler_file, "wb") as scaler_handle:
        pickle.dump(scaler, scaler_handle)

    with open(encoder_file, "wb") as encoder_handle:
        pickle.dump(label_encoder, encoder_handle)

    training_dataset = DisasterDataset(
        training_features,
        training_labels,
    )
    testing_dataset = DisasterDataset(
        testing_features,
        testing_labels,
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    testing_loader = DataLoader(
        testing_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}")
    print(f"Training samples: {len(training_dataset)}")
    print(f"Testing samples: {len(testing_dataset)}")

    model = RadiantNet(
        input_dimension=len(feature_columns),
        hidden_dimension=HIDDEN_DIMENSION,
        number_of_classes=NUMBER_OF_CLASSES,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    training_losses: list[float] = []
    training_accuracies: list[float] = []
    testing_accuracies: list[float] = []

    for epoch in range(NUMBER_OF_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_features, batch_labels in training_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            logits = model(batch_features)
            loss = loss_function(logits, batch_labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(training_loader)

        training_accuracy = calculate_accuracy(
            model,
            training_loader,
            device,
        )

        testing_accuracy = calculate_accuracy(
            model,
            testing_loader,
            device,
        )

        training_losses.append(average_loss)
        training_accuracies.append(training_accuracy)
        testing_accuracies.append(testing_accuracy)

        print(
            f"Epoch [{epoch + 1:02d}/{NUMBER_OF_EPOCHS}] "
            f"Loss: {average_loss:.4f} | "
            f"Training Accuracy: {training_accuracy:.4f} | "
            f"Testing Accuracy: {testing_accuracy:.4f}"
        )

    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []

    with torch.no_grad():
        for batch_features, batch_labels in testing_loader:
            batch_features = batch_features.to(device)

            logits = model(batch_features)
            predictions = torch.argmax(logits, dim=1)

            true_labels.extend(batch_labels.numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    print("\nFinal classification report")
    print("=" * 70)

    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=label_encoder.classes_,
            digits=4,
        )
    )

    print("Confusion matrix")
    print(
        confusion_matrix(
            true_labels,
            predicted_labels,
        )
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dimension": len(feature_columns),
        "hidden_dimension": HIDDEN_DIMENSION,
        "number_of_classes": NUMBER_OF_CLASSES,
        "dropout_rate": DROPOUT_RATE,
        "feature_columns": feature_columns,
        "class_names": list(label_encoder.classes_),
    }

    torch.save(checkpoint, model_file)

    epochs = range(1, NUMBER_OF_EPOCHS + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, training_accuracies, label="Training accuracy")
    plt.plot(epochs, testing_accuracies, label="Testing accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plot_title = {
        "real": "RADIANT-Net Training History (Real Multimodal Data)",
        "imerg-noaa": "RADIANT-Net Training History (NASA IMERG + NOAA)",
        "all": "RADIANT-Net Training History (All Sources + Sentinel-1/2)",
        "complete": "RADIANT-Net (S1/S2 + IMERG + NOAA + FIRMS + OSM + WorldPop)",
    }.get(
        arguments.data,
        "RADIANT-Net Training History (Synthetic Data)",
    )
    plt.title(plot_title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(training_plot, dpi=300)
    plt.close()

    print(f"\nModel saved to: {model_file}")
    print(f"Scaler saved to: {scaler_file}")
    print(f"Label encoder saved to: {encoder_file}")
    print(f"Training graph saved to: {training_plot}")


if __name__ == "__main__":
    main()