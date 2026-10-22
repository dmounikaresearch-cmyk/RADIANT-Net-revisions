from pathlib import Path

# Main project directory
BASE_DIR = Path(__file__).resolve().parent

# Project folders
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# File locations
DATA_FILE = DATA_DIR / "radiant_disaster_data.csv"
REAL_DATA_FILE = DATA_DIR / "radiant_real_disaster_data.csv"
IMERG_NOAA_DATA_FILE = DATA_DIR / "radiant_imerg_noaa_disaster_data.csv"
ALL_MULTIMODAL_DATA_FILE = DATA_DIR / "radiant_all_multimodal_disaster_data.csv"
COMPLETE_MULTIMODAL_DATA_FILE = DATA_DIR / "radiant_complete_multimodal_disaster_data.csv"
MODEL_FILE = MODEL_DIR / "radiant_net_model.pt"
SCALER_FILE = MODEL_DIR / "feature_scaler.pkl"
ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"
TRAINING_PLOT = OUTPUT_DIR / "training_history.png"
REAL_TRAINING_PLOT = OUTPUT_DIR / "training_history_real.png"
REAL_MODEL_FILE = MODEL_DIR / "radiant_net_model_real.pt"
REAL_SCALER_FILE = MODEL_DIR / "feature_scaler_real.pkl"
REAL_ENCODER_FILE = MODEL_DIR / "label_encoder_real.pkl"
IMERG_NOAA_TRAINING_PLOT = OUTPUT_DIR / "training_history_imerg_noaa.png"
IMERG_NOAA_MODEL_FILE = MODEL_DIR / "radiant_net_model_imerg_noaa.pt"
IMERG_NOAA_SCALER_FILE = MODEL_DIR / "feature_scaler_imerg_noaa.pkl"
IMERG_NOAA_ENCODER_FILE = MODEL_DIR / "label_encoder_imerg_noaa.pkl"
ALL_TRAINING_PLOT = OUTPUT_DIR / "training_history_all_multimodal.png"
ALL_MODEL_FILE = MODEL_DIR / "radiant_net_model_all_multimodal.pt"
ALL_SCALER_FILE = MODEL_DIR / "feature_scaler_all_multimodal.pkl"
ALL_ENCODER_FILE = MODEL_DIR / "label_encoder_all_multimodal.pkl"
COMPLETE_TRAINING_PLOT = OUTPUT_DIR / "training_history_complete_multimodal.png"
COMPLETE_MODEL_FILE = MODEL_DIR / "radiant_net_model_complete_multimodal.pt"
COMPLETE_SCALER_FILE = MODEL_DIR / "feature_scaler_complete_multimodal.pkl"
COMPLETE_ENCODER_FILE = MODEL_DIR / "label_encoder_complete_multimodal.pkl"

# Data settings
NUMBER_OF_SAMPLES = 5000
RANDOM_SEED = 42

# Model settings
INPUT_FEATURES = 15
ALL_INPUT_FEATURES = 19
COMPLETE_INPUT_FEATURES = 22
HIDDEN_DIMENSION = 64
NUMBER_OF_CLASSES = 5
DROPOUT_RATE = 0.20

# Training settings
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUMBER_OF_EPOCHS = 30
TEST_SIZE = 0.20

# Disaster classes
DISASTER_CLASSES = [
    "Flood",
    "Cyclone",
    "Landslide",
    "Wildfire",
    "Earthquake",
]