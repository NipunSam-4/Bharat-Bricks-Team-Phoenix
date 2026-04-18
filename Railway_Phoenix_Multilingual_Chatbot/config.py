"""
Rail Drishti Configuration - Updated for Databricks Apps
CPU-optimized settings for train delay prediction and multilingual chatbot
"""
import os

# ==================== PATHS ====================
# Use relative paths for Databricks Apps deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Model files (now in local models directory)
DELAY_MODEL_PATH = os.path.join(MODEL_DIR, "chatbot_train_delay_predictor.pkl")
TRAIN_DATA_PATH = os.path.join(MODEL_DIR, "full_artificial_train_data.csv")
LABEL_ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")

# Delta Lake paths - Updated to use workspace catalog
DELTA_CATALOG = "workspace"
DELTA_SCHEMA = "rail_drishti"
PREDICTIONS_TABLE = f"{DELTA_CATALOG}.{DELTA_SCHEMA}.predictions_log"
FEEDBACK_TABLE = f"{DELTA_CATALOG}.{DELTA_SCHEMA}.feedback_data"
TRAIN_STATUS_TABLE = f"{DELTA_CATALOG}.{DELTA_SCHEMA}.train_status_realtime"
MODEL_METRICS_TABLE = f"{DELTA_CATALOG}.{DELTA_SCHEMA}.model_metrics"

# ==================== API CONFIGURATION ====================
# Indian Language Models
TRANSLATION_MODEL = "indictrans2"  # Using IndicTrans2 for translations

# Sarvam API (for Hindi/Indian language NLU)
SARVAM_API_URL = "https://api.sarvam.ai/v1/translate"
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")  # Set via Databricks secrets

# Alternative: AI21 Jamba for multilingual (if needed)
AI21_API_KEY = os.getenv("AI21_API_KEY", "")

# ==================== LANGUAGE SUPPORT ====================
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "mr": "Marathi (मराठी)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "bn": "Bengali (বাংলা)",
    "gu": "Gujarati (ગુજરાતી)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ml": "Malayalam (മലയാളം)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "or": "Odia (ଓଡ଼ିଆ)"
}

DEFAULT_LANGUAGE = "en"

# ==================== MODEL CONFIGURATION ====================
MAX_BATCH_SIZE = 32
PREDICTION_TIMEOUT = 10  # seconds
RETRAIN_THRESHOLD = 100  # entries before retraining
CONFIDENCE_THRESHOLD = 0.7
MODEL_VERSION = "v1.0.0"

# Dynamic learning settings
FEEDBACK_WINDOW_HOURS = 24  # Collect feedback from last 24 hours
MIN_RETRAINING_SAMPLES = 50  # Minimum samples needed for retraining
RETRAINING_FREQUENCY_HOURS = 6  # Retrain every 6 hours if threshold met

# ==================== CHATBOT SETTINGS ====================
MAX_HISTORY_LENGTH = 5
RESPONSE_MAX_TOKENS = 150
TEMPERATURE = 0.7

# ==================== DELAY CATEGORIES ====================
DELAY_REASONS = {
    "weather": "Weather conditions (rain, fog, storm)",
    "congestion": "Track or platform congestion",
    "technical": "Technical issues with train",
    "operational": "Operational delays",
    "external": "External factors (accident, maintenance)"
}

# ==================== WEATHER MAPPING ====================
WEATHER_CONDITIONS = {
    0: "Clear",
    1: "Rainy",
    2: "Foggy",
    3: "Cloudy",
    4: "Storm"
}

CONGESTION_LEVELS = {
    0: "Low",
    1: "Medium",
    2: "High"
}

# ==================== CACHING ====================
ENABLE_CACHING = True
CACHE_TTL = 300  # 5 minutes

# ==================== UI CONFIGURATION ====================
APP_TITLE = "🚂 Rail Drishti - Multilingual Train Delay Assistant"
APP_DESCRIPTION = "Real-time train delay predictions with AI-powered dynamic learning"
DEFAULT_WEATHER = "Clear"
DEFAULT_CONGESTION = "Low"
