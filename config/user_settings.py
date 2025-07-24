"""
User Configuration Settings for Multilingual TTS System v2.0
Customizable parameters for users to modify system behavior
This file is created during setup and can be modified by users
"""

import torch
from pathlib import Path

# ============================================================================
# 🎯 CORE SYSTEM SETTINGS
# ============================================================================

# Device configuration (auto-detect or specify)
DEVICE = "auto"  # Options: "auto", "cuda", "cpu", "cuda:0", "cuda:1", etc.
# Note: "auto" will use GPU if available, otherwise CPU

# Enable mixed precision training (faster on modern GPUs)
USE_MIXED_PRECISION = True  # Set to False if you have issues

# Number of CPU cores to use for multiprocessing
NUM_WORKERS = 4  # Adjust based on your CPU cores (typically CPU_CORES - 2)

# ============================================================================
# 🎵 AUDIO PROCESSING SETTINGS
# ============================================================================

# Audio specifications
SAMPLE_RATE = 16000  # Hz - Standard for TTS (don't change unless needed)
HOP_LENGTH = 256  # STFT hop length
WIN_LENGTH = 1024  # STFT window length
N_FFT = 1024  # FFT size
N_MELS = 80  # Number of mel filter banks
MAX_WAV_VALUE = 32768.0  # Audio normalization value

# Audio quality thresholds
MIN_AUDIO_DURATION = 1.0  # Minimum segment duration (seconds)
MAX_AUDIO_DURATION = 20.0  # Maximum segment duration (seconds)
MIN_SNR_DB = 10  # Minimum signal-to-noise ratio
MAX_SILENCE_RATIO = 0.4  # Maximum allowed silence in audio

# Enhanced audio processing
ENABLE_NOISE_REDUCTION = True  # Apply noise reduction
ENABLE_AUDIO_ENHANCEMENT = True  # Apply audio enhancement
NORMALIZE_AUDIO = True  # Normalize audio levels

# ============================================================================
# 📚 DATASET COLLECTION SETTINGS
# ============================================================================

# Active languages (modify this to focus on specific languages)
ACTIVE_LANGUAGES = [
    "hi",  # Hindi - Best dataset availability
    "ta",  # Tamil - Good dataset availability
    "te",  # Telugu - Good dataset availability
    "bn",  # Bengali - Good dataset availability
    "mr",  # Marathi - Medium dataset availability
    "gu",  # Gujarati - Medium dataset availability
    "kn",  # Kannada - Lower dataset availability
    "ml",  # Malayalam - Lower dataset availability
    "pa",  # Punjabi - Lower dataset availability
    "or"  # Odia - Lowest dataset availability (mainly custom recordings)
]

# Dataset preferences (in order of priority)
DEFAULT_DATASETS = [
    "common_voice",  # Mozilla Common Voice (highest priority)
    "google_fleurs",  # Google FLEURS (high quality)
    "openslr",  # OpenSLR (when available)
    "indic_tts",  # IITm TTS (research quality)
    "custom_recordings"  # User recordings (always available)
]

# Data collection limits
MAX_HOURS_PER_LANGUAGE = 100  # Maximum hours to collect per language
MIN_HOURS_FOR_TRAINING = 5  # Minimum hours needed to start training
MAX_SEGMENTS_PER_DATASET = 50000  # Limit segments per dataset (0 = no limit)

# Custom recordings settings
ENABLE_CUSTOM_RECORDINGS = True
CUSTOM_RECORDING_QUALITY_CHECK = True
CUSTOM_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".m4a"]

# ============================================================================
# 🧠 LINGUISTIC PROCESSING SETTINGS
# ============================================================================

# Linguistic features to enable
ENABLE_LINGUISTIC_FEATURES = True
ENABLE_G2P_CONVERSION = True  # Grapheme-to-Phoneme conversion
ENABLE_PROSODIC_ANALYSIS = True  # Stress, tone, boundary analysis
ENABLE_MORPHOLOGICAL_ANALYSIS = True  # Compound word analysis
ENABLE_CODE_SWITCH_DETECTION = True  # Multi-language detection

# Text processing
ENABLE_TEXT_NORMALIZATION = True  # Normalize numbers, dates, etc.
ENABLE_PUNCTUATION_PROSODY = True  # Use punctuation for prosody
MIN_TEXT_LENGTH = 3  # Minimum characters per segment
MAX_TEXT_LENGTH = 500  # Maximum characters per segment

# Phoneme coverage settings
TARGET_PHONEME_COVERAGE = 0.85  # Target coverage for training corpus
ENABLE_CORPUS_BALANCING = True  # Create balanced training corpus

# ============================================================================
# 🎙️ SPEAKER PROCESSING SETTINGS
# ============================================================================

# Speaker identification
ENABLE_SPEAKER_DIARIZATION = True
ENABLE_SPEAKER_ENROLLMENT = True
MIN_SEGMENTS_PER_SPEAKER = 10  # Minimum segments to enroll a speaker
MAX_SPEAKERS_PER_LANGUAGE = 200  # Limit speakers per language

# Speaker similarity thresholds
SPEAKER_SIMILARITY_THRESHOLD = 0.75  # Threshold for speaker matching
ENABLE_VOICE_CLONING = True  # Enable voice cloning features

# ============================================================================
# 🤖 TRAINING SETTINGS
# ============================================================================

# Training hyperparameters
BATCH_SIZE = 32  # Reduce if you get out-of-memory errors
LEARNING_RATE = 0.0001  # Learning rate for training
MAX_EPOCHS = 100  # Maximum training epochs
EARLY_STOPPING_PATIENCE = 10  # Epochs to wait before early stopping

# Training data splits
TRAIN_SPLIT = 0.8  # 80% for training
VAL_SPLIT = 0.1  # 10% for validation
TEST_SPLIT = 0.1  # 10% for testing

# Model architecture
MODEL_TYPE = "tacotron2"  # Options: "tacotron2", "fastspeech2"
VOCODER_TYPE = "hifigan"  # Options: "hifigan", "melgan", "waveglow"

# Training features
ENABLE_ATTENTION_FORCING = True  # Improve attention alignment
ENABLE_SPEAKER_CONDITIONING = True  # Multi-speaker support
ENABLE_LANGUAGE_CONDITIONING = True  # Multi-language support

# ============================================================================
# 💾 STORAGE AND PERFORMANCE SETTINGS
# ============================================================================

# Storage paths (relative to project root)
DATA_BASE_DIR = "data"
MODELS_BASE_DIR = "models"
LOGS_BASE_DIR = "logs"
TEMP_BASE_DIR = "temp"
OUTPUTS_BASE_DIR = "outputs"

# Performance settings
ENABLE_CACHING = True  # Cache processed data
CACHE_SIZE_GB = 10  # Maximum cache size in GB
ENABLE_PARALLEL_PROCESSING = True  # Use multiprocessing
MAX_PARALLEL_DOWNLOADS = 3  # Concurrent downloads

# Cleanup settings
AUTO_CLEANUP_TEMP = True  # Auto-cleanup temporary files
KEEP_RAW_DATA = True  # Keep original downloaded data
KEEP_INTERMEDIATE_FILES = False  # Keep intermediate processing files

# ============================================================================
# 📊 LOGGING AND MONITORING
# ============================================================================

# Logging levels
LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
ENABLE_DETAILED_LOGGING = True
ENABLE_PROGRESS_VISUALIZATION = True

# Training monitoring
SAVE_CHECKPOINT_EVERY = 10  # Save checkpoint every N epochs
LOG_TRAINING_EVERY = 100  # Log training metrics every N steps
GENERATE_SAMPLES_EVERY = 1000  # Generate audio samples every N steps

# Quality monitoring
ENABLE_QUALITY_METRICS = True  # Calculate quality metrics
ENABLE_MOS_PREDICTION = True  # Predict Mean Opinion Scores

# ============================================================================
# 🌐 MULTILINGUAL SETTINGS
# ============================================================================

# Multilingual training
ENABLE_UNIFIED_MODEL = True  # Train unified multilingual model
LANGUAGE_EMBEDDING_DIM = 64  # Language embedding dimensions
SHARE_ENCODER_ACROSS_LANGUAGES = True  # Share encoder parameters

# Cross-lingual transfer
ENABLE_TRANSFER_LEARNING = True  # Use transfer learning for low-resource languages
TRANSFER_LEARNING_LANGUAGES = {  # Source -> Target language mapping
    "hi": ["mr", "gu"],  # Hindi -> Marathi, Gujarati
    "ta": ["ml"],  # Tamil -> Malayalam
    "te": ["kn"]  # Telugu -> Kannada
}

# ============================================================================
# 🔧 ADVANCED SETTINGS (Modify carefully)
# ============================================================================

# Forced alignment
ALIGNMENT_METHOD = "auto"  # Options: "mfa", "simple", "auto"
MFA_MODELS_PATH = None  # Path to MFA models (None = auto-download)

# Model optimization
ENABLE_MODEL_PRUNING = False  # Prune models for deployment
ENABLE_QUANTIZATION = False  # Quantize models for speed
TARGET_MODEL_SIZE_MB = 100  # Target size for optimized models

# Experimental features
ENABLE_EXPERIMENTAL_FEATURES = False
ENABLE_REAL_TIME_SYNTHESIS = False
ENABLE_STREAMING_INFERENCE = False

# ============================================================================
# 🎛️ USER INTERFACE SETTINGS
# ============================================================================

# Menu behavior
SHOW_ADVANCED_OPTIONS = False  # Show advanced menu options
ENABLE_INTERACTIVE_MODE = True  # Interactive progress updates
SHOW_DATASET_STATISTICS = True  # Show detailed dataset info

# Output preferences
DEFAULT_OUTPUT_FORMAT = "wav"  # Options: "wav", "mp3", "flac"
DEFAULT_SYNTHESIS_SPEAKER = "auto"  # Default speaker for synthesis
ENABLE_BATCH_SYNTHESIS = True  # Enable batch text-to-speech


# ============================================================================
# 🛠️ HELPER FUNCTIONS (Don't modify)
# ============================================================================

def get_device():
    """Get the appropriate device for training/inference"""
    if DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE


def get_active_languages_info():
    """Get information about active languages"""
    from config.languages import IndianLanguages
    languages = IndianLanguages()

    info = {}
    for lang_code in ACTIVE_LANGUAGES:
        if languages.validate_language_code(lang_code):
            lang_info = languages.get_language_info(lang_code)
            info[lang_code] = {
                'name': lang_info['name'],
                'native_name': lang_info['native_name'],
                'estimated_hours': lang_info['total_estimated_hours'],
                'resource_level': languages.get_resource_level(lang_code)
            }

    return info


def validate_settings():
    """Validate user settings and show warnings"""
    warnings = []

    # Check device availability
    if DEVICE not in ["auto", "cpu"] and not torch.cuda.is_available():
        warnings.append(f"⚠️  CUDA device '{DEVICE}' specified but CUDA not available")

    # Check memory settings
    if BATCH_SIZE > 64:
        warnings.append(f"⚠️  Large batch size ({BATCH_SIZE}) may cause out-of-memory errors")

    # Check storage settings
    data_path = Path(DATA_BASE_DIR)
    if not data_path.exists():
        warnings.append(f"⚠️  Data directory '{DATA_BASE_DIR}' does not exist")

    # Check language settings
    if len(ACTIVE_LANGUAGES) > 5:
        warnings.append(
            f"⚠️  Many active languages ({len(ACTIVE_LANGUAGES)}) will require significant storage and time")

    return warnings


def get_recommended_settings():
    """Get recommended settings based on system capabilities"""
    recommendations = {}

    # GPU-based recommendations
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory >= 16:
            recommendations['batch_size'] = 64
            recommendations['use_mixed_precision'] = True
        elif gpu_memory >= 8:
            recommendations['batch_size'] = 32
            recommendations['use_mixed_precision'] = True
        else:
            recommendations['batch_size'] = 16
            recommendations['use_mixed_precision'] = False
    else:
        recommendations['batch_size'] = 8
        recommendations['use_mixed_precision'] = False
        recommendations['num_workers'] = 2

    return recommendations


def print_current_settings():
    """Print current user settings"""
    print("\n🎛️  CURRENT USER SETTINGS")
    print("=" * 50)

    print(f"🖥️  Device: {get_device()}")
    print(f"📊 Batch Size: {BATCH_SIZE}")
    print(f"🌍 Active Languages: {len(ACTIVE_LANGUAGES)} languages")
    print(f"📦 Default Datasets: {', '.join(DEFAULT_DATASETS)}")
    print(f"🧠 Linguistic Features: {'Enabled' if ENABLE_LINGUISTIC_FEATURES else 'Disabled'}")
    print(f"🎙️  Speaker Features: {'Enabled' if ENABLE_SPEAKER_DIARIZATION else 'Disabled'}")
    print(f"🤖 Model Type: {MODEL_TYPE}")
    print(f"📈 Max Epochs: {MAX_EPOCHS}")

    # Show warnings if any
    warnings = validate_settings()
    if warnings:
        print(f"\n⚠️  Warnings:")
        for warning in warnings:
            print(f"   {warning}")

    print()


# ============================================================================
# 📋 PRESET CONFIGURATIONS
# ============================================================================

def apply_preset_config(preset_name: str):
    """Apply a preset configuration"""
    global ACTIVE_LANGUAGES, BATCH_SIZE, MAX_EPOCHS, DEFAULT_DATASETS

    presets = {
        'quick_test': {
            'description': 'Quick testing with minimal data',
            'ACTIVE_LANGUAGES': ['hi'],
            'DEFAULT_DATASETS': ['google_fleurs'],
            'BATCH_SIZE': 16,
            'MAX_EPOCHS': 10,
            'MAX_HOURS_PER_LANGUAGE': 5
        },

        'hindi_focus': {
            'description': 'Focus on Hindi with all available datasets',
            'ACTIVE_LANGUAGES': ['hi'],
            'DEFAULT_DATASETS': ['common_voice', 'google_fleurs', 'openslr', 'indic_tts'],
            'BATCH_SIZE': 32,
            'MAX_EPOCHS': 100,
            'MAX_HOURS_PER_LANGUAGE': 100
        },

        'high_resource': {
            'description': 'High-resource languages (Hindi, Tamil, Telugu, Bengali)',
            'ACTIVE_LANGUAGES': ['hi', 'ta', 'te', 'bn'],
            'DEFAULT_DATASETS': ['common_voice', 'google_fleurs', 'openslr'],
            'BATCH_SIZE': 32,
            'MAX_EPOCHS': 80,
            'MAX_HOURS_PER_LANGUAGE': 80
        },

        'all_languages': {
            'description': 'All 10 languages with balanced approach',
            'ACTIVE_LANGUAGES': ['hi', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or'],
            'DEFAULT_DATASETS': ['common_voice', 'google_fleurs', 'custom_recordings'],
            'BATCH_SIZE': 24,
            'MAX_EPOCHS': 60,
            'MAX_HOURS_PER_LANGUAGE': 50
        },

        'research_quality': {
            'description': 'Research-grade training with all features',
            'ACTIVE_LANGUAGES': ['hi', 'ta', 'te', 'bn', 'mr'],
            'DEFAULT_DATASETS': ['common_voice', 'google_fleurs', 'openslr', 'indic_tts'],
            'BATCH_SIZE': 16,  # Smaller for stability
            'MAX_EPOCHS': 150,
            'ENABLE_ALL_FEATURES': True
        },

        'low_resource': {
            'description': 'Optimized for low-resource languages',
            'ACTIVE_LANGUAGES': ['kn', 'ml', 'pa', 'or'],
            'DEFAULT_DATASETS': ['google_fleurs', 'custom_recordings'],
            'BATCH_SIZE': 8,
            'MAX_EPOCHS': 120,
            'ENABLE_TRANSFER_LEARNING': True
        }
    }

    if preset_name not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    preset = presets[preset_name]
    print(f"\n🎯 Applying preset: {preset_name}")
    print(f"📝 Description: {preset['description']}")

    # Apply preset values
    for key, value in preset.items():
        if key != 'description' and key in globals():
            globals()[key] = value
            print(f"   Set {key} = {value}")

    print("✅ Preset applied successfully!")


def get_available_presets():
    """Get list of available preset configurations"""
    return [
        ('quick_test', 'Quick testing with minimal data'),
        ('hindi_focus', 'Focus on Hindi with all datasets'),
        ('high_resource', 'High-resource languages only'),
        ('all_languages', 'All 10 languages balanced'),
        ('research_quality', 'Research-grade with all features'),
        ('low_resource', 'Optimized for low-resource languages')
    ]


# ============================================================================
# 💡 USAGE EXAMPLES AND TIPS
# ============================================================================

USAGE_TIPS = """
🎯 CONFIGURATION TIPS:

1. **For Beginners:**
   - Start with preset 'hindi_focus' or 'quick_test'
   - Use default settings for first run
   - Increase BATCH_SIZE if you have more GPU memory

2. **For Research:**
   - Use preset 'research_quality'
   - Enable all linguistic features
   - Set MAX_EPOCHS to 150+ for best quality

3. **For Production:**
   - Focus on 2-3 languages initially
   - Use 'high_resource' preset
   - Enable model optimization features

4. **For Low-Resource Languages:**
   - Use preset 'low_resource'
   - Enable transfer learning
   - Contribute custom recordings

5. **Performance Tuning:**
   - Reduce BATCH_SIZE if out-of-memory errors
   - Increase NUM_WORKERS for faster data loading
   - Enable caching for repeated runs

6. **Storage Management:**
   - Set MAX_HOURS_PER_LANGUAGE to limit data
   - Enable AUTO_CLEANUP_TEMP to save space
   - Disable KEEP_INTERMEDIATE_FILES if space is limited

📋 QUICK PRESET COMMANDS:
```python
# In Python shell or script:
from config.user_settings import apply_preset_config

apply_preset_config('hindi_focus')    # Best for starting
apply_preset_config('quick_test')     # Fast testing
apply_preset_config('all_languages')  # Complete system
```

🔧 CUSTOM MODIFICATIONS:
Edit this file (config/user_settings.py) to customize:
- ACTIVE_LANGUAGES: Choose which languages to process
- BATCH_SIZE: Adjust for your GPU memory
- DEFAULT_DATASETS: Select preferred data sources
- Training parameters: Learning rate, epochs, etc.

⚠️  IMPORTANT NOTES:
- Changes take effect after restarting the system
- Some settings require re-processing data
- GPU settings are auto-detected but can be overridden
- Backup this file before major changes
"""


def show_usage_tips():
    """Display configuration tips and usage examples"""
    print(USAGE_TIPS)


# ============================================================================
# 🧪 EXPERIMENTAL SETTINGS (Use with caution)
# ============================================================================

if ENABLE_EXPERIMENTAL_FEATURES:
    # Advanced TTS features
    ENABLE_EMOTION_CONTROL = False  # Control emotional expression
    ENABLE_SPEAKING_RATE_CONTROL = False  # Control speaking speed
    ENABLE_PITCH_CONTROL = False  # Control pitch/tone

    # Advanced training techniques
    ENABLE_ADVERSARIAL_TRAINING = False  # GAN-based training
    ENABLE_SELF_SUPERVISED_LEARNING = False  # Self-supervised pretraining
    ENABLE_CURRICULUM_LEARNING = False  # Curriculum learning strategy

    # Model architecture experiments
    ENABLE_TRANSFORMER_TTS = False  # Transformer-based TTS
    ENABLE_FLOW_BASED_TTS = False  # Flow-based models
    ENABLE_DIFFUSION_TTS = False  # Diffusion models

    print("🧪 Experimental features enabled - use with caution!")


# ============================================================================
# 📊 SYSTEM INFORMATION DISPLAY
# ============================================================================

def display_system_info():
    """Display comprehensive system information"""
    import psutil
    import platform

    print("\n💻 SYSTEM INFORMATION")
    print("=" * 50)

    # System specs
    print(f"🖥️  OS: {platform.system()} {platform.release()}")
    print(f"🧠 CPU: {psutil.cpu_count()} cores")
    print(f"💾 RAM: {psutil.virtual_memory().total // (1024 ** 3)}GB")

    # GPU info
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)
            print(f"🎮 GPU {i}: {gpu_name} ({gpu_memory}GB)")
    else:
        print("🎮 GPU: Not available (CPU mode)")

    # Storage info
    disk_usage = psutil.disk_usage('.')
    free_gb = disk_usage.free // (1024 ** 3)
    total_gb = disk_usage.total // (1024 ** 3)
    print(f"💽 Storage: {free_gb}GB free / {total_gb}GB total")

    # Python environment
    print(f"🐍 Python: {platform.python_version()}")
    print(f"🔥 PyTorch: {torch.__version__}")

    # Current settings summary
    print(f"\n⚙️  CURRENT CONFIGURATION")
    print(f"   Device: {get_device()}")
    print(f"   Languages: {len(ACTIVE_LANGUAGES)}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Max Epochs: {MAX_EPOCHS}")

    # Recommendations
    recommendations = get_recommended_settings()
    if recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for key, value in recommendations.items():
            current_value = globals().get(key.upper(), 'Not Set')
            if current_value != value:
                print(f"   Consider setting {key.upper()} = {value} (current: {current_value})")


# ============================================================================
# 🔄 CONFIGURATION VALIDATION AND STARTUP
# ============================================================================

def startup_check():
    """Perform startup validation and show important information"""
    print("\n🚀 MULTILINGUAL TTS SYSTEM v2.0")
    print("📝 User Settings Loaded")

    # Validate critical settings
    warnings = validate_settings()
    if warnings:
        print(f"\n⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"   {warning}")
        print(f"   Consider reviewing your settings in config/user_settings.py")

    # Show active configuration
    active_info = get_active_languages_info()
    total_estimated_hours = sum(info['estimated_hours'] for info in active_info.values())

    print(f"\n📊 Active Configuration:")
    print(f"   Languages: {len(ACTIVE_LANGUAGES)} ({', '.join(ACTIVE_LANGUAGES)})")
    print(f"   Estimated Data: {total_estimated_hours:.1f} hours")
    print(f"   Training Device: {get_device().upper()}")
    print(f"   Datasets: {', '.join(DEFAULT_DATASETS)}")

    if total_estimated_hours > 100:
        print(f"\n💡 Tip: Large dataset ({total_estimated_hours:.1f}h) will take significant time and storage")
        print(f"   Consider using a preset configuration or reducing ACTIVE_LANGUAGES")


# Auto-run startup check when imported
if __name__ != "__main__":
    try:
        startup_check()
    except Exception as e:
        print(f"⚠️  Settings validation error: {e}")
        print(f"   System will continue with default settings")

# ============================================================================
# 📚 DOCUMENTATION LINKS
# ============================================================================

DOCUMENTATION_LINKS = {
    'system_architecture': 'docs/ARCHITECTURE.md',
    'dataset_guide': 'docs/DATASET_GUIDE.md',
    'training_guide': 'docs/TRAINING_GUIDE.md',
    'troubleshooting': 'docs/TROUBLESHOOTING.md',
    'linguistic_features': 'docs/LINGUISTIC_FEATURES.md',
    'api_reference': 'docs/API_REFERENCE.md'
}


def show_documentation_links():
    """Show available documentation"""
    print("\n📚 DOCUMENTATION")
    print("=" * 40)
    for doc_name, doc_path in DOCUMENTATION_LINKS.items():
        print(f"📖 {doc_name.replace('_', ' ').title()}: {doc_path}")
    print()


# Export key functions and variables for easy access
__all__ = [
    # Settings validation
    'validate_settings', 'get_recommended_settings', 'startup_check',

    # Preset configurations
    'apply_preset_config', 'get_available_presets',

    # Information display
    'print_current_settings', 'display_system_info', 'show_usage_tips',

    # Helper functions
    'get_device', 'get_active_languages_info',

    # Core settings (most commonly modified)
    'ACTIVE_LANGUAGES', 'DEFAULT_DATASETS', 'BATCH_SIZE', 'MAX_EPOCHS',
    'DEVICE', 'ENABLE_LINGUISTIC_FEATURES', 'ENABLE_SPEAKER_DIARIZATION'
]