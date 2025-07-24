"""
System settings and configuration for the Multilingual TTS System
"""

import os
import torch
from pathlib import Path


class SystemSettings:
    """Global system settings and configuration"""

    def __init__(self):
        # Hardware Configuration
        self.CUDA_AVAILABLE = torch.cuda.is_available()
        self.DEVICE = torch.device("cuda" if self.CUDA_AVAILABLE else "cpu")
        self.NUM_GPUS = torch.cuda.device_count() if self.CUDA_AVAILABLE else 0
        self.NUM_WORKERS = os.cpu_count()

        # Storage Configuration
        self.BASE_DIR = Path.cwd()
        self.DATA_DIR = self.BASE_DIR / "data"
        self.MODELS_DIR = self.BASE_DIR / "models"
        self.LOGS_DIR = self.BASE_DIR / "logs"
        self.TEMP_DIR = self.BASE_DIR / "temp"

        # Target Quality Settings
        self.TARGET_ACCURACY = 80  # 80% accuracy target
        self.STORAGE_REQUIREMENTS = "1-1.5 TB"

        # Audio Processing Settings
        self.SAMPLE_RATE = 16000
        self.AUDIO_FORMAT = "wav"
        self.AUDIO_CHANNELS = 1  # Mono
        self.AUDIO_BITDEPTH = 16

        # Data Collection Settings
        self.MAX_VIDEOS_PER_CHANNEL = 50
        self.MAX_VIDEO_DURATION = 3600  # 1 hour max per video
        self.MIN_SEGMENT_DURATION = 1.0  # 1 second minimum
        self.MAX_SEGMENT_DURATION = 20.0  # 20 seconds maximum

        # TTS Training Settings
        self.BATCH_SIZE = 32 if self.CUDA_AVAILABLE else 8
        self.LEARNING_RATE = 1e-4
        self.MAX_EPOCHS = 1000
        self.EARLY_STOPPING_PATIENCE = 50
        self.VALIDATION_SPLIT = 0.1

        # Speaker Identification Settings
        self.SPEAKER_SIMILARITY_THRESHOLD = 0.7
        self.MIN_SPEAKER_SEGMENTS = 5
        self.MIN_SPEAKER_DURATION = 30.0  # 30 seconds minimum per speaker

        # Forced Alignment Settings
        self.MFA_MODEL_PATH = None
        self.ALIGNMENT_TIER = "words"

        # File Naming Conventions
        self.TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

        # Logging Configuration
        self.LOG_LEVEL = "INFO"
        self.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def get_storage_breakdown(self):
        """Get detailed storage requirements breakdown"""
        return {
            "raw_videos": "500-800 GB",
            "processed_audio": "150-200 GB",
            "aligned_segments": "100-150 GB",
            "model_checkpoints": "50-100 GB",
            "working_space": "200 GB",
            "total_recommended": "1-1.5 TB"
        }

    def get_hardware_info(self):
        """Get hardware configuration information"""
        return {
            "cuda_available": self.CUDA_AVAILABLE,
            "device": str(self.DEVICE),
            "num_gpus": self.NUM_GPUS,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(self.NUM_GPUS)] if self.CUDA_AVAILABLE else [],
            "cpu_count": self.NUM_WORKERS,
            "total_memory": f"{torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)} GB" if self.CUDA_AVAILABLE else "N/A"
        }

    def get_training_config(self, language_code=None):
        """Get training configuration for a specific language or general"""
        config = {
            "batch_size": self.BATCH_SIZE,
            "learning_rate": self.LEARNING_RATE,
            "max_epochs": self.MAX_EPOCHS,
            "early_stopping_patience": self.EARLY_STOPPING_PATIENCE,
            "validation_split": self.VALIDATION_SPLIT,
            "device": str(self.DEVICE),
            "sample_rate": self.SAMPLE_RATE
        }

        if language_code:
            # Language-specific adjustments
            config["language"] = language_code

            # Adjust batch size for complex languages
            from config.languages import IndianLanguages
            lang_info = IndianLanguages().get_language_info(language_code)
            if lang_info.get('complexity') == 'high':
                config["batch_size"] = max(config["batch_size"] // 2, 4)
                config["learning_rate"] = config["learning_rate"] * 0.8

        return config

    def get_data_collection_config(self):
        """Get data collection configuration"""
        return {
            "max_videos_per_channel": self.MAX_VIDEOS_PER_CHANNEL,
            "max_video_duration": self.MAX_VIDEO_DURATION,
            "audio_format": self.AUDIO_FORMAT,
            "sample_rate": self.SAMPLE_RATE,
            "channels": self.AUDIO_CHANNELS,
            "min_segment_duration": self.MIN_SEGMENT_DURATION,
            "max_segment_duration": self.MAX_SEGMENT_DURATION
        }

    def get_speaker_id_config(self):
        """Get speaker identification configuration"""
        return {
            "similarity_threshold": self.SPEAKER_SIMILARITY_THRESHOLD,
            "min_segments": self.MIN_SPEAKER_SEGMENTS,
            "min_duration": self.MIN_SPEAKER_DURATION,
            "sample_rate": self.SAMPLE_RATE
        }

    def update_setting(self, key, value):
        """Update a system setting"""
        if hasattr(self, key):
            setattr(self, key, value)
            return True
        return False

    def validate_storage_space(self, path=None):
        """Validate available storage space"""
        import shutil

        if path is None:
            path = self.BASE_DIR

        try:
            total, used, free = shutil.disk_usage(path)
            free_gb = free // (1024 ** 3)

            # Minimum required: 1TB = 1000GB
            min_required_gb = 1000

            return {
                "total_gb": total // (1024 ** 3),
                "used_gb": used // (1024 ** 3),
                "free_gb": free_gb,
                "sufficient": free_gb >= min_required_gb,
                "min_required_gb": min_required_gb
            }
        except Exception as e:
            return {"error": str(e)}

    def get_recommended_settings_for_hardware(self):
        """Get recommended settings based on available hardware"""
        recommendations = {}

        if self.CUDA_AVAILABLE:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory // (1024 ** 3)

            if gpu_memory_gb >= 24:  # High-end GPU
                recommendations.update({
                    "batch_size": 64,
                    "max_videos_per_channel": 100,
                    "parallel_processing": True,
                    "recommendation": "High-end setup - can handle full dataset"
                })
            elif gpu_memory_gb >= 12:  # Mid-range GPU
                recommendations.update({
                    "batch_size": 32,
                    "max_videos_per_channel": 75,
                    "parallel_processing": True,
                    "recommendation": "Mid-range setup - good for most languages"
                })
            elif gpu_memory_gb >= 6:  # Entry-level GPU
                recommendations.update({
                    "batch_size": 16,
                    "max_videos_per_channel": 50,
                    "parallel_processing": False,
                    "recommendation": "Entry-level setup - process languages sequentially"
                })
            else:  # Low memory GPU
                recommendations.update({
                    "batch_size": 8,
                    "max_videos_per_channel": 25,
                    "parallel_processing": False,
                    "recommendation": "Low memory - consider CPU training or smaller datasets"
                })
        else:  # CPU only
            recommendations.update({
                "batch_size": 4,
                "max_videos_per_channel": 25,
                "parallel_processing": False,
                "recommendation": "CPU only - will be slow, consider using smaller datasets"
            })

        return recommendations


# Quality targets for 80% accuracy
QUALITY_TARGETS = {
    "minimum_hours_per_language": {
        "high_resource": 60,  # Hindi, Tamil, Telugu, Bengali
        "medium_resource": 40,  # Marathi, Gujarati, Kannada
        "low_resource": 25  # Malayalam, Punjabi, Odia
    },
    "minimum_speakers_per_language": 50,
    "minimum_segments_per_speaker": 100,
    "audio_quality": {
        "min_snr_db": 15,  # Signal-to-noise ratio
        "max_background_noise": "moderate",
        "acceptable_compression": "mp3 128kbps or better"
    },
    "text_quality": {
        "min_subtitle_accuracy": 0.85,
        "max_oov_rate": 0.05,  # Out-of-vocabulary words
        "min_text_coverage": 0.9
    }
}

# Default model architectures for different scenarios
MODEL_ARCHITECTURES = {
    "fast_inference": {
        "model_type": "FastSpeech2",
        "vocoder": "HiFiGAN",
        "description": "Fast inference, good quality"
    },
    "high_quality": {
        "model_type": "Tacotron2",
        "vocoder": "WaveGlow",
        "description": "Slower but higher quality"
    },
    "multilingual": {
        "model_type": "YourTTS",
        "vocoder": "HiFiGAN",
        "description": "Supports multiple languages and speakers"
    },
    "low_resource": {
        "model_type": "FastSpeech2",
        "vocoder": "MelGAN",
        "description": "Works well with limited data"
    }
}

# Supported file formats
SUPPORTED_FORMATS = {
    "audio_input": [".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"],
    "audio_output": [".wav"],
    "subtitle_input": [".vtt", ".srt", ".ass"],
    "text_input": [".txt", ".json"],
    "model_formats": [".pth", ".pt", ".ckpt"]
}