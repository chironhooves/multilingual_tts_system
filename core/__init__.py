"""
Core Processing Package for Multilingual TTS System v2.0
Provides main processing components and easy imports
"""

# Core processing modules
from .common_voice_collector import CommonVoiceCollector, AdditionalDatasetCollector
from .data_collector import DataCollector
from .linguistic_processor import LinguisticProcessor
from .preprocessor import AudioPreprocessor, TextPreprocessor
from .speaker_id import SpeakerIdentificationSystem
from .aligner import ForcedAligner
from .trainer import TTSTrainer

# Version and metadata
__version__ = "2.0.0"
__description__ = "Core processing components for multilingual TTS"


# Main processor classes for easy access
class MultlingualTTSCore:
    """Main interface for all core processing components"""

    def __init__(self):
        """Initialize all core components"""
        self.data_collector = DataCollector()
        self.common_voice_collector = CommonVoiceCollector()
        self.additional_collector = AdditionalDatasetCollector()
        self.audio_preprocessor = AudioPreprocessor()
        self.text_preprocessor = TextPreprocessor()
        self.speaker_system = SpeakerIdentificationSystem()
        self.aligner = ForcedAligner()
        self.trainer = TTSTrainer()

        # Linguistic processors (language-specific)
        self.linguistic_processors = {}

    def get_linguistic_processor(self, language_code: str) -> LinguisticProcessor:
        """Get or create linguistic processor for a language"""
        if language_code not in self.linguistic_processors:
            self.linguistic_processors[language_code] = LinguisticProcessor(language_code)
        return self.linguistic_processors[language_code]

    def process_language_pipeline(self, language_code: str, stages: list = None):
        """Run complete processing pipeline for a language"""
        if stages is None:
            stages = ['collect', 'preprocess', 'align', 'train']

        results = {}

        if 'collect' in stages:
            print(f"🔄 Stage 1/4: Data Collection for {language_code}")
            results['collection'] = self.data_collector.collect_language_data(language_code)

        if 'preprocess' in stages:
            print(f"🔄 Stage 2/4: Audio/Text Processing for {language_code}")
            results['audio_processing'] = self.audio_preprocessor.process_language_audio(language_code)
            results['text_processing'] = self.text_preprocessor.process_language_text(language_code)

        if 'align' in stages:
            print(f"🔄 Stage 3/4: Forced Alignment for {language_code}")
            results['alignment'] = self.aligner.align_language_data(language_code)

        if 'train' in stages:
            print(f"🔄 Stage 4/4: TTS Training for {language_code}")
            results['training'] = self.trainer.train_single_language_model(language_code)

        return results

    def get_system_status(self):
        """Get status of all core components"""
        return {
            'data_collector': 'Ready',
            'audio_preprocessor': 'Ready',
            'text_preprocessor': 'Ready',
            'speaker_system': 'Ready',
            'aligner': 'Ready',
            'trainer': 'Ready',
            'linguistic_processors': len(self.linguistic_processors)
        }


# Convenience functions
def create_data_collector():
    """Create a new data collector instance"""
    return DataCollector()


def create_audio_preprocessor():
    """Create a new audio preprocessor instance"""
    return AudioPreprocessor()


def create_speaker_system():
    """Create a new speaker identification system"""
    return SpeakerIdentificationSystem()


def create_linguistic_processor(language_code: str):
    """Create linguistic processor for specific language"""
    return LinguisticProcessor(language_code)


def get_available_components():
    """Get list of available core components"""
    return [
        'CommonVoiceCollector - Open dataset collection',
        'DataCollector - Enhanced data collection with open datasets',
        'LinguisticProcessor - Advanced linguistic analysis',
        'AudioPreprocessor - Enhanced audio processing',
        'TextPreprocessor - Enhanced text processing',
        'SpeakerIdentificationSystem - Multi-speaker support',
        'ForcedAligner - Audio-text alignment',
        'TTSTrainer - Model training with multilingual support'
    ]


def check_dependencies():
    """Check if all required dependencies are available"""
    missing = []

    try:
        import torch
        import torchaudio
    except ImportError:
        missing.append('torch/torchaudio')

    try:
        import librosa
        import soundfile
    except ImportError:
        missing.append('librosa/soundfile')

    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        missing.append('numpy/pandas')

    try:
        import requests
    except ImportError:
        missing.append('requests')

    try:
        from resemblyzer import VoiceEncoder
    except ImportError:
        missing.append('resemblyzer')

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        missing.append('pyannote.audio')

    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All core dependencies available")
        return True


# Create global instance for easy access
multilingual_core = MultlingualTTSCore()

# Export everything
__all__ = [
    # Core classes
    'CommonVoiceCollector', 'AdditionalDatasetCollector',
    'DataCollector', 'LinguisticProcessor',
    'AudioPreprocessor', 'TextPreprocessor',
    'SpeakerIdentificationSystem', 'ForcedAligner', 'TTSTrainer',

    # Main interface
    'MultlingualTTSCore', 'multilingual_core',

    # Convenience functions
    'create_data_collector', 'create_audio_preprocessor',
    'create_speaker_system', 'create_linguistic_processor',
    'get_available_components', 'check_dependencies'
]

# Startup check
try:
    from config import user_settings

    if user_settings.ENABLE_DETAILED_LOGGING:
        deps_ok = check_dependencies()
        if not deps_ok:
            print("⚠️  Some dependencies missing. System may not work properly.")
except Exception:
    pass  # Continue silently if check fails