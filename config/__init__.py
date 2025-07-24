"""
Configuration Package for Multilingual TTS System v2.0
Provides centralized configuration management and easy imports
"""

from .languages import IndianLanguages, indian_languages
from .settings import SystemSettings
from . import user_settings

# Version information
__version__ = "2.0.0"
__author__ = "Multilingual TTS System"
__description__ = "Configuration management for Indian language TTS system"

# Easy access to main configuration objects
languages = indian_languages
system_settings = SystemSettings()


# Quick access functions
def get_supported_languages():
    """Get list of supported language codes"""
    return languages.get_supported_languages()


def get_language_info(language_code):
    """Get information about a specific language"""
    return languages.get_language_info(language_code)


def get_dataset_info():
    """Get information about available datasets"""
    return languages.get_datasets_info()


def get_active_languages():
    """Get currently active languages from user settings"""
    return user_settings.ACTIVE_LANGUAGES


def get_system_device():
    """Get configured device for training"""
    return user_settings.get_device()


# Validation functions
def validate_configuration():
    """Validate current configuration and return warnings"""
    warnings = []

    # Validate user settings
    user_warnings = user_settings.validate_settings()
    warnings.extend(user_warnings)

    # Validate language-dataset compatibility
    for lang_code in user_settings.ACTIVE_LANGUAGES:
        if not languages.validate_language_code(lang_code):
            warnings.append(f"❌ Invalid language code: {lang_code}")
        else:
            lang_info = languages.get_language_info(lang_code)
            available_hours = lang_info['total_estimated_hours']
            if available_hours < user_settings.MIN_HOURS_FOR_TRAINING:
                warnings.append(
                    f"⚠️  {lang_code}: Only {available_hours}h available (need {user_settings.MIN_HOURS_FOR_TRAINING}h minimum)")

    return warnings


def print_configuration_summary():
    """Print a comprehensive configuration summary"""
    print("\n🎛️  SYSTEM CONFIGURATION SUMMARY")
    print("=" * 60)

    # Basic info
    print(f"📦 Version: {__version__}")
    print(f"🖥️  Device: {get_system_device()}")
    print(f"🌍 Active Languages: {len(get_active_languages())}")

    # Language details
    print(f"\n📋 Language Details:")
    for lang_code in user_settings.ACTIVE_LANGUAGES[:5]:  # Show first 5
        if languages.validate_language_code(lang_code):
            lang_info = languages.get_language_info(lang_code)
            print(f"   {lang_code}: {lang_info['native_name']} ({lang_info['total_estimated_hours']:.1f}h)")

    if len(user_settings.ACTIVE_LANGUAGES) > 5:
        print(f"   ... and {len(user_settings.ACTIVE_LANGUAGES) - 5} more")

    # Dataset info
    print(f"\n📊 Datasets: {', '.join(user_settings.DEFAULT_DATASETS)}")

    # Training settings
    print(f"\n🤖 Training Settings:")
    print(f"   Batch Size: {user_settings.BATCH_SIZE}")
    print(f"   Max Epochs: {user_settings.MAX_EPOCHS}")
    print(f"   Learning Rate: {user_settings.LEARNING_RATE}")

    # Warnings
    warnings = validate_configuration()
    if warnings:
        print(f"\n⚠️  Configuration Warnings ({len(warnings)}):")
        for warning in warnings[:3]:  # Show first 3 warnings
            print(f"   {warning}")
        if len(warnings) > 3:
            print(f"   ... and {len(warnings) - 3} more warnings")
    else:
        print(f"\n✅ Configuration validated successfully!")


# Export main objects for easy importing
__all__ = [
    'IndianLanguages', 'indian_languages', 'SystemSettings',
    'languages', 'system_settings', 'user_settings',
    'get_supported_languages', 'get_language_info', 'get_dataset_info',
    'get_active_languages', 'get_system_device',
    'validate_configuration', 'print_configuration_summary'
]

# Auto-validation on import (optional, can be disabled)
try:
    if user_settings.ENABLE_DETAILED_LOGGING:
        warnings = validate_configuration()
        if warnings and len(warnings) <= 3:  # Only show if few warnings
            print("⚠️  Configuration warnings found. Run config.print_configuration_summary() for details.")
except Exception:
    pass  # Silently continue if validation fails during import