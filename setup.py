#!/usr/bin/env python3
"""
Enhanced Setup Script for Multilingual TTS System v2.0
Handles installation, dependency checking, and system configuration
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import json


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║              🌟 MULTILINGUAL TTS SYSTEM v2.0 🌟               ║
║                   Enhanced Open Datasets Edition              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_system_requirements():
    """Check system requirements"""
    print("\n🔍 CHECKING SYSTEM REQUIREMENTS...")

    # Check available memory
    try:
        if platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                mem_total = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1]) // 1024
        else:
            import psutil
            mem_total = psutil.virtual_memory().total // (1024 ** 3)

        if mem_total < 8:
            print(f"⚠️  RAM: {mem_total}GB (16GB+ recommended)")
        else:
            print(f"✅ RAM: {mem_total}GB")
    except:
        print("⚠️  Could not check RAM")

    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (1024 ** 3)
        if free_gb < 50:
            print(f"⚠️  Free disk space: {free_gb}GB (100GB+ recommended)")
        else:
            print(f"✅ Free disk space: {free_gb}GB")
    except:
        print("⚠️  Could not check disk space")


def install_python_packages():
    """Install required Python packages"""
    print("\n📦 INSTALLING PYTHON PACKAGES...")

    packages = [
        # Core ML/AI
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "TTS>=0.22.0",
        "transformers>=4.20.0",
        "huggingface_hub>=0.16.0",

        # Audio processing
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "pyaudio>=0.2.11",
        "pydub>=0.25.0",
        "resemblyzer>=0.1.1",
        "pyannote.audio>=3.1.0",

        # Data processing
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "webvtt-py>=0.4.6",
        "datasets>=2.0.0",

        # Visualization
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "plotly>=5.0.0",

        # Web/Download
        "requests>=2.28.0",
        "tqdm>=4.64.0",

        # Language processing
        "indic-transliteration>=2.3.0",
        "regex>=2022.0.0"
    ]

    failed_packages = []

    for package in packages:
        try:
            print(f"Installing {package}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", package],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                failed_packages.append(package)
                print(f"❌ Failed to install {package}")
            else:
                print(f"✅ Installed {package}")
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")
            failed_packages.append(package)

    if failed_packages:
        print(f"\n⚠️  {len(failed_packages)} packages failed to install:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
        print("\nTry installing them manually with:")
        print(f"pip install {' '.join(failed_packages)}")
    else:
        print("\n✅ All Python packages installed successfully!")


def check_system_tools():
    """Check for required system tools"""
    print("\n🔧 CHECKING SYSTEM TOOLS...")

    tools = {
        'ffmpeg': 'sudo apt install ffmpeg',
        'espeak': 'sudo apt install espeak espeak-data libespeak1 libespeak-dev',
        'git': 'sudo apt install git'
    }

    missing_tools = []

    for tool, install_cmd in tools.items():
        try:
            result = subprocess.run([tool, '--version'],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {tool} is installed")
            else:
                missing_tools.append((tool, install_cmd))
        except FileNotFoundError:
            missing_tools.append((tool, install_cmd))
            print(f"❌ {tool} not found")

    if missing_tools:
        print(f"\n⚠️  Missing {len(missing_tools)} system tools:")
        for tool, cmd in missing_tools:
            print(f"   {tool}: {cmd}")
        return False

    return True


def create_directory_structure():
    """Create the required directory structure"""
    print("\n📁 CREATING DIRECTORY STRUCTURE...")

    directories = [
        "config",
        "core",
        "utils",
        "data",
        "models/individual",
        "models/unified",
        "models/checkpoints",
        "models/pretrained",
        "logs/training_logs",
        "logs/system_logs",
        "logs/visualizations",
        "temp/downloads",
        "temp/processing",
        "temp/alignment"
    ]

    # Add language-specific directories
    languages = ['hi', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or']

    for lang in languages:
        directories.extend([
            f"data/{lang}/raw_audio",
            f"data/{lang}/raw_text",
            f"data/{lang}/processed_audio",
            f"data/{lang}/processed_subtitles",
            f"data/{lang}/processed_text",
            f"data/{lang}/linguistic_features",
            f"data/{lang}/linguistic_text",
            f"data/{lang}/balanced_corpus",
            f"data/{lang}/diarization",
            f"data/{lang}/aligned",
            f"data/{lang}/aligned_segments",
            f"data/{lang}/manifests",
            f"data/{lang}/metadata",
            f"data/{lang}/logs",
            f"data/{lang}/common_voice",
            f"data/{lang}/openslr",
            f"data/{lang}/fleurs",
            f"data/{lang}/custom_recordings/audio",
            f"data/{lang}/custom_recordings/text",
            f"models/individual/{lang}"
        ])

    created_count = 0
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_count += 1

    print(f"✅ Created {created_count} directories")

    # Create __init__.py files
    init_files = ["config/__init__.py", "core/__init__.py", "utils/__init__.py"]
    for init_file in init_files:
        Path(init_file).touch()

    print("✅ Created Python package structure")


def create_requirements_file():
    """Create requirements.txt file"""
    print("\n📝 CREATING REQUIREMENTS FILE...")

    requirements = """# Core ML/AI Libraries
torch>=2.0.0
torchaudio>=2.0.0
TTS>=0.22.0
resemblyzer>=0.1.1
pyannote.audio>=3.1.0
transformers>=4.20.0
huggingface_hub>=0.16.0

# Audio Processing
librosa>=0.10.0
soundfile>=0.12.0
pyaudio>=0.2.11
pydub>=0.25.0

# Data Processing
numpy>=1.21.0
pandas>=1.5.0
scikit-learn>=1.2.0
webvtt-py>=0.4.6
datasets>=2.0.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.0.0

# Web/Download
requests>=2.28.0
tqdm>=4.64.0

# Language Processing
indic-transliteration>=2.3.0
regex>=2022.0.0

# Optional but recommended
# montreal-forced-alignment  # Install via conda
"""

    with open("requirements.txt", "w") as f:
        f.write(requirements)

    print("✅ Created requirements.txt")


def create_config_files():
    """Create basic configuration files"""
    print("\n⚙️  CREATING CONFIGURATION FILES...")

    # Basic system settings
    settings_config = {
        "SAMPLE_RATE": 16000,
        "HOP_LENGTH": 256,
        "WIN_LENGTH": 1024,
        "N_FFT": 1024,
        "N_MELS": 80,
        "MAX_WAV_VALUE": 32768.0,
        "DEVICE": "auto",  # auto-detect GPU/CPU
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.0001,
        "MAX_VIDEOS_PER_CHANNEL": 100,
        "MIN_AUDIO_DURATION": 1.0,
        "MAX_AUDIO_DURATION": 20.0,
        "ACTIVE_LANGUAGES": ["hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa", "or"],
        "DEFAULT_DATASETS": ["common_voice", "google_fleurs", "custom_recordings"]
    }

    config_dir = Path("config")

    # Save settings
    with open(config_dir / "user_settings.py", "w") as f:
        f.write("# User Configuration Settings\n")
        f.write("# Modify these values to customize the system\n\n")
        for key, value in settings_config.items():
            if isinstance(value, str):
                f.write(f'{key} = "{value}"\n')
            else:
                f.write(f'{key} = {value}\n')

    print("✅ Created configuration files")


def check_gpu_availability():
    """Check for GPU availability"""
    print("\n🎮 CHECKING GPU AVAILABILITY...")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA GPU available: {gpu_name} ({gpu_count} device(s))")
            return True
        else:
            print("⚠️  No CUDA GPU detected - using CPU (training will be slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not yet installed - GPU check will be done later")
        return False


def create_getting_started_guide():
    """Create a getting started guide"""
    print("\n📖 CREATING GETTING STARTED GUIDE...")

    guide = """# 🚀 Getting Started with Multilingual TTS System v2.0

## Quick Start (5 minutes)
1. Run the system: `python main.py`
2. Select: Option 1 - System Information (verify setup)
3. Select: Option 3 - Data Collection → Single Language
4. Choose Hindi for best results
5. Wait 30-60 minutes for data download
6. Select: Option 4 - Audio Processing  
7. Select: Option 6 - TTS Training

## Expected Timeline
- **Setup**: 5-10 minutes
- **Data Collection (Hindi)**: 30-60 minutes  
- **Processing**: 15-30 minutes
- **Training**: 2-6 hours (depending on GPU)
- **Total**: 3-8 hours for first working model

## System Features
✅ 100% Legal open datasets (Mozilla Common Voice, Google FLEURS)
✅ 10 Indian languages supported
✅ Advanced linguistic processing  
✅ Multi-speaker voice cloning
✅ Professional TTS quality
✅ No YouTube dependencies

## Architecture
```
main.py → Data Collection → Audio Processing → Training → TTS Model
```

## Troubleshooting
- **Out of memory**: Reduce BATCH_SIZE in config/user_settings.py
- **No GPU**: System works on CPU (slower but functional)
- **Download fails**: Check internet connection, try different dataset

## Next Steps
1. Start with Hindi (best dataset availability)
2. Add other languages gradually  
3. Experiment with custom recordings
4. Try multilingual unified model

For detailed documentation, see the code comments and docstrings.
Happy TTS building! 🎉
"""

    with open("GETTING_STARTED.md", "w") as f:
        f.write(guide)

    print("✅ Created GETTING_STARTED.md")


def run_system_test():
    """Run a basic system test"""
    print("\n🧪 RUNNING SYSTEM TEST...")

    try:
        # Test imports
        print("Testing core imports...")

        test_imports = [
            "import torch",
            "import librosa",
            "import soundfile",
            "import numpy as np",
            "import pandas as pd",
            "import requests"
        ]

        for import_stmt in test_imports:
            try:
                exec(import_stmt)
                print(f"✅ {import_stmt}")
            except ImportError as e:
                print(f"❌ {import_stmt} - {e}")
                return False

        # Test basic functionality
        print("Testing basic audio processing...")
        import librosa
        import numpy as np

        # Create dummy audio
        dummy_audio = np.random.randn(16000)  # 1 second at 16kHz

        # Test librosa functions
        mfccs = librosa.feature.mfcc(y=dummy_audio, sr=16000, n_mfcc=13)
        print(f"✅ Audio processing works (MFCC shape: {mfccs.shape})")

        print("\n🎉 SYSTEM TEST PASSED!")
        return True

    except Exception as e:
        print(f"\n❌ SYSTEM TEST FAILED: {e}")
        return False


def main():
    """Main setup function"""
    print_banner()

    # Check prerequisites
    if not check_python_version():
        sys.exit(1)

    check_system_requirements()
    tools_ok = check_system_tools()

    if not tools_ok:
        print("\n⚠️  Please install missing system tools and run setup again")

    # Create structure
    create_directory_structure()
    create_requirements_file()
    create_config_files()
    create_getting_started_guide()

    # Install packages
    install_python_packages()

    # Check GPU
    check_gpu_availability()

    # Run test
    test_passed = run_system_test()

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)

    if test_passed:
        print("✅ System is ready to use!")
        print("\nNext steps:")
        print("1. Run: python main.py")
        print("2. Select Option 1 to verify system")
        print("3. Select Option 3 to start data collection")
        print("4. See GETTING_STARTED.md for detailed guide")
    else:
        print("⚠️  Some issues detected. Check error messages above.")
        print("You can still try running: python main.py")

    print(f"\n📊 System Overview:")
    print(f"   - 10 Indian languages supported")
    print(f"   - 100% legal open datasets")
    print(f"   - Advanced linguistic processing")
    print(f"   - Multi-speaker support")
    print(f"   - Expected data: 500+ hours total")

    print(f"\n💡 Pro tip: Start with Hindi for best results!")


if __name__ == "__main__":
    main()