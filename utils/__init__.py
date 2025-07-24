"""
Utilities Package for Multilingual TTS System v2.0
Provides utility functions and helper classes
"""

from .visualization import ProgressVisualizer, TrainingVisualizer
from .audio_utils import AudioUtils
from .text_utils import TextUtils

# Version info
__version__ = "2.0.0"
__description__ = "Utility functions and helpers for multilingual TTS"

# Utility classes instances for easy access
progress_viz = ProgressVisualizer()
training_viz = TrainingVisualizer()
audio_utils = AudioUtils()
text_utils = TextUtils()


# Common utility functions
def format_duration(seconds):
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_file_size(bytes_size):
    """Format file size in bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"


def format_number(number):
    """Format large numbers with appropriate suffixes"""
    if number < 1000:
        return str(int(number))
    elif number < 1000000:
        return f"{number / 1000:.1f}K"
    elif number < 1000000000:
        return f"{number / 1000000:.1f}M"
    else:
        return f"{number / 1000000000:.1f}B"


def create_progress_bar(current, total, width=50, prefix="", suffix=""):
    """Create a text-based progress bar"""
    if total == 0:
        percent = 0
    else:
        percent = (current / total) * 100

    filled_width = int(width * current // total) if total > 0 else 0
    bar = '█' * filled_width + '-' * (width - filled_width)

    return f"{prefix}[{bar}] {percent:.1f}% {suffix}"


def safe_divide(numerator, denominator, default=0):
    """Safely divide two numbers, returning default if division by zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def ensure_dir(directory_path):
    """Ensure directory exists, create if it doesn't"""
    from pathlib import Path
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp():
    """Get current timestamp in ISO format"""
    from datetime import datetime
    return datetime.now().isoformat()


def load_json_safe(file_path, default=None):
    """Safely load JSON file, return default if failed"""
    import json
    from pathlib import Path

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return default if default is not None else {}


def save_json_safe(data, file_path):
    """Safely save data to JSON file"""
    import json
    from pathlib import Path

    try:
        # Ensure directory exists
        ensure_dir(Path(file_path).parent)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")
        return False


class PerformanceTimer:
    """Simple performance timer context manager"""

    def __init__(self, description="Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"⏱️  {self.description}: {format_duration(duration)}")


class SystemResourceMonitor:
    """Monitor system resources during processing"""

    def __init__(self):
        self.initial_memory = None
        self.peak_memory = None

    def start_monitoring(self):
        """Start monitoring system resources"""
        try:
            import psutil
            self.initial_memory = psutil.virtual_memory().used
            self.peak_memory = self.initial_memory
        except ImportError:
            print("⚠️  psutil not available for resource monitoring")

    def update_peak(self):
        """Update peak memory usage"""
        try:
            import psutil
            current_memory = psutil.virtual_memory().used
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
        except ImportError:
            pass

    def get_usage_summary(self):
        """Get resource usage summary"""
        if self.initial_memory is None:
            return "Resource monitoring not available"

        memory_used = self.peak_memory - self.initial_memory
        return f"Memory used: {format_file_size(memory_used)}"


def log_system_info():
    """Log comprehensive system information"""
    import platform
    import sys

    print(f"\n💻 SYSTEM INFORMATION")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version}")
    print(f"   Architecture: {platform.machine()}")

    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   CUDA: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"   CUDA: Not available")
    except ImportError:
        print(f"   PyTorch: Not installed")

    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   RAM: {format_file_size(memory.total)} total, {format_file_size(memory.available)} available")

        disk = psutil.disk_usage('.')
        print(f"   Disk: {format_file_size(disk.free)} free of {format_file_size(disk.total)}")
    except ImportError:
        print(f"   System info: Limited (install psutil for details)")


# Export all utilities
__all__ = [
    # Classes
    'ProgressVisualizer', 'TrainingVisualizer', 'AudioUtils', 'TextUtils',
    'PerformanceTimer', 'SystemResourceMonitor',

    # Class instances
    'progress_viz', 'training_viz', 'audio_utils', 'text_utils',

    # Utility functions
    'format_duration', 'format_file_size', 'format_number',
    'create_progress_bar', 'safe_divide', 'ensure_dir',
    'get_timestamp', 'load_json_safe', 'save_json_safe',
    'log_system_info'
]

# Quick system check on import
try:
    from config import user_settings

    if user_settings.ENABLE_DETAILED_LOGGING:
        print("🛠️  Utils package loaded successfully")
except Exception:
    pass  # Continue silently