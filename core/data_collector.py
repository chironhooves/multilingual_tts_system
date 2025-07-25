"""
Data Collection Module for Multilingual TTS System
Handles open dataset downloading and processing (Common Voice, OpenSLR, FLEURS, etc.)
Replaces YouTube dependency with legal, open datasets
"""

import os
import subprocess
import json
import logging
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional, Callable
import time
import re
from datetime import datetime
import librosa
import soundfile as sf

from config.languages import IndianLanguages
from config.settings import SystemSettings
from core.common_voice_collector import CommonVoiceCollector, AdditionalDatasetCollector

logger = logging.getLogger(__name__)


class DataCollector:
    """Enhanced data collector using open datasets instead of YouTube"""

    def __init__(self, base_dir='data'):
        self.base_dir = Path(base_dir)
        self.settings = SystemSettings()
        self.languages = IndianLanguages()

        # Initialize dataset collectors
        self.common_voice_collector = CommonVoiceCollector()
        self.additional_collector = AdditionalDatasetCollector()

        # Statistics
        self.stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_duration': 0,
            'languages_processed': set()
        }

        logger.info("📊 Enhanced DataCollector with open datasets initialized")

    def setup_directories(self, language_code: str) -> Dict[str, Path]:
        """Setup directory structure for a language"""
        lang_dir = self.base_dir / language_code

        directories = {
            'raw_audio': lang_dir / 'raw_audio',
            'raw_text': lang_dir / 'raw_text',
            'processed_audio': lang_dir / 'processed_audio',
            'processed_subtitles': lang_dir / 'processed_subtitles',
            'processed_text': lang_dir / 'processed_text',
            'metadata': lang_dir / 'metadata',
            'logs': lang_dir / 'logs',
            # Open dataset specific directories
            'common_voice': lang_dir / 'common_voice',
            'openslr': lang_dir / 'openslr',
            'fleurs': lang_dir / 'fleurs',
            'custom_recordings': lang_dir / 'custom_recordings'
        }

        for dir_path in directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Created directory structure for {language_code}")
        return directories

    def check_dependencies(self) -> bool:
        """Check if required tools are installed"""
        required_tools = ['ffmpeg']  # Removed yt-dlp dependency
        missing_tools = []

        for tool in required_tools:
            try:
                result = subprocess.run([tool, '--version'],
                                        capture_output=True, text=True)
                if result.returncode != 0:
                    missing_tools.append(tool)
            except FileNotFoundError:
                missing_tools.append(tool)

        # Check Python packages
        try:
            import requests
            import pandas as pd
        except ImportError as e:
            logger.error(f"Missing Python package: {e}")
            return False

        if missing_tools:
            logger.error(f"Missing required tools: {', '.join(missing_tools)}")
            logger.info("Install with: sudo apt install ffmpeg")
            return False

        logger.info("✅ All required tools are available")
        return True

    def list_available_datasets(self, language_code: str = None) -> Dict:
        """List all available open datasets"""
        logger.info(f"📋 Listing available datasets for {language_code or 'all languages'}")

        available_datasets = {}

        # Common Voice datasets
        cv_datasets = self.common_voice_collector.list_available_datasets(language_code)
        if cv_datasets:
            available_datasets.update(cv_datasets)

        # Additional datasets
        additional_sources = self.additional_collector.additional_sources
        for source_name, source_info in additional_sources.items():
            if not language_code or language_code in source_info.get('languages', []):
                available_datasets[source_name] = {
                    'name': source_info['name'],
                    'description': source_info.get('description', ''),
                    'url': source_info.get('url', ''),
                    'available_for_language': language_code in source_info.get('languages', [])
                }

        return available_datasets

    def collect_language_data(self, language_code: str, datasets: List[str] = None,
                              callback: Optional[Callable] = None) -> Dict:
        """Collect data for a specific language from open datasets"""
        logger.info(f"🌟 Starting open dataset collection for {language_code}")

        if not self.check_dependencies():
            return {'success': False, 'error': 'Missing dependencies'}

        # Validate language
        if not self.languages.validate_language_code(language_code):
            return {'success': False, 'error': f'Unsupported language: {language_code}'}

        # Setup directories
        directories = self.setup_directories(language_code)

        # Default datasets to try
        if datasets is None:
            datasets = ['common_voice', 'google_fleurs', 'openslr', 'custom_recordings']

        # Collection results
        results = {
            'language_code': language_code,
            'datasets_attempted': len(datasets),
            'datasets_successful': 0,
            'total_segments': 0,
            'total_duration': 0,
            'collection_method': 'open_datasets',
            'results_by_dataset': {}
        }

        # Process each dataset
        for dataset_name in datasets:
            logger.info(f"📥 Collecting data from {dataset_name} for {language_code}")

            try:
                if dataset_name == 'common_voice':
                    dataset_result = self.collect_common_voice_data(language_code)
                elif dataset_name == 'google_fleurs':
                    dataset_result = self.collect_fleurs_data(language_code)
                elif dataset_name == 'openslr':
                    dataset_result = self.collect_openslr_data(language_code)
                elif dataset_name == 'custom_recordings':
                    dataset_result = self.collect_custom_recordings(language_code)
                else:
                    dataset_result = {'success': False, 'error': f'Unknown dataset: {dataset_name}'}

                results['results_by_dataset'][dataset_name] = dataset_result

                if dataset_result.get('success', False):
                    results['datasets_successful'] += 1
                    segments = dataset_result.get('segments', dataset_result.get('processed_segments', 0))
                    results['total_segments'] += segments

                    # Estimate duration (approximate)
                    estimated_duration = segments * 5  # Assume 5 seconds per segment
                    results['total_duration'] += estimated_duration

                    logger.info(f"✅ {dataset_name}: {segments} segments collected")

                    # Update callback
                    if callback:
                        callback(language_code, 'data_collected', True)
                else:
                    logger.warning(
                        f"❌ {dataset_name} collection failed: {dataset_result.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Error collecting from {dataset_name}: {e}")
                results['results_by_dataset'][dataset_name] = {'success': False, 'error': str(e)}

        # Process collected data
        if results['total_segments'] > 0:
            processing_result = self.process_collected_data(language_code)
            results['processing_result'] = processing_result

        # Update global stats
        self.stats['languages_processed'].add(language_code)
        self.stats['total_downloads'] += results['datasets_attempted']
        self.stats['successful_downloads'] += results['datasets_successful']
        self.stats['failed_downloads'] += (results['datasets_attempted'] - results['datasets_successful'])

        # Save collection summary
        summary_file = directories[
                           'metadata'] / f"open_dataset_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Open dataset collection completed for {language_code}")
        logger.info(f"   Successful datasets: {results['datasets_successful']}/{results['datasets_attempted']}")
        logger.info(f"   Total segments: {results['total_segments']}")
        logger.info(f"   Estimated duration: {results['total_duration'] / 3600:.1f} hours")
        logger.info(f"   Summary saved: {summary_file}")

        results['success'] = results['datasets_successful'] > 0
        return results

    def collect_common_voice_data(self, language_code: str) -> Dict:
        """Collect Common Voice data for a language"""
        try:
            # Try both train and dev sets
            results = {'success': False, 'total_segments': 0, 'subsets': {}}

            for subset in ['train', 'dev']:
                try:
                    result = self.common_voice_collector.download_common_voice_dataset(language_code, subset)
                    results['subsets'][subset] = result

                    if result.get('success', False):
                        results['total_segments'] += result.get('processed_segments', 0)
                        results['success'] = True

                except Exception as e:
                    logger.warning(f"Failed to collect Common Voice {subset} for {language_code}: {e}")
                    results['subsets'][subset] = {'success': False, 'error': str(e)}

            results['segments'] = results['total_segments']
            return results

        except Exception as e:
            logger.error(f"Common Voice collection failed: {e}")
            return {'success': False, 'error': str(e)}

    def collect_fleurs_data(self, language_code: str) -> Dict:
        """Collect Google FLEURS data"""
        try:
            return self.common_voice_collector.download_fleurs_dataset(language_code)
        except Exception as e:
            logger.error(f"FLEURS collection failed: {e}")
            return {'success': False, 'error': str(e)}

    def collect_openslr_data(self, language_code: str) -> Dict:
        """Collect OpenSLR data"""
        try:
            return self.common_voice_collector.download_openslr_dataset(language_code)
        except Exception as e:
            logger.error(f"OpenSLR collection failed: {e}")
            return {'success': False, 'error': str(e)}

    def collect_custom_recordings(self, language_code: str) -> Dict:
        """Collect and process custom recordings"""
        try:
            # First setup the interface if it doesn't exist
            setup_result = self.additional_collector.setup_custom_recording_interface(language_code)
            if not setup_result['success']:
                return setup_result

            # Check if user has added any recordings
            custom_dir = Path("data") / language_code / "custom_recordings"
            audio_dir = custom_dir / "audio"

            audio_files = []
            if audio_dir.exists():
                for ext in ['.wav', '.mp3', '.flac', '.m4a']:
                    audio_files.extend(audio_dir.glob(f'*{ext}'))

            if not audio_files:
                return {
                    'success': True,
                    'segments': 0,
                    'message': f'Custom recording interface ready at {custom_dir}. Add your recordings and run again.',
                    'instructions': setup_result['instructions']
                }

            # Process the recordings
            return self.additional_collector.process_custom_recordings(language_code)

        except Exception as e:
            logger.error(f"Custom recordings collection failed: {e}")
            return {'success': False, 'error': str(e)}

    def process_collected_data(self, language_code: str) -> Dict:
        """Process and standardize collected data from various sources"""
        logger.info(f"🔄 Processing collected data for {language_code}")

        base_dir = Path("data") / language_code
        raw_audio_dir = base_dir / "raw_audio"
        raw_text_dir = base_dir / "raw_text"
        processed_audio_dir = base_dir / "processed_audio"

        processed_audio_dir.mkdir(exist_ok=True)

        if not raw_audio_dir.exists():
            return {'success': False, 'error': 'No raw audio directory found'}

        # Get all audio files
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.m4a']:
            audio_files.extend(raw_audio_dir.glob(f'*{ext}'))

        if not audio_files:
            return {'success': False, 'error': 'No audio files found to process'}

        logger.info(f"Found {len(audio_files)} audio files to process")

        processed_count = 0
        total_duration = 0

        for audio_file in audio_files:
            try:
                # Load and standardize audio
                audio, sr = librosa.load(audio_file, sr=self.settings.SAMPLE_RATE, mono=True)

                # Basic quality checks
                duration = len(audio) / sr
                if duration < 0.5 or duration > 30:  # Skip very short or very long clips
                    continue

                # Normalize audio
                audio = librosa.util.normalize(audio)

                # Save processed audio
                output_file = processed_audio_dir / f"{audio_file.stem}_processed.wav"
                sf.write(output_file, audio, sr)

                processed_count += 1
                total_duration += duration

                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} files...")

            except Exception as e:
                logger.warning(f"Error processing {audio_file}: {e}")

        result = {
            'success': processed_count > 0,
            'processed_files': processed_count,
            'total_files': len(audio_files),
            'total_duration': total_duration,
            'processing_rate': processed_count / len(audio_files) if audio_files else 0
        }

        logger.info(f"✅ Data processing completed: {processed_count}/{len(audio_files)} files processed")
        logger.info(f"   Total duration: {total_duration / 3600:.2f} hours")

        return result

    def collect_all_languages(self, datasets: List[str] = None, max_workers: int = 2,
                              callback: Optional[Callable] = None) -> Dict:
        """Collect data for all supported languages from open datasets"""
        logger.info("🌍 Starting data collection for all languages from open datasets")

        supported_languages = self.languages.get_supported_languages()

        all_results = {
            'total_languages': len(supported_languages),
            'successful_languages': 0,
            'failed_languages': 0,
            'start_time': datetime.now().isoformat(),
            'collection_method': 'open_datasets',
            'results_by_language': {}
        }

        # Process languages with limited parallelism to avoid overwhelming servers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_lang = {
                executor.submit(self.collect_language_data, lang_code, datasets, callback): lang_code
                for lang_code in supported_languages
            }

            for future in concurrent.futures.as_completed(future_to_lang):
                lang_code = future_to_lang[future]
                try:
                    result = future.result()
                    all_results['results_by_language'][lang_code] = result

                    if result.get('success', False):
                        all_results['successful_languages'] += 1
                        logger.info(f"✅ {lang_code}: {result.get('total_segments', 0)} segments collected")
                    else:
                        all_results['failed_languages'] += 1
                        logger.warning(f"❌ {lang_code}: Collection failed")

                except Exception as e:
                    logger.error(f"Error collecting data for {lang_code}: {e}")
                    all_results['failed_languages'] += 1
                    all_results['results_by_language'][lang_code] = {
                        'success': False,
                        'error': str(e)
                    }

        all_results['end_time'] = datetime.now().isoformat()

        # Save overall summary
        summary_file = self.base_dir / f"all_languages_open_datasets_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logger.info("✅ All languages data collection from open datasets completed")
        logger.info(f"   Successful: {all_results['successful_languages']}/{all_results['total_languages']}")
        logger.info(f"   Summary saved: {summary_file}")

        return all_results

    def show_collection_menu(self):
        """Interactive menu for data collection options"""
        print("\n" + "=" * 60)
        print("📊 OPEN DATASET COLLECTION OPTIONS")
        print("=" * 60)
        print("1. Single Language - All Available Datasets")
        print("2. Single Language - Specific Dataset")
        print("3. All Languages - Common Voice Only")
        print("4. All Languages - All Available Datasets")
        print("5. Setup Custom Recordings Interface")
        print("6. List Available Datasets")
        print("7. Collection Statistics")
        print("8. Back to Main Menu")

        choice = input("\nSelect option (1-8): ").strip()

        if choice == '1':
            self.single_language_all_datasets()
        elif choice == '2':
            self.single_language_specific_dataset()
        elif choice == '3':
            self.all_languages_common_voice()
        elif choice == '4':
            self.all_languages_all_datasets()
        elif choice == '5':
            self.setup_custom_recordings_menu()
        elif choice == '6':
            self.show_available_datasets()
        elif choice == '7':
            self.show_collection_statistics()

    def single_language_all_datasets(self):
        """Collect all available datasets for a single language"""
        print("\n📋 Select Language:")
        languages = list(self.languages.LANGUAGES.items())
        for i, (code, info) in enumerate(languages, 1):
            print(f"{i:2d}. {info['native_name']} ({info['name']})")

        try:
            choice = int(input("\nEnter language number: "))
            if 1 <= choice <= len(languages):
                lang_code = languages[choice - 1][0]

                print(f"\n🚀 Collecting all available datasets for {lang_code}...")
                result = self.collect_language_data(lang_code)

                if result['success']:
                    print(f"✅ Collection successful!")
                    print(f"   Datasets collected: {result['datasets_successful']}/{result['datasets_attempted']}")
                    print(f"   Total segments: {result['total_segments']}")
                    print(f"   Estimated duration: {result['total_duration'] / 3600:.1f} hours")
                else:
                    print(f"❌ Collection failed: {result.get('error', 'Unknown error')}")
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Please enter a valid number")

    def show_available_datasets(self):
        """Show all available datasets"""
        print("\n📊 Available Open Datasets:")
        print("=" * 50)

        available = self.list_available_datasets()

        for dataset_name, dataset_info in available.items():
            print(f"\n🔹 {dataset_info['name']}")
            if 'description' in dataset_info:
                print(f"   Description: {dataset_info['description']}")
            if 'languages' in dataset_info:
                langs = list(dataset_info['languages'].keys())
                print(f"   Languages: {', '.join(langs[:5])}")
                if len(langs) > 5:
                    print(f"   ... and {len(langs) - 5} more")

        print(f"\n📈 Total datasets available: {len(available)}")
        input("\nPress Enter to continue...")

    def show_collection_statistics(self):
        """Show collection statistics"""
        print("\n📊 Collection Statistics:")
        print("=" * 40)
        print(f"Total downloads attempted: {self.stats['total_downloads']}")
        print(f"Successful downloads: {self.stats['successful_downloads']}")
        print(f"Failed downloads: {self.stats['failed_downloads']}")
        print(f"Languages processed: {len(self.stats['languages_processed'])}")
        print(f"Estimated total duration: {self.stats['total_duration'] / 3600:.1f} hours")

        if self.stats['languages_processed']:
            print(f"Processed languages: {', '.join(self.stats['languages_processed'])}")

        input("\nPress Enter to continue...")

    def setup_custom_recordings_menu(self):
        """Setup custom recordings for selected language"""
        print("\n🎙️ Custom Recordings Setup")
        print("=" * 40)

        languages = list(self.languages.LANGUAGES.items())
        for i, (code, info) in enumerate(languages, 1):
            print(f"{i:2d}. {info['native_name']} ({info['name']})")

        try:
            choice = int(input("\nSelect language for custom recordings: "))
            if 1 <= choice <= len(languages):
                lang_code = languages[choice - 1][0]

                result = self.additional_collector.setup_custom_recording_interface(lang_code)

                if result['success']:
                    print(f"\n✅ Custom recording interface ready!")
                    print(f"📁 Directory: {result['custom_dir']}")
                    print("\n📋 Next Steps:")
                    print("1. Add your audio files to the 'audio' folder")
                    print("2. Add corresponding text files to the 'text' folder")
                    print("3. Run data collection again to process your recordings")
                    print("\n💡 See README.md in the custom_recordings folder for detailed instructions")
                else:
                    print(f"❌ Setup failed: {result.get('error')}")
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Please enter a valid number")

        input("\nPress Enter to continue...")


def main():
    """Test the enhanced data collector"""
    collector = DataCollector()

    # Show available datasets
    available = collector.list_available_datasets()
    print("Available open datasets:")
    for dataset, info in available.items():
        print(f"  {dataset}: {info['name']}")

    # Show YouTube alternative message
    print(collector.get_youtube_alternative_message())


if __name__ == "__main__":
    main()

