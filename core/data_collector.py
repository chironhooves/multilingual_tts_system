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

    """Get message about YouTube alternatives"""
    return """
🚨 YOUTUBE ALTERNATIVES - LEGAL & RELIABLE DATASETS

Instead of YouTube (which has legal/technical limitations), this system uses:

✅ OPEN & LEGAL DATASETS:
• Mozilla Common Voice - Crowd-sourced recordings with CC-0 license
• Google FLEURS - Multilingual speech corpus  
• OpenSLR - Open speech and language resources
• IIT/Academic datasets - Research-grade corpora
• Custom recordings - Your own contributed data

🎯 ADVANTAGES:
• 100% Legal and redistributable
• High-quality, clean recordings
• Consistent format and metadata
• No rate limiting or blocking
• Research-grade quality
• Designed specifically for TTS training

📊 EXPECTED DATA VOLUMES:
• Hindi: 50-100 hours (Common Voice + FLEURS)
• Tamil: 30-50 hours
• Telugu: 25-40 hours  
• Other languages: 15-30 hours each

🚀 GETTING STARTED:
Run: python main.py -> Option 3 -> Data Collection
The system will automatically download and process these legal datasets!
"""
"""
Data Collection Module for Multilingual TTS System
Handles YouTube video downloading, audio extraction, and subtitle processing
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
import webvtt
from datetime import datetime
import librosa
import soundfile as sf

from config.languages import IndianLanguages
from config.settings import SystemSettings

logger = logging.getLogger(__name__)


class DataCollector:
    """Handles data collection from YouTube sources"""

    def __init__(self, base_dir='data'):
        self.base_dir = Path(base_dir)
        self.settings = SystemSettings()
        self.languages = IndianLanguages()
        self.stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_duration': 0,
            'languages_processed': set()
        }

    def setup_directories(self, language_code: str) -> Dict[str, Path]:
        """Setup directory structure for a language"""
        lang_dir = self.base_dir / language_code

        directories = {
            'raw_videos': lang_dir / 'raw_videos',
            'raw_audio': lang_dir / 'raw_audio',
            'processed_audio': lang_dir / 'processed_audio',
            'subtitles': lang_dir / 'subtitles',
            'processed_subtitles': lang_dir / 'processed_subtitles',
            'metadata': lang_dir / 'metadata',
            'logs': lang_dir / 'logs'
        }

        for dir_path in directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Created directory structure for {language_code}")
        return directories

    def check_dependencies(self) -> bool:
        """Check if required tools are installed"""
        required_tools = ['yt-dlp', 'ffmpeg']
        missing_tools = []

        for tool in required_tools:
            try:
                result = subprocess.run([tool, '--version'],
                                        capture_output=True, text=True)
                if result.returncode != 0:
                    missing_tools.append(tool)
            except FileNotFoundError:
                missing_tools.append(tool)

        if missing_tools:
            logger.error(f"Missing required tools: {', '.join(missing_tools)}")
            logger.info("Install with: sudo apt install yt-dlp ffmpeg")
            return False

        logger.info("✅ All required tools are available")
        return True

    def get_channel_videos(self, channel_url: str, max_videos: int = 50,
                           language_code: str = None) -> List[str]:
        """Get list of video URLs from a channel"""
        logger.info(f"📺 Getting videos from channel: {channel_url}")

        cmd = [
            'yt-dlp',
            '--flat-playlist',
            '--print', 'webpage_url',
            '--playlist-end', str(max_videos),
            channel_url
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                video_urls = [url.strip() for url in result.stdout.split('\n') if url.strip()]
                logger.info(f"✅ Found {len(video_urls)} videos")
                return video_urls
            else:
                logger.error(f"Failed to get videos: {result.stderr}")
                return []
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout getting videos from {channel_url}")
            return []
        except Exception as e:
            logger.error(f"Error getting videos: {e}")
            return []

    def download_video_with_subtitles(self, video_url: str, output_dir: Path,
                                      language_code: str) -> Dict:
        """Download a single video with audio and subtitles"""
        logger.info(f"⬇️ Downloading: {video_url}")

        # Create unique filename based on video ID
        video_id = self.extract_video_id(video_url)
        if not video_id:
            return {'success': False, 'error': 'Could not extract video ID'}

        base_filename = f"{video_id}"

        # Download audio
        audio_cmd = [
            'yt-dlp',
            '--extract-audio',
            '--audio-format', 'wav',
            '--audio-quality', '0',
            '--output', str(output_dir / 'raw_audio' / f'{base_filename}.%(ext)s'),
            '--max-duration', str(self.settings.MAX_VIDEO_DURATION),
            video_url
        ]

        # Download subtitles
        subtitle_cmd = [
            'yt-dlp',
            '--write-auto-sub',
            '--write-sub',
            '--sub-lang', language_code,
            '--skip-download',
            '--output', str(output_dir / 'subtitles' / f'{base_filename}.%(ext)s'),
            video_url
        ]

        results = {
            'video_id': video_id,
            'video_url': video_url,
            'success': False,
            'audio_file': None,
            'subtitle_file': None,
            'metadata': {},
            'errors': []
        }

        try:
            # Download audio
            logger.info(f"🎵 Downloading audio for {video_id}")
            audio_result = subprocess.run(audio_cmd, capture_output=True, text=True, timeout=600)

            if audio_result.returncode == 0:
                audio_file = output_dir / 'raw_audio' / f'{base_filename}.wav'
                if audio_file.exists():
                    results['audio_file'] = str(audio_file)
                    logger.info(f"✅ Audio downloaded: {audio_file}")
                else:
                    results['errors'].append("Audio file not found after download")
            else:
                results['errors'].append(f"Audio download failed: {audio_result.stderr}")

            # Download subtitles
            logger.info(f"📝 Downloading subtitles for {video_id}")
            subtitle_result = subprocess.run(subtitle_cmd, capture_output=True, text=True, timeout=300)

            if subtitle_result.returncode == 0:
                # Check for subtitle files
                subtitle_files = list((output_dir / 'subtitles').glob(f'{base_filename}.*'))
                if subtitle_files:
                    results['subtitle_file'] = str(subtitle_files[0])
                    logger.info(f"✅ Subtitles downloaded: {subtitle_files[0]}")
                else:
                    results['errors'].append("No subtitle files found")
            else:
                results['errors'].append(f"Subtitle download failed: {subtitle_result.stderr}")

            # Get metadata
            metadata = self.get_video_metadata(video_url)
            results['metadata'] = metadata

            # Success if we have either audio or subtitles
            results['success'] = bool(results['audio_file'] or results['subtitle_file'])

            if results['success']:
                self.stats['successful_downloads'] += 1
            else:
                self.stats['failed_downloads'] += 1

        except subprocess.TimeoutExpired:
            results['errors'].append("Download timeout")
            self.stats['failed_downloads'] += 1
        except Exception as e:
            results['errors'].append(f"Download error: {str(e)}")
            self.stats['failed_downloads'] += 1

        self.stats['total_downloads'] += 1
        return results

    def extract_video_id(self, video_url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/watch\?.*v=([^&\n?#]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, video_url)
            if match:
                return match.group(1)

        return None

    def get_video_metadata(self, video_url: str) -> Dict:
        """Get metadata for a video"""
        cmd = [
            'yt-dlp',
            '--print', '%(title)s|%(duration)s|%(uploader)s|%(upload_date)s|%(view_count)s',
            '--no-download',
            video_url
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                parts = result.stdout.strip().split('|')
                if len(parts) >= 5:
                    return {
                        'title': parts[0],
                        'duration': int(parts[1]) if parts[1] != 'NA' else 0,
                        'uploader': parts[2],
                        'upload_date': parts[3],
                        'view_count': int(parts[4]) if parts[4] != 'NA' else 0
                    }
        except Exception as e:
            logger.warning(f"Could not get metadata: {e}")

        return {}

    def process_audio_file(self, audio_file: Path, output_dir: Path) -> Optional[Path]:
        """Process raw audio file to required format"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=self.settings.SAMPLE_RATE, mono=True)

            # Basic quality checks
            if len(audio) < self.settings.SAMPLE_RATE:  # Less than 1 second
                logger.warning(f"Audio too short: {audio_file}")
                return None

            # Normalize audio
            audio = librosa.util.normalize(audio)

            # Save processed audio
            output_file = output_dir / 'processed_audio' / f"{audio_file.stem}_processed.wav"
            sf.write(output_file, audio, self.settings.SAMPLE_RATE)

            # Update duration stats
            duration = len(audio) / self.settings.SAMPLE_RATE
            self.stats['total_duration'] += duration

            logger.info(f"✅ Processed audio: {output_file} ({duration:.2f}s)")
            return output_file

        except Exception as e:
            logger.error(f"Error processing audio {audio_file}: {e}")
            return None

    def process_subtitle_file(self, subtitle_file: Path, output_dir: Path) -> Optional[Path]:
        """Process subtitle file to clean text format"""
        try:
            if subtitle_file.suffix.lower() == '.vtt':
                return self.process_vtt_subtitle(subtitle_file, output_dir)
            elif subtitle_file.suffix.lower() == '.srt':
                return self.process_srt_subtitle(subtitle_file, output_dir)
            else:
                logger.warning(f"Unsupported subtitle format: {subtitle_file}")
                return None

        except Exception as e:
            logger.error(f"Error processing subtitle {subtitle_file}: {e}")
            return None

    def process_vtt_subtitle(self, vtt_file: Path, output_dir: Path) -> Optional[Path]:
        """Process VTT subtitle file"""
        try:
            vtt = webvtt.read(str(vtt_file))

            transcript = []
            for caption in vtt:
                text = self.clean_subtitle_text(caption.text)

                if text:
                    transcript.append({
                        'start': caption.start_in_seconds,
                        'end': caption.end_in_seconds,
                        'duration': caption.end_in_seconds - caption.start_in_seconds,
                        'text': text
                    })

            # Save processed subtitle
            output_file = output_dir / 'processed_subtitles' / f"{vtt_file.stem}_processed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Processed subtitles: {output_file} ({len(transcript)} segments)")
            return output_file

        except Exception as e:
            logger.error(f"Error processing VTT file: {e}")
            return None

    def process_srt_subtitle(self, srt_file: Path, output_dir: Path) -> Optional[Path]:
        """Process SRT subtitle file"""
        # Implementation for SRT files
        # This is a simplified version - you might want to use a proper SRT parser
        try:
            with open(srt_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Basic SRT parsing
            blocks = content.split('\n\n')
            transcript = []

            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    # Parse time
                    time_line = lines[1]
                    if '-->' in time_line:
                        start_str, end_str = time_line.split(' --> ')
                        start_time = self.parse_srt_time(start_str)
                        end_time = self.parse_srt_time(end_str)

                        # Get text
                        text = ' '.join(lines[2:])
                        text = self.clean_subtitle_text(text)

                        if text and start_time is not None and end_time is not None:
                            transcript.append({
                                'start': start_time,
                                'end': end_time,
                                'duration': end_time - start_time,
                                'text': text
                            })

            # Save processed subtitle
            output_file = output_dir / 'processed_subtitles' / f"{srt_file.stem}_processed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Processed SRT subtitles: {output_file} ({len(transcript)} segments)")
            return output_file

        except Exception as e:
            logger.error(f"Error processing SRT file: {e}")
            return None

    def parse_srt_time(self, time_str: str) -> Optional[float]:
        """Parse SRT time format to seconds"""
        try:
            # Format: HH:MM:SS,mmm
            time_str = time_str.strip()
            if ',' in time_str:
                time_part, ms_part = time_str.split(',')
                ms = int(ms_part) / 1000.0
            else:
                time_part = time_str
                ms = 0.0

            h, m, s = map(int, time_part.split(':'))
            total_seconds = h * 3600 + m * 60 + s + ms
            return total_seconds
        except:
            return None

    def clean_subtitle_text(self, text: str) -> str:
        """Clean subtitle text"""
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove speaker labels in brackets
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)

        # Remove music notation
        text = re.sub(r'♪.*?♪', '', text)
        text = re.sub(r'♫.*?♫', '', text)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Filter out very short texts
        if len(text) < 3:
            return ""

        return text

    def collect_language_data(self, language_code: str, max_videos: int = 50,
                              callback: Optional[Callable] = None) -> Dict:
        """Collect data for a specific language"""
        logger.info(f"🌟 Starting data collection for {language_code}")

        if not self.check_dependencies():
            return {'success': False, 'error': 'Missing dependencies'}

        # Validate language
        if not self.languages.validate_language_code(language_code):
            return {'success': False, 'error': f'Unsupported language: {language_code}'}

        # Setup directories
        directories = self.setup_directories(language_code)

        # Get language info
        lang_info = self.languages.get_language_info(language_code)
        channels = lang_info.get('youtube_channels', [])

        if not channels:
            return {'success': False, 'error': f'No channels configured for {language_code}'}

        # Collection results
        results = {
            'language_code': language_code,
            'total_videos_attempted': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_duration': 0,
            'channels_processed': 0,
            'files': {
                'audio_files': [],
                'subtitle_files': [],
                'processed_audio': [],
                'processed_subtitles': []
            }
        }

        # Process each channel
        for channel_idx, channel_url in enumerate(channels):
            logger.info(f"📺 Processing channel {channel_idx + 1}/{len(channels)}: {channel_url}")

            try:
                # Get video URLs from channel
                video_urls = self.get_channel_videos(channel_url, max_videos, language_code)

                if not video_urls:
                    logger.warning(f"No videos found for channel: {channel_url}")
                    continue

                results['channels_processed'] += 1

                # Download videos with limited parallelism
                max_workers = min(3, len(video_urls))  # Limit concurrent downloads

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    download_futures = {
                        executor.submit(
                            self.download_video_with_subtitles,
                            video_url,
                            directories[''],
                            language_code
                        ): video_url
                        for video_url in video_urls
                    }

                    for future in concurrent.futures.as_completed(download_futures):
                        video_url = download_futures[future]
                        results['total_videos_attempted'] += 1

                        try:
                            download_result = future.result(timeout=900)  # 15 min timeout

                            if download_result['success']:
                                results['successful_downloads'] += 1

                                # Process audio if available
                                if download_result['audio_file']:
                                    audio_file = Path(download_result['audio_file'])
                                    results['files']['audio_files'].append(str(audio_file))

                                    # Process audio
                                    processed_audio = self.process_audio_file(audio_file, directories[''])
                                    if processed_audio:
                                        results['files']['processed_audio'].append(str(processed_audio))

                                # Process subtitle if available
                                if download_result['subtitle_file']:
                                    subtitle_file = Path(download_result['subtitle_file'])
                                    results['files']['subtitle_files'].append(str(subtitle_file))

                                    # Process subtitle
                                    processed_subtitle = self.process_subtitle_file(subtitle_file, directories[''])
                                    if processed_subtitle:
                                        results['files']['processed_subtitles'].append(str(processed_subtitle))

                                # Update duration from metadata
                                duration = download_result['metadata'].get('duration', 0)
                                results['total_duration'] += duration

                            else:
                                results['failed_downloads'] += 1
                                logger.warning(f"Failed to download {video_url}: {download_result['errors']}")

                            # Callback for progress updates
                            if callback:
                                callback(language_code, 'data_collected', True)

                        except concurrent.futures.TimeoutError:
                            logger.error(f"Timeout downloading {video_url}")
                            results['failed_downloads'] += 1
                        except Exception as e:
                            logger.error(f"Error downloading {video_url}: {e}")
                            results['failed_downloads'] += 1

            except Exception as e:
                logger.error(f"Error processing channel {channel_url}: {e}")