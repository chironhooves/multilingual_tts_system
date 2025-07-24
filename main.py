#!/usr/bin/env python3
"""
Multilingual Indian TTS System - Main Entry Point
Integrates data collection, speaker identification, and TTS training
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from config.languages import IndianLanguages
from config.settings import SystemSettings
from core.speaker_id import AdvancedVoiceSystem
from core.data_collector import DataCollector
from core.preprocessor import AudioPreprocessor
from core.aligner import ForcedAligner
from core.trainer import TTSTrainer
from utils.visualization import ProgressVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tts_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultilingualTTSSystem:
    """Main orchestrator for the multilingual TTS system"""

    def __init__(self):
        """Initialize the complete system"""
        logger.info("🚀 Initializing Multilingual Indian TTS System")

        # Initialize components
        self.settings = SystemSettings()
        self.languages = IndianLanguages()
        self.speaker_system = AdvancedVoiceSystem()
        self.data_collector = DataCollector()
        self.preprocessor = AudioPreprocessor()
        self.aligner = ForcedAligner()
        self.trainer = TTSTrainer(self.speaker_system)
        self.visualizer = ProgressVisualizer()

        # Progress tracking
        self.progress = {
            lang_code: {
                'data_collected': False,
                'preprocessed': False,
                'aligned': False,
                'trained': False,
                'speakers_enrolled': 0,
                'total_duration': 0
            } for lang_code in self.languages.get_supported_languages()
        }

        # Setup directories
        self.setup_directories()

        logger.info("✅ System initialization complete")

    def setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            'data', 'models', 'logs', 'temp',
            'data/raw', 'data/processed', 'data/aligned', 'data/manifests',
            'models/individual', 'models/unified', 'models/checkpoints'
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

        logger.info("📁 Directory structure created")

    def display_system_info(self):
        """Display system information and requirements"""
        print("\n" + "=" * 80)
        print("🎤 MULTILINGUAL INDIAN TTS SYSTEM")
        print("=" * 80)

        print(f"\n📊 System Configuration:")
        print(f"   • Supported Languages: {len(self.languages.get_supported_languages())}")
        print(f"   • Storage Requirements: {self.settings.STORAGE_REQUIREMENTS}")
        print(f"   • GPU Available: {'✅' if self.settings.CUDA_AVAILABLE else '❌'}")
        print(f"   • Target Accuracy: {self.settings.TARGET_ACCURACY}%")

        print(f"\n🌍 Supported Languages:")
        for lang_code, lang_info in self.languages.LANGUAGES.items():
            status = "✅" if self.progress[lang_code]['trained'] else "⏳"
            print(f"   {status} {lang_info['native_name']} ({lang_info['name']})")

        print(f"\n💾 Storage Breakdown:")
        print(f"   • Raw Videos: ~500-800 GB")
        print(f"   • Processed Audio: ~150-200 GB")
        print(f"   • Aligned Data: ~100-150 GB")
        print(f"   • Models: ~50-100 GB")
        print(f"   • Working Space: ~200 GB")
        print(f"   • TOTAL RECOMMENDED: 1-1.5 TB")

    def run_interactive_menu(self):
        """Main interactive menu"""
        while True:
            print("\n" + "=" * 80)
            print("🎤 MULTILINGUAL INDIAN TTS SYSTEM - MAIN MENU")
            print("=" * 80)
            print("1.  📋 System Information & Requirements")
            print("2.  🔧 Setup & Environment Check")
            print("3.  📥 Data Collection")
            print("4.  🔊 Audio Processing & Speaker Analysis")
            print("5.  🎯 Forced Alignment")
            print("6.  🤖 TTS Model Training")
            print("7.  🌐 Unified Multilingual Model")
            print("8.  🎙️  Speaker Identification Tools")
            print("9.  📊 Progress & Statistics")
            print("10. 🧪 Testing & Inference")
            print("11. 🛠️  Advanced Options")
            print("12. ❌ Exit")

            choice = input("\n🔹 Select option (1-12): ").strip()

            try:
                if choice == '1':
                    self.display_system_info()
                elif choice == '2':
                    self.setup_environment_menu()
                elif choice == '3':
                    self.data_collection_menu()
                elif choice == '4':
                    self.audio_processing_menu()
                elif choice == '5':
                    self.alignment_menu()
                elif choice == '6':
                    self.training_menu()
                elif choice == '7':
                    self.unified_model_menu()
                elif choice == '8':
                    self.speaker_identification_menu()
                elif choice == '9':
                    self.progress_menu()
                elif choice == '10':
                    self.testing_menu()
                elif choice == '11':
                    self.advanced_menu()
                elif choice == '12':
                    print("\n👋 Thank you for using Multilingual TTS System!")
                    break
                else:
                    print("❌ Invalid option. Please try again.")

            except KeyboardInterrupt:
                print("\n\n⏸️  Operation interrupted by user.")
            except Exception as e:
                logger.error(f"Menu error: {e}")
                print(f"❌ Error: {e}")

    def setup_environment_menu(self):
        """Environment setup and dependency checking"""
        print("\n" + "=" * 60)
        print("🔧 ENVIRONMENT SETUP")
        print("=" * 60)

        print("1. Check Dependencies")
        print("2. Install Missing Packages")
        print("3. Download Language Models")
        print("4. Configure Storage")
        print("5. Test GPU Setup")
        print("6. Back to Main Menu")

        choice = input("\nSelect option: ").strip()

        if choice == '1':
            self.check_dependencies()
        elif choice == '2':
            self.install_dependencies()
        elif choice == '3':
            self.download_language_models()
        elif choice == '4':
            self.configure_storage()
        elif choice == '5':
            self.test_gpu_setup()

    def data_collection_menu(self):
        """Enhanced data collection menu with open datasets"""
        print("\n" + "=" * 60)
        print("📥 OPEN DATASET COLLECTION")
        print("=" * 60)
        print("🎉 YOUTUBE-FREE! Using legal, open datasets:")
        print("   • Mozilla Common Voice (CC-0 license)")
        print("   • Google FLEURS (Apache 2.0)")
        print("   • OpenSLR datasets")
        print("   • Custom recordings")
        print()

        print("1. Single Language - All Available Datasets")
        print("2. Single Language - Specific Dataset")
        print("3. All Languages - Common Voice Only")
        print("4. All Languages - All Available Datasets")
        print("5. Setup Custom Recordings")
        print("6. List Available Datasets")
        print("7. Collection Statistics")
        print("8. YouTube Alternative Info")
        print("9. Back to Main Menu")

        choice = input("\nSelect option (1-9): ").strip()

        if choice == '1':
            self.single_language_collection()
        elif choice == '2':
            self.specific_dataset_collection()
        elif choice == '3':
            self.common_voice_all_languages()
        elif choice == '4':
            self.all_datasets_all_languages()
        elif choice == '5':
            self.setup_custom_recordings()
        elif choice == '6':
            self.list_available_datasets()
        elif choice == '7':
            self.show_collection_stats()
        elif choice == '8':
            self.show_youtube_alternatives()

    def single_language_collection(self):
        """Collect all available datasets for a single language"""
        print("\n📋 Select Language for Complete Dataset Collection:")

        languages = list(self.languages.LANGUAGES.items())
        for i, (code, info) in enumerate(languages, 1):
            available_datasets = self.data_collector.list_available_datasets(code)
            dataset_count = len(available_datasets)
            print(f"{i:2d}. {info['native_name']} ({info['name']}) - {dataset_count} datasets available")

        try:
            choice = int(input("\nEnter language number: "))
            if 1 <= choice <= len(languages):
                lang_code = languages[choice - 1][0]

                print(f"\n🚀 Collecting all available datasets for {lang_code}...")
                print("This may take 30-60 minutes depending on dataset sizes...")

                result = self.data_collector.collect_language_data(
                    lang_code,
                    callback=self.update_progress
                )

                if result.get('success', False):
                    print(f"\n✅ Collection completed successfully!")
                    print(f"   Datasets successful: {result['datasets_successful']}/{result['datasets_attempted']}")
                    print(f"   Total segments: {result['total_segments']}")
                    print(f"   Estimated duration: {result['total_duration'] / 3600:.1f} hours")

                    # Show breakdown by dataset
                    print(f"\n📊 Dataset Breakdown:")
                    for dataset, dataset_result in result['results_by_dataset'].items():
                        if dataset_result.get('success'):
                            segments = dataset_result.get('segments', dataset_result.get('processed_segments', 0))
                            print(f"   ✅ {dataset}: {segments} segments")
                        else:
                            print(f"   ❌ {dataset}: Failed - {dataset_result.get('error', 'Unknown error')}")

                    self.progress[lang_code]['data_collected'] = True
                else:
                    print(f"\n❌ Collection failed: {result.get('error', 'Unknown error')}")

            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n⏸️  Collection interrupted by user")

    def specific_dataset_collection(self):
        """Collect from a specific dataset for a language"""
        print("\n📋 Select Language:")
        languages = list(self.languages.LANGUAGES.items())
        for i, (code, info) in enumerate(languages, 1):
            print(f"{i:2d}. {info['native_name']} ({info['name']})")

        try:
            lang_choice = int(input("\nEnter language number: "))
            if 1 <= lang_choice <= len(languages):
                lang_code = languages[lang_choice - 1][0]

                # Show available datasets for this language
                available_datasets = self.data_collector.list_available_datasets(lang_code)

                if not available_datasets:
                    print(f"❌ No datasets available for {lang_code}")
                    return

                print(f"\n📊 Available datasets for {lang_code}:")
                dataset_list = list(available_datasets.items())
                for i, (dataset_name, dataset_info) in enumerate(dataset_list, 1):
                    print(f"{i}. {dataset_info['name']}")
                    if 'description' in dataset_info:
                        print(f"   {dataset_info['description']}")

                dataset_choice = int(input(f"\nSelect dataset (1-{len(dataset_list)}): "))
                if 1 <= dataset_choice <= len(dataset_list):
                    selected_dataset = dataset_list[dataset_choice - 1][0]

                    print(f"\n🚀 Collecting {selected_dataset} data for {lang_code}...")

                    result = self.data_collector.collect_language_data(
                        lang_code,
                        datasets=[selected_dataset],
                        callback=self.update_progress
                    )

                    if result.get('success', False):
                        print(f"✅ Collection successful!")
                        segments = result.get('total_segments', 0)
                        duration = result.get('total_duration', 0)
                        print(f"   Segments: {segments}")
                        print(f"   Duration: {duration / 3600:.1f} hours")
                    else:
                        print(f"❌ Collection failed: {result.get('error')}")
                else:
                    print("❌ Invalid dataset selection")
            else:
                print("❌ Invalid language selection")
        except ValueError:
            print("❌ Please enter a valid number")

    def common_voice_all_languages(self):
        """Collect Common Voice data for all supported languages"""
        print("\n🌍 Collecting Mozilla Common Voice for all languages...")
        print("⚠️  This will download large datasets and may take 2-4 hours!")

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm == 'yes':
            print("\n🚀 Starting Common Voice collection for all languages...")

            result = self.data_collector.collect_all_languages(
                datasets=['common_voice'],
                callback=self.update_progress
            )

            print(f"\n✅ Common Voice collection completed!")
            print(f"   Successful languages: {result['successful_languages']}/{result['total_languages']}")

            # Show breakdown
            for lang_code, lang_result in result['results_by_language'].items():
                if lang_result.get('success', False):
                    segments = lang_result.get('total_segments', 0)
                    print(f"   ✅ {lang_code}: {segments} segments")
                    self.progress[lang_code]['data_collected'] = True
                else:
                    print(f"   ❌ {lang_code}: Failed")

    def all_datasets_all_languages(self):
        """Collect all available datasets for all languages"""
        print("\n🌍 Collecting ALL available datasets for ALL languages...")
        print("⚠️  WARNING: This is a massive download (potentially 10+ GB)")
        print("   Estimated time: 4-8 hours depending on internet speed")
        print("   Storage needed: 50-100 GB")

        confirm = input("\nAre you sure you want to proceed? (type 'YES' to confirm): ").strip()
        if confirm == 'YES':
            print("\n🚀 Starting comprehensive data collection...")

            result = self.data_collector.collect_all_languages(
                callback=self.update_progress
            )

            print(f"\n🎉 Comprehensive data collection completed!")
            print(f"   Successful languages: {result['successful_languages']}/{result['total_languages']}")

            # Update progress for successful languages
            for lang_code, lang_result in result['results_by_language'].items():
                if lang_result.get('success', False):
                    self.progress[lang_code]['data_collected'] = True
        else:
            print("❌ Collection cancelled")

    def setup_custom_recordings(self):
        """Setup custom recordings interface"""
        print("\n🎙️  CUSTOM RECORDINGS SETUP")
        print("=" * 50)
        print("Set up interface for adding your own voice recordings")

        languages = list(self.languages.LANGUAGES.items())
        for i, (code, info) in enumerate(languages, 1):
            print(f"{i:2d}. {info['native_name']} ({info['name']})")

        try:
            choice = int(input("\nSelect language for custom recordings: "))
            if 1 <= choice <= len(languages):
                lang_code = languages[choice - 1][0]

                from core.common_voice_collector import AdditionalDatasetCollector
                additional_collector = AdditionalDatasetCollector()

                result = additional_collector.setup_custom_recording_interface(lang_code)

                if result['success']:
                    print(f"\n✅ Custom recordings interface ready!")
                    print(f"📁 Directory: {result['custom_dir']}")
                    print("\n📋 NEXT STEPS:")
                    print("1. Navigate to the custom_recordings folder")
                    print("2. Add your audio files (.wav, .mp3) to the 'audio' folder")
                    print("3. Add corresponding text files (.txt) to the 'text' folder")
                    print("4. Run this menu again and select 'Process Custom Recordings'")
                    print("\n💡 See README.md in the folder for detailed instructions")

                    # Ask if user wants to process existing recordings
                    process_now = input("\nProcess existing recordings now? (y/n): ").strip().lower()
                    if process_now == 'y':
                        process_result = additional_collector.process_custom_recordings(lang_code)
                        if process_result.get('success', False):
                            segments = process_result.get('segments', 0)
                            print(f"✅ Processed {segments} custom recordings!")
                            if segments > 0:
                                self.progress[lang_code]['data_collected'] = True
                        else:
                            print(f"ℹ️  {process_result.get('message', 'No recordings found to process')}")
                else:
                    print(f"❌ Setup failed: {result.get('error')}")
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Please enter a valid number")

    def list_available_datasets(self):
        """List all available open datasets"""
        print("\n📊 AVAILABLE OPEN DATASETS")
        print("=" * 60)

        all_datasets = self.data_collector.list_available_datasets()

        dataset_count = 0
        for dataset_name, dataset_info in all_datasets.items():
            dataset_count += 1
            print(f"\n🔹 {dataset_info['name']}")

            if 'description' in dataset_info:
                print(f"   📝 {dataset_info['description']}")

            if 'url' in dataset_info and dataset_info['url']:
                print(f"   🔗 {dataset_info['url']}")

            if 'languages' in dataset_info:
                langs = list(dataset_info['languages'].keys())
                if len(langs) <= 5:
                    print(f"   🌍 Languages: {', '.join(langs)}")
                else:
                    print(f"   🌍 Languages: {', '.join(langs[:5])} ... and {len(langs) - 5} more")
            elif 'available_for_language' in dataset_info:
                print(f"   🌍 Available for selected language: {'✅' if dataset_info['available_for_language'] else '❌'}")

        print(f"\n📈 Total datasets available: {dataset_count}")
        print("\n💡 To collect data, use options 1-4 in the data collection menu")

        input("\nPress Enter to continue...")

    def show_collection_stats(self):
        """Show data collection statistics"""
        print("\n📊 DATA COLLECTION STATISTICS")
        print("=" * 50)

        stats = self.data_collector.get_collection_statistics()

        print(f"📥 Downloads attempted: {stats['total_downloads']}")
        print(f"✅ Successful downloads: {stats['successful_downloads']}")
        print(f"❌ Failed downloads: {stats['failed_downloads']}")

        if stats['total_downloads'] > 0:
            success_rate = (stats['successful_downloads'] / stats['total_downloads']) * 100
            print(f"📊 Success rate: {success_rate:.1f}%")

        print(f"🌍 Languages processed: {len(stats['languages_processed'])}")
        if stats['languages_processed']:
            print(f"   Languages: {', '.join(stats['languages_processed'])}")

        print(f"⏱️  Estimated total duration: {stats.get('total_duration', 0) / 3600:.1f} hours")

        input("\nPress Enter to continue...")

    def show_youtube_alternatives(self):
        """Show information about YouTube alternatives"""
        message = self.data_collector.get_youtube_alternative_message()
        print(message)

        print("\n🔗 USEFUL LINKS:")
        print("• Mozilla Common Voice: https://commonvoice.mozilla.org/")
        print("• Google FLEURS: https://huggingface.co/datasets/google/fleurs")
        print("• OpenSLR: https://www.openslr.org/")
        print("• AI4Bharat: https://ai4bharat.iitm.ac.in/")

        input("\nPress Enter to continue...")

    def all_languages_collection(self):
        """Enhanced all languages collection with open datasets"""
        print("\n🌍 COMPREHENSIVE DATA COLLECTION")
        print("=" * 50)
        print("Options:")
        print("1. Common Voice only (Recommended - 2-4 hours)")
        print("2. All available datasets (Complete - 6-12 hours)")
        print("3. Back")

        choice = input("\nSelect option (1-3): ").strip()

        if choice == '1':
            self.common_voice_all_languages()
        elif choice == '2':
            self.all_datasets_all_languages()

    def custom_channel_collection(self):
        """Custom recordings setup (replaces custom channel collection)"""
        print("\n🎙️  CUSTOM RECORDINGS")
        print("=" * 40)
        print("Instead of custom YouTube channels, you can add your own recordings!")
        print()
        self.setup_custom_recordings()

    def resume_collection(self):
        """Resume interrupted collection"""
        print("\n🔄 RESUME COLLECTION")
        print("=" * 30)
        print("Check for incomplete downloads and resume...")

        # Check which languages have incomplete data
        incomplete_languages = []

        for lang_code in self.languages.get_supported_languages():
            if not self.progress[lang_code].get('data_collected', False):
                # Check if there's partial data
                lang_dir = Path("data") / lang_code
                if lang_dir.exists() and any(lang_dir.iterdir()):
                    incomplete_languages.append(lang_code)

        if not incomplete_languages:
            print("✅ No incomplete collections found")
            return

        print(f"Found {len(incomplete_languages)} languages with incomplete data:")
        for i, lang_code in enumerate(incomplete_languages, 1):
            lang_info = self.languages.get_language_info(lang_code)
            print(f"{i}. {lang_info['native_name']} ({lang_info['name']})")

        print(f"{len(incomplete_languages) + 1}. Resume all incomplete")

        try:
            choice = int(input(f"\nSelect option (1-{len(incomplete_languages) + 1}): "))

            if 1 <= choice <= len(incomplete_languages):
                # Resume single language
                lang_code = incomplete_languages[choice - 1]
                print(f"\n🔄 Resuming collection for {lang_code}...")

                result = self.data_collector.collect_language_data(
                    lang_code,
                    callback=self.update_progress
                )

                if result.get('success', False):
                    print(f"✅ Resume successful for {lang_code}!")
                    self.progress[lang_code]['data_collected'] = True
                else:
                    print(f"❌ Resume failed: {result.get('error')}")

            elif choice == len(incomplete_languages) + 1:
                # Resume all
                print(f"\n🔄 Resuming collection for all {len(incomplete_languages)} languages...")

                successful = 0
                for lang_code in incomplete_languages:
                    try:
                        result = self.data_collector.collect_language_data(
                            lang_code,
                            callback=self.update_progress
                        )

                        if result.get('success', False):
                            self.progress[lang_code]['data_collected'] = True
                            successful += 1
                            print(f"✅ {lang_code}: Resumed successfully")
                        else:
                            print(f"❌ {lang_code}: Resume failed")

                    except Exception as e:
                        print(f"❌ {lang_code}: Error - {e}")

                print(f"\n🎉 Resume completed: {successful}/{len(incomplete_languages)} successful")
            else:
                print("❌ Invalid selection")

        except ValueError:
            print("❌ Please enter a valid number")

    def verify_data(self):
        """Verify collected open dataset data"""
        print("\n🔍 DATA VERIFICATION")
        print("=" * 30)
        print("Verifying integrity of collected open datasets...")

        languages_with_data = []

        for lang_code in self.languages.get_supported_languages():
            lang_dir = Path("data") / lang_code

            if lang_dir.exists():
                # Count different types of data
                raw_audio = len(list((lang_dir / "raw_audio").glob("*.wav"))) if (
                            lang_dir / "raw_audio").exists() else 0
                raw_audio += len(list((lang_dir / "raw_audio").glob("*.mp3"))) if (
                            lang_dir / "raw_audio").exists() else 0

                processed_audio = len(list((lang_dir / "processed_audio").glob("*.wav"))) if (
                            lang_dir / "processed_audio").exists() else 0

                metadata_files = len(list((lang_dir / "metadata").glob("*.json"))) if (
                            lang_dir / "metadata").exists() else 0

                if raw_audio > 0 or processed_audio > 0:
                    languages_with_data.append({
                        'code': lang_code,
                        'name': self.languages.get_language_info(lang_code)['name'],
                        'raw_audio': raw_audio,
                        'processed_audio': processed_audio,
                        'metadata': metadata_files
                    })

        if not languages_with_data:
            print("❌ No data found. Run data collection first.")
            return

        print(f"\n📊 Found data for {len(languages_with_data)} languages:")
        print("-" * 60)
        print(f"{'Language':<15} {'Raw Audio':<12} {'Processed':<12} {'Metadata':<10}")
        print("-" * 60)

        total_raw = 0
        total_processed = 0

        for lang_data in languages_with_data:
            print(
                f"{lang_data['name']:<15} {lang_data['raw_audio']:<12} {lang_data['processed_audio']:<12} {lang_data['metadata']:<10}")
            total_raw += lang_data['raw_audio']
            total_processed += lang_data['processed_audio']

        print("-" * 60)
        print(f"{'TOTAL':<15} {total_raw:<12} {total_processed:<12}")

        # Estimate total duration
        avg_duration_per_file = 5  # seconds
        total_duration_hours = (total_raw * avg_duration_per_file) / 3600

        print(f"\n📈 Summary:")
        print(f"   Total audio files: {total_raw}")
        print(f"   Estimated duration: {total_duration_hours:.1f} hours")
        print(f"   Languages ready for training: {len([l for l in languages_with_data if l['raw_audio'] > 100])}")

        input("\nPress Enter to continue...")

    def audio_processing_menu(self):
        """Audio processing and speaker analysis menu"""
        print("\n" + "=" * 60)
        print("🔊 AUDIO PROCESSING & SPEAKER ANALYSIS")
        print("=" * 60)

        print("1. Process Raw Audio Files")
        print("2. Speaker Diarization")
        print("3. Speaker Enrollment")
        print("4. Quality Assessment")
        print("5. Generate Statistics")
        print("6. Back to Main Menu")

        choice = input("\nSelect option: ").strip()

        if choice == '1':
            self.process_audio_files()
        elif choice == '2':
            self.run_speaker_diarization()
        elif choice == '3':
            self.enroll_speakers()
        elif choice == '4':
            self.assess_audio_quality()
        elif choice == '5':
            self.generate_audio_statistics()

    def training_menu(self):
        """Training menu"""
        print("\n" + "=" * 60)
        print("🤖 TTS MODEL TRAINING")
        print("=" * 60)

        print("1. Train Single Language Model")
        print("2. Train All Language Models")
        print("3. Resume Training")
        print("4. Evaluate Models")
        print("5. Fine-tune Existing Model")
        print("6. Back to Main Menu")

        choice = input("\nSelect option: ").strip()

        if choice == '1':
            self.train_single_language()
        elif choice == '2':
            self.train_all_languages()
        elif choice == '3':
            self.resume_training()
        elif choice == '4':
            self.evaluate_models()
        elif choice == '5':
            self.finetune_model()

    def speaker_identification_menu(self):
        """Speaker identification menu - integrates your existing code"""
        print("\n" + "=" * 60)
        print("🎙️ SPEAKER IDENTIFICATION TOOLS")
        print("=" * 60)

        print("1. Single Audio File Analysis")
        print("2. Batch Audio Processing")
        print("3. Speaker Enrollment")
        print("4. Speaker Database Management")
        print("5. Voice Similarity Analysis")
        print("6. Back to Main Menu")

        choice = input("\nSelect option: ").strip()

        if choice == '1':
            self.speaker_system.process_single_audio_file()
        elif choice == '2':
            self.speaker_system.process_multiple_audio_files()
        elif choice == '3':
            self.speaker_system.enrollment_mode()
        elif choice == '4':
            self.speaker_database_menu()
        elif choice == '5':
            self.voice_similarity_analysis()

    def progress_menu(self):
        """Progress tracking and statistics"""
        print("\n" + "=" * 60)
        print("📊 PROGRESS & STATISTICS")
        print("=" * 60)

        self.visualizer.show_overall_progress(self.progress)
        self.visualizer.show_language_breakdown(self.progress, self.languages)
        self.visualizer.show_storage_usage()

        input("\nPress Enter to continue...")

    def single_language_collection(self):
        """Collect data for a single language"""
        print("\n📋 Select Language for Data Collection:")

        languages = list(self.languages.LANGUAGES.items())
        for i, (code, info) in enumerate(languages, 1):
            status = "✅" if self.progress[code]['data_collected'] else "⏳"
            print(f"{i:2d}. {status} {info['native_name']} ({info['name']})")

        try:
            choice = int(input("\nEnter language number: "))
            if 1 <= choice <= len(languages):
                lang_code = languages[choice - 1][0]
                max_videos = int(input("Max videos per channel (default 50): ") or "50")

                print(f"\n🚀 Starting data collection for {lang_code}...")
                self.data_collector.collect_language_data(
                    lang_code,
                    max_videos=max_videos,
                    callback=self.update_progress
                )
                self.progress[lang_code]['data_collected'] = True

            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Please enter a valid number")

    def all_languages_collection(self):
        """Collect data for all languages"""
        print("\n🌍 Starting data collection for all languages...")
        print("⚠️  This will require significant time and storage space!")

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm == 'yes':
            max_videos = int(input("Max videos per channel per language (default 50): ") or "50")

            self.data_collector.collect_all_languages(
                max_videos=max_videos,
                callback=self.update_progress
            )

    def run_speaker_diarization(self):
        """Run speaker diarization on collected data"""
        print("\n🎯 Speaker Diarization")

        # Get available languages with data
        available_langs = [
            code for code, progress in self.progress.items()
            if progress['data_collected']
        ]

        if not available_langs:
            print("❌ No data available. Please collect data first.")
            return

        print("Available languages:")
        for i, lang_code in enumerate(available_langs, 1):
            lang_info = self.languages.LANGUAGES[lang_code]
            print(f"{i}. {lang_info['native_name']} ({lang_info['name']})")

        choice = input("Select language (number) or 'all': ").strip()

        if choice == 'all':
            for lang_code in available_langs:
                print(f"\n🔄 Processing {lang_code}...")
                self.preprocessor.diarize_language_data(lang_code)
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(available_langs):
                    lang_code = available_langs[idx]
                    self.preprocessor.diarize_language_data(lang_code)
                else:
                    print("❌ Invalid selection")
            except ValueError:
                print("❌ Please enter a valid number")

    def train_single_language(self):
        """Train TTS model for a single language"""
        print("\n🤖 Single Language Training")

        # Get available languages with aligned data
        available_langs = [
            code for code, progress in self.progress.items()
            if progress['aligned']
        ]

        if not available_langs:
            print("❌ No aligned data available. Please run alignment first.")
            return

        print("Available languages:")
        for i, lang_code in enumerate(available_langs, 1):
            lang_info = self.languages.LANGUAGES[lang_code]
            print(f"{i}. {lang_info['native_name']} ({lang_info['name']})")

        try:
            choice = int(input("Select language number: "))
            if 1 <= choice <= len(available_langs):
                lang_code = available_langs[choice - 1]

                print(f"\n🚀 Starting training for {lang_code}...")
                success = self.trainer.train_language_model(lang_code)

                if success:
                    self.progress[lang_code]['trained'] = True
                    print(f"✅ Training completed for {lang_code}")
                else:
                    print(f"❌ Training failed for {lang_code}")
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Please enter a valid number")

    def update_progress(self, lang_code, stage, value):
        """Update progress callback"""
        if lang_code in self.progress:
            self.progress[lang_code][stage] = value
            logger.info(f"Progress update: {lang_code} - {stage}: {value}")

    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        print("\n🔍 Checking Dependencies...")

        dependencies = {
            'yt-dlp': 'yt-dlp --version',
            'ffmpeg': 'ffmpeg -version',
            'torch': 'python -c "import torch; print(torch.__version__)"',
            'TTS': 'python -c "import TTS; print(TTS.__version__)"',
            'resemblyzer': 'python -c "import resemblyzer; print(\'OK\')"',
            'pyannote.audio': 'python -c "import pyannote.audio; print(\'OK\')"',
            'librosa': 'python -c "import librosa; print(librosa.__version__)"',
        }

        for package, command in dependencies.items():
            try:
                import subprocess
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ {package}: OK")
                else:
                    print(f"❌ {package}: Not found")
            except Exception as e:
                print(f"❌ {package}: Error - {e}")

        print("\n💡 Install missing packages with: pip install -r requirements.txt")

    def install_dependencies(self):
        """Install missing dependencies"""
        print("\n📦 Installing Dependencies...")

        requirements = [
            "torch",
            "torchaudio",
            "TTS",
            "resemblyzer",
            "pyannote.audio",
            "librosa",
            "yt-dlp",
            "webvtt-py",
            "matplotlib",
            "seaborn",
            "scikit-learn",
            "pandas",
            "numpy",
            "soundfile",
            "pyaudio"
        ]

        import subprocess
        for package in requirements:
            try:
                print(f"Installing {package}...")
                subprocess.run(["pip", "install", package], check=True)
                print(f"✅ {package} installed")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")

    def speaker_database_menu(self):
        """Manage speaker database"""
        print("\n" + "=" * 50)
        print("👥 SPEAKER DATABASE MANAGEMENT")
        print("=" * 50)

        print("1. Show Database Statistics")
        print("2. Export Database")
        print("3. Import Database")
        print("4. Merge Databases")
        print("5. Clean Database")
        print("6. Back")

        choice = input("\nSelect option: ").strip()

        if choice == '1':
            self.speaker_system.display_database_info()
        elif choice == '2':
            self.export_speaker_database()
        elif choice == '3':
            self.import_speaker_database()
        elif choice == '4':
            self.merge_speaker_databases()
        elif choice == '5':
            self.clean_speaker_database()


def main():
    """Main entry point"""
    try:
        system = MultilingualTTSSystem()
        system.run_interactive_menu()
    except KeyboardInterrupt:
        print("\n\n⏸️  System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        print(f"\n❌ System error: {e}")


if __name__ == "__main__":
    main()