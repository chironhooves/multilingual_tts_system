"""
Common Voice and Open Dataset Integration
Handles downloading and processing of Mozilla Common Voice and other open datasets
for Indian languages, replacing YouTube dependency
"""

import os
import requests
import tarfile
import zipfile
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import concurrent.futures
import hashlib
import shutil
from urllib.parse import urlparse
import time

from config.languages import IndianLanguages
from config.settings import SystemSettings

logger = logging.getLogger(__name__)


class CommonVoiceCollector:
    """Handles Common Voice dataset collection and processing"""

    def __init__(self):
        self.settings = SystemSettings()
        self.languages = IndianLanguages()
        self.base_url = "https://commonvoice.mozilla.org/api/v1/bucket"
        self.datasets_info = self._load_datasets_info()

        # Create downloads directory
        self.downloads_dir = Path("temp/downloads")
        self.downloads_dir.mkdir(parents=True, exist_ok=True)

        logger.info("🎙️ Common Voice Collector initialized")

    def _load_datasets_info(self) -> Dict:
        """Load information about available open datasets"""
        return {
            'common_voice': {
                'name': 'Mozilla Common Voice',
                'base_url': 'https://commonvoice.mozilla.org/api/v1/bucket',
                'languages': {
                    'hi': {'name': 'Hindi', 'code': 'hi', 'available': True, 'version': '13.0'},
                    'ta': {'name': 'Tamil', 'code': 'ta', 'available': True, 'version': '13.0'},
                    'te': {'name': 'Telugu', 'code': 'te', 'available': True, 'version': '13.0'},
                    'bn': {'name': 'Bengali', 'code': 'bn', 'available': True, 'version': '13.0'},
                    'mr': {'name': 'Marathi', 'code': 'mr', 'available': True, 'version': '13.0'},
                    'gu': {'name': 'Gujarati', 'code': 'gu', 'available': True, 'version': '13.0'},
                    'kn': {'name': 'Kannada', 'code': 'kn', 'available': True, 'version': '13.0'},
                    'ml': {'name': 'Malayalam', 'code': 'ml', 'available': True, 'version': '13.0'},
                    'pa': {'name': 'Punjabi', 'code': 'pa', 'available': True, 'version': '13.0'},
                    'or': {'name': 'Odia', 'code': 'or', 'available': False, 'version': 'N/A'}  # Not yet available
                }
            },
            'openslr': {
                'name': 'OpenSLR Indian Language Datasets',
                'base_url': 'https://www.openslr.org/resources',
                'languages': {
                    'hi': {'resource_id': '103', 'name': 'Hindi', 'available': True},
                    'te': {'resource_id': '66', 'name': 'Telugu', 'available': True},
                    'ta': {'resource_id': '65', 'name': 'Tamil', 'available': True},
                    'gu': {'resource_id': '78', 'name': 'Gujarati', 'available': True},
                    'bn': {'resource_id': '37', 'name': 'Bengali', 'available': True}
                }
            },
            'google_fleurs': {
                'name': 'Google FLEURS',
                'base_url': 'https://huggingface.co/datasets/google/fleurs',
                'languages': {
                    'hi': {'name': 'Hindi', 'available': True},
                    'ta': {'name': 'Tamil', 'available': True},
                    'te': {'name': 'Telugu', 'available': True},
                    'bn': {'name': 'Bengali', 'available': True},
                    'mr': {'name': 'Marathi', 'available': True},
                    'gu': {'name': 'Gujarati', 'available': True},
                    'kn': {'name': 'Kannada', 'available': True},
                    'ml': {'name': 'Malayalam', 'available': True},
                    'pa': {'name': 'Punjabi', 'available': True},
                    'or': {'name': 'Odia', 'available': True}
                }
            },
            'indic_tts': {
                'name': 'IITm Indic TTS Database',
                'base_url': 'https://www.iitm.ac.in/donlab/tts/',
                'languages': {
                    'hi': {'name': 'Hindi', 'available': True},
                    'ta': {'name': 'Tamil', 'available': True},
                    'te': {'name': 'Telugu', 'available': True},
                    'bn': {'name': 'Bengali', 'available': True},
                    'mr': {'name': 'Marathi', 'available': True},
                    'gu': {'name': 'Gujarati', 'available': True},
                    'kn': {'name': 'Kannada', 'available': True},
                    'ml': {'name': 'Malayalam', 'available': True}
                }
            }
        }

    def list_available_datasets(self, language_code: str = None) -> Dict:
        """List all available datasets for languages"""
        available = {}

        for dataset_name, dataset_info in self.datasets_info.items():
            dataset_langs = dataset_info['languages']

            if language_code:
                if language_code in dataset_langs and dataset_langs[language_code].get('available', False):
                    available[dataset_name] = {
                        'name': dataset_info['name'],
                        'language': dataset_langs[language_code]
                    }
            else:
                # List all available languages for this dataset
                available_langs = {
                    lang: info for lang, info in dataset_langs.items()
                    if info.get('available', False)
                }
                if available_langs:
                    available[dataset_name] = {
                        'name': dataset_info['name'],
                        'languages': available_langs
                    }

        return available

    def download_common_voice_dataset(self, language_code: str, subset: str = 'train') -> Dict:
        """Download Common Voice dataset for a language"""
        logger.info(f"📥 Downloading Common Voice dataset for {language_code}")

        if language_code not in self.datasets_info['common_voice']['languages']:
            return {'success': False, 'error': f'Language {language_code} not available in Common Voice'}

        lang_info = self.datasets_info['common_voice']['languages'][language_code]
        if not lang_info.get('available', False):
            return {'success': False, 'error': f'Common Voice dataset not available for {language_code}'}

        # Construct download URL
        version = lang_info.get('version', '13.0')
        filename = f"cv-corpus-{version}-2023-09-08-{language_code}.tar.gz"
        download_url = f"https://commonvoice.mozilla.org/datasets/{filename}"

        # Alternative direct download URLs (Common Voice provides these)
        alt_urls = [
            f"https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-{version}-2023-09-08/{language_code}.tar.gz",
            f"https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-{version}-2023-09-08-{language_code}.tar.gz"
        ]

        # Try downloading from different URLs
        download_result = None
        for url in [download_url] + alt_urls:
            try:
                logger.info(f"Attempting download from: {url}")
                download_result = self._download_file(url, filename)
                if download_result['success']:
                    break
            except Exception as e:
                logger.warning(f"Download failed from {url}: {e}")
                continue

        if not download_result or not download_result['success']:
            return {'success': False, 'error': 'All download attempts failed'}

        # Extract the dataset
        extract_result = self._extract_common_voice_dataset(
            download_result['file_path'], language_code, subset
        )

        if extract_result['success']:
            # Clean up downloaded archive
            os.remove(download_result['file_path'])
            logger.info(f"✅ Common Voice dataset ready for {language_code}")

        return extract_result

    def _download_file(self, url: str, filename: str, chunk_size: int = 8192) -> Dict:
        """Download a file with progress tracking"""
        file_path = self.downloads_dir / filename

        try:
            logger.info(f"Starting download: {filename}")

            # Check if file already exists and verify
            if file_path.exists():
                logger.info(f"File already exists: {filename}")
                return {'success': True, 'file_path': str(file_path)}

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        # Progress logging every 10MB
                        if downloaded_size % (10 * 1024 * 1024) == 0:
                            progress = (downloaded_size / total_size * 100) if total_size > 0 else 0
                            logger.info(f"Download progress: {progress:.1f}% ({downloaded_size // (1024 * 1024)} MB)")

            logger.info(f"✅ Download completed: {filename}")
            return {'success': True, 'file_path': str(file_path)}

        except Exception as e:
            logger.error(f"Download failed: {e}")
            if file_path.exists():
                file_path.unlink()  # Clean up partial download
            return {'success': False, 'error': str(e)}

    def _extract_common_voice_dataset(self, archive_path: str, language_code: str, subset: str) -> Dict:
        """Extract Common Voice dataset"""
        logger.info(f"📦 Extracting Common Voice dataset for {language_code}")

        try:
            # Setup extraction directory
            extract_dir = Path("data") / language_code / "common_voice"
            extract_dir.mkdir(parents=True, exist_ok=True)

            # Extract tar.gz file
            with tarfile.open(archive_path, 'r:gz') as tar:
                # Find the correct directory structure
                members = tar.getnames()
                base_dir = None

                for member in members:
                    if f'cv-corpus-' in member and language_code in member:
                        base_dir = member.split('/')[0]
                        break

                if not base_dir:
                    return {'success': False, 'error': 'Could not find dataset structure in archive'}

                # Extract specific files we need
                files_extracted = 0
                audio_files = []

                for member in tar.getmembers():
                    # Extract TSV files (metadata)
                    if member.name.endswith('.tsv') and subset in member.name:
                        tar.extract(member, extract_dir)
                        files_extracted += 1

                    # Extract audio clips
                    elif member.name.endswith('.mp3') and 'clips' in member.name:
                        tar.extract(member, extract_dir)
                        audio_files.append(member.name)
                        files_extracted += 1

                        # Progress logging
                        if len(audio_files) % 1000 == 0:
                            logger.info(f"Extracted {len(audio_files)} audio files...")

            # Process the extracted data
            process_result = self._process_common_voice_data(extract_dir, language_code, subset)

            result = {
                'success': True,
                'language_code': language_code,
                'extract_dir': str(extract_dir),
                'files_extracted': files_extracted,
                'audio_files': len(audio_files),
                'processed_segments': process_result.get('segments', 0)
            }

            logger.info(f"✅ Extraction completed: {files_extracted} files extracted")
            return result

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {'success': False, 'error': str(e)}

    def _process_common_voice_data(self, extract_dir: Path, language_code: str, subset: str) -> Dict:
        """Process extracted Common Voice data into our format"""
        logger.info(f"🔄 Processing Common Voice data for {language_code}")

        try:
            # Find the TSV file
            tsv_files = list(extract_dir.rglob(f'{subset}.tsv'))
            if not tsv_files:
                return {'success': False, 'error': f'No {subset}.tsv file found'}

            tsv_file = tsv_files[0]

            # Read the TSV file
            df = pd.read_csv(tsv_file, sep='\t')
            logger.info(f"Found {len(df)} entries in {subset}.tsv")

            # Setup output directories
            base_dir = Path("data") / language_code
            raw_audio_dir = base_dir / "raw_audio"
            raw_text_dir = base_dir / "raw_text"
            metadata_dir = base_dir / "metadata"

            for dir_path in [raw_audio_dir, raw_text_dir, metadata_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

            # Process each entry
            processed_segments = []
            successful_copies = 0

            for idx, row in df.iterrows():
                try:
                    # Get audio file path
                    audio_filename = row['path']
                    source_audio = None

                    # Find the actual audio file
                    for audio_file in extract_dir.rglob(audio_filename):
                        source_audio = audio_file
                        break

                    if not source_audio or not source_audio.exists():
                        continue

                    # Copy audio file to our format
                    target_audio = raw_audio_dir / f"cv_{language_code}_{idx:06d}.mp3"
                    shutil.copy2(source_audio, target_audio)

                    # Create text entry
                    text_data = {
                        'text': row['sentence'],
                        'speaker_id': row.get('client_id', f'speaker_{idx}'),
                        'age': row.get('age', 'unknown'),
                        'gender': row.get('gender', 'unknown'),
                        'accent': row.get('accent', 'unknown'),
                        'duration': 0,  # Will be calculated during preprocessing
                        'votes_up': row.get('up_votes', 0),
                        'votes_down': row.get('down_votes', 0),
                        'source': 'common_voice',
                        'language': language_code
                    }

                    # Save text file
                    text_file = raw_text_dir / f"cv_{language_code}_{idx:06d}.json"
                    with open(text_file, 'w', encoding='utf-8') as f:
                        json.dump(text_data, f, ensure_ascii=False, indent=2)

                    processed_segments.append({
                        'audio_file': str(target_audio),
                        'text_file': str(text_file),
                        'text': text_data['text'],
                        'speaker_id': text_data['speaker_id']
                    })

                    successful_copies += 1

                    # Progress logging
                    if successful_copies % 500 == 0:
                        logger.info(f"Processed {successful_copies} segments...")

                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {e}")
                    continue

            # Save manifest
            manifest_data = {
                'language_code': language_code,
                'source': 'common_voice',
                'subset': subset,
                'total_segments': len(processed_segments),
                'created_at': datetime.now().isoformat(),
                'segments': processed_segments
            }

            manifest_file = metadata_dir / f"common_voice_{subset}_manifest.json"
            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest_data, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Processed {successful_copies} Common Voice segments")

            return {
                'success': True,
                'segments': successful_copies,
                'manifest_file': str(manifest_file)
            }

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {'success': False, 'error': str(e)}

    def download_openslr_dataset(self, language_code: str) -> Dict:
        """Download OpenSLR dataset for a language"""
        logger.info(f"📥 Downloading OpenSLR dataset for {language_code}")

        if language_code not in self.datasets_info['openslr']['languages']:
            return {'success': False, 'error': f'Language {language_code} not available in OpenSLR'}

        lang_info = self.datasets_info['openslr']['languages'][language_code]
        resource_id = lang_info['resource_id']

        # OpenSLR URLs
        base_url = f"https://www.openslr.org/resources/{resource_id}"
        dataset_urls = [
            f"{base_url}/data_train.tar.gz",
            f"{base_url}/data_test.tar.gz",
            f"{base_url}/line_index.tsv"
        ]

        extract_dir = Path("data") / language_code / "openslr"
        extract_dir.mkdir(parents=True, exist_ok=True)

        downloaded_files = []

        for url in dataset_urls:
            filename = Path(urlparse(url).path).name
            try:
                download_result = self._download_file(url, f"openslr_{language_code}_{filename}")
                if download_result['success']:
                    downloaded_files.append(download_result['file_path'])

                    # Extract if it's a tar file
                    if filename.endswith('.tar.gz'):
                        with tarfile.open(download_result['file_path'], 'r:gz') as tar:
                            tar.extractall(extract_dir)
                        os.remove(download_result['file_path'])  # Clean up

            except Exception as e:
                logger.warning(f"Failed to download {url}: {e}")

        if not downloaded_files:
            return {'success': False, 'error': 'No files could be downloaded'}

        # Process OpenSLR data
        process_result = self._process_openslr_data(extract_dir, language_code)

        return {
            'success': True,
            'language_code': language_code,
            'extract_dir': str(extract_dir),
            'files_downloaded': len(downloaded_files),
            'processed_segments': process_result.get('segments', 0)
        }

    def _process_openslr_data(self, extract_dir: Path, language_code: str) -> Dict:
        """Process OpenSLR data into our format"""
        # Implementation similar to Common Voice processing
        # OpenSLR has different structure, adapt accordingly
        logger.info(f"🔄 Processing OpenSLR data for {language_code}")

        # This would be implemented based on OpenSLR's specific format
        # Each dataset has slightly different structure
        return {'success': True, 'segments': 0}

    def download_fleurs_dataset(self, language_code: str) -> Dict:
        """Download Google FLEURS dataset"""
        logger.info(f"📥 Downloading FLEURS dataset for {language_code}")

        try:
            # FLEURS is available via Hugging Face datasets
            from datasets import load_dataset

            # Map our language codes to FLEURS codes
            fleurs_code_map = {
                'hi': 'hi_in', 'ta': 'ta_in', 'te': 'te_in', 'bn': 'bn_in',
                'mr': 'mr_in', 'gu': 'gu_in', 'kn': 'kn_in', 'ml': 'ml_in',
                'pa': 'pa_in', 'or': 'or_in'
            }

            fleurs_code = fleurs_code_map.get(language_code)
            if not fleurs_code:
                return {'success': False, 'error': f'FLEURS not available for {language_code}'}

            # Download the dataset
            dataset = load_dataset("google/fleurs", fleurs_code, split="train")

            # Process and save
            extract_dir = Path("data") / language_code / "fleurs"
            extract_dir.mkdir(parents=True, exist_ok=True)

            processed_segments = self._process_fleurs_data(dataset, extract_dir, language_code)

            return {
                'success': True,
                'language_code': language_code,
                'segments': len(processed_segments)
            }

        except ImportError:
            return {'success': False, 'error': 'datasets library not installed. Install with: pip install datasets'}
        except Exception as e:
            logger.error(f"FLEURS download failed: {e}")
            return {'success': False, 'error': str(e)}

    def _process_fleurs_data(self, dataset, extract_dir: Path, language_code: str) -> List[Dict]:
        """Process FLEURS dataset"""
        logger.info(f"🔄 Processing FLEURS data for {language_code}")

        # Setup directories
        base_dir = Path("data") / language_code
        raw_audio_dir = base_dir / "raw_audio"
        raw_text_dir = base_dir / "raw_text"

        for dir_path in [raw_audio_dir, raw_text_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        processed_segments = []

        for idx, item in enumerate(dataset):
            try:
                # Save audio
                audio_file = raw_audio_dir / f"fleurs_{language_code}_{idx:06d}.wav"

                # FLEURS provides audio as dict with 'array' and 'sampling_rate'
                import soundfile as sf
                sf.write(audio_file, item['audio']['array'], item['audio']['sampling_rate'])

                # Save text
                text_data = {
                    'text': item['transcription'],
                    'speaker_id': f"fleurs_speaker_{item.get('speaker_id', idx)}",
                    'gender': item.get('gender', 'unknown'),
                    'source': 'fleurs',
                    'language': language_code
                }

                text_file = raw_text_dir / f"fleurs_{language_code}_{idx:06d}.json"
                with open(text_file, 'w', encoding='utf-8') as f:
                    json.dump(text_data, f, ensure_ascii=False, indent=2)

                processed_segments.append({
                    'audio_file': str(audio_file),
                    'text_file': str(text_file),
                    'text': text_data['text'],
                    'speaker_id': text_data['speaker_id']
                })

                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1} FLEURS segments...")

            except Exception as e:
                logger.warning(f"Error processing FLEURS item {idx}: {e}")

        return processed_segments

    def collect_all_available_data(self, language_code: str) -> Dict:
        """Collect data from all available sources for a language"""
        logger.info(f"🌟 Collecting all available data for {language_code}")

        available_datasets = self.list_available_datasets(language_code)
        results = {
            'language_code': language_code,
            'datasets_attempted': len(available_datasets),
            'datasets_successful': 0,
            'total_segments': 0,
            'results_by_dataset': {}
        }

        for dataset_name in available_datasets:
            logger.info(f"📥 Attempting to download {dataset_name} for {language_code}")

            try:
                if dataset_name == 'common_voice':
                    result = self.download_common_voice_dataset(language_code)
                elif dataset_name == 'openslr':
                    result = self.download_openslr_dataset(language_code)
                elif dataset_name == 'google_fleurs':
                    result = self.download_fleurs_dataset(language_code)
                else:
                    result = {'success': False, 'error': 'Dataset not implemented'}

                results['results_by_dataset'][dataset_name] = result

                if result['success']:
                    results['datasets_successful'] += 1
                    results['total_segments'] += result.get('segments', result.get('processed_segments', 0))
                    logger.info(f"✅ {dataset_name} downloaded successfully")
                else:
                    logger.warning(f"❌ {dataset_name} download failed: {result.get('error')}")

            except Exception as e:
                logger.error(f"Error downloading {dataset_name}: {e}")
                results['results_by_dataset'][dataset_name] = {'success': False, 'error': str(e)}

        # Save combined results
        base_dir = Path("data") / language_code
        metadata_dir = base_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        results_file = metadata_dir / f"dataset_collection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        logger.info(f"🎉 Data collection completed for {language_code}")
        logger.info(f"   Successful datasets: {results['datasets_successful']}/{results['datasets_attempted']}")
        logger.info(f"   Total segments: {results['total_segments']}")
        logger.info(f"   Results saved: {results_file}")

        return results


class AdditionalDatasetCollector:
    """Collector for additional open datasets and custom sources"""

    def __init__(self):
        self.additional_sources = self._load_additional_sources()

    def _load_additional_sources(self) -> Dict:
        """Load information about additional data sources"""
        return {
            'ai4bharat_indicwav2vec': {
                'name': 'AI4Bharat IndicWav2Vec',
                'url': 'https://github.com/AI4Bharat/IndicWav2Vec',
                'languages': ['hi', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or'],
                'description': 'Large-scale multilingual speech corpus'
            },
            'mucs_corpus': {
                'name': 'MUCS (Multilingual and Code-Switching) Corpus',
                'url': 'https://github.com/iitbhi/MUCS-Corpus',
                'languages': ['hi', 'bn'],  # Hindi-English, Bengali-English
                'description': 'Code-switching speech corpus'
            },
            'iisc_mile_corpus': {
                'name': 'IISc MILE Speech Corpus',
                'url': 'https://mile.ee.iisc.ac.in/MILE/',
                'languages': ['hi', 'ta', 'te', 'kn'],
                'description': 'Multi-lingual Indian Language speech corpus'
            },
            'custom_recordings': {
                'name': 'Custom Recordings',
                'description': 'User-provided recordings',
                'languages': ['hi', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or']
            }
        }

    def setup_custom_recording_interface(self, language_code: str) -> Dict:
        """Setup interface for custom recordings"""
        logger.info(f"🎙️ Setting up custom recording interface for {language_code}")

        base_dir = Path("data") / language_code
        custom_dir = base_dir / "custom_recordings"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Create templates for custom data
        templates = {
            'recording_template.json': {
                'instructions': [
                    f"Record audio files in {language_code} and place them in the 'audio' folder",
                    "Create corresponding text files with the same name in the 'text' folder",
                    "Use the manifest template to describe your recordings",
                    "Supported audio formats: WAV, MP3, FLAC",
                    "Recommended: 16kHz sample rate, mono channel"
                ],
                'example_manifest': {
                    'recording_001.wav': {
                        'text': 'Example text in your language',
                        'speaker_id': 'speaker_001',
                        'gender': 'male/female/other',
                        'age_group': 'young/middle/senior',
                        'accent': 'region_name',
                        'quality': 'high/medium/low',
                        'notes': 'Any additional notes'
                    }
                }
            },
            'README.md': f"""# Custom Recordings for {language_code.upper()}

## Quick Start
1. Place audio files in `audio/` folder
2. Place corresponding text files in `text/` folder  
3. Update the manifest.json with your recording details
4. Run the system to process your custom data

## File Structure
```
custom_recordings/
├── audio/
│   ├── recording_001.wav
│   ├── recording_002.wav
│   └── ...
├── text/
│   ├── recording_001.txt
│   ├── recording_002.txt
│   └── ...
├── manifest.json
└── README.md
```

## Audio Requirements
- Format: WAV (preferred), MP3, FLAC
- Sample Rate: 16kHz recommended
- Channels: Mono preferred
- Duration: 2-20 seconds per clip
- Quality: Clear speech, minimal background noise

## Text Requirements
- Use native script for {language_code}
- One sentence per file
- Avoid very short (<5 words) or very long (>25 words) sentences
- Include punctuation for natural prosody

## Processing
Your recordings will be automatically processed when you run:
```bash
python main.py -> Option 3 -> Data Collection -> Custom Recordings
```
"""
        }

        # Create template files
        for filename, content in templates.items():
            file_path = custom_dir / filename
            if isinstance(content, dict):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

        # Create directories
        (custom_dir / 'audio').mkdir(exist_ok=True)
        (custom_dir / 'text').mkdir(exist_ok=True)

        logger.info(f"✅ Custom recording interface ready at: {custom_dir}")

        return {
            'success': True,
            'custom_dir': str(custom_dir),
            'instructions': templates['README.md']
        }

    def process_custom_recordings(self, language_code: str) -> Dict:
        """Process user-provided custom recordings"""
        logger.info(f"🔄 Processing custom recordings for {language_code}")

        base_dir = Path("data") / language_code
        custom_dir = base_dir / "custom_recordings"

        if not custom_dir.exists():
            return {'success': False, 'error': 'Custom recordings directory not found'}

        audio_dir = custom_dir / 'audio'
        text_dir = custom_dir / 'text'

        if not audio_dir.exists() or not text_dir.exists():
            return {'success': False, 'error': 'Audio or text directories not found'}

        # Find audio files
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.m4a']:
            audio_files.extend(audio_dir.glob(f'*{ext}'))

        if not audio_files:
            return {'success': False, 'error': 'No audio files found'}

        # Process each audio file
        processed_segments = []
        successful_processes = 0

        for audio_file in audio_files:
            try:
                # Find corresponding text file
                text_file = text_dir / f"{audio_file.stem}.txt"
                if not text_file.exists():
                    text_file = text_dir / f"{audio_file.stem}.json"

                if not text_file.exists():
                    logger.warning(f"No text file found for {audio_file.name}")
                    continue

                # Read text content
                if text_file.suffix == '.json':
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text_data = json.load(f)
                        text_content = text_data.get('text', '')
                else:
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text_content = f.read().strip()

                if not text_content:
                    logger.warning(f"Empty text for {audio_file.name}")
                    continue

                # Copy to our standard format
                target_audio = base_dir / "raw_audio" / f"custom_{audio_file.name}"
                target_audio.parent.mkdir(exist_ok=True)

                # Convert audio to standard format if needed
                if audio_file.suffix != '.wav':
                    self._convert_audio_to_wav(audio_file, target_audio.with_suffix('.wav'))
                    target_audio = target_audio.with_suffix('.wav')
                else:
                    import shutil
                    shutil.copy2(audio_file, target_audio)

                # Create metadata
                segment_data = {
                    'audio_file': str(target_audio),
                    'text': text_content,
                    'speaker_id': f"custom_speaker_{successful_processes}",
                    'source': 'custom_recordings',
                    'language': language_code,
                    'processed_at': datetime.now().isoformat()
                }

                processed_segments.append(segment_data)
                successful_processes += 1

            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")

        if processed_segments:
            # Save manifest
            manifest_file = base_dir / "metadata" / f"custom_recordings_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            manifest_file.parent.mkdir(exist_ok=True)

            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'language_code': language_code,
                    'source': 'custom_recordings',
                    'total_segments': len(processed_segments),
                    'created_at': datetime.now().isoformat(),
                    'segments': processed_segments
                }, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Processed {successful_processes} custom recordings")

            return {
                'success': True,
                'segments': successful_processes,
                'manifest_file': str(manifest_file)
            }
        else:
            return {'success': False, 'error': 'No recordings could be processed'}

    def _convert_audio_to_wav(self, input_file: Path, output_file: Path):
        """Convert audio file to WAV format"""
        try:
            import librosa
            import soundfile as sf

            # Load audio
            audio, sr = librosa.load(input_file, sr=16000, mono=True)

            # Save as WAV
            sf.write(output_file, audio, sr)

        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            # Fallback: copy original file
            import shutil
            shutil.copy2(input_file, output_file)


def main():
    """Test the Common Voice collector"""
    collector = CommonVoiceCollector()

    # Show available datasets
    available = collector.list_available_datasets()
    print("Available datasets:")
    for dataset, info in available.items():
        print(f"  {dataset}: {info['name']}")
        print(f"    Languages: {list(info['languages'].keys())}")

    # Test download for Hindi
    # result = collector.download_common_voice_dataset('hi')
    # print(f"Download result: {result}")


if __name__ == "__main__":
    main()