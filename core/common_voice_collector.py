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
from typing import Dict, List
from datetime import datetime
import shutil
from urllib.parse import urlparse

from config.languages import IndianLanguages
from config.settings import SystemSettings

logger = logging.getLogger(__name__)


class CommonVoiceCollector:
    """Handles Common Voice dataset collection and processing"""

    def __init__(self):
        self.settings = SystemSettings()
        self.languages = IndianLanguages()
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
                'languages': {
                    'hi': {'name': 'Hindi', 'available': True, 'version': '13.0'},
                    'ta': {'name': 'Tamil', 'available': True, 'version': '13.0'},
                    'te': {'name': 'Telugu', 'available': True, 'version': '13.0'},
                    'bn': {'name': 'Bengali', 'available': True, 'version': '13.0'},
                    'mr': {'name': 'Marathi', 'available': True, 'version': '13.0'},
                    'gu': {'name': 'Gujarati', 'available': True, 'version': '13.0'},
                    'kn': {'name': 'Kannada', 'available': True, 'version': '13.0'},
                    'ml': {'name': 'Malayalam', 'available': True, 'version': '13.0'},
                    'pa': {'name': 'Punjabi', 'available': True, 'version': '13.0'},
                    'or': {'name': 'Odia', 'available': False, 'version': 'N/A'}
                }
            },
            'openslr': {
                'name': 'OpenSLR Indian Language Datasets',
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
                'languages': {
                    'hi': {'available': True},
                    'ta': {'available': True},
                    'te': {'available': True},
                    'bn': {'available': True},
                    'mr': {'available': True},
                    'gu': {'available': True},
                    'kn': {'available': True},
                    'ml': {'available': True},
                    'pa': {'available': True},
                    'or': {'available': True}
                }
            },
            'indic_tts': {
                'name': 'AI4Bharat Indic-TTS',
                'languages': {
                    'hi': {'available': True},
                    'ta': {'available': True},
                    'te': {'available': True},
                    'bn': {'available': True},
                    'mr': {'available': True},
                    'gu': {'available': True},
                    'kn': {'available': True},
                    'ml': {'available': True}
                }
            }
        }

    def list_available_datasets(self, language_code: str = None) -> Dict:
        """List all available datasets for a language (or all languages)"""
        available = {}
        for name, info in self.datasets_info.items():
            langs = info['languages']
            if language_code:
                if langs.get(language_code, {}).get('available'):
                    available[name] = info['name']
            else:
                if any(l.get('available') for l in langs.values()):
                    available[name] = info['name']
        return available

    def download_common_voice_dataset(self, language_code: str, subset: str = 'train') -> Dict:
        """Download and extract Common Voice"""
        logger.info(f"📥 Downloading Common Voice for {language_code}")
        lang_info = self.datasets_info['common_voice']['languages'][language_code]
        version = lang_info['version']

        filename = f"cv-corpus-{version}-2023-09-08-{language_code}.tar.gz"
        # use known-working HF endpoint
        download_url = (
            f"https://huggingface.co/datasets/common_voice/cv-corpus-{version}/"
            f"resolve/main/{language_code}.tar.gz"
        )

        res = self._download_file(download_url, filename)
        if not res['success']:
            return {'success': False, 'error': res['error']}

        extract_res = self._extract_common_voice_dataset(res['file_path'], language_code, subset)
        if extract_res['success']:
            os.remove(res['file_path'])
        return extract_res

    def _extract_common_voice_dataset(self, archive_path: str, language_code: str, subset: str) -> Dict:
        """Extract and process Common Voice"""
        try:
            extract_dir = Path("data") / language_code / "common_voice"
            extract_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=extract_dir)

            proc = self._process_common_voice_data(extract_dir, language_code, subset)
            return proc
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _process_common_voice_data(self, extract_dir: Path, language_code: str, subset: str) -> Dict:
        """Process extracted Common Voice data"""
        tsvs = list(extract_dir.rglob(f"{subset}.tsv"))
        if not tsvs:
            return {'success': False, 'error': f"No {subset}.tsv found"}
        df = pd.read_csv(tsvs[0], sep='\t')
        base = Path("data") / language_code
        raw_audio = base / "raw_audio"; raw_text = base / "raw_text"; meta = base / "metadata"
        for d in [raw_audio, raw_text, meta]:
            d.mkdir(parents=True, exist_ok=True)

        count = 0
        for idx, row in df.iterrows():
            src = extract_dir / "clips" / row['path']
            if not src.exists(): continue
            tgt_audio = raw_audio / f"{language_code}_{idx:06d}.mp3"
            shutil.copy2(src, tgt_audio)
            data = {'text': row['sentence'], 'speaker_id': row.get('client_id', ''),
                    'source':'common_voice','language':language_code}
            tgt_txt = raw_text / f"{language_code}_{idx:06d}.json"
            with open(tgt_txt,'w',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False)
            count += 1

        manifest = meta / f"common_voice_{subset}_manifest.json"
        with open(manifest,'w',encoding='utf-8') as f:
            json.dump({'segments':count}, f)
        return {'success':True, 'segments':count, 'manifest_file':str(manifest)}

    def download_openslr_dataset(self, language_code: str) -> Dict:
        """Download and process OpenSLR"""
        info = self.datasets_info['openslr']['languages'][language_code]
        rid = info['resource_id']
        urls = [
            f"https://www.openslr.org/resources/{rid}/data_train.tar.gz",
            f"https://www.openslr.org/resources/{rid}/data_test.tar.gz",
            f"https://www.openslr.org/resources/{rid}/line_index.tsv"
        ]
        extract_dir = Path("data")/language_code/"openslr"
        extract_dir.mkdir(parents=True,exist_ok=True)
        downloaded = []
        for url in urls:
            fn = Path(urlparse(url).path).name
            r = self._download_file(url,f"openslr_{language_code}_{fn}")
            if r['success']:
                downloaded.append(r['file_path'])
                if fn.endswith('.tar.gz'):
                    with tarfile.open(r['file_path'],'r:gz') as tar:
                        tar.extractall(extract_dir)
                    os.remove(r['file_path'])
        if not downloaded: return {'success':False,'error':'no openslr files'}
        return self._process_openslr_data(extract_dir,language_code)

    def _process_openslr_data(self, extract_dir: Path, language_code: str) -> Dict:
        """Process OpenSLR files into raw_audio/raw_text"""
        tsv = extract_dir/"line_index.tsv"
        transcripts = {}
        if tsv.exists():
            df = pd.read_csv(tsv, sep='\t', header=None, names=['file','text'])
            transcripts = dict(zip(df['file'],df['text']))
        base = Path("data")/language_code
        raw_audio = base/"raw_audio"; raw_text=base/"raw_text"; meta=base/"metadata"
        for d in [raw_audio, raw_text, meta]:
            d.mkdir(parents=True, exist_ok=True)
        count=0
        for wav in extract_dir.rglob('*.wav'):
            txt = transcripts.get(wav.name,'')
            tgt_audio = raw_audio/wav.name
            shutil.copy2(wav,tgt_audio)
            tgt_txt = raw_text/wav.with_suffix('.json').name
            with open(tgt_txt,'w',encoding='utf-8') as f:
                json.dump({'text':txt,'speaker_id':''},f)
            count+=1
        manifest=meta/f"openslr_manifest.json"
        with open(manifest,'w',encoding='utf-8') as f: json.dump({'segments':count},f)
        return {'success':True,'segments':count,'manifest_file':str(manifest)}

    def download_fleurs_dataset(self, language_code: str) -> Dict:
        """Download and process Google FLEURS"""
        try:
            from datasets import load_dataset
        except ImportError:
            return {'success': False, 'error': 'Install with: pip install datasets huggingface_hub'}
        code = f"{language_code}_in"
        ds = load_dataset("google/fleurs", code, split="train")
        base=Path("data")/language_code
        raw_audio=base/"raw_audio"; raw_text=base/"raw_text"; meta=base/"metadata"
        for d in [raw_audio, raw_text, meta]:
            d.mkdir(parents=True,exist_ok=True)
        count=0
        import soundfile as sf
        for idx,item in enumerate(ds):
            af= raw_audio/f"fleurs_{language_code}_{idx:06d}.wav"
            sf.write(af,item['audio']['array'],item['audio']['sampling_rate'])
            tf= raw_text/f"fleurs_{language_code}_{idx:06d}.json"
            with open(tf,'w',encoding='utf-8') as f:
                json.dump({'text':item['transcription'],'speaker_id':''},f)
            count+=1
        m=meta/f"fleurs_manifest.json"
        with open(m,'w',encoding='utf-8') as f: json.dump({'segments':count},f)
        return {'success':True,'segments':count,'manifest_file':str(m)}

    def download_indic_tts_dataset(self, language_code: str) -> Dict:
        """Download and process AI4Bharat Indic-TTS from Hugging Face"""
        try:
            from datasets import load_dataset
        except ImportError:
            return {'success': False, 'error': 'Install with: pip install datasets huggingface_hub'}
        ds = load_dataset("ai4bharat/indic-tts", language_code, split="train")
        base=Path("data")/language_code
        raw_audio=base/"raw_audio"; raw_text=base/"raw_text"; meta=base/"metadata"
        for d in [raw_audio, raw_text, meta]:
            d.mkdir(parents=True,exist_ok=True)
        count=0
        import soundfile as sf
        for idx,item in enumerate(ds):
            af= raw_audio/f"indictts_{language_code}_{idx:06d}.wav"
            sf.write(af,item['audio']['array'],item['audio']['sampling_rate'])
            tf= raw_text/f"indictts_{language_code}_{idx:06d}.json"
            with open(tf,'w',encoding='utf-8') as f:
                json.dump({'text':item['text'],'speaker_id':''},f)
            count+=1
        m=meta/f"indictts_manifest.json"
        with open(m,'w',encoding='utf-8') as f: json.dump({'segments':count},f)
        return {'success':True,'segments':count,'manifest_file':str(m)}

    def collect_all_available_data(self, language_code: str) -> Dict:
        """Collect data from all available sources for a language"""
        results = {'language_code':language_code,'results_by_dataset':{}}
        for ds in self.list_available_datasets(language_code):
            try:
                if ds=='common_voice':
                    r=self.download_common_voice_dataset(language_code)
                elif ds=='openslr':
                    r=self.download_openslr_dataset(language_code)
                elif ds=='google_fleurs':
                    r=self.download_fleurs_dataset(language_code)
                elif ds=='indic_tts':
                    r=self.download_indic_tts_dataset(language_code)
                else:
                    r={'success':False,'error':'not implemented'}
            except Exception as e:
                r={'success':False,'error':str(e)}
            results['results_by_dataset'][ds]=r
        return results

    def _download_file(self, url: str, filename: str, chunk_size: int = 8192) -> Dict:
        """Download a file with progress tracking"""
        file_path = self.downloads_dir / filename
        try:
            if file_path.exists():
                return {'success': True, 'file_path': str(file_path)}
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            with open(file_path, 'wb') as f:
                for c in resp.iter_content(chunk_size):
                    f.write(c)
            return {'success': True, 'file_path': str(file_path)}
        except Exception as e:
            if file_path.exists(): file_path.unlink()
            return {'success': False, 'error': str(e)}


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
