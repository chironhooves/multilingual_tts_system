"""
Forced Alignment Module
Handles Montreal Forced Alignment (MFA) for creating time-aligned text-audio pairs
"""

import os
import subprocess
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import tempfile
import re

from config.languages import IndianLanguages
from config.settings import SystemSettings

logger = logging.getLogger(__name__)


class ForcedAligner:
    """Handles forced alignment using Montreal Forced Aligner"""

    def __init__(self):
        self.settings = SystemSettings()
        self.languages = IndianLanguages()
        self.mfa_available = False
        self.mfa_models = {}

        # Check MFA availability
        self.check_mfa_installation()

    def check_mfa_installation(self) -> bool:
        """Check if Montreal Forced Aligner is installed"""
        try:
            result = subprocess.run(['mfa', '--help'], capture_output=True, text=True)
            if result.returncode == 0:
                self.mfa_available = True
                logger.info("✅ Montreal Forced Aligner is available")
                self.check_available_models()
            else:
                logger.warning("⚠️ MFA command failed")
                self.mfa_available = False
        except FileNotFoundError:
            logger.warning("⚠️ Montreal Forced Aligner not found")
            logger.info("Install with: conda install -c conda-forge montreal-forced-alignment")
            self.mfa_available = False
        except Exception as e:
            logger.error(f"Error checking MFA: {e}")
            self.mfa_available = False

        return self.mfa_available

    def check_available_models(self):
        """Check what MFA models are available"""
        try:
            # List acoustic models
            result = subprocess.run(['mfa', 'model', 'download', 'acoustic', '--list'],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                models = result.stdout.strip().split('\n')
                for model in models:
                    if any(lang in model.lower() for lang in ['hindi', 'tamil', 'bengali']):
                        self.mfa_models[model] = 'acoustic'

            # List G2P models
            result = subprocess.run(['mfa', 'model', 'download', 'g2p', '--list'],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                models = result.stdout.strip().split('\n')
                for model in models:
                    if any(lang in model.lower() for lang in ['hindi', 'tamil', 'bengali']):
                        self.mfa_models[model] = 'g2p'

            logger.info(f"Found {len(self.mfa_models)} relevant MFA models")

        except Exception as e:
            logger.warning(f"Could not check MFA models: {e}")

    def install_mfa(self) -> bool:
        """Install Montreal Forced Aligner via conda"""
        logger.info("📦 Installing Montreal Forced Aligner...")

        try:
            # Check if conda is available
            subprocess.run(['conda', '--version'], check=True, capture_output=True)

            # Install MFA
            cmd = ['conda', 'install', '-c', 'conda-forge', 'montreal-forced-alignment', '-y']
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("✅ MFA installed successfully")
                self.mfa_available = True
                return True
            else:
                logger.error(f"MFA installation failed: {result.stderr}")
                return False

        except subprocess.CalledProcessError:
            logger.error("Conda not available. Please install conda first.")
            return False
        except Exception as e:
            logger.error(f"Error installing MFA: {e}")
            return False

    def download_language_model(self, language_code: str) -> Optional[str]:
        """Download appropriate model for a language"""
        if not self.mfa_available:
            logger.error("MFA not available")
            return None

        # Language code to MFA model mapping
        model_mapping = {
            'hi': 'hindi_mfa',
            'ta': 'tamil_mfa',
            'te': 'generic',  # Use generic if specific not available
            'bn': 'generic',
            'mr': 'generic',
            'gu': 'generic',
            'kn': 'generic',
            'ml': 'generic',
            'pa': 'generic',
            'or': 'generic'
        }

        model_name = model_mapping.get(language_code, 'generic')

        try:
            logger.info(f"📥 Downloading MFA model: {model_name}")

            # Try to download acoustic model
            cmd = ['mfa', 'model', 'download', 'acoustic', model_name]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"✅ Downloaded acoustic model: {model_name}")
                return model_name
            else:
                logger.warning(f"Could not download {model_name}, trying generic")

                # Fallback to generic model
                cmd = ['mfa', 'model', 'download', 'acoustic', 'generic']
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    logger.info("✅ Downloaded generic acoustic model")
                    return 'generic'
                else:
                    logger.error("Failed to download any acoustic model")
                    return None

        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return None

    def prepare_alignment_data(self, language_code: str) -> Optional[Path]:
        """Prepare data for MFA alignment"""
        logger.info(f"🔄 Preparing alignment data for {language_code}")

        base_dir = Path("data") / language_code
        audio_dir = base_dir / "processed_audio"
        text_dir = base_dir / "processed_text"

        # Create MFA input directory
        mfa_input_dir = base_dir / "mfa_input"
        if mfa_input_dir.exists():
            shutil.rmtree(mfa_input_dir)
        mfa_input_dir.mkdir(parents=True)

        # Get audio and text files
        audio_files = list(audio_dir.glob("*_processed.wav"))
        text_files = list(text_dir.glob("*_clean.json"))

        if not audio_files or not text_files:
            logger.error("No audio or text files found for alignment")
            return None

        logger.info(f"Found {len(audio_files)} audio files and {len(text_files)} text files")

        # Match audio and text files
        paired_files = 0

        for audio_file in audio_files:
            # Find corresponding text file
            base_name = audio_file.stem.replace('_processed', '')
            text_file = None

            for tf in text_files:
                if base_name in tf.stem:
                    text_file = tf
                    break

            if text_file:
                try:
                    # Load text data
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text_data = json.load(f)

                    # Create combined text for the entire file
                    combined_text = ' '.join(
                        [segment.get('clean_text', '') for segment in text_data if segment.get('clean_text')])

                    if combined_text.strip():
                        # Copy audio file
                        mfa_audio_file = mfa_input_dir / f"{base_name}.wav"
                        shutil.copy2(audio_file, mfa_audio_file)

                        # Create text file
                        mfa_text_file = mfa_input_dir / f"{base_name}.txt"
                        with open(mfa_text_file, 'w', encoding='utf-8') as f:
                            f.write(combined_text)

                        paired_files += 1

                except Exception as e:
                    logger.warning(f"Error processing {audio_file}: {e}")

        logger.info(f"✅ Prepared {paired_files} file pairs for alignment")

        if paired_files == 0:
            logger.error("No valid file pairs created")
            return None

        return mfa_input_dir

    def run_alignment(self, language_code: str, input_dir: Path, model_name: str = None) -> Optional[Path]:
        """Run MFA alignment"""
        if not self.mfa_available:
            logger.error("MFA not available")
            return None

        logger.info(f"🎯 Running forced alignment for {language_code}")

        # Setup output directory
        base_dir = Path("data") / language_code
        output_dir = base_dir / "aligned"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        # Use provided model or download one
        if model_name is None:
            model_name = self.download_language_model(language_code)
            if model_name is None:
                logger.error("No suitable model available")
                return None

        try:
            # Run MFA alignment
            cmd = [
                'mfa', 'align',
                str(input_dir),
                'generic',  # Use generic lexicon for now
                model_name,
                str(output_dir),
                '--clean'
            ]

            logger.info(f"Running MFA command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout

            if result.returncode == 0:
                logger.info("✅ MFA alignment completed successfully")
                return output_dir
            else:
                logger.error(f"MFA alignment failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("MFA alignment timed out")
            return None
        except Exception as e:
            logger.error(f"Error running MFA: {e}")
            return None

    def parse_textgrid_files(self, alignment_dir: Path, language_code: str) -> Dict:
        """Parse TextGrid files to extract aligned segments"""
        logger.info(f"📖 Parsing TextGrid files for {language_code}")

        base_dir = Path("data") / language_code
        segments_dir = base_dir / "aligned_segments"
        segments_dir.mkdir(exist_ok=True)

        # Get all TextGrid files
        textgrid_files = list(alignment_dir.glob("*.TextGrid"))

        if not textgrid_files:
            logger.warning("No TextGrid files found")
            return {'success': False, 'error': 'No TextGrid files'}

        results = {
            'total_files': len(textgrid_files),
            'processed_files': 0,
            'total_segments': 0,
            'segments': []
        }

        for textgrid_file in textgrid_files:
            try:
                segments = self.parse_single_textgrid(textgrid_file, language_code)

                if segments:
                    # Save segments for this file
                    segments_file = segments_dir / f"{textgrid_file.stem}_segments.json"
                    with open(segments_file, 'w', encoding='utf-8') as f:
                        json.dump(segments, f, ensure_ascii=False, indent=2)

                    results['processed_files'] += 1
                    results['total_segments'] += len(segments)
                    results['segments'].extend(segments)

            except Exception as e:
                logger.error(f"Error parsing {textgrid_file}: {e}")

        # Save combined segments
        all_segments_file = base_dir / "metadata" / f"all_aligned_segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(all_segments_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Parsed {results['processed_files']} TextGrid files")
        logger.info(f"   Total segments: {results['total_segments']}")
        logger.info(f"   Segments saved: {all_segments_file}")

        return results

    def parse_single_textgrid(self, textgrid_file: Path, language_code: str) -> List[Dict]:
        """Parse a single TextGrid file"""
        try:
            with open(textgrid_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple TextGrid parsing (basic implementation)
            # For production, consider using a proper TextGrid library like praatio
            segments = []

            # Extract intervals from TextGrid
            lines = content.split('\n')
            in_intervals = False
            current_interval = {}

            for line in lines:
                line = line.strip()

                if 'intervals:' in line:
                    in_intervals = True
                    continue

                if in_intervals:
                    if line.startswith('xmin'):
                        current_interval['start'] = float(line.split('=')[1].strip())
                    elif line.startswith('xmax'):
                        current_interval['end'] = float(line.split('=')[1].strip())
                    elif line.startswith('text'):
                        text = line.split('=')[1].strip().strip('"')
                        current_interval['text'] = text

                        # If we have all components, add segment
                        if all(key in current_interval for key in ['start', 'end', 'text']):
                            if current_interval['text'].strip():  # Only non-empty segments
                                duration = current_interval['end'] - current_interval['start']

                                # Filter by duration
                                if self.settings.MIN_SEGMENT_DURATION <= duration <= self.settings.MAX_SEGMENT_DURATION:
                                    segments.append({
                                        'start': current_interval['start'],
                                        'end': current_interval['end'],
                                        'duration': duration,
                                        'text': current_interval['text'],
                                        'audio_file': textgrid_file.stem,
                                        'language': language_code
                                    })

                            current_interval = {}

            return segments

        except Exception as e:
            logger.error(f"Error parsing TextGrid {textgrid_file}: {e}")
            return []

    def create_training_manifest(self, language_code: str) -> Optional[Path]:
        """Create training manifest from aligned segments"""
        logger.info(f"📋 Creating training manifest for {language_code}")

        base_dir = Path("data") / language_code
        segments_dir = base_dir / "aligned_segments"
        manifests_dir = base_dir / "manifests"
        manifests_dir.mkdir(exist_ok=True)

        # Load all segment files
        segment_files = list(segments_dir.glob("*_segments.json"))

        if not segment_files:
            logger.error("No segment files found")
            return None

        all_segments = []

        for segment_file in segment_files:
            try:
                with open(segment_file, 'r', encoding='utf-8') as f:
                    segments = json.load(f)
                all_segments.extend(segments)
            except Exception as e:
                logger.error(f"Error loading {segment_file}: {e}")

        if not all_segments:
            logger.error("No segments loaded")
            return None

        # Create manifest entries
        manifest_entries = []

        for segment in all_segments:
            # Find corresponding audio file
            audio_file = base_dir / "processed_audio" / f"{segment['audio_file']}_processed.wav"

            if audio_file.exists():
                manifest_entry = {
                    "audio_filepath": str(audio_file),
                    "text": segment['text'],
                    "duration": segment['duration'],
                    "start": segment['start'],
                    "end": segment['end'],
                    "language": segment['language']
                }
                manifest_entries.append(manifest_entry)

        # Save manifest
        manifest_file = manifests_dir / f"train_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        with open(manifest_file, 'w', encoding='utf-8') as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        logger.info(f"✅ Created training manifest: {manifest_file}")
        logger.info(f"   Entries: {len(manifest_entries)}")
        logger.info(f"   Total duration: {sum(e['duration'] for e in manifest_entries) / 3600:.2f} hours")

        return manifest_file

    def align_language_data(self, language_code: str) -> Dict:
        """Complete alignment pipeline for a language"""
        logger.info(f"🎯 Starting alignment pipeline for {language_code}")

        if not self.mfa_available:
            if not self.install_mfa():
                return {'success': False, 'error': 'MFA not available and installation failed'}

        # Step 1: Prepare data
        input_dir = self.prepare_alignment_data(language_code)
        if input_dir is None:
            return {'success': False, 'error': 'Failed to prepare alignment data'}

        # Step 2: Run alignment
        alignment_dir = self.run_alignment(language_code, input_dir)
        if alignment_dir is None:
            return {'success': False, 'error': 'Alignment failed'}

        # Step 3: Parse results
        parse_results = self.parse_textgrid_files(alignment_dir, language_code)
        if not parse_results.get('success', True):
            return {'success': False, 'error': 'Failed to parse alignment results'}

        # Step 4: Create training manifest
        manifest_file = self.create_training_manifest(language_code)
        if manifest_file is None:
            return {'success': False, 'error': 'Failed to create training manifest'}

        # Cleanup temporary files
        try:
            shutil.rmtree(input_dir)
        except:
            pass

        results = {
            'success': True,
            'language_code': language_code,
            'alignment_dir': str(alignment_dir),
            'manifest_file': str(manifest_file),
            'total_segments': parse_results['total_segments'],
            'processed_files': parse_results['processed_files']
        }

        logger.info(f"✅ Alignment pipeline completed for {language_code}")
        return results

    def validate_alignment_quality(self, language_code: str) -> Dict:
        """Validate the quality of alignment results"""
        logger.info(f"🔍 Validating alignment quality for {language_code}")

        base_dir = Path("data") / language_code
        manifests_dir = base_dir / "manifests"

        # Find latest manifest
        manifest_files = list(manifests_dir.glob("train_manifest_*.jsonl"))
        if not manifest_files:
            return {'success': False, 'error': 'No manifest files found'}

        latest_manifest = max(manifest_files, key=lambda x: x.stat().st_mtime)

        # Load manifest
        segments = []
        with open(latest_manifest, 'r', encoding='utf-8') as f:
            for line in f:
                segments.append(json.loads(line))

        # Quality metrics
        validation_results = {
            'total_segments': len(segments),
            'duration_stats': {
                'min_duration': min(s['duration'] for s in segments),
                'max_duration': max(s['duration'] for s in segments),
                'avg_duration': sum(s['duration'] for s in segments) / len(segments),
                'total_duration': sum(s['duration'] for s in segments)
            },
            'text_stats': {
                'min_length': min(len(s['text']) for s in segments),
                'max_length': max(len(s['text']) for s in segments),
                'avg_length': sum(len(s['text']) for s in segments) / len(segments)
            },
            'quality_issues': []
        }

        # Check for quality issues
        for i, segment in enumerate(segments):
            # Very short segments
            if segment['duration'] < 0.5:
                validation_results['quality_issues'].append(
                    f"Segment {i}: Very short duration ({segment['duration']:.2f}s)")

            # Very long segments
            if segment['duration'] > 25.0:
                validation_results['quality_issues'].append(
                    f"Segment {i}: Very long duration ({segment['duration']:.2f}s)")

            # Very short text
            if len(segment['text']) < 3:
                validation_results['quality_issues'].append(f"Segment {i}: Very short text")

            # Check if audio file exists
            if not Path(segment['audio_filepath']).exists():
                validation_results['quality_issues'].append(f"Segment {i}: Audio file missing")

        # Calculate quality score
        issue_ratio = len(validation_results['quality_issues']) / len(segments)
        validation_results['quality_score'] = max(0, 1.0 - issue_ratio)

        # Save validation report
        report_file = base_dir / "metadata" / f"alignment_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Alignment validation completed")
        logger.info(f"   Quality score: {validation_results['quality_score']:.2f}")
        logger.info(f"   Issues found: {len(validation_results['quality_issues'])}")
        logger.info(f"   Report: {report_file}")

        return validation_results

    def extract_segments_from_audio(self, language_code: str, manifest_file: Path) -> Dict:
        """Extract individual audio segments based on alignment"""
        logger.info(f"✂️ Extracting audio segments for {language_code}")

        base_dir = Path("data") / language_code
        segments_audio_dir = base_dir / "segment_audio"
        segments_audio_dir.mkdir(exist_ok=True)

        # Load manifest
        segments = []
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                segments.append(json.loads(line))

        extraction_results = {
            'total_segments': len(segments),
            'extracted_segments': 0,
            'failed_extractions': 0,
            'total_duration': 0
        }

        for i, segment in enumerate(segments):
            try:
                import librosa
                import soundfile as sf

                # Load full audio file
                audio, sr = librosa.load(segment['audio_filepath'], sr=self.settings.SAMPLE_RATE)

                # Extract segment
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)
                segment_audio = audio[start_sample:end_sample]

                # Save segment
                segment_filename = f"segment_{i:06d}_{segment['language']}.wav"
                segment_path = segments_audio_dir / segment_filename

                sf.write(segment_path, segment_audio, sr)

                extraction_results['extracted_segments'] += 1
                extraction_results['total_duration'] += segment['duration']

                # Update manifest entry with segment path
                segment['segment_audio_filepath'] = str(segment_path)

            except Exception as e:
                logger.warning(f"Failed to extract segment {i}: {e}")
                extraction_results['failed_extractions'] += 1

        # Save updated manifest
        updated_manifest_file = manifest_file.parent / f"train_manifest_with_segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(updated_manifest_file, 'w', encoding='utf-8') as f:
            for segment in segments:
                f.write(json.dumps(segment, ensure_ascii=False) + '\n')

        logger.info(f"✅ Segment extraction completed")
        logger.info(f"   Extracted: {extraction_results['extracted_segments']}/{extraction_results['total_segments']}")
        logger.info(f"   Total duration: {extraction_results['total_duration'] / 3600:.2f} hours")
        logger.info(f"   Updated manifest: {updated_manifest_file}")

        return extraction_results


class SimpleAligner:
    """Fallback aligner when MFA is not available"""

    def __init__(self):
        self.settings = SystemSettings()

    def simple_align(self, language_code: str) -> Dict:
        """Simple alignment using subtitle timestamps"""
        logger.info(f"🔄 Running simple alignment for {language_code} (MFA fallback)")

        base_dir = Path("data") / language_code
        text_dir = base_dir / "processed_text"
        audio_dir = base_dir / "processed_audio"
        aligned_dir = base_dir / "aligned_simple"
        aligned_dir.mkdir(exist_ok=True)

        # Get text files with timestamps
        text_files = list(text_dir.glob("*_clean.json"))

        all_segments = []

        for text_file in text_files:
            try:
                # Load subtitle data
                with open(text_file, 'r', encoding='utf-8') as f:
                    subtitles = json.load(f)

                # Find corresponding audio file
                base_name = text_file.stem.replace('_clean', '')
                audio_file = audio_dir / f"{base_name}_processed.wav"

                if audio_file.exists():
                    # Create segments using subtitle timestamps
                    for subtitle in subtitles:
                        if subtitle.get('clean_text'):
                            segment = {
                                'audio_filepath': str(audio_file),
                                'text': subtitle['clean_text'],
                                'start': subtitle['start'],
                                'end': subtitle['end'],
                                'duration': subtitle['duration'],
                                'language': language_code
                            }
                            all_segments.append(segment)

            except Exception as e:
                logger.error(f"Error processing {text_file}: {e}")

        # Save simple alignment manifest
        manifest_file = base_dir / "manifests" / f"simple_align_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        manifest_file.parent.mkdir(exist_ok=True)

        with open(manifest_file, 'w', encoding='utf-8') as f:
            for segment in all_segments:
                f.write(json.dumps(segment, ensure_ascii=False) + '\n')

        results = {
            'success': True,
            'method': 'simple_alignment',
            'language_code': language_code,
            'manifest_file': str(manifest_file),
            'total_segments': len(all_segments),
            'total_duration': sum(s['duration'] for s in all_segments)
        }

        logger.info(f"✅ Simple alignment completed for {language_code}")
        logger.info(f"   Segments: {len(all_segments)}")
        logger.info(f"   Duration: {results['total_duration'] / 3600:.2f} hours")

        return results


def main():
    """Test the alignment module"""
    aligner = ForcedAligner()

    if aligner.mfa_available:
        result = aligner.align_language_data('hi')
        print(f"Alignment result: {result}")
    else:
        simple_aligner = SimpleAligner()
        result = simple_aligner.simple_align('hi')
        print(f"Simple alignment result: {result}")


if __name__ == "__main__":
    main()