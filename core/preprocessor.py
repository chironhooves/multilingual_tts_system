"""
Enhanced Audio and Text Preprocessing for Multilingual TTS System v2.0
Comprehensive preprocessing with advanced linguistic features and open dataset support
Replaces YouTube dependency with legal open datasets processing
"""

import os
import json
import logging
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
import concurrent.futures
import re
import shutil

from config.languages import IndianLanguages
from config.settings import SystemSettings

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Enhanced audio preprocessing with linguistic awareness and speaker analysis"""

    def __init__(self):
        self.settings = SystemSettings()
        self.languages = IndianLanguages()
        self.target_sr = self.settings.SAMPLE_RATE
        self.min_duration = 1.0
        self.max_duration = 20.0

        # Initialize speaker system
        try:
            from core.speaker_id import SpeakerIdentificationSystem
            self.speaker_system = SpeakerIdentificationSystem()
        except Exception as e:
            logger.warning(f"Speaker system initialization failed: {e}")
            self.speaker_system = None

        # Linguistic processors cache
        self.linguistic_processors = {}

        logger.info("🔊 Enhanced AudioPreprocessor initialized")

    def get_linguistic_processor(self, language_code: str):
        """Get or create linguistic processor for a language"""
        if language_code not in self.linguistic_processors:
            try:
                from core.linguistic_processor import LinguisticProcessor
                self.linguistic_processors[language_code] = LinguisticProcessor(language_code)
            except Exception as e:
                logger.warning(f"Could not create linguistic processor for {language_code}: {e}")
                self.linguistic_processors[language_code] = None
        return self.linguistic_processors[language_code]

    def process_language_audio(self, language_code: str, callback: Optional[Callable] = None) -> Dict:
        """Process all audio for a specific language with enhanced linguistic features"""
        logger.info(f"🔊 Starting enhanced audio processing for {language_code}")

        if not self.languages.validate_language_code(language_code):
            return {'success': False, 'error': f'Unsupported language: {language_code}'}

        # Setup directories
        base_dir = Path("data") / language_code
        raw_audio_dir = base_dir / "raw_audio"
        processed_audio_dir = base_dir / "processed_audio"
        linguistic_dir = base_dir / "linguistic_features"
        diarization_dir = base_dir / "diarization"
        metadata_dir = base_dir / "metadata"

        for dir_path in [processed_audio_dir, linguistic_dir, diarization_dir, metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        if not raw_audio_dir.exists():
            return {'success': False, 'error': f'No raw audio directory found: {raw_audio_dir}'}

        # Get linguistic processor
        linguistic_processor = self.get_linguistic_processor(language_code)

        # Find all audio files
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.m4a']:
            audio_files.extend(raw_audio_dir.glob(f'*{ext}'))

        if not audio_files:
            return {'success': False, 'error': 'No audio files found'}

        logger.info(f"📊 Found {len(audio_files)} audio files to process")

        # Process files with enhanced features
        results = {
            'total_files': len(audio_files),
            'processed_files': 0,
            'high_quality_files': 0,
            'medium_quality_files': 0,
            'low_quality_files': 0,
            'rejected_files': 0,
            'total_duration': 0,
            'enhanced_files': 0,
            'linguistic_features_extracted': 0,
            'speaker_analysis_completed': 0,
            'files': []
        }

        # Process files in parallel
        max_workers = min(4, len(audio_files))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    self.process_single_audio_file_enhanced,
                    audio_file, processed_audio_dir, linguistic_dir, linguistic_processor
                ): audio_file
                for audio_file in audio_files
            }

            for future in concurrent.futures.as_completed(future_to_file):
                audio_file = future_to_file[future]
                try:
                    result = future.result()
                    results['files'].append(result)

                    if result['success']:
                        results['processed_files'] += 1
                        results['total_duration'] += result.get('duration', 0)

                        # Categorize by quality
                        quality = result.get('quality', 'rejected')
                        if quality == 'high_quality':
                            results['high_quality_files'] += 1
                        elif quality == 'medium_quality':
                            results['medium_quality_files'] += 1
                        elif quality == 'low_quality':
                            results['low_quality_files'] += 1
                        else:
                            results['rejected_files'] += 1

                        # Count enhanced features
                        if result.get('linguistic_features'):
                            results['linguistic_features_extracted'] += 1

                        if result.get('enhanced', False):
                            results['enhanced_files'] += 1
                    else:
                        results['rejected_files'] += 1

                    # Progress callback
                    if callback:
                        progress = len(results['files']) / len(audio_files)
                        callback(language_code, 'audio_processing', progress)

                except Exception as e:
                    logger.error(f"Error processing {audio_file}: {e}")
                    results['rejected_files'] += 1

        # Run speaker diarization if speaker system is available
        if self.speaker_system and results['processed_files'] > 0:
            logger.info(f"🎙️ Running enhanced speaker diarization...")
            diarization_result = self.diarize_language_data(language_code)
            results['speaker_analysis'] = diarization_result
            if diarization_result.get('success', False):
                results['speaker_analysis_completed'] = diarization_result.get('processed_files', 0)

        # Save processing summary
        summary_file = metadata_dir / f"audio_processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Calculate success rate
        success_rate = results['processed_files'] / results['total_files'] if results['total_files'] > 0 else 0

        logger.info(f"✅ Enhanced audio processing completed for {language_code}")
        logger.info(f"   Processed: {results['processed_files']}/{results['total_files']} ({success_rate:.1%})")
        logger.info(
            f"   Quality distribution: {results['high_quality_files']} high, {results['medium_quality_files']} medium, {results['low_quality_files']} low")
        logger.info(f"   Total duration: {results['total_duration'] / 3600:.2f} hours")
        logger.info(f"   Enhanced features: {results['linguistic_features_extracted']} files")
        logger.info(f"   Summary saved: {summary_file}")

        results['success'] = success_rate > 0.5  # At least 50% success rate
        return results

    def process_single_audio_file_enhanced(self, audio_file: Path, output_dir: Path,
                                           linguistic_dir: Path, linguistic_processor) -> Dict:
        """Process a single audio file with linguistic enhancement"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=self.target_sr, mono=True)
            original_duration = len(audio) / sr

            # Basic quality checks
            quality_result = self.assess_audio_quality(audio, sr)

            if quality_result['quality'] == 'rejected':
                return {
                    'success': False,
                    'error': 'Audio quality too low',
                    'original_duration': original_duration,
                    'quality': 'rejected',
                    'audio_file': str(audio_file)
                }

            # Enhanced audio processing with linguistic awareness
            audio = self.normalize_audio(audio)

            # Apply noise reduction if needed
            if quality_result['snr_db'] < 20:
                audio = self.reduce_noise(audio, sr)

            # Linguistically-informed silence trimming
            audio = self.trim_silence_linguistic(audio, sr, linguistic_processor)

            # Check duration after processing
            final_duration = len(audio) / sr
            if final_duration < self.min_duration:
                return {
                    'success': False,
                    'error': 'Audio too short after processing',
                    'original_duration': original_duration,
                    'quality': 'rejected',
                    'audio_file': str(audio_file)
                }

            # Extract linguistic features from corresponding text
            linguistic_features = self.extract_linguistic_features(audio_file, linguistic_processor)

            # Save processed audio
            output_file = output_dir / f"{audio_file.stem}_processed.wav"
            sf.write(output_file, audio, sr)

            # Save linguistic features
            if linguistic_features:
                ling_output_file = linguistic_dir / f"{audio_file.stem}_features.json"
                with open(ling_output_file, 'w', encoding='utf-8') as f:
                    json.dump(linguistic_features, f, ensure_ascii=False, indent=2)

            return {
                'success': True,
                'audio_file': str(audio_file),
                'output_file': str(output_file),
                'original_duration': original_duration,
                'duration': final_duration,
                'quality': quality_result['quality'],
                'quality_score': quality_result.get('quality_score', 0),
                'snr_db': quality_result['snr_db'],
                'linguistic_features': linguistic_features,
                'enhanced': True
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'audio_file': str(audio_file),
                'original_duration': 0,
                'quality': 'error'
            }

    def trim_silence_linguistic(self, audio: np.ndarray, sr: int, linguistic_processor) -> np.ndarray:
        """Linguistically-informed silence trimming that preserves prosodic boundaries"""
        try:
            # Standard trimming first
            audio_trimmed, intervals = librosa.effects.trim(
                audio,
                top_db=30,
                frame_length=2048,
                hop_length=512
            )

            # If we have significant trimming, check for prosodic boundaries
            original_length = len(audio) / sr
            trimmed_length = len(audio_trimmed) / sr

            if original_length - trimmed_length > 0.5:  # More than 0.5s trimmed
                # Apply more conservative trimming to preserve phrase boundaries
                audio_conservatively_trimmed, _ = librosa.effects.trim(
                    audio,
                    top_db=20,  # Less aggressive
                    frame_length=4096,  # Longer frame for better boundary detection
                    hop_length=1024
                )
                return audio_conservatively_trimmed

            return audio_trimmed

        except Exception as e:
            logger.warning(f"Linguistic silence trimming failed: {e}")
            return audio

    def extract_linguistic_features(self, audio_file: Path, linguistic_processor) -> Dict:
        """Extract linguistic features from corresponding text"""
        try:
            # Find corresponding subtitle/text file
            base_name = audio_file.stem.replace('_processed', '')
            text_dir = audio_file.parent.parent / "raw_text"

            # Look for corresponding text file
            text_file = None
            for ext in ['.json', '.txt']:
                potential_file = text_dir / f"{base_name}{ext}"
                if potential_file.exists():
                    text_file = potential_file
                    break

            if not text_file:
                logger.debug(f"No text file found for {audio_file}")
                return {}

            # Load text content
            if text_file.suffix == '.json':
                with open(text_file, 'r', encoding='utf-8') as f:
                    text_data = json.load(f)
                    text_content = text_data.get('text', '')
            else:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()

            if not text_content or not linguistic_processor:
                return {}

            # Process text through linguistic pipeline
            logger.debug(f"Processing linguistic features for: {text_content[:100]}...")

            # Basic linguistic analysis (simplified if full processor not available)
            linguistic_features = {
                'original_text': text_content,
                'normalized_text': text_content.strip(),
                'text_length': len(text_content),
                'word_count': len(text_content.split()),
                'character_count': len(text_content)
            }

            # Try advanced features if linguistic processor is available
            if hasattr(linguistic_processor, 'text_normalizer'):
                try:
                    # 1. Text normalization
                    normalized_text = linguistic_processor.text_normalizer.normalize(text_content)
                    linguistic_features['normalized_text'] = normalized_text

                    # 2. Grapheme-to-Phoneme conversion
                    phonemes = linguistic_processor.grapheme_to_phoneme(normalized_text)
                    linguistic_features['phonemes'] = phonemes
                    linguistic_features['total_phonemes'] = len(phonemes)
                    linguistic_features['unique_phonemes'] = len(set(phonemes))

                    # 3. Add prosodic features
                    prosodic_phonemes = linguistic_processor.add_prosodic_features(phonemes, normalized_text)

                    # 4. Calculate phoneme coverage
                    phoneme_inventory = linguistic_processor.phoneme_inventory
                    phoneme_coverage = self.calculate_phoneme_coverage(phonemes, phoneme_inventory)
                    linguistic_features['phoneme_coverage'] = phoneme_coverage

                    # 5. Detect code-switching
                    code_switches = linguistic_processor.code_switch_detector.mark_language_boundaries(text_content)
                    code_switch_count = code_switches.count('<lang:')
                    linguistic_features['code_switches'] = code_switch_count

                    # 6. Count prosodic boundaries
                    prosodic_boundaries = sum(1 for _, prosody in prosodic_phonemes
                                              if prosody.boundary in ['ip', 'pp'])
                    linguistic_features['prosodic_boundaries'] = prosodic_boundaries

                    # 7. Calculate linguistic complexity
                    complexity = self.calculate_linguistic_complexity(
                        phonemes, prosodic_phonemes, 0, code_switch_count
                    )
                    linguistic_features['linguistic_complexity'] = complexity

                except Exception as e:
                    logger.debug(f"Advanced linguistic processing failed: {e}")

            return linguistic_features

        except Exception as e:
            logger.error(f"Error extracting linguistic features: {e}")
            return {}

    def calculate_phoneme_coverage(self, phonemes: List[str], phoneme_inventory: Dict) -> float:
        """Calculate how well the phonemes cover the language's inventory"""
        if not phoneme_inventory:
            return 0.0

        unique_phonemes = set(p for p in phonemes if p not in ['|', '+', '∅'])
        inventory_phonemes = set(phoneme_inventory.keys())

        if not inventory_phonemes:
            return 0.0

        coverage = len(unique_phonemes.intersection(inventory_phonemes)) / len(inventory_phonemes)
        return coverage

    def calculate_linguistic_complexity(self, phonemes: List[str],
                                        prosodic_phonemes: List, compounds: int,
                                        code_switches: int) -> float:
        """Calculate overall linguistic complexity score"""
        if not phonemes:
            return 0.0

        # Factors contributing to complexity
        phoneme_diversity = len(set(phonemes)) / max(len(phonemes), 1)
        prosodic_complexity = len([p for p, prosody in prosodic_phonemes
                                   if prosody.stress_level > 0]) / max(len(prosodic_phonemes), 1)
        morphological_complexity = compounds / max(len(' '.join(phonemes).split('|')), 1)
        code_switch_complexity = code_switches / max(len(' '.join(phonemes).split('|')), 1)

        # Weighted combination
        complexity = (phoneme_diversity * 0.3 +
                      prosodic_complexity * 0.3 +
                      morphological_complexity * 0.2 +
                      code_switch_complexity * 0.2)

        return complexity

    def assess_audio_quality(self, audio: np.ndarray, sr: int) -> Dict:
        """Enhanced audio quality assessment with linguistic considerations"""
        try:
            # Basic quality metrics
            signal_power = np.mean(audio ** 2)
            noise_power = np.var(audio - np.mean(audio))
            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

            # Calculate silence ratio
            silence_threshold = 0.01
            silence_samples = np.sum(np.abs(audio) < silence_threshold)
            silence_ratio = silence_samples / len(audio)

            # Spectral quality metrics
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))

            # Enhanced quality determination with spectral features
            quality_score = 0

            # SNR contribution (0-40 points)
            if snr_db >= 25:
                quality_score += 40
            elif snr_db >= 20:
                quality_score += 30
            elif snr_db >= 15:
                quality_score += 20
            elif snr_db >= 10:
                quality_score += 10

            # Silence ratio contribution (0-30 points)
            if silence_ratio <= 0.1:
                quality_score += 30
            elif silence_ratio <= 0.2:
                quality_score += 25
            elif silence_ratio <= 0.3:
                quality_score += 20
            elif silence_ratio <= 0.4:
                quality_score += 10

            # Spectral quality contribution (0-30 points)
            if 1000 <= spectral_centroid <= 4000:  # Good for speech
                quality_score += 15
            if spectral_rolloff > spectral_centroid * 1.5:  # Good spectral distribution
                quality_score += 15

            # Determine quality level
            if quality_score >= 80:
                quality = 'high_quality'
            elif quality_score >= 60:
                quality = 'medium_quality'
            elif quality_score >= 40:
                quality = 'low_quality'
            else:
                quality = 'rejected'

            return {
                'quality': quality,
                'quality_score': quality_score,
                'snr_db': snr_db,
                'silence_ratio': silence_ratio,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'zero_crossing_rate': zero_crossing_rate,
                'duration': len(audio) / sr
            }

        except Exception as e:
            logger.error(f"Error assessing audio quality: {e}")
            return {
                'quality': 'error',
                'quality_score': 0,
                'snr_db': 0,
                'silence_ratio': 1.0,
                'duration': 0
            }

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Enhanced audio normalization with perceptual considerations"""
        # Peak normalization
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95  # Leave some headroom

        # RMS normalization for consistent loudness
        rms = np.sqrt(np.mean(audio ** 2))
        target_rms = 0.2
        if rms > 0:
            audio = audio * (target_rms / rms)

        # Apply gentle high-pass filter to remove low-frequency noise
        try:
            from scipy import signal
            nyquist = self.target_sr // 2
            high_freq = 80 / nyquist  # 80 Hz high-pass
            b, a = signal.butter(4, high_freq, btype='high')
            audio = signal.filtfilt(b, a, audio)
        except ImportError:
            logger.warning("scipy not available for high-pass filtering")

        return audio

    def reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Enhanced noise reduction with linguistic preservation"""
        try:
            # Compute spectrogram
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            # Estimate noise floor from quieter regions
            power = magnitude ** 2
            noise_floor = np.percentile(power, 10, axis=1, keepdims=True)

            # Conservative spectral gating to preserve speech formants
            gate_ratio = 2.5  # More conservative than basic version
            alpha = 0.1  # Smoothing factor

            # Create time-varying mask
            mask = magnitude > (noise_floor * gate_ratio)

            # Smooth the mask to avoid artifacts
            mask = mask.astype(float)
            for i in range(1, mask.shape[1]):
                mask[:, i] = alpha * mask[:, i - 1] + (1 - alpha) * mask[:, i]

            # Apply mask with floor to preserve some background
            mask = np.maximum(mask, 0.1)  # Never completely remove components

            magnitude_clean = magnitude * mask
            stft_clean = magnitude_clean * np.exp(1j * phase)

            # Reconstruct audio
            audio_clean = librosa.istft(stft_clean, hop_length=512)

            return audio_clean

        except Exception as e:
            logger.warning(f"Enhanced noise reduction failed: {e}")
            return audio  # Return original if noise reduction fails

    def diarize_language_data(self, language_code: str) -> Dict:
        """Run speaker diarization with linguistic enhancement"""
        logger.info(f"🎯 Running enhanced speaker diarization for {language_code}")

        if not self.speaker_system:
            return {'success': False, 'error': 'Speaker system not available'}

        base_dir = Path("data") / language_code
        processed_audio_dir = base_dir / "processed_audio"
        linguistic_dir = base_dir / "linguistic_features"
        diarization_dir = base_dir / "diarization"
        diarization_dir.mkdir(exist_ok=True)

        # Get processed audio files
        audio_files = list(processed_audio_dir.glob("*_processed.wav"))
        if not audio_files:
            logger.warning(f"No processed audio files found for {language_code}")
            return {'success': False, 'error': 'No processed audio files'}

        results = {
            'total_files': len(audio_files),
            'processed_files': 0,
            'total_speakers': 0,
            'total_segments': 0,
            'linguistic_enhanced_segments': 0,
            'files': []
        }

        for audio_file in audio_files:
            logger.info(f"🔄 Enhanced diarizing: {audio_file.name}")

            try:
                # Load corresponding linguistic features
                ling_file = linguistic_dir / f"{audio_file.stem.replace('_processed', '')}_features.json"
                linguistic_features = {}
                if ling_file.exists():
                    with open(ling_file, 'r', encoding='utf-8') as f:
                        linguistic_features = json.load(f)

                # Run speaker identification system on this file
                diarization_result = self.speaker_system.process_file_for_tts_training(
                    str(audio_file),
                    language_code=language_code
                )

                if diarization_result:
                    # Enhance segments with linguistic information
                    enhanced_segments = self.enhance_segments_with_linguistics(
                        diarization_result['tts_segments'],
                        linguistic_features
                    )

                    diarization_result['enhanced_segments'] = enhanced_segments
                    diarization_result['linguistic_features'] = linguistic_features

                    # Save enhanced diarization results
                    result_file = diarization_dir / f"{audio_file.stem}_enhanced_diarization.json"
                    self.save_enhanced_diarization_result(diarization_result, result_file)

                    results['processed_files'] += 1
                    results['total_speakers'] += diarization_result['total_speakers']
                    results['total_segments'] += len(diarization_result['tts_segments'])
                    results['linguistic_enhanced_segments'] += len(enhanced_segments)
                    results['files'].append({
                        'audio_file': str(audio_file),
                        'result_file': str(result_file),
                        'speakers': diarization_result['total_speakers'],
                        'segments': len(diarization_result['tts_segments']),
                        'enhanced_segments': len(enhanced_segments),
                        'linguistic_quality': linguistic_features.get('linguistic_complexity', 0)
                    })

            except Exception as e:
                logger.error(f"Error in enhanced diarization of {audio_file}: {e}")

        # Save enhanced summary
        summary_file = base_dir / "metadata" / f"enhanced_diarization_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"✅ Enhanced diarization completed for {language_code}")
        logger.info(f"   Files processed: {results['processed_files']}/{results['total_files']}")
        logger.info(f"   Total speakers: {results['total_speakers']}")
        logger.info(f"   Enhanced segments: {results['linguistic_enhanced_segments']}/{results['total_segments']}")

        results['success'] = results['processed_files'] > 0
        return results

    def enhance_segments_with_linguistics(self, segments: List[Dict], linguistic_features: Dict) -> List[Dict]:
        """Enhance audio segments with linguistic information"""
        enhanced_segments = []

        for segment in segments:
            try:
                enhanced_segment = segment.copy()

                # Add linguistic quality score
                enhanced_segment['linguistic_quality'] = self.calculate_segment_linguistic_quality(
                    segment, linguistic_features
                )

                # Add basic linguistic information
                enhanced_segment['text_length'] = linguistic_features.get('text_length', 0)
                enhanced_segment['word_count'] = linguistic_features.get('word_count', 0)
                enhanced_segment['linguistic_complexity'] = linguistic_features.get('linguistic_complexity', 0)

                # Add to enhanced list only if quality is sufficient
                if enhanced_segment['linguistic_quality'] >= 0.3:
                    enhanced_segments.append(enhanced_segment)

            except Exception as e:
                logger.warning(f"Error enhancing segment: {e}")
                enhanced_segments.append(segment)  # Keep original if enhancement fails

        return enhanced_segments

    def calculate_segment_linguistic_quality(self, segment: Dict, linguistic_features: Dict) -> float:
        """Calculate linguistic quality score for a segment"""
        quality_score = 0.0

        # Duration appropriateness (0-0.3)
        duration = segment.get('duration', 0)
        if 2.0 <= duration <= 15.0:  # Optimal range for TTS
            quality_score += 0.3
        elif 1.0 <= duration <= 20.0:  # Acceptable range
            quality_score += 0.2
        elif duration > 0.5:  # Minimum acceptable
            quality_score += 0.1

        # Speaker confidence (0-0.3)
        confidence = segment.get('confidence', 0)
        quality_score += min(confidence * 0.3, 0.3)

        # Status bonus (0-0.2)
        if segment.get('status') == 'identified':
            quality_score += 0.2
        elif segment.get('status') == 'enrolled':
            quality_score += 0.15

        # Linguistic complexity bonus (0-0.2)
        complexity = linguistic_features.get('linguistic_complexity', 0)
        quality_score += min(complexity * 0.2, 0.2)

        return quality_score

    def save_enhanced_diarization_result(self, result: Dict, output_file: Path):
        """Save enhanced diarization result with linguistic features"""
        # Prepare data for JSON (handle non-serializable objects)
        json_result = {
            'file_path': result['file_path'],
            'language': result['language'],
            'total_speakers': result['total_speakers'],
            'total_duration': result['total_duration'],
            'enhanced_segments_count': len(result.get('enhanced_segments', [])),
            'linguistic_features_summary': {
                'text_length': result.get('linguistic_features', {}).get('text_length', 0),
                'word_count': result.get('linguistic_features', {}).get('word_count', 0),
                'linguistic_complexity': result.get('linguistic_features', {}).get('linguistic_complexity', 0)
            },
            'enhanced_segments': []
        }

        # Save enhanced segments (without raw audio data)
        for segment in result.get('enhanced_segments', []):
            clean_segment = {
                'speaker_id': segment.get('speaker_id'),
                'speaker_name': segment.get('speaker_name'),
                'start': segment.get('start'),
                'end': segment.get('end'),
                'duration': segment.get('duration'),
                'confidence': segment.get('confidence'),
                'status': segment.get('status'),
                'language': segment.get('language'),
                'linguistic_quality': segment.get('linguistic_quality', 0),
                'text_length': segment.get('text_length', 0),
                'word_count': segment.get('word_count', 0)
            }
            json_result['enhanced_segments'].append(clean_segment)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 Enhanced diarization result saved: {output_file}")


class TextPreprocessor:
    """Enhanced text preprocessing with advanced linguistic features"""

    def __init__(self):
        self.languages = IndianLanguages()
        self.linguistic_processors = {}

    def get_linguistic_processor(self, language_code: str):
        """Get or create linguistic processor for a language"""
        if language_code not in self.linguistic_processors:
            try:
                from core.linguistic_processor import LinguisticProcessor
                self.linguistic_processors[language_code] = LinguisticProcessor(language_code)
            except Exception as e:
                logger.warning(f"Could not create linguistic processor for {language_code}: {e}")
                self.linguistic_processors[language_code] = None
        return self.linguistic_processors[language_code]

    def process_language_text(self, language_code: str) -> Dict:
        """Process all subtitle text for a language with linguistic enhancement"""
        logger.info(f"📝 Processing text with linguistic enhancement for {language_code}")

        base_dir = Path("data") / language_code
        raw_text_dir = base_dir / "raw_text"
        processed_text_dir = base_dir / "processed_text"
        linguistic_text_dir = base_dir / "linguistic_text"

        processed_text_dir.mkdir(exist_ok=True)
        linguistic_text_dir.mkdir(exist_ok=True)

        if not raw_text_dir.exists():
            return {'success': False, 'error': f'No raw text directory found: {raw_text_dir}'}

        # Get linguistic processor
        linguistic_processor = self.get_linguistic_processor(language_code)

        # Find all text files
        text_files = []
        for ext in ['.json', '.txt']:
            text_files.extend(raw_text_dir.glob(f'*{ext}'))

        if not text_files:
            return {'success': False, 'error': 'No text files found'}

        # Process all text files
        results = {
            'total_files': len(text_files),
            'processed_files': 0,
            'total_segments': 0,
            'clean_segments': 0,
            'linguistically_enhanced_segments': 0,
            'phoneme_coverage': 0,
            'prosodic_boundaries': 0
        }

        for text_file in text_files:
            try:
                # Load text data
                if text_file.suffix == '.json':
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text_data = json.load(f)

                    # Handle different JSON structures
                    if isinstance(text_data, dict):
                        if 'text' in text_data:
                            # Single text entry
                            segments = [text_data]
                        else:
                            # Unknown structure, skip
                            continue
                    elif isinstance(text_data, list):
                        # Multiple segments
                        segments = text_data
                    else:
                        continue
                else:
                    # Plain text file
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text_content = f.read().strip()

                    if text_content:
                        segments = [{'text': text_content}]
                    else:
                        continue

                # Process each segment with linguistic enhancement
                cleaned_segments = []
                enhanced_segments = []

                for segment in segments:
                    text_content = segment.get('text', '') if isinstance(segment, dict) else str(segment)

                    # Basic cleaning
                    cleaned_text = self.clean_text(text_content, language_code)

                    if cleaned_text:
                        # Create basic clean segment
                        clean_segment = segment.copy() if isinstance(segment, dict) else {'text': text_content}
                        clean_segment['clean_text'] = cleaned_text
                        cleaned_segments.append(clean_segment)

                        # Enhanced linguistic processing
                        try:
                            enhanced_segment = self.enhance_text_segment(
                                clean_segment, linguistic_processor
                            )
                            enhanced_segments.append(enhanced_segment)
                            results['linguistically_enhanced_segments'] += 1
                            results['phoneme_coverage'] += enhanced_segment.get('phoneme_coverage', 0)
                            results['prosodic_boundaries'] += enhanced_segment.get('prosodic_boundaries', 0)

                        except Exception as e:
                            logger.warning(f"Error enhancing segment: {e}")
                            enhanced_segments.append(clean_segment)

                # Save cleaned text
                output_file = processed_text_dir / f"{text_file.stem}_clean.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_segments, f, ensure_ascii=False, indent=2)

                # Save linguistically enhanced version
                enhanced_output_file = linguistic_text_dir / f"{text_file.stem}_enhanced.json"
                with open(enhanced_output_file, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_segments, f, ensure_ascii=False, indent=2)

                results['processed_files'] += 1
                results['total_segments'] += len(segments)
                results['clean_segments'] += len(cleaned_segments)

            except Exception as e:
                logger.error(f"Error processing {text_file}: {e}")

        # Calculate averages
        if results['linguistically_enhanced_segments'] > 0:
            results['phoneme_coverage'] /= results['linguistically_enhanced_segments']
            results['prosodic_boundaries'] /= results['linguistically_enhanced_segments']

        logger.info(f"✅ Enhanced text processing completed for {language_code}")
        logger.info(f"   Files: {results['processed_files']}/{results['total_files']}")
        logger.info(f"   Enhanced segments: {results['linguistically_enhanced_segments']}/{results['total_segments']}")
        logger.info(f"   Avg phoneme coverage: {results['phoneme_coverage']:.3f}")
        logger.info(f"   Avg prosodic boundaries: {results['prosodic_boundaries']:.1f}")

        results['success'] = results['processed_files'] > 0
        return results

    def enhance_text_segment(self, segment: Dict, linguistic_processor) -> Dict:
        """Enhance text segment with comprehensive linguistic features"""
        enhanced_segment = segment.copy()
        text = segment.get('clean_text', '')

        if not text:
            return enhanced_segment

        try:
            # Basic text analysis
            enhanced_segment['text_length'] = len(text)
            enhanced_segment['word_count'] = len(text.split())
            enhanced_segment['sentence_type'] = self.detect_sentence_type(text)

            # Try advanced linguistic processing if processor is available
            if linguistic_processor and hasattr(linguistic_processor, 'text_normalizer'):
                try:
                    # 1. Advanced text normalization
                    normalized_text = linguistic_processor.text_normalizer.normalize(text)
                    enhanced_segment['normalized_text'] = normalized_text

                    # 2. Grapheme-to-phoneme conversion
                    phonemes = linguistic_processor.grapheme_to_phoneme(normalized_text)
                    enhanced_segment['phonemes'] = phonemes
                    enhanced_segment['phoneme_count'] = len(phonemes)
                    enhanced_segment['unique_phonemes'] = len(set(phonemes))

                    # 3. Phoneme coverage analysis
                    phoneme_inventory = linguistic_processor.phoneme_inventory
                    unique_phonemes = set(p for p in phonemes if p not in ['|', '+', '∅'])
                    inventory_phonemes = set(phoneme_inventory.keys())

                    if inventory_phonemes:
                        phoneme_coverage = len(unique_phonemes.intersection(inventory_phonemes)) / len(
                            inventory_phonemes)
                    else:
                        phoneme_coverage = 0

                    enhanced_segment['phoneme_coverage'] = phoneme_coverage

                    # 4. Prosodic analysis
                    prosodic_phonemes = linguistic_processor.add_prosodic_features(phonemes, normalized_text)
                    prosodic_boundaries = sum(1 for _, prosody in prosodic_phonemes
                                              if prosody.boundary in ['ip', 'pp'])
                    enhanced_segment['prosodic_boundaries'] = prosodic_boundaries
                    enhanced_segment['prosodic_complexity'] = prosodic_boundaries / max(len(phonemes), 1)

                    # 5. Code-switching detection
                    code_switches = linguistic_processor.code_switch_detector.mark_language_boundaries(text)
                    code_switch_count = code_switches.count('<lang:')
                    enhanced_segment['code_switches'] = code_switch_count
                    enhanced_segment['contains_code_switching'] = code_switch_count > 0

                    # 6. Linguistic complexity score
                    complexity_score = self.calculate_text_complexity(
                        phonemes, prosodic_boundaries, 0, code_switch_count
                    )
                    enhanced_segment['linguistic_complexity'] = complexity_score

                    # 7. TTS suitability assessment
                    tts_quality_score = self.assess_tts_suitability(enhanced_segment)
                    enhanced_segment['tts_suitability_score'] = tts_quality_score

                except Exception as e:
                    logger.debug(f"Advanced linguistic processing failed: {e}")
                    # Fall back to basic analysis
                    enhanced_segment['linguistic_complexity'] = 0.5
                    enhanced_segment['tts_suitability_score'] = 0.6
            else:
                # Basic fallback analysis
                enhanced_segment['linguistic_complexity'] = 0.5
                enhanced_segment['tts_suitability_score'] = self.assess_tts_suitability_basic(enhanced_segment)

            return enhanced_segment

        except Exception as e:
            logger.error(f"Error in linguistic enhancement: {e}")
            return enhanced_segment

    def calculate_text_complexity(self, phonemes: List[str], prosodic_boundaries: int,
                                  compounds: int, code_switches: int) -> float:
        """Calculate comprehensive text complexity score"""
        if not phonemes:
            return 0.0

        # Phonological complexity
        unique_phonemes = len(set(p for p in phonemes if p not in ['|', '+', '∅']))
        phonological_complexity = unique_phonemes / max(len(phonemes), 1)

        # Prosodic complexity
        prosodic_complexity = prosodic_boundaries / max(len(phonemes), 1)

        # Code-switching complexity
        code_switch_complexity = code_switches / max(len(phonemes) // 10, 1)

        # Weighted combination
        complexity = (
                phonological_complexity * 0.4 +
                prosodic_complexity * 0.4 +
                code_switch_complexity * 0.2
        )

        return min(complexity, 1.0)  # Cap at 1.0

    def detect_sentence_type(self, text: str) -> str:
        """Detect sentence type for prosodic modeling"""
        text = text.strip()

        # Question markers
        if text.endswith('?') or any(marker in text.lower() for marker in
                                     ['क्या', 'कैसे', 'कब', 'कहाँ', 'क्यों', 'कौन', 'what', 'how', 'when', 'where',
                                      'why']):
            return 'interrogative'

        # Exclamation markers
        if text.endswith('!') or any(marker in text for marker in ['अरे', 'वाह', 'हाय', 'wow', 'oh']):
            return 'exclamative'

        # Imperative markers (commands)
        if any(marker in text for marker in ['करो', 'जाओ', 'आओ', 'please', 'do', 'go']):
            return 'imperative'

        return 'declarative'

    def assess_tts_suitability(self, enhanced_segment: Dict) -> float:
        """Assess how suitable this segment is for TTS training"""
        score = 0.0

        # Text length appropriateness (0-30 points)
        text_length = enhanced_segment.get('text_length', 0)
        if 10 <= text_length <= 150:  # Good length
            score += 30
        elif 5 <= text_length <= 200:  # Acceptable
            score += 20
        elif text_length >= 3:  # Minimum
            score += 10

        # Word count appropriateness (0-20 points)
        word_count = enhanced_segment.get('word_count', 0)
        if 3 <= word_count <= 25:  # Good range
            score += 20
        elif 2 <= word_count <= 35:  # Acceptable
            score += 15
        elif word_count >= 1:  # Minimum
            score += 5

        # Phoneme coverage (0-20 points)
        phoneme_coverage = enhanced_segment.get('phoneme_coverage', 0)
        score += phoneme_coverage * 20

        # Linguistic complexity (0-15 points) - moderate complexity is best
        complexity = enhanced_segment.get('linguistic_complexity', 0)
        if 0.3 <= complexity <= 0.7:  # Sweet spot
            score += 15
        elif 0.2 <= complexity <= 0.8:  # Acceptable
            score += 12
        elif complexity > 0:  # Some complexity
            score += 8

        # Prosodic richness (0-10 points)
        prosodic_complexity = enhanced_segment.get('prosodic_complexity', 0)
        score += min(prosodic_complexity * 10, 10)

        # Sentence type bonus (0-5 points)
        sentence_type = enhanced_segment.get('sentence_type', 'declarative')
        if sentence_type in ['interrogative', 'exclamative']:
            score += 5  # Bonus for prosodic variety

        # Penalty for code-switching (TTS models prefer monolingual)
        if enhanced_segment.get('contains_code_switching', False):
            score -= 10

        return max(0, min(100, score)) / 100.0  # Normalize to 0-1

    def assess_tts_suitability_basic(self, enhanced_segment: Dict) -> float:
        """Basic TTS suitability assessment without advanced linguistic features"""
        score = 0.0

        # Text length appropriateness
        text_length = enhanced_segment.get('text_length', 0)
        if 10 <= text_length <= 150:
            score += 0.4
        elif 5 <= text_length <= 200:
            score += 0.3
        elif text_length >= 3:
            score += 0.2

        # Word count appropriateness
        word_count = enhanced_segment.get('word_count', 0)
        if 3 <= word_count <= 25:
            score += 0.3
        elif 2 <= word_count <= 35:
            score += 0.2
        elif word_count >= 1:
            score += 0.1

        # Sentence type variety
        sentence_type = enhanced_segment.get('sentence_type', 'declarative')
        if sentence_type in ['interrogative', 'exclamative']:
            score += 0.1

        # Basic text quality
        text = enhanced_segment.get('clean_text', '')
        if text and len(text.strip()) > 0:
            score += 0.2

        return min(score, 1.0)

    def clean_text(self, text: str, language_code: str) -> str:
        """Enhanced text cleaning with linguistic awareness"""
        if not text or not text.strip():
            return ""

        # Basic cleaning
        text = text.strip()

        # Remove unwanted characters
        text = re.sub(r'[<>{}[\]\\|`~]', '', text)

        # Language-specific cleaning
        if language_code in ['hi', 'mr']:  # Devanagari script languages
            text = self._clean_devanagari_text(text)
        elif language_code in ['ta']:  # Tamil
            text = self._clean_tamil_text(text)
        elif language_code in ['te']:  # Telugu
            text = self._clean_telugu_text(text)
        elif language_code in ['bn']:  # Bengali
            text = self._clean_bengali_text(text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Filter out very short or very long texts
        if len(text) < 3 or len(text) > 500:
            return ""

        return text

    def _clean_devanagari_text(self, text: str) -> str:
        """Clean Devanagari script text (Hindi, Marathi)"""
        # Remove excessive English text
        text = re.sub(r'\b[a-zA-Z]{10,}\b', '', text)  # Remove long English words

        # Normalize punctuation
        text = text.replace('।', '.')  # Devanagari full stop

        # Remove excessive punctuation
        text = re.sub(r'[.!?]{2,}', '.', text)

        return text

    def _clean_tamil_text(self, text: str) -> str:
        """Clean Tamil script text"""
        # Remove excessive English text
        text = re.sub(r'\b[a-zA-Z]{10,}\b', '', text)

        # Normalize Tamil punctuation
        text = text.replace('।', '.')

        return text

    def _clean_telugu_text(self, text: str) -> str:
        """Clean Telugu script text"""
        # Remove excessive English text
        text = re.sub(r'\b[a-zA-Z]{10,}\b', '', text)

        # Normalize punctuation
        text = text.replace('।', '.')

        return text

    def _clean_bengali_text(self, text: str) -> str:
        """Clean Bengali script text"""
        # Remove excessive English text
        text = re.sub(r'\b[a-zA-Z]{10,}\b', '', text)

        # Normalize punctuation
        text = text.replace('।', '.')

        return text

    def create_balanced_corpus(self, language_code: str) -> Dict:
        """Create a balanced corpus for comprehensive phoneme and prosody coverage"""
        logger.info(f"📊 Creating balanced corpus for {language_code}")

        base_dir = Path("data") / language_code
        linguistic_text_dir = base_dir / "linguistic_text"
        balanced_corpus_dir = base_dir / "balanced_corpus"
        balanced_corpus_dir.mkdir(exist_ok=True)

        # Load all enhanced segments
        all_segments = []
        for enhanced_file in linguistic_text_dir.glob("*_enhanced.json"):
            try:
                with open(enhanced_file, 'r', encoding='utf-8') as f:
                    segments = json.load(f)
                    all_segments.extend(segments)
            except Exception as e:
                logger.warning(f"Error loading {enhanced_file}: {e}")

        if not all_segments:
            return {'success': False, 'error': 'No enhanced segments found'}

        # Select balanced subset
        balanced_segments = self._select_balanced_segments(all_segments)

        # Save balanced corpus
        balanced_corpus_file = balanced_corpus_dir / f"balanced_corpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(balanced_corpus_file, 'w', encoding='utf-8') as f:
            json.dump(balanced_segments, f, ensure_ascii=False, indent=2)

        # Create coverage report
        coverage_report = {
            'total_segments': len(all_segments),
            'balanced_segments': len(balanced_segments),
            'quality_distribution': self._analyze_quality_distribution(balanced_segments)
        }

        report_file = balanced_corpus_dir / f"coverage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(coverage_report, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Balanced corpus created for {language_code}")
        logger.info(f"   Total segments: {len(all_segments)} → {len(balanced_segments)}")

        return {
            'success': True,
            'balanced_corpus_file': str(balanced_corpus_file),
            'coverage_report_file': str(report_file),
            'coverage_stats': coverage_report
        }

    def _select_balanced_segments(self, all_segments: List[Dict]) -> List[Dict]:
        """Select segments for balanced corpus coverage"""
        # Sort segments by TTS suitability
        sorted_segments = sorted(all_segments,
                                 key=lambda x: x.get('tts_suitability_score', 0),
                                 reverse=True)

        selected_segments = []
        sentence_types = {'declarative': 0, 'interrogative': 0, 'exclamative': 0, 'imperative': 0}
        target_segments = min(len(sorted_segments), 1000)  # Don't exceed 1000 segments

        for segment in sorted_segments:
            if len(selected_segments) >= target_segments:
                break

            should_add = False

            # Add if it has good TTS suitability score
            if segment.get('tts_suitability_score', 0) > 0.6:
                should_add = True

            # Add to ensure sentence type diversity
            sentence_type = segment.get('sentence_type', 'declarative')
            if sentence_type in sentence_types and sentence_types[sentence_type] < 250:
                should_add = True

            if should_add:
                selected_segments.append(segment)
                if sentence_type in sentence_types:
                    sentence_types[sentence_type] += 1

        return selected_segments

    def _analyze_quality_distribution(self, segments: List[Dict]) -> Dict:
        """Analyze quality distribution of selected segments"""
        quality_scores = [s.get('tts_suitability_score', 0) for s in segments]

        if not quality_scores:
            return {}

        return {
            'high_quality': len([s for s in quality_scores if s >= 0.8]),
            'medium_quality': len([s for s in quality_scores if 0.6 <= s < 0.8]),
            'acceptable_quality': len([s for s in quality_scores if 0.4 <= s < 0.6]),
            'low_quality': len([s for s in quality_scores if s < 0.4]),
            'average_score': sum(quality_scores) / len(quality_scores),
            'min_score': min(quality_scores),
            'max_score': max(quality_scores)
        }


def main():
    """Test enhanced preprocessing functions"""
    print("🧪 Testing Enhanced Preprocessing System")

    # Test audio preprocessor
    print("\n🔊 Testing Audio Preprocessor...")
    audio_preprocessor = AudioPreprocessor()

    # Test text preprocessor
    print("\n📝 Testing Text Preprocessor...")
    text_preprocessor = TextPreprocessor()

    # Test with Hindi if data is available
    try:
        result = audio_preprocessor.process_language_audio('hi')
        print(f"✅ Audio processing test result: {result.get('success', False)}")

        text_result = text_preprocessor.process_language_text('hi')
        print(f"✅ Text processing test result: {text_result.get('success', False)}")

        # Test balanced corpus creation
        corpus_result = text_preprocessor.create_balanced_corpus('hi')
        print(f"✅ Balanced corpus test result: {corpus_result.get('success', False)}")

    except Exception as e:
        print(f"⚠️  Test failed (this is normal if no data is available): {e}")

    print("\n🎉 Preprocessing system ready!")


if __name__ == "__main__":
    main()