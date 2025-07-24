"""
Audio Processing Utilities for Multilingual TTS System v2.0
Common audio operations, format conversions, and quality assessment
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import json
from datetime import datetime
import subprocess
import shutil

logger = logging.getLogger(__name__)


class AudioUtils:
    """Comprehensive audio processing utilities"""

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self.supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']

    def load_audio(self, file_path: Union[str, Path],
                   sr: Optional[int] = None,
                   mono: bool = True,
                   normalize: bool = False) -> Tuple[np.ndarray, int]:
        """
        Load audio file with robust error handling

        Args:
            file_path: Path to audio file
            sr: Target sample rate (None = keep original)
            mono: Convert to mono
            normalize: Apply normalization

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")

            # Use target sample rate if not specified
            if sr is None:
                sr = self.target_sr

            # Load audio
            audio, sample_rate = librosa.load(file_path, sr=sr, mono=mono)

            # Apply normalization if requested
            if normalize:
                audio = self.normalize_audio(audio)

            logger.debug(f"Loaded audio: {file_path} ({len(audio) / sample_rate:.2f}s, {sample_rate}Hz)")
            return audio, sample_rate

        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise

    def save_audio(self, audio: np.ndarray,
                   file_path: Union[str, Path],
                   sr: int,
                   format: str = 'wav',
                   quality: str = 'high') -> bool:
        """
        Save audio to file with format conversion

        Args:
            audio: Audio array
            file_path: Output file path
            sr: Sample rate
            format: Output format ('wav', 'mp3', 'flac')
            quality: Quality setting ('high', 'medium', 'low')

        Returns:
            Success status
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == 'wav':
                # Save as WAV
                sf.write(file_path, audio, sr)

            elif format.lower() == 'mp3':
                # Save as MP3 (requires ffmpeg)
                temp_wav = file_path.with_suffix('.temp.wav')
                sf.write(temp_wav, audio, sr)

                # Convert to MP3
                quality_settings = {
                    'high': '128k',
                    'medium': '96k',
                    'low': '64k'
                }
                bitrate = quality_settings.get(quality, '128k')

                cmd = [
                    'ffmpeg', '-i', str(temp_wav),
                    '-codec:a', 'mp3', '-b:a', bitrate,
                    '-y', str(file_path)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    temp_wav.unlink()  # Remove temporary file
                else:
                    logger.warning(f"MP3 conversion failed, saving as WAV instead")
                    shutil.move(temp_wav, file_path.with_suffix('.wav'))

            elif format.lower() == 'flac':
                # Save as FLAC
                sf.write(file_path, audio, sr, format='FLAC')

            else:
                # Default to WAV
                sf.write(file_path, audio, sr)

            logger.debug(f"Saved audio: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving audio to {file_path}: {e}")
            return False

    def convert_format(self, input_path: Union[str, Path],
                       output_path: Union[str, Path],
                       target_format: str = 'wav',
                       target_sr: Optional[int] = None,
                       quality: str = 'high') -> bool:
        """
        Convert audio file format and sample rate

        Args:
            input_path: Input audio file
            output_path: Output file path
            target_format: Target format
            target_sr: Target sample rate
            quality: Conversion quality

        Returns:
            Success status
        """
        try:
            # Load audio
            audio, sr = self.load_audio(input_path, sr=target_sr)

            # Save in target format
            return self.save_audio(audio, output_path, sr, target_format, quality)

        except Exception as e:
            logger.error(f"Error converting {input_path} to {output_path}: {e}")
            return False

    def normalize_audio(self, audio: np.ndarray,
                        method: str = 'peak',
                        target_level: float = 0.95) -> np.ndarray:
        """
        Normalize audio using different methods

        Args:
            audio: Input audio array
            method: Normalization method ('peak', 'rms', 'lufs')
            target_level: Target level (0-1 for peak, dB for others)

        Returns:
            Normalized audio array
        """
        try:
            if method == 'peak':
                # Peak normalization
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio * (target_level / peak)

            elif method == 'rms':
                # RMS normalization
                rms = np.sqrt(np.mean(audio ** 2))
                if rms > 0:
                    target_rms = target_level if target_level <= 1 else target_level / 100
                    audio = audio * (target_rms / rms)

            elif method == 'lufs':
                # LUFS normalization (simplified)
                # This is a basic implementation - for precise LUFS, use specialized libraries
                rms = np.sqrt(np.mean(audio ** 2))
                if rms > 0:
                    # Convert target LUFS to linear scale (approximate)
                    target_linear = 10 ** (target_level / 20)
                    audio = audio * (target_linear / rms)

            # Ensure no clipping
            audio = np.clip(audio, -1.0, 1.0)

            return audio

        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return audio

    def trim_silence(self, audio: np.ndarray,
                     sr: int,
                     top_db: int = 30,
                     frame_length: int = 2048,
                     hop_length: int = 512) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Trim silence from beginning and end of audio

        Args:
            audio: Input audio array
            sr: Sample rate
            top_db: Silence threshold in dB below peak
            frame_length: Frame length for analysis
            hop_length: Hop length for analysis

        Returns:
            Tuple of (trimmed_audio, (start_sample, end_sample))
        """
        try:
            # Use librosa's trim function
            trimmed_audio, intervals = librosa.effects.trim(
                audio,
                top_db=top_db,
                frame_length=frame_length,
                hop_length=hop_length
            )

            # Calculate trim indices
            start_sample = intervals[0]
            end_sample = intervals[1]

            logger.debug(f"Trimmed silence: {start_sample / sr:.3f}s to {end_sample / sr:.3f}s")

            return trimmed_audio, (start_sample, end_sample)

        except Exception as e:
            logger.error(f"Error trimming silence: {e}")
            return audio, (0, len(audio))

    def split_audio(self, audio: np.ndarray,
                    sr: int,
                    segment_length: float = 10.0,
                    overlap: float = 0.5) -> List[Tuple[np.ndarray, float, float]]:
        """
        Split audio into segments with optional overlap

        Args:
            audio: Input audio array
            sr: Sample rate
            segment_length: Length of each segment in seconds
            overlap: Overlap between segments in seconds

        Returns:
            List of (segment_audio, start_time, end_time) tuples
        """
        try:
            segments = []
            segment_samples = int(segment_length * sr)
            overlap_samples = int(overlap * sr)
            step_samples = segment_samples - overlap_samples

            total_samples = len(audio)

            for start in range(0, total_samples - segment_samples + 1, step_samples):
                end = start + segment_samples
                segment = audio[start:end]

                start_time = start / sr
                end_time = end / sr

                segments.append((segment, start_time, end_time))

            # Handle remaining audio if any
            if total_samples % step_samples > overlap_samples:
                start = total_samples - segment_samples
                if start > 0:
                    segment = audio[start:]
                    start_time = start / sr
                    end_time = total_samples / sr
                    segments.append((segment, start_time, end_time))

            logger.debug(f"Split audio into {len(segments)} segments")
            return segments

        except Exception as e:
            logger.error(f"Error splitting audio: {e}")
            return [(audio, 0.0, len(audio) / sr)]

    def merge_audio_segments(self, segments: List[Tuple[np.ndarray, float, float]],
                             total_duration: float,
                             sr: int,
                             fade_duration: float = 0.01) -> np.ndarray:
        """
        Merge audio segments back into continuous audio

        Args:
            segments: List of (audio, start_time, end_time) tuples
            total_duration: Total duration of output audio
            sr: Sample rate
            fade_duration: Crossfade duration in seconds

        Returns:
            Merged audio array
        """
        try:
            total_samples = int(total_duration * sr)
            merged_audio = np.zeros(total_samples)
            fade_samples = int(fade_duration * sr)

            for segment_audio, start_time, end_time in segments:
                start_sample = int(start_time * sr)
                end_sample = min(start_sample + len(segment_audio), total_samples)

                # Apply fade in/out for smooth transitions
                if fade_samples > 0 and len(segment_audio) > 2 * fade_samples:
                    # Fade in
                    fade_in = np.linspace(0, 1, fade_samples)
                    segment_audio[:fade_samples] *= fade_in

                    # Fade out
                    fade_out = np.linspace(1, 0, fade_samples)
                    segment_audio[-fade_samples:] *= fade_out

                # Add to merged audio
                segment_length = end_sample - start_sample
                merged_audio[start_sample:end_sample] += segment_audio[:segment_length]

            return merged_audio

        except Exception as e:
            logger.error(f"Error merging audio segments: {e}")
            return np.array([])

    def calculate_audio_features(self, audio: np.ndarray,
                                 sr: int) -> Dict:
        """
        Calculate comprehensive audio features

        Args:
            audio: Input audio array
            sr: Sample rate

        Returns:
            Dictionary of audio features
        """
        try:
            features = {}

            # Basic properties
            features['duration'] = len(audio) / sr
            features['sample_rate'] = sr
            features['num_samples'] = len(audio)

            # Amplitude features
            features['rms'] = np.sqrt(np.mean(audio ** 2))
            features['peak'] = np.max(np.abs(audio))
            features['dynamic_range'] = features['peak'] - np.min(np.abs(audio[audio != 0])) if np.any(
                audio != 0) else 0

            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)

            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)

            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            features['zero_crossing_rate_mean'] = np.mean(zcr)

            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
            features['mfcc_std'] = np.std(mfccs, axis=1).tolist()

            # Pitch features (fundamental frequency)
            try:
                pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
                pitches_clean = pitches[magnitudes > np.percentile(magnitudes, 85)]
                if len(pitches_clean) > 0:
                    features['pitch_mean'] = np.mean(pitches_clean[pitches_clean > 0])
                    features['pitch_std'] = np.std(pitches_clean[pitches_clean > 0])
                else:
                    features['pitch_mean'] = 0
                    features['pitch_std'] = 0
            except:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0

            # Signal-to-noise ratio estimation
            signal_power = np.mean(audio ** 2)
            noise_power = np.var(audio - np.mean(audio))
            features['snr_db'] = 10 * np.log10(signal_power / (noise_power + 1e-10))

            # Silence analysis
            silence_threshold = 0.01
            silence_samples = np.sum(np.abs(audio) < silence_threshold)
            features['silence_ratio'] = silence_samples / len(audio)

            return features

        except Exception as e:
            logger.error(f"Error calculating audio features: {e}")
            return {}

    def assess_audio_quality(self, audio: np.ndarray,
                             sr: int,
                             min_duration: float = 1.0,
                             max_duration: float = 30.0) -> Dict:
        """
        Assess audio quality for TTS training

        Args:
            audio: Input audio array
            sr: Sample rate
            min_duration: Minimum acceptable duration
            max_duration: Maximum acceptable duration

        Returns:
            Quality assessment dictionary
        """
        try:
            assessment = {
                'overall_quality': 'unknown',
                'quality_score': 0.0,
                'issues': [],
                'recommendations': []
            }

            # Calculate features
            features = self.calculate_audio_features(audio, sr)
            duration = features['duration']

            # Duration check
            if duration < min_duration:
                assessment['issues'].append(f"Too short: {duration:.2f}s < {min_duration}s")
                assessment['recommendations'].append("Use longer audio clips")
            elif duration > max_duration:
                assessment['issues'].append(f"Too long: {duration:.2f}s > {max_duration}s")
                assessment['recommendations'].append("Split into shorter segments")

            # SNR check
            snr_db = features.get('snr_db', 0)
            if snr_db < 10:
                assessment['issues'].append(f"Low SNR: {snr_db:.1f}dB")
                assessment['recommendations'].append("Apply noise reduction")

            # Silence ratio check
            silence_ratio = features.get('silence_ratio', 0)
            if silence_ratio > 0.5:
                assessment['issues'].append(f"Too much silence: {silence_ratio:.1%}")
                assessment['recommendations'].append("Trim excessive silence")

            # Dynamic range check
            dynamic_range = features.get('dynamic_range', 0)
            if dynamic_range < 0.1:
                assessment['issues'].append("Low dynamic range")
                assessment['recommendations'].append("Check audio normalization")

            # Calculate overall quality score
            score = 0

            # Duration score (0-25 points)
            if min_duration <= duration <= max_duration:
                if 2 <= duration <= 15:  # Optimal range
                    score += 25
                else:
                    score += 20

            # SNR score (0-30 points)
            if snr_db >= 20:
                score += 30
            elif snr_db >= 15:
                score += 25
            elif snr_db >= 10:
                score += 20
            elif snr_db >= 5:
                score += 10

            # Silence ratio score (0-25 points)
            if silence_ratio <= 0.1:
                score += 25
            elif silence_ratio <= 0.2:
                score += 20
            elif silence_ratio <= 0.3:
                score += 15
            elif silence_ratio <= 0.4:
                score += 10

            # Spectral quality score (0-20 points)
            spectral_centroid = features.get('spectral_centroid_mean', 0)
            if 1000 <= spectral_centroid <= 4000:  # Good for speech
                score += 20
            elif 500 <= spectral_centroid <= 6000:
                score += 15
            elif spectral_centroid > 0:
                score += 10

            assessment['quality_score'] = score / 100.0

            # Determine overall quality
            if score >= 80:
                assessment['overall_quality'] = 'excellent'
            elif score >= 60:
                assessment['overall_quality'] = 'good'
            elif score >= 40:
                assessment['overall_quality'] = 'acceptable'
            elif score >= 20:
                assessment['overall_quality'] = 'poor'
            else:
                assessment['overall_quality'] = 'unusable'

            return assessment

        except Exception as e:
            logger.error(f"Error assessing audio quality: {e}")
            return {
                'overall_quality': 'error',
                'quality_score': 0.0,
                'issues': [f"Error during assessment: {e}"],
                'recommendations': []
            }

    def batch_process_audio_files(self, input_dir: Union[str, Path],
                                  output_dir: Union[str, Path],
                                  operation: str = 'convert',
                                  **kwargs) -> Dict:
        """
        Batch process multiple audio files

        Args:
            input_dir: Directory containing input audio files
            output_dir: Directory for output files
            operation: Operation to perform ('convert', 'normalize', 'trim')
            **kwargs: Additional arguments for the operation

        Returns:
            Processing results dictionary
        """
        try:
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Find all audio files
            audio_files = []
            for ext in self.supported_formats:
                audio_files.extend(input_dir.glob(f'*{ext}'))

            if not audio_files:
                return {
                    'success': False,
                    'error': 'No audio files found',
                    'processed': 0,
                    'total': 0
                }

            results = {
                'success': True,
                'processed': 0,
                'total': len(audio_files),
                'failed': [],
                'details': []
            }

            for audio_file in audio_files:
                try:
                    output_file = output_dir / f"{audio_file.stem}_processed{audio_file.suffix}"

                    if operation == 'convert':
                        target_format = kwargs.get('target_format', 'wav')
                        target_sr = kwargs.get('target_sr', self.target_sr)
                        success = self.convert_format(audio_file, output_file, target_format, target_sr)

                    elif operation == 'normalize':
                        audio, sr = self.load_audio(audio_file)
                        normalized_audio = self.normalize_audio(audio, **kwargs)
                        success = self.save_audio(normalized_audio, output_file, sr)

                    elif operation == 'trim':
                        audio, sr = self.load_audio(audio_file)
                        trimmed_audio, _ = self.trim_silence(audio, sr, **kwargs)
                        success = self.save_audio(trimmed_audio, output_file, sr)

                    else:
                        success = False
                        logger.error(f"Unknown operation: {operation}")

                    if success:
                        results['processed'] += 1
                        results['details'].append({
                            'input': str(audio_file),
                            'output': str(output_file),
                            'status': 'success'
                        })
                    else:
                        results['failed'].append(str(audio_file))

                except Exception as e:
                    logger.error(f"Error processing {audio_file}: {e}")
                    results['failed'].append(str(audio_file))

            logger.info(f"Batch processing completed: {results['processed']}/{results['total']} files")
            return results

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'processed': 0,
                'total': 0
            }

    def get_audio_info(self, file_path: Union[str, Path]) -> Dict:
        """
        Get detailed information about an audio file

        Args:
            file_path: Path to audio file

        Returns:
            Audio information dictionary
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                return {'error': 'File not found'}

            # Get file info
            info = {
                'file_path': str(file_path),
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'format': file_path.suffix.lower()
            }

            # Get audio properties using soundfile
            try:
                with sf.SoundFile(file_path) as f:
                    info['duration'] = len(f) / f.samplerate
                    info['sample_rate'] = f.samplerate
                    info['channels'] = f.channels
                    info['frames'] = len(f)
                    info['subtype'] = f.subtype
            except:
                # Fallback to librosa
                audio, sr = librosa.load(file_path, sr=None)
                info['duration'] = len(audio) / sr
                info['sample_rate'] = sr
                info['channels'] = 1 if audio.ndim == 1 else audio.shape[1]
                info['frames'] = len(audio)

            # Calculate bitrate (approximate)
            if info['duration'] > 0:
                info['bitrate_kbps'] = (info['file_size_mb'] * 8 * 1024) / info['duration']

            return info

        except Exception as e:
            logger.error(f"Error getting audio info for {file_path}: {e}")
            return {'error': str(e)}


# Create global instance
audio_utils = AudioUtils()