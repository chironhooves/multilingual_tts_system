"""
Speaker Diarization and Identification System for Multilingual TTS v2.0
Custom implementation without pyannote dependency
Based on energy-based segmentation and MFCC features
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging
from datetime import datetime
import pickle
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import find_peaks
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SpeakerIdentificationSystem:
    """Custom speaker identification and diarization system"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.speaker_embeddings = {}
        self.speaker_models = {}
        self.embedding_dim = 128
        self.similarity_threshold = 0.75

        # Diarization parameters
        self.frame_length = 2048
        self.hop_length = 512
        self.min_segment_length = 1.0  # seconds
        self.max_segment_length = 30.0  # seconds

        # Feature extraction parameters
        self.n_mfcc = 13
        self.n_mels = 40

        logger.info("✅ Custom Speaker Identification System initialized")
        logger.info(f"   Sample rate: {self.sample_rate}Hz")
        logger.info(f"   Embedding dimension: {self.embedding_dim}")
        logger.info(f"   Similarity threshold: {self.similarity_threshold}")

    def extract_speaker_embedding(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """Extract speaker embedding using MFCC and spectral features"""
        try:
            if sr is None:
                sr = self.sample_rate

            # Resample if needed
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

            # Ensure minimum length
            min_samples = int(0.5 * self.sample_rate)  # 0.5 seconds minimum
            if len(audio) < min_samples:
                audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')

            # Extract comprehensive features
            features = self._extract_comprehensive_features(audio)

            # Normalize features
            features = self._normalize_features(features)

            return features

        except Exception as e:
            logger.error(f"Error extracting speaker embedding: {e}")
            return np.zeros(self.embedding_dim)

    def _extract_comprehensive_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract comprehensive speaker features"""
        features = []

        # 1. MFCC features (most important for speaker recognition)
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_delta = np.mean(librosa.feature.delta(mfccs), axis=1)
        mfcc_delta2 = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)

        features.extend(mfcc_mean)
        features.extend(mfcc_std)
        features.extend(mfcc_delta)
        features.extend(mfcc_delta2)

        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)

        features.extend([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.mean(zero_crossing_rate),
            np.std(zero_crossing_rate)
        ])

        # 3. Pitch features (fundamental frequency)
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if pitch_values:
                features.extend([
                    np.mean(pitch_values),
                    np.std(pitch_values),
                    np.min(pitch_values),
                    np.max(pitch_values)
                ])
            else:
                features.extend([0, 0, 0, 0])
        except:
            features.extend([0, 0, 0, 0])

        # 4. Energy features
        rms_energy = librosa.feature.rms(y=audio)
        features.extend([
            np.mean(rms_energy),
            np.std(rms_energy),
            np.max(rms_energy),
            np.min(rms_energy)
        ])

        # 5. Chroma features (harmonic content)
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            chroma_mean = np.mean(chroma, axis=1)
            features.extend(chroma_mean)
        except:
            features.extend([0] * 12)

        # Convert to numpy array and handle any NaN/inf values
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        return features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to fixed dimension"""
        # Pad or truncate to embedding_dim
        if len(features) < self.embedding_dim:
            features = np.pad(features, (0, self.embedding_dim - len(features)), mode='constant')
        elif len(features) > self.embedding_dim:
            features = features[:self.embedding_dim]

        # L2 normalization
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features

    def diarize_audio(self, audio_file: Union[str, Path]) -> Dict:
        """Perform speaker diarization using energy-based segmentation and clustering"""
        try:
            audio_file = Path(audio_file)
            logger.info(f"🎯 Starting diarization for: {audio_file.name}")

            # Load audio
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            total_duration = len(audio) / sr

            logger.info(f"   Audio duration: {total_duration:.2f}s")

            # Step 1: Voice Activity Detection (VAD)
            speech_segments = self._voice_activity_detection(audio)
            logger.info(f"   Found {len(speech_segments)} speech segments")

            if len(speech_segments) == 0:
                return {
                    'success': False,
                    'error': 'No speech detected',
                    'file_path': str(audio_file)
                }

            # Step 2: Extract speaker embeddings for each segment
            segment_embeddings = []
            valid_segments = []

            for start_time, end_time in speech_segments:
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment_audio = audio[start_sample:end_sample]

                # Skip very short segments
                if len(segment_audio) < sr * self.min_segment_length:
                    continue

                embedding = self.extract_speaker_embedding(segment_audio)
                segment_embeddings.append(embedding)
                valid_segments.append((start_time, end_time))

            if len(segment_embeddings) < 2:
                # Single speaker scenario
                return {
                    'success': True,
                    'segments': [
                        {
                            'start': seg[0],
                            'end': seg[1],
                            'duration': seg[1] - seg[0],
                            'speaker': 'SPEAKER_0',
                            'confidence': 0.9
                        }
                        for seg in valid_segments
                    ],
                    'total_speakers': 1,
                    'method': 'single_speaker',
                    'file_path': str(audio_file)
                }

            # Step 3: Cluster embeddings to identify speakers
            speaker_labels = self._cluster_speakers(segment_embeddings)

            # Step 4: Create final segments with speaker labels
            diarized_segments = []
            for i, (start_time, end_time) in enumerate(valid_segments):
                speaker_id = f"SPEAKER_{speaker_labels[i]}"
                confidence = self._calculate_segment_confidence(
                    segment_embeddings[i], segment_embeddings, speaker_labels, speaker_labels[i]
                )

                diarized_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time,
                    'speaker': speaker_id,
                    'confidence': confidence
                })

            # Step 5: Post-process segments (merge adjacent same-speaker segments)
            final_segments = self._merge_adjacent_segments(diarized_segments)

            num_speakers = len(set(speaker_labels))

            logger.info(f"✅ Diarization completed")
            logger.info(f"   Total speakers identified: {num_speakers}")
            logger.info(f"   Final segments: {len(final_segments)}")

            return {
                'success': True,
                'segments': final_segments,
                'total_speakers': num_speakers,
                'method': 'energy_clustering',
                'file_path': str(audio_file),
                'total_duration': total_duration
            }

        except Exception as e:
            logger.error(f"Error in diarization: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': str(audio_file)
            }

    def _voice_activity_detection(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Detect speech segments using energy-based VAD"""
        # Calculate frame-wise RMS energy
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]

        # Calculate time axis
        times = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )

        # Adaptive thresholding
        energy_threshold = np.percentile(rms, 30)  # 30th percentile
        max_energy = np.max(rms)

        # Adjust threshold based on dynamic range
        if max_energy > 0:
            dynamic_threshold = energy_threshold + 0.1 * (max_energy - energy_threshold)
        else:
            dynamic_threshold = energy_threshold

        # Find speech frames
        speech_frames = rms > dynamic_threshold

        # Convert to segments
        segments = []
        in_speech = False
        start_time = 0

        for i, (time, is_speech) in enumerate(zip(times, speech_frames)):
            if is_speech and not in_speech:
                # Start of speech
                start_time = time
                in_speech = True
            elif not is_speech and in_speech:
                # End of speech
                if time - start_time >= self.min_segment_length:
                    segments.append((start_time, time))
                in_speech = False

        # Handle case where audio ends with speech
        if in_speech and len(times) > 0:
            if times[-1] - start_time >= self.min_segment_length:
                segments.append((start_time, times[-1]))

        return segments

    def _cluster_speakers(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Cluster speaker embeddings to identify different speakers"""
        if len(embeddings) < 2:
            return np.array([0])

        # Convert to matrix
        embedding_matrix = np.array(embeddings)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embedding_matrix)

        # Determine optimal number of clusters using similarity threshold
        n_clusters = self._estimate_num_speakers(similarity_matrix)

        # Perform clustering
        if n_clusters == 1:
            return np.zeros(len(embeddings), dtype=int)

        try:
            # Use Agglomerative Clustering for better results with small datasets
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward',
                metric='euclidean'
            )
            labels = clustering.fit_predict(embedding_matrix)
        except:
            # Fallback to KMeans
            try:
                clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = clustering.fit_predict(embedding_matrix)
            except:
                # Ultimate fallback: assign all to one speaker
                labels = np.zeros(len(embeddings), dtype=int)

        return labels

    def _estimate_num_speakers(self, similarity_matrix: np.ndarray) -> int:
        """Estimate number of speakers based on similarity matrix"""
        # Get upper triangular part (excluding diagonal)
        n = similarity_matrix.shape[0]
        similarities = []

        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(similarity_matrix[i, j])

        if not similarities:
            return 1

        similarities = np.array(similarities)

        # If most similarities are above threshold, likely 1 speaker
        high_sim_ratio = np.mean(similarities > self.similarity_threshold)

        if high_sim_ratio > 0.7:
            return 1
        elif high_sim_ratio > 0.3:
            return 2
        else:
            # Estimate based on similarity distribution
            return min(3, max(2, int(n / 3)))  # Conservative estimate

    def _calculate_segment_confidence(self, embedding: np.ndarray, all_embeddings: List[np.ndarray],
                                      labels: np.ndarray, assigned_label: int) -> float:
        """Calculate confidence score for segment assignment"""
        try:
            # Find embeddings with same label
            same_speaker_embeddings = [
                emb for i, emb in enumerate(all_embeddings)
                if labels[i] == assigned_label
            ]

            if len(same_speaker_embeddings) <= 1:
                return 0.8  # Default confidence for single segment

            # Calculate average similarity to same speaker
            similarities = [
                cosine_similarity([embedding], [other_emb])[0, 0]
                for other_emb in same_speaker_embeddings
                if not np.array_equal(embedding, other_emb)
            ]

            if similarities:
                avg_similarity = np.mean(similarities)
                # Convert similarity to confidence (0.5 to 1.0 range)
                confidence = 0.5 + 0.5 * avg_similarity
                return min(1.0, max(0.0, confidence))

            return 0.8

        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.7

    def _merge_adjacent_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge adjacent segments from the same speaker"""
        if len(segments) <= 1:
            return segments

        merged = []
        current_segment = segments[0].copy()

        for next_segment in segments[1:]:
            # Check if same speaker and segments are close
            if (current_segment['speaker'] == next_segment['speaker'] and
                    next_segment['start'] - current_segment['end'] < 0.5):  # 0.5s gap tolerance

                # Merge segments
                current_segment['end'] = next_segment['end']
                current_segment['duration'] = current_segment['end'] - current_segment['start']
                current_segment['confidence'] = min(current_segment['confidence'], next_segment['confidence'])
            else:
                # Different speaker or large gap
                merged.append(current_segment)
                current_segment = next_segment.copy()

        merged.append(current_segment)
        return merged

    def identify_speaker(self, audio: np.ndarray, sr: int = None) -> Dict:
        """Identify speaker from audio segment"""
        try:
            if sr is None:
                sr = self.sample_rate

            # Extract embedding
            embedding = self.extract_speaker_embedding(audio, sr)

            if not self.speaker_embeddings:
                return {
                    'speaker_id': 'unknown',
                    'confidence': 0.0,
                    'status': 'no_enrolled_speakers'
                }

            # Compare with enrolled speakers
            best_match = None
            best_similarity = 0

            for speaker_id, enrolled_embedding in self.speaker_embeddings.items():
                similarity = cosine_similarity([embedding], [enrolled_embedding])[0, 0]

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = speaker_id

            # Determine if match is confident enough
            if best_similarity > self.similarity_threshold:
                return {
                    'speaker_id': best_match,
                    'confidence': best_similarity,
                    'status': 'identified'
                }
            else:
                return {
                    'speaker_id': f'unknown_speaker_{len(self.speaker_embeddings)}',
                    'confidence': best_similarity,
                    'status': 'new_speaker'
                }

        except Exception as e:
            logger.error(f"Error identifying speaker: {e}")
            return {
                'speaker_id': 'error',
                'confidence': 0.0,
                'status': 'error'
            }

    def enroll_speaker(self, speaker_id: str, audio: np.ndarray, sr: int = None) -> bool:
        """Enroll a new speaker"""
        try:
            if sr is None:
                sr = self.sample_rate

            embedding = self.extract_speaker_embedding(audio, sr)
            self.speaker_embeddings[speaker_id] = embedding

            logger.info(f"✅ Speaker enrolled: {speaker_id}")
            return True

        except Exception as e:
            logger.error(f"Error enrolling speaker {speaker_id}: {e}")
            return False

    def process_file_for_tts_training(self, audio_file: Union[str, Path],
                                      language_code: str = None) -> Dict:
        """Process audio file for TTS training with speaker analysis"""
        try:
            audio_file = Path(audio_file)
            logger.info(f"🎯 Processing file for TTS training: {audio_file.name}")

            # Perform diarization
            diarization_result = self.diarize_audio(audio_file)

            if not diarization_result['success']:
                return {
                    'success': False,
                    'error': 'Diarization failed',
                    'file_path': str(audio_file)
                }

            # Load audio for speaker identification
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            total_duration = len(audio) / sr

            # Process each segment
            tts_segments = []
            for segment in diarization_result['segments']:
                try:
                    # Extract segment audio
                    start_sample = int(segment['start'] * sr)
                    end_sample = int(segment['end'] * sr)
                    segment_audio = audio[start_sample:end_sample]

                    # Skip very short segments
                    if len(segment_audio) < sr * 0.5:  # Less than 0.5 seconds
                        continue

                    # Identify speaker (if enrolled speakers exist)
                    speaker_result = self.identify_speaker(segment_audio, sr)

                    # Create TTS segment
                    tts_segment = {
                        'start': segment['start'],
                        'end': segment['end'],
                        'duration': segment['duration'],
                        'speaker_id': speaker_result['speaker_id'],
                        'speaker_name': speaker_result['speaker_id'],
                        'diarization_speaker': segment['speaker'],
                        'confidence': segment['confidence'],
                        'identification_confidence': speaker_result['confidence'],
                        'status': speaker_result['status'],
                        'language': language_code,
                        'audio_file': str(audio_file),
                        'sample_rate': sr
                    }

                    tts_segments.append(tts_segment)

                except Exception as e:
                    logger.warning(f"Error processing segment: {e}")
                    continue

            logger.info(f"✅ TTS processing completed")
            logger.info(f"   Processed segments: {len(tts_segments)}")

            return {
                'success': True,
                'file_path': str(audio_file),
                'language': language_code,
                'total_duration': total_duration,
                'total_speakers': diarization_result['total_speakers'],
                'tts_segments': tts_segments,
                'diarization_method': diarization_result['method']
            }

        except Exception as e:
            logger.error(f"Error processing file for TTS: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': str(audio_file)
            }

    def save_speaker_models(self, output_dir: Union[str, Path]) -> bool:
        """Save speaker models and embeddings"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save embeddings
            embeddings_file = output_dir / "speaker_embeddings.pkl"
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.speaker_embeddings, f)

            # Save metadata
            metadata = {
                'num_speakers': len(self.speaker_embeddings),
                'embedding_dim': self.embedding_dim,
                'similarity_threshold': self.similarity_threshold,
                'sample_rate': self.sample_rate,
                'created_at': datetime.now().isoformat(),
                'speaker_ids': list(self.speaker_embeddings.keys())
            }

            metadata_file = output_dir / "speaker_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"✅ Speaker models saved to {output_dir}")
            logger.info(f"   Saved {len(self.speaker_embeddings)} speaker embeddings")
            return True

        except Exception as e:
            logger.error(f"Error saving speaker models: {e}")
            return False

    def load_speaker_models(self, input_dir: Union[str, Path]) -> bool:
        """Load speaker models and embeddings"""
        try:
            input_dir = Path(input_dir)

            # Load embeddings
            embeddings_file = input_dir / "speaker_embeddings.pkl"
            if embeddings_file.exists():
                with open(embeddings_file, 'rb') as f:
                    self.speaker_embeddings = pickle.load(f)

            # Load metadata
            metadata_file = input_dir / "speaker_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.similarity_threshold = metadata.get('similarity_threshold', self.similarity_threshold)
                    self.sample_rate = metadata.get('sample_rate', self.sample_rate)

            logger.info(f"✅ Speaker models loaded from {input_dir}")
            logger.info(f"   Loaded {len(self.speaker_embeddings)} speakers")
            return True

        except Exception as e:
            logger.error(f"Error loading speaker models: {e}")
            return False


# Alias for backward compatibility
AdvancedVoiceSystem = SpeakerIdentificationSystem


def test_diarization_system():
    """Test the custom diarization system"""
    print("🧪 Testing Custom Speaker Diarization System")
    print("=" * 50)

    # Initialize system
    system = SpeakerIdentificationSystem()
    print(f"✅ System initialized")
    print(f"   Sample rate: {system.sample_rate}Hz")
    print(f"   Embedding dimension: {system.embedding_dim}")

    # Test with synthetic audio
    print("\n🎵 Testing with synthetic audio...")

    # Create synthetic multi-speaker audio
    duration = 10  # seconds
    sr = system.sample_rate
    t = np.linspace(0, duration, duration * sr)

    # Speaker 1: Lower frequency (male voice simulation)
    speaker1_audio = 0.3 * np.sin(2 * np.pi * 150 * t) * np.random.randn(len(t)) * 0.1

    # Speaker 2: Higher frequency (female voice simulation)
    speaker2_audio = 0.3 * np.sin(2 * np.pi * 250 * t) * np.random.randn(len(t)) * 0.1

    # Combine: first 5 seconds speaker 1, next 5 seconds speaker 2
    combined_audio = np.concatenate([
        speaker1_audio[:sr * 5],
        np.zeros(sr // 2),  # 0.5s silence
        speaker2_audio[sr * 5:]
    ])

    # Save temporary file
    temp_file = Path("temp_test_audio.wav")
    sf.write(temp_file, combined_audio, sr)

    try:
        # Test diarization
        print("🎯 Testing diarization...")
        result = system.diarize_audio(temp_file)

        if result['success']:
            print(f"✅ Diarization successful!")
            print(f"   Total speakers: {result['total_speakers']}")
            print(f"   Segments found: {len(result['segments'])}")
            print(f"   Method: {result['method']}")

            for i, segment in enumerate(result['segments']):
                print(f"   Segment {i + 1}: {segment['start']:.2f}s - {segment['end']:.2f}s, "
                      f"Speaker: {segment['speaker']}, Confidence: {segment['confidence']:.3f}")
        else:
            print(f"❌ Diarization failed: {result.get('error')}")

        # Test speaker enrollment
        print("\n👤 Testing speaker enrollment...")
        speaker1_segment = combined_audio[:sr * 3]  # First 3 seconds
        success = system.enroll_speaker("test_speaker_1", speaker1_segment)
        print(f"✅ Speaker enrollment: {success}")

        # Test speaker identification
        print("\n🔍 Testing speaker identification...")
        test_segment = combined_audio[sr * 1:sr * 2]  # 1-2 seconds (should be speaker 1)
        identification = system.identify_speaker(test_segment)
        print(f"✅ Speaker identification result:")
        print(f"   Speaker ID: {identification['speaker_id']}")
        print(f"   Confidence: {identification['confidence']:.3f}")
        print(f"   Status: {identification['status']}")

        # Test TTS processing
        print("\n🎤 Testing TTS processing...")
        tts_result = system.process_file_for_tts_training(temp_file, 'hi')

        if tts_result['success']:
            print(f"✅ TTS processing successful!")
            print(f"   Total duration: {tts_result['total_duration']:.2f}s")
            print(f"   TTS segments: {len(tts_result['tts_segments'])}")

            for i, segment in enumerate(tts_result['tts_segments'][:3]):  # Show first 3
                print(f"   TTS Segment {i + 1}: {segment['start']:.2f}s - {segment['end']:.2f}s, "
                      f"Speaker: {segment['speaker_id']}")
        else:
            print(f"❌ TTS processing failed: {tts_result.get('error')}")

        print("\n🎉 All tests completed successfully!")

    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()


if __name__ == "__main__":
    test_diarization_system()