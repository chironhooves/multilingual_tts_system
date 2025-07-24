"""
TTS Model Training Module
Handles training of individual and multilingual TTS models
"""

import os
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import yaml
import torch

from config.languages import IndianLanguages
from config.settings import SystemSettings, MODEL_ARCHITECTURES
from core.speaker_id import AdvancedVoiceSystem

logger = logging.getLogger(__name__)


class TTSTrainer:
    """Handles TTS model training using Coqui TTS"""

    def __init__(self, speaker_system: AdvancedVoiceSystem = None):
        self.settings = SystemSettings()
        self.languages = IndianLanguages()
        self.speaker_system = speaker_system

        # Training parameters
        self.device = self.settings.DEVICE
        self.batch_size = self.settings.BATCH_SIZE
        self.learning_rate = self.settings.LEARNING_RATE

        # Model directories
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

        # Check TTS installation
        self.check_tts_installation()

    def check_tts_installation(self) -> bool:
        """Check if Coqui TTS is properly installed"""
        try:
            import TTS
            logger.info(f"✅ Coqui TTS {TTS.__version__} is available")
            return True
        except ImportError:
            logger.error("❌ Coqui TTS not found. Install with: pip install TTS")
            return False

    def create_training_config(self, language_code: str, manifest_file: Path,
                               model_type: str = "tacotron2") -> Path:
        """Create training configuration for a language"""
        logger.info(f"⚙️ Creating training config for {language_code}")

        # Get language info
        lang_info = self.languages.get_language_info(language_code)

        # Create config directory
        config_dir = self.models_dir / language_code / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        # Get model architecture
        model_config = MODEL_ARCHITECTURES.get(model_type, MODEL_ARCHITECTURES["fast_inference"])

        # Base configuration
        config = {
            "model": model_config["model_type"],
            "run_name": f"{language_code}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "run_description": f"TTS training for {lang_info['name']} ({lang_info['native_name']})",

            # Dataset configuration
            "datasets": [{
                "name": f"{language_code}_dataset",
                "path": str(manifest_file.parent),
                "meta_file_train": str(manifest_file),
                "language": language_code
            }],

            # Audio configuration
            "audio": {
                "sample_rate": self.settings.SAMPLE_RATE,
                "resample": True,
                "do_trim_silence": True,
                "trim_db": 60,
                "signal_norm": True,
                "symmetric_norm": True,
                "max_norm": 4.0,
                "clip_norm": True,
                "mel_fmin": 0.0,
                "mel_fmax": 8000.0,
                "spec_gain": 1.0,
                "do_amp_to_db_linear": True,
                "do_amp_to_db_mel": True,
                "do_sound_norm": False
            },

            # Training configuration
            "batch_size": self.settings.get_training_config(language_code)["batch_size"],
            "eval_batch_size": 4,
            "num_loader_workers": 4,
            "num_eval_loader_workers": 2,
            "run_eval": True,
            "test_delay_epochs": -1,
            "epochs": self.settings.MAX_EPOCHS,
            "text_cleaner": "phoneme_cleaners",
            "enable_eos_bos_chars": False,
            "test_sentences_file": "",
            "phoneme_cache_path": str(config_dir / "phoneme_cache"),
            "precompute_num_workers": 4,

            # Optimizer
            "lr": self.settings.get_training_config(language_code)["learning_rate"],
            "optimizer": "RAdam",
            "optimizer_params": {"betas": [0.9, 0.998], "weight_decay": 1e-6},
            "lr_scheduler": "NoamLR",
            "lr_scheduler_params": {"warmup_steps": 4000},
            "grad_clip": 1.0,

            # Checkpoint and logging
            "output_path": str(self.models_dir / language_code / "checkpoints"),
            "save_step": 1000,
            "checkpoint": True,
            "keep_all_best": False,
            "keep_after": 10000,
            "num_checkpoints": 5,
            "log_model_step": 1000,
            "wandb_project": f"multilingual_tts_{language_code}",

            # Validation
            "run_eval_steps": 500,
            "print_step": 50,
            "print_eval": True,
            "mixed_precision": True if self.device.type == "cuda" else False,

            # Language specific settings
            "language": language_code,
            "phoneme_language": language_code,
            "use_phonemes": True,
            "phoneme_level": "phoneme"
        }

        # Model-specific configurations
        if model_config["model_type"] == "tacotron2":
            config.update({
                "r": 1,  # Reduction factor
                "memory_size": 5,
                "prenet_type": "original",
                "prenet_dropout": True,
                "forward_attn": True,
                "transition_agent": False,
                "forward_attn_mask": True,
                "location_attn": True,
                "attention_norm": "sigmoid",
                "prenet_dropout_at_inference": False,
                "double_decoder_consistency": True,
                "ddc_r": 6
            })
        elif model_config["model_type"] == "FastSpeech2":
            config.update({
                "pitch_embedding": True,
                "energy_embedding": True,
                "duration_predictor_hidden_size": 256,
                "duration_predictor_kernel_size": 3,
                "duration_predictor_dropout": 0.5
            })

        # Add speaker embedding if multiple speakers detected
        if self.speaker_system:
            speaker_count = self.get_speaker_count_for_language(language_code)
            if speaker_count > 1:
                config.update({
                    "use_speaker_embedding": True,
                    "num_speakers": speaker_count,
                    "speakers_file": str(self.create_speakers_file(language_code)),
                    "speaker_embedding_dim": 512
                })

        # Save configuration
        config_file = config_dir / f"config_{model_type}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Training config created: {config_file}")
        return config_file

    def get_speaker_count_for_language(self, language_code: str) -> int:
        """Get number of speakers for a language from speaker database"""
        if not self.speaker_system:
            return 1

        speakers = [
            name for name in self.speaker_system.voice_database.keys()
            if name.endswith(f"_{language_code}")
        ]
        return max(len(speakers), 1)

    def create_speakers_file(self, language_code: str) -> Path:
        """Create speakers.json file for multi-speaker training"""
        speakers_file = self.models_dir / language_code / "speakers.json"

        if not self.speaker_system:
            # Create dummy speaker file
            speakers_data = {"speakers": ["default_speaker"]}
        else:
            # Get speakers from database
            speakers = [
                name.replace(f"_{language_code}", "") for name in self.speaker_system.voice_database.keys()
                if name.endswith(f"_{language_code}")
            ]
            if not speakers:
                speakers = ["default_speaker"]
            speakers_data = {"speakers": speakers}

        with open(speakers_file, 'w', encoding='utf-8') as f:
            json.dump(speakers_data, f, indent=2)

        return speakers_file

    def prepare_training_data(self, language_code: str) -> Optional[Path]:
        """Prepare and validate training data"""
        logger.info(f"📋 Preparing training data for {language_code}")

        base_dir = Path("data") / language_code
        manifests_dir = base_dir / "manifests"

        # Find latest manifest file
        manifest_files = list(manifests_dir.glob("*.jsonl"))
        if not manifest_files:
            logger.error(f"No manifest files found for {language_code}")
            return None

        # Use the latest manifest
        latest_manifest = max(manifest_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using manifest: {latest_manifest}")

        # Validate manifest
        validation_result = self.validate_training_manifest(latest_manifest)
        if not validation_result["valid"]:
            logger.error(f"Manifest validation failed: {validation_result['errors']}")
            return None

        # Convert to TTS format if needed
        tts_manifest = self.convert_to_tts_format(latest_manifest, language_code)

        return tts_manifest

    def validate_training_manifest(self, manifest_file: Path) -> Dict:
        """Validate training manifest"""
        logger.info(f"🔍 Validating manifest: {manifest_file}")

        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "total_samples": 0,
                "valid_samples": 0,
                "total_duration": 0,
                "avg_duration": 0,
                "min_duration": float('inf'),
                "max_duration": 0
            }
        }

        try:
            with open(manifest_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        sample = json.loads(line.strip())
                        validation_result["stats"]["total_samples"] += 1

                        # Check required fields
                        required_fields = ["audio_filepath", "text", "duration"]
                        missing_fields = [field for field in required_fields if field not in sample]

                        if missing_fields:
                            validation_result["errors"].append(
                                f"Line {line_num}: Missing fields {missing_fields}"
                            )
                            continue

                        # Check if audio file exists
                        if not Path(sample["audio_filepath"]).exists():
                            validation_result["errors"].append(
                                f"Line {line_num}: Audio file not found: {sample['audio_filepath']}"
                            )
                            continue

                        # Check duration
                        duration = sample["duration"]
                        if duration < self.settings.MIN_SEGMENT_DURATION:
                            validation_result["warnings"].append(
                                f"Line {line_num}: Duration too short: {duration}s"
                            )
                        elif duration > self.settings.MAX_SEGMENT_DURATION:
                            validation_result["warnings"].append(
                                f"Line {line_num}: Duration too long: {duration}s"
                            )

                        # Check text
                        if len(sample["text"].strip()) < 3:
                            validation_result["warnings"].append(
                                f"Line {line_num}: Text too short"
                            )

                        # Update stats
                        validation_result["stats"]["valid_samples"] += 1
                        validation_result["stats"]["total_duration"] += duration
                        validation_result["stats"]["min_duration"] = min(
                            validation_result["stats"]["min_duration"], duration
                        )
                        validation_result["stats"]["max_duration"] = max(
                            validation_result["stats"]["max_duration"], duration
                        )

                    except json.JSONDecodeError:
                        validation_result["errors"].append(f"Line {line_num}: Invalid JSON")

            # Calculate average duration
            if validation_result["stats"]["valid_samples"] > 0:
                validation_result["stats"]["avg_duration"] = (
                        validation_result["stats"]["total_duration"] /
                        validation_result["stats"]["valid_samples"]
                )

            # Set validation status
            if validation_result["errors"]:
                validation_result["valid"] = False

            # Check minimum requirements
            if validation_result["stats"]["valid_samples"] < 100:
                validation_result["warnings"].append("Very few samples - training may not be effective")

            if validation_result["stats"]["total_duration"] < 3600:  # Less than 1 hour
                validation_result["warnings"].append("Insufficient training data - consider collecting more")

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Error reading manifest: {e}")

        logger.info(f"Validation result: {validation_result['stats']['valid_samples']} valid samples")
        return validation_result

    def convert_to_tts_format(self, manifest_file: Path, language_code: str) -> Path:
        """Convert manifest to TTS training format"""
        logger.info(f"🔄 Converting manifest to TTS format")

        # Create TTS manifest file
        tts_manifest_file = manifest_file.parent / f"tts_{manifest_file.name}"

        with open(manifest_file, 'r', encoding='utf-8') as infile, \
                open(tts_manifest_file, 'w', encoding='utf-8') as outfile:

            for line in infile:
                try:
                    sample = json.loads(line.strip())

                    # Convert to TTS format
                    tts_sample = {
                        "audio_file": sample["audio_filepath"],
                        "text": sample["text"],
                        "speaker_name": sample.get("speaker_name", "default_speaker"),
                        "language": sample.get("language", language_code)
                    }

                    outfile.write(json.dumps(tts_sample, ensure_ascii=False) + '\n')

                except json.JSONDecodeError:
                    continue

        return tts_manifest_file

    def train_language_model(self, language_code: str, model_type: str = "tacotron2",
                             resume_checkpoint: str = None) -> Dict:
        """Train TTS model for a specific language"""
        logger.info(f"🚀 Starting TTS training for {language_code}")

        # Prepare training data
        manifest_file = self.prepare_training_data(language_code)
        if manifest_file is None:
            return {"success": False, "error": "Failed to prepare training data"}

        # Create training configuration
        config_file = self.create_training_config(language_code, manifest_file, model_type)

        # Setup output directory
        output_dir = self.models_dir / language_code / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create training command
        cmd = [
            "python", "-m", "TTS.bin.train_tts",
            "--config_path", str(config_file),
            "--restore_path", resume_checkpoint if resume_checkpoint else ""
        ]

        # Remove empty restore_path if no checkpoint
        if not resume_checkpoint:
            cmd = cmd[:-2]

        training_result = {
            "success": False,
            "language_code": language_code,
            "model_type": model_type,
            "config_file": str(config_file),
            "output_dir": str(output_dir),
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "training_log": None,
            "best_checkpoint": None,
            "final_loss": None
        }

        try:
            logger.info(f"Running training command: {' '.join(cmd)}")

            # Create log file
            log_file = output_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            training_result["training_log"] = str(log_file)

            # Start training process
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    cwd=str(Path.cwd())
                )

                # Monitor training progress
                final_loss = None
                for line in process.stdout:
                    log.write(line)
                    log.flush()

                    # Extract loss information
                    if "loss" in line.lower():
                        try:
                            # Simple loss extraction - could be improved
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if "loss" in part.lower() and i + 1 < len(parts):
                                    try:
                                        loss_val = float(parts[i + 1].replace(',', ''))
                                        final_loss = loss_val
                                    except ValueError:
                                        pass
                        except:
                            pass

                    # Print progress
                    if any(keyword in line.lower() for keyword in ["epoch", "step", "loss"]):
                        logger.info(f"Training: {line.strip()}")

                # Wait for completion
                return_code = process.wait()

                if return_code == 0:
                    training_result["success"] = True
                    training_result["final_loss"] = final_loss

                    # Find best checkpoint
                    checkpoint_files = list(output_dir.glob("best_model.pth"))
                    if checkpoint_files:
                        training_result["best_checkpoint"] = str(checkpoint_files[0])

                    logger.info(f"✅ Training completed successfully for {language_code}")
                else:
                    training_result["error"] = f"Training process failed with return code {return_code}"
                    logger.error(f"❌ Training failed for {language_code}")

        except Exception as e:
            training_result["error"] = str(e)
            logger.error(f"Training error for {language_code}: {e}")

        finally:
            training_result["end_time"] = datetime.now().isoformat()

        # Save training result
        result_file = output_dir / f"training_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(training_result, f, indent=2)

        return training_result

    def train_all_languages(self, model_type: str = "tacotron2") -> Dict:
        """Train models for all available languages"""
        logger.info("🌍 Starting training for all languages")

        supported_languages = self.languages.get_supported_languages()

        all_results = {
            "total_languages": len(supported_languages),
            "successful_trainings": 0,
            "failed_trainings": 0,
            "start_time": datetime.now().isoformat(),
            "results_by_language": {}
        }

        for lang_code in supported_languages:
            logger.info(f"🔄 Training {lang_code}...")

            try:
                result = self.train_language_model(lang_code, model_type)
                all_results["results_by_language"][lang_code] = result

                if result["success"]:
                    all_results["successful_trainings"] += 1
                else:
                    all_results["failed_trainings"] += 1

            except Exception as e:
                logger.error(f"Error training {lang_code}: {e}")
                all_results["failed_trainings"] += 1
                all_results["results_by_language"][lang_code] = {
                    "success": False,
                    "error": str(e)
                }

        all_results["end_time"] = datetime.now().isoformat()

        # Save overall results
        results_file = self.models_dir / f"all_languages_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info("✅ All languages training completed")
        logger.info(f"   Successful: {all_results['successful_trainings']}/{all_results['total_languages']}")

        return all_results

    def create_unified_multilingual_model(self, base_model_type: str = "tacotron2") -> Dict:
        """Create a unified multilingual model"""
        logger.info("🌐 Creating unified multilingual model")

        # Collect all training data
        all_manifests = []
        supported_languages = self.languages.get_supported_languages()

        for lang_code in supported_languages:
            manifests_dir = Path("data") / lang_code / "manifests"
            if manifests_dir.exists():
                manifest_files = list(manifests_dir.glob("tts_*.jsonl"))
                if manifest_files:
                    latest_manifest = max(manifest_files, key=lambda x: x.stat().st_mtime)
                    all_manifests.append((lang_code, latest_manifest))

        if not all_manifests:
            return {"success": False, "error": "No training data found for any language"}

        # Create unified manifest
        unified_dir = self.models_dir / "unified"
        unified_dir.mkdir(exist_ok=True)

        unified_manifest = unified_dir / f"unified_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        total_samples = 0
        with open(unified_manifest, 'w', encoding='utf-8') as outfile:
            for lang_code, manifest_file in all_manifests:
                with open(manifest_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        try:
                            sample = json.loads(line.strip())
                            sample["language"] = lang_code  # Ensure language is set
                            outfile.write(json.dumps(sample, ensure_ascii=False) + '\n')
                            total_samples += 1
                        except json.JSONDecodeError:
                            continue

        logger.info(f"Created unified manifest with {total_samples} samples")

        # Create unified training config
        config = {
            "model": "YourTTS",  # Use multilingual model
            "run_name": f"unified_multilingual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "run_description": "Unified multilingual Indian TTS model",

            # Dataset configuration
            "datasets": [{
                "name": "unified_multilingual_dataset",
                "path": str(unified_dir),
                "meta_file_train": str(unified_manifest),
                "language": "multilingual"
            }],

            # Multilingual settings
            "use_language_embedding": True,
            "language_ids_file": str(self.create_language_ids_file()),
            "use_speaker_embedding": True,
            "speakers_file": str(self.create_unified_speakers_file()),

            # Audio configuration
            "audio": {
                "sample_rate": self.settings.SAMPLE_RATE,
                "resample": True,
                "do_trim_silence": True,
                "trim_db": 60,
                "signal_norm": True,
                "symmetric_norm": True,
                "max_norm": 4.0,
                "clip_norm": True,
                "mel_fmin": 0.0,
                "mel_fmax": 8000.0
            },

            # Training configuration
            "batch_size": max(self.batch_size // 2, 4),  # Reduce batch size for multilingual
            "eval_batch_size": 2,
            "num_loader_workers": 4,
            "epochs": self.settings.MAX_EPOCHS,
            "lr": self.learning_rate * 0.8,  # Slightly lower learning rate
            "optimizer": "RAdam",
            "lr_scheduler": "NoamLR",
            "mixed_precision": True if self.device.type == "cuda" else False,

            # Output
            "output_path": str(unified_dir / "checkpoints"),
            "save_step": 2000,
            "log_model_step": 2000,
            "run_eval_steps": 1000
        }

        # Save unified config
        unified_config_file = unified_dir / "unified_config.json"
        with open(unified_config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Train unified model
        training_result = {
            "success": False,
            "model_type": "unified_multilingual",
            "languages": [lang for lang, _ in all_manifests],
            "total_samples": total_samples,
            "config_file": str(unified_config_file),
            "manifest_file": str(unified_manifest),
            "start_time": datetime.now().isoformat()
        }

        try:
            cmd = [
                "python", "-m", "TTS.bin.train_tts",
                "--config_path", str(unified_config_file)
            ]

            logger.info("🚀 Starting unified model training...")

            # Run training (similar to single language training)
            log_file = unified_dir / f"unified_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )

                for line in process.stdout:
                    log.write(line)
                    log.flush()

                    if any(keyword in line.lower() for keyword in ["epoch", "step", "loss"]):
                        logger.info(f"Unified Training: {line.strip()}")

                return_code = process.wait()

                if return_code == 0:
                    training_result["success"] = True
                    logger.info("✅ Unified model training completed")
                else:
                    training_result["error"] = f"Training failed with return code {return_code}"

        except Exception as e:
            training_result["error"] = str(e)
            logger.error(f"Unified training error: {e}")

        training_result["end_time"] = datetime.now().isoformat()

        # Save result
        result_file = unified_dir / f"unified_training_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(training_result, f, indent=2)

        return training_result

    def create_language_ids_file(self) -> Path:
        """Create language IDs file for multilingual training"""
        language_ids_file = self.models_dir / "unified" / "language_ids.json"

        supported_languages = self.languages.get_supported_languages()
        language_ids = {lang: i for i, lang in enumerate(supported_languages)}

        with open(language_ids_file, 'w') as f:
            json.dump(language_ids, f, indent=2)

        return language_ids_file

    def create_unified_speakers_file(self) -> Path:
        """Create unified speakers file from all languages"""
        speakers_file = self.models_dir / "unified" / "speakers.json"

        all_speakers = set()

        if self.speaker_system:
            # Get all speakers from database
            for speaker_key in self.speaker_system.voice_database.keys():
                # Extract speaker name without language suffix
                if '_' in speaker_key:
                    speaker_name = '_'.join(speaker_key.split('_')[:-1])
                else:
                    speaker_name = speaker_key
                all_speakers.add(speaker_name)

        if not all_speakers:
            all_speakers = {"default_speaker"}

        speakers_data = {"speakers": list(all_speakers)}

        with open(speakers_file, 'w') as f:
            json.dump(speakers_data, f, indent=2)

        return speakers_file

    def evaluate_model(self, language_code: str, checkpoint_path: str) -> Dict:
        """Evaluate a trained model"""
        logger.info(f"📊 Evaluating model for {language_code}")

        # This would involve:
        # 1. Loading the model
        # 2. Running inference on test set
        # 3. Computing metrics (MOS, intelligibility, etc.)
        # 4. Generating sample audio

        evaluation_result = {
            "language_code": language_code,
            "checkpoint_path": checkpoint_path,
            "evaluation_time": datetime.now().isoformat(),
            "metrics": {
                "inference_speed": 0.0,
                "model_size_mb": 0.0,
                "sample_rate": self.settings.SAMPLE_RATE
            },
            "sample_outputs": []
        }

        try:
            # Get model size
            if Path(checkpoint_path).exists():
                model_size = Path(checkpoint_path).stat().st_size / (1024 * 1024)  # MB
                evaluation_result["metrics"]["model_size_mb"] = model_size

            # TODO: Add actual model evaluation
            # This would require loading the model and running inference

            logger.info(f"✅ Model evaluation completed for {language_code}")

        except Exception as e:
            evaluation_result["error"] = str(e)
            logger.error(f"Evaluation error: {e}")

        return evaluation_result

    def get_training_progress(self, language_code: str) -> Dict:
        """Get training progress for a language"""
        output_dir = self.models_dir / language_code / "checkpoints"

        progress = {
            "language_code": language_code,
            "status": "not_started",
            "checkpoints": [],
            "latest_checkpoint": None,
            "training_log": None
        }

        if output_dir.exists():
            # Find checkpoints
            checkpoint_files = list(output_dir.glob("*.pth"))
            progress["checkpoints"] = [str(f) for f in checkpoint_files]

            if checkpoint_files:
                progress["latest_checkpoint"] = str(max(checkpoint_files, key=lambda x: x.stat().st_mtime))
                progress["status"] = "completed" if any(
                    "best_model" in f.name for f in checkpoint_files) else "in_progress"

            # Find training log
            log_files = list(output_dir.glob("training_log_*.txt"))
            if log_files:
                progress["training_log"] = str(max(log_files, key=lambda x: x.stat().st_mtime))

        return progress


def main():
    """Test the trainer module"""
    from core.speaker_id import AdvancedVoiceSystem

    speaker_system = AdvancedVoiceSystem()
    trainer = TTSTrainer(speaker_system)

    # Test training preparation
    manifest = trainer.prepare_training_data('hi')
    if manifest:
        print(f"Training data prepared: {manifest}")

        # Test config creation
        config = trainer.create_training_config('hi', manifest)
        print(f"Config created: {config}")


if __name__ == "__main__":
    main()