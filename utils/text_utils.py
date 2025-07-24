"""
Text Processing Utilities for Multilingual TTS System v2.0
Common text operations, normalization, and language detection
"""

import re
import unicodedata
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TextUtils:
    """Comprehensive text processing utilities"""

    def __init__(self):
        self.supported_scripts = {
            'hi': 'Devanagari',
            'mr': 'Devanagari',
            'ta': 'Tamil',
            'te': 'Telugu',
            'bn': 'Bengali',
            'gu': 'Gujarati',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'pa': 'Gurmukhi',
            'or': 'Odia'
        }

        # Unicode ranges for Indian scripts
        self.script_ranges = {
            'Devanagari': (0x0900, 0x097F),
            'Bengali': (0x0980, 0x09FF),
            'Gurmukhi': (0x0A00, 0x0A7F),
            'Gujarati': (0x0A80, 0x0AFF),
            'Odia': (0x0B00, 0x0B7F),
            'Tamil': (0x0B80, 0x0BFF),
            'Telugu': (0x0C00, 0x0C7F),
            'Kannada': (0x0C80, 0x0CFF),
            'Malayalam': (0x0D00, 0x0D7F)
        }

        # Common punctuation mappings
        self.punctuation_map = {
            '।': '.',  # Devanagari danda
            '॥': '.',  # Double danda
            '‌': '',  # Zero width non-joiner
            '‍': '',  # Zero width joiner
            ''': "'",  # Left single quotation
            ''': "'",  # Right single quotation
            '"': '"',  # Left double quotation
            '"': '"',  # Right double quotation
            '–': '-',  # En dash
            '—': '-',  # Em dash
            '…': '...'  # Ellipsis
        }

    def normalize_unicode(self, text: str,
                          form: str = 'NFC') -> str:
        """
        Normalize Unicode text

        Args:
            text: Input text
            form: Normalization form ('NFC', 'NFD', 'NFKC', 'NFKD')

        Returns:
            Normalized text
        """
        try:
            return unicodedata.normalize(form, text)
        except Exception as e:
            logger.error(f"Error normalizing Unicode: {e}")
            return text

    def clean_text(self, text: str,
                   language_code: Optional[str] = None,
                   remove_punctuation: bool = False,
                   normalize_whitespace: bool = True,
                   remove_numbers: bool = False) -> str:
        """
        Clean and normalize text

        Args:
            text: Input text
            language_code: Target language code
            remove_punctuation: Remove punctuation marks
            normalize_whitespace: Normalize whitespace
            remove_numbers: Remove numeric characters

        Returns:
            Cleaned text
        """
        try:
            if not text or not isinstance(text, str):
                return ""

            # Unicode normalization
            text = self.normalize_unicode(text)

            # Remove control characters
            text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')

            # Normalize punctuation
            for old_punct, new_punct in self.punctuation_map.items():
                text = text.replace(old_punct, new_punct)

            # Language-specific cleaning
            if language_code:
                text = self._clean_language_specific(text, language_code)

            # Remove numbers if requested
            if remove_numbers:
                text = re.sub(r'\d+', '', text)

            # Remove punctuation if requested
            if remove_punctuation:
                text = re.sub(r'[^\w\s]', '', text)

            # Normalize whitespace
            if normalize_whitespace:
                text = re.sub(r'\s+', ' ', text).strip()

            return text

        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text

    def _clean_language_specific(self, text: str, language_code: str) -> str:
        """Apply language-specific cleaning rules"""
        try:
            if language_code in ['hi', 'mr']:  # Devanagari
                # Remove excessive English mixed with Devanagari
                text = re.sub(r'\b[a-zA-Z]{15,}\b', '', text)  # Very long English words

            elif language_code == 'ta':  # Tamil
                # Tamil-specific cleaning
                text = re.sub(r'[a-zA-Z]{10,}', '', text)

            elif language_code == 'te':  # Telugu
                # Telugu-specific cleaning
                text = re.sub(r'[a-zA-Z]{10,}', '', text)

            elif language_code == 'bn':  # Bengali
                # Bengali-specific cleaning
                text = re.sub(r'[a-zA-Z]{10,}', '', text)

            # Remove excessive repetition
            text = re.sub(r'(.)\1{3,}', r'\1\1', text)  # Max 2 repeated characters

            return text

        except Exception as e:
            logger.error(f"Error in language-specific cleaning: {e}")
            return text

    def detect_language_script(self, text: str) -> Dict[str, float]:
        """
        Detect the script(s) used in text

        Args:
            text: Input text

        Returns:
            Dictionary of script: percentage mappings
        """
        try:
            if not text:
                return {}

            # Count characters by script
            script_counts = {}
            total_chars = 0

            for char in text:
                if char.isspace() or not char.isalnum():
                    continue

                total_chars += 1
                code_point = ord(char)

                # Check which script this character belongs to
                detected_script = None
                for script, (start, end) in self.script_ranges.items():
                    if start <= code_point <= end:
                        detected_script = script
                        break

                if detected_script:
                    script_counts[detected_script] = script_counts.get(detected_script, 0) + 1
                elif char.isascii():
                    script_counts['Latin'] = script_counts.get('Latin', 0) + 1
                else:
                    script_counts['Other'] = script_counts.get('Other', 0) + 1

            # Calculate percentages
            if total_chars == 0:
                return {}

            script_percentages = {
                script: count / total_chars
                for script, count in script_counts.items()
            }

            return script_percentages

        except Exception as e:
            logger.error(f"Error detecting language script: {e}")
            return {}

    def detect_language_from_script(self, text: str) -> Optional[str]:
        """
        Detect most likely language based on script analysis

        Args:
            text: Input text

        Returns:
            Most likely language code or None
        """
        try:
            script_percentages = self.detect_language_script(text)

            if not script_percentages:
                return None

            # Find dominant script
            dominant_script = max(script_percentages.items(), key=lambda x: x[1])[0]

            # Map script to language (simplified - assumes most common language for script)
            script_to_language = {
                'Devanagari': 'hi',  # Could be Hindi or Marathi - default to Hindi
                'Tamil': 'ta',
                'Telugu': 'te',
                'Bengali': 'bn',
                'Gujarati': 'gu',
                'Kannada': 'kn',
                'Malayalam': 'ml',
                'Gurmukhi': 'pa',
                'Odia': 'or'
            }

            return script_to_language.get(dominant_script)

        except Exception as e:
            logger.error(f"Error detecting language from script: {e}")
            return None

    def is_mixed_script(self, text: str, threshold: float = 0.1) -> bool:
        """
        Check if text contains mixed scripts

        Args:
            text: Input text
            threshold: Minimum percentage to consider a script present

        Returns:
            True if mixed scripts detected
        """
        try:
            script_percentages = self.detect_language_script(text)

            # Count scripts above threshold
            significant_scripts = sum(1 for percentage in script_percentages.values()
                                      if percentage >= threshold)

            return significant_scripts > 1

        except Exception as e:
            logger.error(f"Error checking mixed script: {e}")
            return False

    def extract_sentences(self, text: str,
                          language_code: Optional[str] = None) -> List[str]:
        """
        Extract sentences from text

        Args:
            text: Input text
            language_code: Language code for language-specific rules

        Returns:
            List of sentences
        """
        try:
            if not text:
                return []

            # Define sentence terminators based on language
            if language_code in ['hi', 'mr', 'ta', 'te', 'bn', 'gu', 'kn', 'ml', 'pa', 'or']:
                # Indian languages - include Devanagari danda
                terminators = r'[.!?।॥]'
            else:
                # Default to common punctuation
                terminators = r'[.!?]'

            # Split by sentence terminators
            sentences = re.split(terminators, text)

            # Clean and filter sentences
            cleaned_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 3:  # Minimum length
                    cleaned_sentences.append(sentence)

            return cleaned_sentences

        except Exception as e:
            logger.error(f"Error extracting sentences: {e}")
            return [text] if text else []

    def count_words(self, text: str,
                    language_code: Optional[str] = None) -> int:
        """
        Count words in text (language-aware)

        Args:
            text: Input text
            language_code: Language code for language-specific counting

        Returns:
            Word count
        """
        try:
            if not text:
                return 0

            # For Indian languages, word boundaries can be complex
            # Simplified approach: split by whitespace and punctuation
            words = re.findall(r'\S+', text)

            # Filter out pure punctuation
            word_count = 0
            for word in words:
                # Check if word contains at least one alphanumeric character
                if re.search(r'\w', word):
                    word_count += 1

            return word_count

        except Exception as e:
            logger.error(f"Error counting words: {e}")
            return 0

    def count_characters(self, text: str,
                         include_spaces: bool = False,
                         include_punctuation: bool = False) -> int:
        """
        Count characters in text with options

        Args:
            text: Input text
            include_spaces: Include space characters
            include_punctuation: Include punctuation marks

        Returns:
            Character count
        """
        try:
            if not text:
                return 0

            if include_spaces and include_punctuation:
                return len(text)

            count = 0
            for char in text:
                if char.isalnum():
                    count += 1
                elif include_spaces and char.isspace():
                    count += 1
                elif include_punctuation and not char.isalnum() and not char.isspace():
                    count += 1

            return count

        except Exception as e:
            logger.error(f"Error counting characters: {e}")
            return 0

    def validate_text_for_tts(self, text: str,
                              language_code: Optional[str] = None,
                              min_length: int = 3,
                              max_length: int = 500) -> Dict:
        """
        Validate text for TTS training suitability

        Args:
            text: Input text
            language_code: Target language code
            min_length: Minimum character length
            max_length: Maximum character length

        Returns:
            Validation results dictionary
        """
        try:
            validation = {
                'is_valid': True,
                'issues': [],
                'warnings': [],
                'recommendations': [],
                'metrics': {}
            }

            # Basic checks
            if not text or not isinstance(text, str):
                validation['is_valid'] = False
                validation['issues'].append("Text is empty or invalid")
                return validation

            # Length checks
            char_count = len(text.strip())
            validation['metrics']['character_count'] = char_count

            if char_count < min_length:
                validation['is_valid'] = False
                validation['issues'].append(f"Text too short: {char_count} < {min_length} characters")
            elif char_count > max_length:
                validation['is_valid'] = False
                validation['issues'].append(f"Text too long: {char_count} > {max_length} characters")

            # Word count
            word_count = self.count_words(text, language_code)
            validation['metrics']['word_count'] = word_count

            if word_count < 2:
                validation['warnings'].append("Very few words - may not be suitable for TTS")
            elif word_count > 50:
                validation['warnings'].append("Many words - consider splitting into shorter segments")

            # Script validation
            if language_code:
                expected_script = self.supported_scripts.get(language_code)
                if expected_script:
                    script_percentages = self.detect_language_script(text)

                    if expected_script not in script_percentages:
                        validation['warnings'].append(f"Expected {expected_script} script not found")
                    elif script_percentages[expected_script] < 0.7:
                        validation['warnings'].append(
                            f"Low {expected_script} script percentage: {script_percentages[expected_script]:.1%}")

                    # Check for mixed scripts
                    if self.is_mixed_script(text):
                        validation['warnings'].append("Mixed scripts detected - may affect TTS quality")

            # Content quality checks
            # Check for excessive repetition
            if re.search(r'(.{3,})\1{2,}', text):
                validation['warnings'].append("Excessive repetition detected")

            # Check for unusual characters
            unusual_chars = re.findall(r'[^\w\s\p{P}]', text)
            if unusual_chars:
                validation['warnings'].append(f"Unusual characters found: {set(unusual_chars)}")

            # Check capitalization (for scripts that have it)
            if re.search(r'[A-Z]', text):
                caps_ratio = len(re.findall(r'[A-Z]', text)) / len(re.findall(r'[A-Za-z]', text))
                if caps_ratio > 0.3:
                    validation['warnings'].append("High capitalization ratio")

            # Sentence structure
            sentences = self.extract_sentences(text, language_code)
            validation['metrics']['sentence_count'] = len(sentences)

            if len(sentences) == 0:
                validation['warnings'].append("No clear sentence structure")
            elif len(sentences) > 5:
                validation['recommendations'].append("Consider splitting into multiple segments")

            # Generate recommendations
            if not validation['issues'] and not validation['warnings']:
                validation['recommendations'].append("Text appears suitable for TTS training")
            elif validation['warnings']:
                validation['recommendations'].append("Text may work for TTS but has minor issues")

            return validation

        except Exception as e:
            logger.error(f"Error validating text: {e}")
            return {
                'is_valid': False,
                'issues': [f"Validation error: {e}"],
                'warnings': [],
                'recommendations': [],
                'metrics': {}
            }

    def normalize_numbers(self, text: str,
                          language_code: Optional[str] = None) -> str:
        """
        Normalize numbers to text representation

        Args:
            text: Input text
            language_code: Target language for number words

        Returns:
            Text with numbers converted to words
        """
        try:
            # This is a simplified implementation
            # For production, use language-specific number-to-word libraries

            # Common number patterns
            number_patterns = [
                (r'\b(\d{1,2}):(\d{2})\b', r'\1 बजकर \2 मिनट'),  # Time (Hindi example)
                (r'\b(\d+)\b', self._convert_number_to_words)  # General numbers
            ]

            for pattern, replacement in number_patterns:
                if callable(replacement):
                    text = re.sub(pattern, lambda m: replacement(m.group(0), language_code), text)
                else:
                    text = re.sub(pattern, replacement, text)

            return text

        except Exception as e:
            logger.error(f"Error normalizing numbers: {e}")
            return text

    def _convert_number_to_words(self, number_str: str,
                                 language_code: Optional[str] = None) -> str:
        """Convert a number string to words (simplified implementation)"""
        try:
            number = int(number_str)

            # This is a very basic implementation
            # For production, use proper number-to-word libraries for each language

            if language_code == 'hi':
                # Basic Hindi number words
                hindi_numbers = {
                    0: 'शून्य', 1: 'एक', 2: 'दो', 3: 'तीन', 4: 'चार', 5: 'पांच',
                    6: 'छह', 7: 'सात', 8: 'आठ', 9: 'नौ', 10: 'दस'
                }
                if number in hindi_numbers:
                    return hindi_numbers[number]

            # Fallback to English
            english_numbers = {
                0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'
            }

            if number in english_numbers:
                return english_numbers[number]
            else:
                return number_str  # Return original if can't convert

        except:
            return number_str

    def normalize_punctuation(self, text: str) -> str:
        """
        Normalize punctuation marks

        Args:
            text: Input text

        Returns:
            Text with normalized punctuation
        """
        try:
            # Apply punctuation mappings
            for old_punct, new_punct in self.punctuation_map.items():
                text = text.replace(old_punct, new_punct)

            # Normalize multiple punctuation
            text = re.sub(r'[.]{2,}', '...', text)  # Multiple dots to ellipsis
            text = re.sub(r'[!]{2,}', '!', text)  # Multiple exclamation marks
            text = re.sub(r'[?]{2,}', '?', text)  # Multiple question marks

            # Normalize spacing around punctuation
            text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)
            text = re.sub(r'\s*,\s*', ', ', text)
            text = re.sub(r'\s*;\s*', '; ', text)
            text = re.sub(r'\s*:\s*', ': ', text)

            return text.strip()

        except Exception as e:
            logger.error(f"Error normalizing punctuation: {e}")
            return text

    def extract_text_features(self, text: str,
                              language_code: Optional[str] = None) -> Dict:
        """
        Extract comprehensive text features for analysis

        Args:
            text: Input text
            language_code: Language code

        Returns:
            Dictionary of text features
        """
        try:
            features = {}

            # Basic metrics
            features['character_count'] = len(text)
            features['character_count_no_spaces'] = len(text.replace(' ', ''))
            features['word_count'] = self.count_words(text, language_code)
            features['sentence_count'] = len(self.extract_sentences(text, language_code))

            # Ratios
            if features['character_count'] > 0:
                features['space_ratio'] = text.count(' ') / features['character_count']
                features['punctuation_ratio'] = len(re.findall(r'[^\w\s]', text)) / features['character_count']

            # Average lengths
            if features['word_count'] > 0:
                features['avg_word_length'] = features['character_count_no_spaces'] / features['word_count']

            if features['sentence_count'] > 0:
                features['avg_sentence_length'] = features['word_count'] / features['sentence_count']

            # Script analysis
            features['script_distribution'] = self.detect_language_script(text)
            features['is_mixed_script'] = self.is_mixed_script(text)
            features['detected_language'] = self.detect_language_from_script(text)

            # Content analysis
            features['has_numbers'] = bool(re.search(r'\d', text))
            features['has_punctuation'] = bool(re.search(r'[^\w\s]', text))
            features['has_uppercase'] = bool(re.search(r'[A-Z]', text))
            features['has_lowercase'] = bool(re.search(r'[a-z]', text))

            # Complexity indicators
            features['unique_characters'] = len(set(text.lower()))
            features['repetition_score'] = self._calculate_repetition_score(text)

            return features

        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return {}

    def _calculate_repetition_score(self, text: str) -> float:
        """Calculate text repetition score (0-1, higher = more repetitive)"""
        try:
            if len(text) < 10:
                return 0.0

            # Check for character repetition
            char_counts = {}
            for char in text.lower():
                if char.isalnum():
                    char_counts[char] = char_counts.get(char, 0) + 1

            if not char_counts:
                return 0.0

            # Calculate entropy-based repetition score
            total_chars = sum(char_counts.values())
            entropy = 0
            for count in char_counts.values():
                p = count / total_chars
                entropy -= p * (p.bit_length() - 1) if p > 0 else 0

            # Normalize entropy to 0-1 scale (higher = less repetitive)
            max_entropy = (len(char_counts).bit_length() - 1) if len(char_counts) > 1 else 1
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            # Return repetition score (inverse of normalized entropy)
            return 1 - normalized_entropy

        except Exception as e:
            logger.error(f"Error calculating repetition score: {e}")
            return 0.0

    def batch_process_text_files(self, input_dir: Union[str, Path],
                                 output_dir: Union[str, Path],
                                 operation: str = 'clean',
                                 language_code: Optional[str] = None,
                                 **kwargs) -> Dict:
        """
        Batch process multiple text files

        Args:
            input_dir: Directory containing input text files
            output_dir: Directory for output files
            operation: Operation to perform ('clean', 'validate', 'analyze')
            language_code: Target language code
            **kwargs: Additional arguments for the operation

        Returns:
            Processing results dictionary
        """
        try:
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Find all text files
            text_files = []
            for ext in ['.txt', '.json']:
                text_files.extend(input_dir.glob(f'*{ext}'))

            if not text_files:
                return {
                    'success': False,
                    'error': 'No text files found',
                    'processed': 0,
                    'total': 0
                }

            results = {
                'success': True,
                'processed': 0,
                'total': len(text_files),
                'failed': [],
                'details': []
            }

            for text_file in text_files:
                try:
                    # Read input file
                    if text_file.suffix == '.json':
                        with open(text_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            text_content = data.get('text', str(data))
                    else:
                        with open(text_file, 'r', encoding='utf-8') as f:
                            text_content = f.read()

                    output_file = output_dir / f"{text_file.stem}_processed{text_file.suffix}"

                    if operation == 'clean':
                        processed_text = self.clean_text(text_content, language_code, **kwargs)

                        if text_file.suffix == '.json':
                            output_data = {'text': processed_text, 'original_file': str(text_file)}
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(output_data, f, ensure_ascii=False, indent=2)
                        else:
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(processed_text)

                    elif operation == 'validate':
                        validation_result = self.validate_text_for_tts(text_content, language_code, **kwargs)

                        output_data = {
                            'original_file': str(text_file),
                            'validation': validation_result,
                            'text': text_content
                        }

                        with open(output_file.with_suffix('.json'), 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, ensure_ascii=False, indent=2)

                    elif operation == 'analyze':
                        features = self.extract_text_features(text_content, language_code)

                        output_data = {
                            'original_file': str(text_file),
                            'features': features,
                            'text': text_content
                        }

                        with open(output_file.with_suffix('.json'), 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, ensure_ascii=False, indent=2)

                    results['processed'] += 1
                    results['details'].append({
                        'input': str(text_file),
                        'output': str(output_file),
                        'status': 'success'
                    })

                except Exception as e:
                    logger.error(f"Error processing {text_file}: {e}")
                    results['failed'].append(str(text_file))

            logger.info(f"Batch text processing completed: {results['processed']}/{results['total']} files")
            return results

        except Exception as e:
            logger.error(f"Error in batch text processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'processed': 0,
                'total': 0
            }


# Create global instance
text_utils = TextUtils()