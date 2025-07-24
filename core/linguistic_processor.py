"""
Advanced Linguistic Processing Module
Handles phonetics, phonology, morphology, prosody, and text normalization
for Indian languages to improve TTS quality and training efficiency
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import unicodedata

from config.languages import IndianLanguages

logger = logging.getLogger(__name__)


@dataclass
class Phoneme:
    """Phoneme representation with linguistic features"""
    symbol: str
    features: Dict[str, str]  # articulatory features
    allophones: List[str] = None
    contexts: List[str] = None  # contextual variants


@dataclass
class ProsodyMarker:
    """Prosodic information for synthesis"""
    stress_level: int  # 0=unstressed, 1=secondary, 2=primary
    tone: str  # 'H', 'L', 'R' (rising), 'F' (falling)
    boundary: str  # 'ip' (intonational phrase), 'pp' (phonological phrase), 'w' (word)
    duration_modifier: float = 1.0


class LinguisticProcessor:
    """Advanced linguistic processing for Indian languages"""

    def __init__(self, language_code: str):
        self.language_code = language_code
        self.languages = IndianLanguages()
        self.lang_info = self.languages.get_language_info(language_code)

        # Initialize linguistic components
        self.phoneme_inventory = self.load_phoneme_inventory()
        self.g2p_rules = self.load_g2p_rules()
        self.prosody_rules = self.load_prosody_rules()
        self.morphological_rules = self.load_morphological_rules()
        self.text_normalizer = TextNormalizer(language_code)
        self.code_switch_detector = CodeSwitchDetector(language_code)

        logger.info(f"✅ Linguistic processor initialized for {language_code}")

    def load_phoneme_inventory(self) -> Dict[str, Phoneme]:
        """Load comprehensive phoneme inventory with articulatory features"""
        phoneme_sets = {
            'hi': self._get_hindi_phonemes(),
            'ta': self._get_tamil_phonemes(),
            'te': self._get_telugu_phonemes(),
            'bn': self._get_bengali_phonemes(),
            'mr': self._get_marathi_phonemes(),
            'gu': self._get_gujarati_phonemes(),
            'kn': self._get_kannada_phonemes(),
            'ml': self._get_malayalam_phonemes(),
            'pa': self._get_punjabi_phonemes(),
            'or': self._get_odia_phonemes()
        }

        return phoneme_sets.get(self.language_code, {})

    def _get_hindi_phonemes(self) -> Dict[str, Phoneme]:
        """Detailed Hindi phoneme inventory with articulatory features"""
        phonemes = {}

        # Vowels with features
        vowels = [
            ('a', {'height': 'low', 'backness': 'central', 'roundness': 'unrounded', 'length': 'short'}),
            ('aː', {'height': 'low', 'backness': 'central', 'roundness': 'unrounded', 'length': 'long'}),
            ('i', {'height': 'high', 'backness': 'front', 'roundness': 'unrounded', 'length': 'short'}),
            ('iː', {'height': 'high', 'backness': 'front', 'roundness': 'unrounded', 'length': 'long'}),
            ('u', {'height': 'high', 'backness': 'back', 'roundness': 'rounded', 'length': 'short'}),
            ('uː', {'height': 'high', 'backness': 'back', 'roundness': 'rounded', 'length': 'long'}),
            ('e', {'height': 'mid', 'backness': 'front', 'roundness': 'unrounded', 'length': 'long'}),
            ('o', {'height': 'mid', 'backness': 'back', 'roundness': 'rounded', 'length': 'long'}),
            ('ɛ', {'height': 'low-mid', 'backness': 'front', 'roundness': 'unrounded', 'length': 'long'}),
            ('ɔ', {'height': 'low-mid', 'backness': 'back', 'roundness': 'rounded', 'length': 'long'})
        ]

        for symbol, features in vowels:
            features['type'] = 'vowel'
            phonemes[symbol] = Phoneme(symbol, features)

        # Consonants with detailed articulatory features
        consonants = [
            # Stops (plosives)
            ('p', {'manner': 'stop', 'place': 'bilabial', 'voice': 'voiceless', 'aspiration': 'unaspirated'}),
            ('pʰ', {'manner': 'stop', 'place': 'bilabial', 'voice': 'voiceless', 'aspiration': 'aspirated'}),
            ('b', {'manner': 'stop', 'place': 'bilabial', 'voice': 'voiced', 'aspiration': 'unaspirated'}),
            ('bʱ', {'manner': 'stop', 'place': 'bilabial', 'voice': 'voiced', 'aspiration': 'breathy'}),

            ('t̪', {'manner': 'stop', 'place': 'dental', 'voice': 'voiceless', 'aspiration': 'unaspirated'}),
            ('t̪ʰ', {'manner': 'stop', 'place': 'dental', 'voice': 'voiceless', 'aspiration': 'aspirated'}),
            ('d̪', {'manner': 'stop', 'place': 'dental', 'voice': 'voiced', 'aspiration': 'unaspirated'}),
            ('d̪ʱ', {'manner': 'stop', 'place': 'dental', 'voice': 'voiced', 'aspiration': 'breathy'}),

            # Retroflexes (crucial for Indian languages)
            ('ʈ', {'manner': 'stop', 'place': 'retroflex', 'voice': 'voiceless', 'aspiration': 'unaspirated'}),
            ('ʈʰ', {'manner': 'stop', 'place': 'retroflex', 'voice': 'voiceless', 'aspiration': 'aspirated'}),
            ('ɖ', {'manner': 'stop', 'place': 'retroflex', 'voice': 'voiced', 'aspiration': 'unaspirated'}),
            ('ɖʱ', {'manner': 'stop', 'place': 'retroflex', 'voice': 'voiced', 'aspiration': 'breathy'}),

            ('k', {'manner': 'stop', 'place': 'velar', 'voice': 'voiceless', 'aspiration': 'unaspirated'}),
            ('kʰ', {'manner': 'stop', 'place': 'velar', 'voice': 'voiceless', 'aspiration': 'aspirated'}),
            ('ɡ', {'manner': 'stop', 'place': 'velar', 'voice': 'voiced', 'aspiration': 'unaspirated'}),
            ('ɡʱ', {'manner': 'stop', 'place': 'velar', 'voice': 'voiced', 'aspiration': 'breathy'}),

            # Nasals
            ('m', {'manner': 'nasal', 'place': 'bilabial', 'voice': 'voiced'}),
            ('n̪', {'manner': 'nasal', 'place': 'dental', 'voice': 'voiced'}),
            ('ɳ', {'manner': 'nasal', 'place': 'retroflex', 'voice': 'voiced'}),
            ('n', {'manner': 'nasal', 'place': 'alveolar', 'voice': 'voiced'}),
            ('ŋ', {'manner': 'nasal', 'place': 'velar', 'voice': 'voiced'}),

            # Liquids
            ('r', {'manner': 'trill', 'place': 'alveolar', 'voice': 'voiced'}),
            ('ɽ', {'manner': 'tap', 'place': 'retroflex', 'voice': 'voiced'}),
            ('l', {'manner': 'lateral', 'place': 'alveolar', 'voice': 'voiced'}),

            # Fricatives
            ('s', {'manner': 'fricative', 'place': 'alveolar', 'voice': 'voiceless'}),
            ('ʃ', {'manner': 'fricative', 'place': 'postalveolar', 'voice': 'voiceless'}),
            ('h', {'manner': 'fricative', 'place': 'glottal', 'voice': 'voiceless'}),

            # Approximants
            ('j', {'manner': 'approximant', 'place': 'palatal', 'voice': 'voiced'}),
            ('w', {'manner': 'approximant', 'place': 'labio-velar', 'voice': 'voiced'}),
            ('ʋ', {'manner': 'approximant', 'place': 'labio-dental', 'voice': 'voiced'})
        ]

        for symbol, features in consonants:
            features['type'] = 'consonant'
            phonemes[symbol] = Phoneme(symbol, features)

        # Add allophonic variants and contexts
        self._add_allophonic_rules(phonemes)

        return phonemes

    def _get_tamil_phonemes(self) -> Dict[str, Phoneme]:
        """Tamil phoneme inventory with Dravidian-specific features"""
        phonemes = {}

        # Tamil-specific features
        vowels = [
            ('a', {'height': 'low', 'backness': 'central', 'roundness': 'unrounded', 'length': 'short'}),
            ('aː', {'height': 'low', 'backness': 'central', 'roundness': 'unrounded', 'length': 'long'}),
            ('i', {'height': 'high', 'backness': 'front', 'roundness': 'unrounded', 'length': 'short'}),
            ('iː', {'height': 'high', 'backness': 'front', 'roundness': 'unrounded', 'length': 'long'}),
            ('u', {'height': 'high', 'backness': 'back', 'roundness': 'rounded', 'length': 'short'}),
            ('uː', {'height': 'high', 'backness': 'back', 'roundness': 'rounded', 'length': 'long'}),
            ('e', {'height': 'mid', 'backness': 'front', 'roundness': 'unrounded', 'length': 'long'}),
            ('eː', {'height': 'mid', 'backness': 'front', 'roundness': 'unrounded', 'length': 'long'}),
            ('o', {'height': 'mid', 'backness': 'back', 'roundness': 'rounded', 'length': 'long'}),
            ('oː', {'height': 'mid', 'backness': 'back', 'roundness': 'rounded', 'length': 'long'})
        ]

        for symbol, features in vowels:
            features['type'] = 'vowel'
            phonemes[symbol] = Phoneme(symbol, features)

        # Tamil consonants (no aspiration contrast)
        consonants = [
            ('k', {'manner': 'stop', 'place': 'velar', 'voice': 'voiceless'}),
            ('ŋ', {'manner': 'nasal', 'place': 'velar', 'voice': 'voiced'}),
            ('tʃ', {'manner': 'affricate', 'place': 'postalveolar', 'voice': 'voiceless'}),
            ('ɲ', {'manner': 'nasal', 'place': 'palatal', 'voice': 'voiced'}),
            ('ʈ', {'manner': 'stop', 'place': 'retroflex', 'voice': 'voiceless'}),
            ('ɳ', {'manner': 'nasal', 'place': 'retroflex', 'voice': 'voiced'}),
            ('t̪', {'manner': 'stop', 'place': 'dental', 'voice': 'voiceless'}),
            ('n̪', {'manner': 'nasal', 'place': 'dental', 'voice': 'voiced'}),
            ('p', {'manner': 'stop', 'place': 'bilabial', 'voice': 'voiceless'}),
            ('m', {'manner': 'nasal', 'place': 'bilabial', 'voice': 'voiced'}),
            ('j', {'manner': 'approximant', 'place': 'palatal', 'voice': 'voiced'}),
            ('r', {'manner': 'trill', 'place': 'alveolar', 'voice': 'voiced'}),
            ('l', {'manner': 'lateral', 'place': 'alveolar', 'voice': 'voiced'}),
            ('ʋ', {'manner': 'approximant', 'place': 'labio-dental', 'voice': 'voiced'}),
            ('ɻ', {'manner': 'approximant', 'place': 'retroflex', 'voice': 'voiced'}),  # Tamil-specific
            ('ɭ', {'manner': 'lateral', 'place': 'retroflex', 'voice': 'voiced'})  # Tamil-specific
        ]

        for symbol, features in consonants:
            features['type'] = 'consonant'
            phonemes[symbol] = Phoneme(symbol, features)

        return phonemes

    def _add_allophonic_rules(self, phonemes: Dict[str, Phoneme]):
        """Add allophonic variation rules"""
        # Hindi schwa deletion context
        if 'a' in phonemes:
            phonemes['a'].contexts = ['word_final_delete', 'unstressed_delete']
            phonemes['a'].allophones = ['ə', 'ɐ', '∅']  # including deletion

        # Retroflex assimilation
        for symbol in ['ʈ', 'ɖ', 'ɳ']:
            if symbol in phonemes:
                phonemes[symbol].contexts = ['before_retroflex', 'after_retroflex']

    def load_g2p_rules(self) -> Dict:
        """Load grapheme-to-phoneme conversion rules"""
        g2p_rules = {
            'hi': self._get_hindi_g2p_rules(),
            'ta': self._get_tamil_g2p_rules(),
            'te': self._get_telugu_g2p_rules(),
            'bn': self._get_bengali_g2p_rules(),
            'mr': self._get_marathi_g2p_rules(),
            'gu': self._get_gujarati_g2p_rules(),
            'kn': self._get_kannada_g2p_rules(),
            'ml': self._get_malayalam_g2p_rules(),
            'pa': self._get_punjabi_g2p_rules(),
            'or': self._get_odia_g2p_rules()
        }

        return g2p_rules.get(self.language_code, {})

    def _get_hindi_g2p_rules(self) -> Dict:
        """Hindi grapheme-to-phoneme rules with context sensitivity"""
        return {
            'single_chars': {
                # Vowels
                'अ': 'a', 'आ': 'aː', 'इ': 'i', 'ई': 'iː',
                'उ': 'u', 'ऊ': 'uː', 'ए': 'e', 'ै': 'ɛ',
                'ओ': 'o', 'ौ': 'ɔ',

                # Consonants with inherent vowel
                'क': 'ka', 'ख': 'kʰa', 'ग': 'ɡa', 'घ': 'ɡʱa', 'ङ': 'ŋa',
                'च': 'tʃa', 'छ': 'tʃʰa', 'ज': 'dʒa', 'झ': 'dʒʱa', 'ञ': 'ɲa',
                'ट': 'ʈa', 'ठ': 'ʈʰa', 'ड': 'ɖa', 'ढ': 'ɖʱa', 'ण': 'ɳa',
                'त': 't̪a', 'थ': 't̪ʰa', 'द': 'd̪a', 'ध': 'd̪ʱa', 'न': 'n̪a',
                'प': 'pa', 'फ': 'pʰa', 'ब': 'ba', 'भ': 'bʱa', 'म': 'ma',
                'य': 'ja', 'र': 'ra', 'ल': 'la', 'व': 'ʋa',
                'श': 'ʃa', 'ष': 'ʃa', 'स': 'sa', 'ह': 'ha'
            },

            'contextual_rules': [
                # Schwa deletion rules
                {
                    'pattern': r'([कखगघचछजझटठडढतथदधपफबभयरलवशषसह])्?a([कखगघचछजझटठडढतथदधपफबभयरलवशषसह])',
                    'replacement': r'\1∅\2',  # Delete schwa in medial position
                    'condition': 'medial_schwa'
                },
                {
                    'pattern': r'([कखगघचछजझटठडढतथदधपफबभयरलवशषसह])a$',
                    'replacement': r'\1∅',  # Delete word-final schwa
                    'condition': 'final_schwa'
                },

                # Nasalization rules
                {
                    'pattern': r'([aeiouािीुूेैोौ])ं',
                    'replacement': r'\1̃',  # Add nasalization
                    'condition': 'anusvara'
                },

                # Retroflexion assimilation
                {
                    'pattern': r'([ʈɖɳ])([त̪द̪न̪])',
                    'replacement': r'\1\2[+retroflex]',
                    'condition': 'retroflex_assimilation'
                }
            ],

            'compound_handling': {
                'split_markers': ['्', '़'],
                'joiner_markers': ['‌', '‍']
            }
        }

    def load_prosody_rules(self) -> Dict:
        """Load prosodic rules for natural speech rhythm"""
        return {
            'stress_patterns': self._get_stress_patterns(),
            'tone_rules': self._get_tone_rules(),
            'boundary_markers': self._get_boundary_markers(),
            'duration_rules': self._get_duration_rules()
        }

    def _get_stress_patterns(self) -> Dict:
        """Language-specific stress assignment rules"""
        stress_rules = {
            'hi': {
                'primary_stress': 'first_heavy_syllable',  # Heavy syllable gets primary stress
                'secondary_stress': 'alternating',
                'clash_resolution': 'move_left',
                'weight_sensitive': True,
                'heavy_syllable_pattern': r'[aeiouािीुूेैोौ][ःंँ]|[aeiouािीुूेैोौ][कखगघचछजझटठडढतथदधपफबभयरलवशषसह]'
            },
            'ta': {
                'primary_stress': 'initial',  # Tamil typically has initial stress
                'secondary_stress': 'none',
                'weight_sensitive': False
            },
            'te': {
                'primary_stress': 'penultimate',  # Second-to-last syllable
                'secondary_stress': 'initial',
                'weight_sensitive': True
            }
        }

        return stress_rules.get(self.language_code, stress_rules['hi'])

    def _get_tone_rules(self) -> Dict:
        """Intonational patterns for different sentence types"""
        return {
            'declarative': {
                'initial': 'H*',  # High pitch accent
                'medial': 'L+H*',  # Rising accent
                'final': 'L-L%'  # Low boundary tone
            },
            'interrogative': {
                'initial': 'H*',
                'medial': 'L+H*',
                'final': 'H-H%'  # High boundary tone for questions
            },
            'exclamative': {
                'initial': 'H*',
                'medial': '!H*',  # Downstepped high
                'final': 'L-L%'
            }
        }

    def _get_boundary_markers(self) -> Dict:
        """Prosodic boundary detection rules"""
        return {
            'punctuation_boundaries': {
                '.': 'ip',  # Intonational phrase
                '।': 'ip',  # Devanagari danda
                ',': 'pp',  # Phonological phrase
                ';': 'pp',
                ':': 'pp',
                '?': 'ip',
                '!': 'ip'
            },
            'syntactic_boundaries': {
                'clause_boundary': 'pp',
                'phrase_boundary': 'pw',  # Prosodic word
                'compound_boundary': 'pw'
            }
        }

    def load_morphological_rules(self) -> Dict:
        """Load morphological analysis rules"""
        return {
            'compound_splitting': self._get_compound_rules(),
            'inflectional_analysis': self._get_inflection_rules(),
            'derivational_analysis': self._get_derivation_rules()
        }

    def _get_compound_rules(self) -> Dict:
        """Rules for splitting compound words"""
        return {
            'hi': {
                'compound_markers': ['्', '-'],
                'common_prefixes': ['अ', 'अन्', 'अप्', 'अभि', 'अव', 'आ', 'उप', 'नि', 'परि', 'प्र', 'वि', 'सम्'],
                'common_suffixes': ['कार', 'ता', 'त्व', 'पन', 'आई', 'इया'],
                'splitting_patterns': [
                    r'([अआइईउऊएऐओऔ][कखगघचछजझटठडढतथदधपफबभयरलवशषसह]+)([कखगघचछजझटठडढतथदधपफबभयरलवशषसह][अआइईउऊएऐओऔ])',
                    r'([कखगघचछजझटठडढतथदधपफबभयरलवशषसह]+)(कार|ता|त्व|पन)$'
                ]
            }
        }

    def grapheme_to_phoneme(self, text: str) -> List[str]:
        """Convert text to phonemic representation"""
        logger.info(f"Converting to phonemes: {text[:50]}...")

        # Normalize text
        text = self.text_normalizer.normalize(text)

        # Handle code-switching
        text = self.code_switch_detector.mark_language_boundaries(text)

        # Apply G2P rules
        phonemes = []
        words = text.split()

        for word in words:
            word_phonemes = self._convert_word_to_phonemes(word)
            phonemes.extend(word_phonemes)
            phonemes.append('|')  # Word boundary marker

        return phonemes[:-1]  # Remove final boundary marker

    def _convert_word_to_phonemes(self, word: str) -> List[str]:
        """Convert single word to phonemes"""
        if not word:
            return []

        # Check for compound splitting
        if self._is_compound(word):
            parts = self._split_compound(word)
            phonemes = []
            for part in parts:
                phonemes.extend(self._convert_simple_word(part))
                phonemes.append('+')  # Morpheme boundary
            return phonemes[:-1]  # Remove final boundary
        else:
            return self._convert_simple_word(word)

    def _convert_simple_word(self, word: str) -> List[str]:
        """Convert simple (non-compound) word to phonemes"""
        phonemes = []
        g2p_rules = self.g2p_rules

        # Apply contextual rules first
        processed_word = word
        for rule in g2p_rules.get('contextual_rules', []):
            processed_word = re.sub(rule['pattern'], rule['replacement'], processed_word)

        # Convert character by character
        i = 0
        while i < len(processed_word):
            char = processed_word[i]

            # Check for multi-character sequences
            if i + 1 < len(processed_word):
                two_char = processed_word[i:i + 2]
                if two_char in g2p_rules.get('single_chars', {}):
                    phonemes.append(g2p_rules['single_chars'][two_char])
                    i += 2
                    continue

            # Single character conversion
            if char in g2p_rules.get('single_chars', {}):
                phonemes.append(g2p_rules['single_chars'][char])
            elif char not in [' ', '\t', '\n']:
                # Unknown character - use character itself
                phonemes.append(char)

            i += 1

        return phonemes

    def add_prosodic_features(self, phonemes: List[str], text: str) -> List[Tuple[str, ProsodyMarker]]:
        """Add prosodic information to phoneme sequence"""
        logger.info("Adding prosodic features...")

        # Detect sentence type for intonation
        sentence_type = self._detect_sentence_type(text)

        # Syllabify and assign stress
        syllables = self._syllabify(phonemes)
        stressed_syllables = self._assign_stress(syllables)

        # Add intonational patterns
        prosodic_phonemes = []
        tone_rules = self.prosody_rules['tone_rules'][sentence_type]

        for i, (phoneme, syllable_info) in enumerate(stressed_syllables):
            # Determine prosodic position
            if i == 0:
                position = 'initial'
            elif i == len(stressed_syllables) - 1:
                position = 'final'
            else:
                position = 'medial'

            # Create prosody marker
            prosody = ProsodyMarker(
                stress_level=syllable_info.get('stress', 0),
                tone=tone_rules.get(position, 'L'),
                boundary=syllable_info.get('boundary', 'none'),
                duration_modifier=self._calculate_duration_modifier(phoneme, syllable_info)
            )

            prosodic_phonemes.append((phoneme, prosody))

        return prosodic_phonemes

    def _detect_sentence_type(self, text: str) -> str:
        """Detect sentence type for appropriate intonation"""
        text = text.strip()
        if text.endswith('?') or 'क्या' in text or 'कैसे' in text:
            return 'interrogative'
        elif text.endswith('!') or any(excl in text for excl in ['अरे', 'वाह', 'हाय']):
            return 'exclamative'
        else:
            return 'declarative'

    def _syllabify(self, phonemes: List[str]) -> List[List[str]]:
        """Break phoneme sequence into syllables"""
        syllables = []
        current_syllable = []

        for phoneme in phonemes:
            if phoneme == '|':  # Word boundary
                if current_syllable:
                    syllables.append(current_syllable)
                    current_syllable = []
                continue

            current_syllable.append(phoneme)

            # Simple syllable boundary detection
            # In Indian languages, typically CV(C) structure
            if self._is_vowel(phoneme):
                # Look ahead for consonant clusters
                next_consonants = []
                j = phonemes.index(phoneme) + 1
                while j < len(phonemes) and not self._is_vowel(phonemes[j]) and phonemes[j] != '|':
                    next_consonants.append(phonemes[j])
                    j += 1

                # Syllable boundary after vowel if followed by consonant cluster
                if len(next_consonants) > 1:
                    current_syllable.append(next_consonants[0])  # Take first consonant
                    syllables.append(current_syllable)
                    current_syllable = []

        if current_syllable:
            syllables.append(current_syllable)

        return syllables

    def _assign_stress(self, syllables: List[List[str]]) -> List[Tuple[str, Dict]]:
        """Assign stress to syllables based on language rules"""
        stress_pattern = self.prosody_rules['stress_patterns']
        stressed_syllables = []

        if not syllables:
            return stressed_syllables

        # Determine heavy vs light syllables
        syllable_weights = []
        for syllable in syllables:
            weight = self._calculate_syllable_weight(syllable)
            syllable_weights.append(weight)

        # Apply stress rules based on language
        primary_stress_pos = self._find_primary_stress_position(syllable_weights, stress_pattern)
        secondary_stress_positions = self._find_secondary_stress_positions(syllable_weights, primary_stress_pos,
                                                                           stress_pattern)

        # Create stressed syllable list
        for i, syllable in enumerate(syllables):
            for phoneme in syllable:
                stress_level = 0
                if i == primary_stress_pos:
                    stress_level = 2  # Primary stress
                elif i in secondary_stress_positions:
                    stress_level = 1  # Secondary stress

                syllable_info = {
                    'stress': stress_level,
                    'weight': syllable_weights[i],
                    'position': i,
                    'boundary': 'syllable'
                }

                stressed_syllables.append((phoneme, syllable_info))

        return stressed_syllables

    def _calculate_syllable_weight(self, syllable: List[str]) -> str:
        """Calculate syllable weight (heavy/light)"""
        # Heavy syllable: long vowel, vowel + consonant, diphthong
        has_long_vowel = any('ː' in phoneme for phoneme in syllable)
        vowel_count = sum(1 for p in syllable if self._is_vowel(p))
        consonant_after_vowel = False

        vowel_found = False
        for phoneme in syllable:
            if self._is_vowel(phoneme):
                vowel_found = True
            elif vowel_found and not self._is_vowel(phoneme):
                consonant_after_vowel = True
                break

        if has_long_vowel or consonant_after_vowel or vowel_count > 1:
            return 'heavy'
        else:
            return 'light'

    def _find_primary_stress_position(self, weights: List[str], pattern: Dict) -> int:
        """Find primary stress position based on language rules"""
        if not weights:
            return 0

        rule = pattern.get('primary_stress', 'initial')

        if rule == 'initial':
            return 0
        elif rule == 'final':
            return len(weights) - 1
        elif rule == 'penultimate':
            return max(0, len(weights) - 2)
        elif rule == 'first_heavy_syllable':
            for i, weight in enumerate(weights):
                if weight == 'heavy':
                    return i
            return 0  # Default to initial if no heavy syllable
        else:
            return 0

    def _find_secondary_stress_positions(self, weights: List[str], primary_pos: int, pattern: Dict) -> List[int]:
        """Find secondary stress positions"""
        secondary_rule = pattern.get('secondary_stress', 'none')
        positions = []

        if secondary_rule == 'alternating':
            # Place secondary stress every other syllable, avoiding primary
            for i in range(0, len(weights), 2):
                if i != primary_pos and len(weights) > 2:
                    positions.append(i)
        elif secondary_rule == 'initial' and primary_pos != 0:
            positions.append(0)

        return positions

    def _calculate_duration_modifier(self, phoneme: str, syllable_info: Dict) -> float:
        """Calculate duration modification based on prosodic context"""
        base_duration = 1.0

        # Stress-based lengthening
        if syllable_info.get('stress', 0) == 2:  # Primary stress
            base_duration *= 1.2
        elif syllable_info.get('stress', 0) == 1:  # Secondary stress
            base_duration *= 1.1

        # Vowel lengthening
        if self._is_vowel(phoneme):
            if 'ː' in phoneme:  # Already long vowel
                base_duration *= 1.3
            elif syllable_info.get('weight') == 'heavy':
                base_duration *= 1.15

        # Boundary effects
        boundary = syllable_info.get('boundary', 'none')
        if boundary == 'ip':  # Intonational phrase
            base_duration *= 1.5  # Phrase-final lengthening
        elif boundary == 'pp':  # Phonological phrase
            base_duration *= 1.25

        return base_duration

    def _is_vowel(self, phoneme: str) -> bool:
        """Check if phoneme is a vowel"""
        vowel_symbols = {'a', 'e', 'i', 'o', 'u', 'ə', 'ɛ', 'ɔ', 'ɐ'}
        # Remove length and nasalization markers for checking
        base_phoneme = phoneme.replace('ː', '').replace('̃', '').replace('̥', '')
        return any(v in base_phoneme for v in vowel_symbols)

    def _is_compound(self, word: str) -> bool:
        """Check if word is a compound"""
        compound_rules = self.morphological_rules.get('compound_splitting', {})
        markers = compound_rules.get('compound_markers', [])

        # Check for compound markers
        if any(marker in word for marker in markers):
            return True

        # Check for common prefixes/suffixes pattern
        prefixes = compound_rules.get('common_prefixes', [])
        suffixes = compound_rules.get('common_suffixes', [])

        has_prefix = any(word.startswith(prefix) for prefix in prefixes)
        has_suffix = any(word.endswith(suffix) for suffix in suffixes)

        return has_prefix and len(word) > 6  # Heuristic for compound length

    def _split_compound(self, word: str) -> List[str]:
        """Split compound word into morphemes"""
        compound_rules = self.morphological_rules.get('compound_splitting', {})

        # Try splitting patterns
        for pattern in compound_rules.get('splitting_patterns', []):
            match = re.search(pattern, word)
            if match:
                return [match.group(1), match.group(2)]

        # Fallback: split by markers
        markers = compound_rules.get('compound_markers', [])
        for marker in markers:
            if marker in word:
                return word.split(marker)

        return [word]  # Return as single word if no split found


class TextNormalizer:
    """Advanced text normalization for Indian languages"""

    def __init__(self, language_code: str):
        self.language_code = language_code
        self.number_rules = self._load_number_rules()
        self.abbreviation_rules = self._load_abbreviation_rules()
        self.date_time_rules = self._load_datetime_rules()

    def normalize(self, text: str) -> str:
        """Comprehensive text normalization"""
        logger.info(f"Normalizing text: {text[:100]}...")

        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)

        # Expand numbers
        text = self._expand_numbers(text)

        # Expand dates and times
        text = self._expand_datetime(text)

        # Expand abbreviations
        text = self._expand_abbreviations(text)

        # Handle currency and measurements
        text = self._expand_currency_measurements(text)

        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _load_number_rules(self) -> Dict:
        """Load number expansion rules"""
        rules = {
            'hi': {
                'ones': ['', 'एक', 'दो', 'तीन', 'चार', 'पांच', 'छह', 'सात', 'आठ', 'नौ'],
                'teens': ['दस', 'ग्यारह', 'बारह', 'तेरह', 'चौदह', 'पंद्रह', 'सोलह', 'सत्रह', 'अठारह', 'उन्नीस'],
                'tens': ['', '', 'बीस', 'तीस', 'चालीस', 'पचास', 'साठ', 'सत्तर', 'अस्सी', 'नब्बे'],
                'hundreds': 'सौ',
                'thousands': 'हज़ार',
                'lakhs': 'लाख',
                'crores': 'करोड़'
            },
            'ta': {
                'ones': ['', 'ஒன்று', 'இரண்டு', 'மூன்று', 'நான்கு', 'ஐந்து', 'ஆறு', 'ஏழு', 'எட்டு', 'ஒன்பது'],
                'teens': ['பத்து', 'பதினொன்று', 'பன்னிரண்டு', 'பதின்மூன்று', 'பதினான்கு', 'பதினைந்து', 'பதினாறு',
                          'பதினேழு', 'பதினெட்டு', 'பத்தொன்பது'],
                'tens': ['', '', 'இருபது', 'முப்பது', 'நாற்பது', 'ஐம்பது', 'அறுபது', 'எழுபது', 'எண்பது', 'தொண்ணூறு'],
                'hundreds': 'நூறு',
                'thousands': 'ஆயிரம்',
                'lakhs': 'லட்சம்',
                'crores': 'கோடி'
            }
        }

        return rules.get(self.language_code, rules['hi'])

    def _expand_numbers(self, text: str) -> str:
        """Expand numbers to words"""
        # Devanagari numerals
        devanagari_map = {'०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
                          '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'}

        for dev_num, arab_num in devanagari_map.items():
            text = text.replace(dev_num, arab_num)

        # Expand Arabic numerals
        def replace_number(match):
            number = int(match.group())
            return self._number_to_words(number)

        # Match various number patterns
        patterns = [
            r'\b\d{1,2}:\d{2}\b',  # Time format
            r'\b\d+\.\d+\b',  # Decimal numbers
            r'\b\d+\b'  # Integers
        ]

        for pattern in patterns:
            if ':' in pattern:  # Time
                text = re.sub(pattern, self._expand_time, text)
            elif '\\.' in pattern:  # Decimal
                text = re.sub(pattern, self._expand_decimal, text)
            else:  # Integer
                text = re.sub(pattern, replace_number, text)

        return text

    def _number_to_words(self, number: int) -> str:
        """Convert number to words in target language"""
        if number == 0:
            return 'शून्य' if self.language_code == 'hi' else 'zero'

        rules = self.number_rules

        if number < 10:
            return rules['ones'][number]
        elif number < 20:
            return rules['teens'][number - 10]
        elif number < 100:
            tens = number // 10
            ones = number % 10
            result = rules['tens'][tens]
            if ones > 0:
                result += ' ' + rules['ones'][ones]
            return result
        elif number < 1000:
            hundreds = number // 100
            remainder = number % 100
            result = rules['ones'][hundreds] + ' ' + rules['hundreds']
            if remainder > 0:
                result += ' ' + self._number_to_words(remainder)
            return result
        elif number < 100000:  # Indian numbering system
            thousands = number // 1000
            remainder = number % 1000
            result = self._number_to_words(thousands) + ' ' + rules['thousands']
            if remainder > 0:
                result += ' ' + self._number_to_words(remainder)
            return result
        elif number < 10000000:
            lakhs = number // 100000
            remainder = number % 100000
            result = self._number_to_words(lakhs) + ' ' + rules['lakhs']
            if remainder > 0:
                result += ' ' + self._number_to_words(remainder)
            return result
        else:
            crores = number // 10000000
            remainder = number % 10000000
            result = self._number_to_words(crores) + ' ' + rules['crores']
            if remainder > 0:
                result += ' ' + self._number_to_words(remainder)
            return result

    def _expand_time(self, match) -> str:
        """Expand time format (HH:MM)"""
        time_str = match.group()
        hours, minutes = map(int, time_str.split(':'))

        if self.language_code == 'hi':
            hour_word = self._number_to_words(hours)
            minute_word = self._number_to_words(minutes) if minutes > 0 else ''

            if minutes == 0:
                return f"{hour_word} बजे"
            else:
                return f"{hour_word} बजकर {minute_word} मिनट"
        else:
            return f"{self._number_to_words(hours)} {self._number_to_words(minutes)}"

    def _expand_decimal(self, match) -> str:
        """Expand decimal numbers"""
        decimal_str = match.group()
        integer_part, decimal_part = decimal_str.split('.')

        integer_words = self._number_to_words(int(integer_part))

        if self.language_code == 'hi':
            decimal_words = ' '.join(self._number_to_words(int(d)) for d in decimal_part)
            return f"{integer_words} दशमलव {decimal_words}"
        else:
            decimal_words = ' '.join(self._number_to_words(int(d)) for d in decimal_part)
            return f"{integer_words} point {decimal_words}"

    def _load_abbreviation_rules(self) -> Dict:
        """Load abbreviation expansion rules"""
        return {
            'hi': {
                'डॉ.': 'डॉक्टर',
                'श्री': 'श्री',
                'श्रीमती': 'श्रीमती',
                'प्रो.': 'प्रोफेसर',
                'इ.पू.': 'ईसा पूर्व',
                'ई.स्वी.': 'ईसवी सन',
                'आदि': 'आदि',
                'etc.': 'इत्यादि',
                'vs.': 'बनाम',
                'i.e.': 'यानी',
                'e.g.': 'जैसे'
            },
            'ta': {
                'டாக்.': 'டாக்டர்',
                'திரு.': 'திரு',
                'திருமती.': 'திருமती',
                'பேரா.': 'பேராசிரியர்'
            }
        }

    def _expand_abbreviations(self, text: str) -> str:
        """Expand abbreviations"""
        abbrev_rules = self.abbreviation_rules.get(self.language_code, {})

        for abbrev, expansion in abbrev_rules.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text)

        return text

    def _load_datetime_rules(self) -> Dict:
        """Load date and time expansion rules"""
        return {
            'hi': {
                'months': ['जनवरी', 'फरवरी', 'मार्च', 'अप्रैल', 'मई', 'जून',
                           'जुलाई', 'अगस्त', 'सितंबर', 'अक्टूबर', 'नवंबर', 'दिसंबर'],
                'days': ['सोमवार', 'मंगलवार', 'बुधवार', 'गुरुवार', 'शुक्रवार', 'शनिवार', 'रविवार']
            }
        }

    def _expand_datetime(self, text: str) -> str:
        """Expand date and time expressions"""
        # Date patterns: DD/MM/YYYY, DD-MM-YYYY
        date_pattern = r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b'

        def replace_date(match):
            day, month, year = match.groups()
            day_word = self._number_to_words(int(day))
            month_word = self.date_time_rules.get('months', [])[int(month) - 1] if int(month) <= 12 else month
            year_word = self._number_to_words(int(year))

            if self.language_code == 'hi':
                return f"{day_word} {month_word} {year_word}"
            else:
                return f"{day_word} {month_word} {year_word}"

        text = re.sub(date_pattern, replace_date, text)

        return text

    def _expand_currency_measurements(self, text: str) -> str:
        """Expand currency and measurement expressions"""
        currency_map = {
            '₹': 'रुपये',
            '#': 'डॉलर',
            '€': 'यूरो',
            'Rs.': 'रुपये',
            'USD': 'अमेरिकी डॉलर'
        }

        for symbol, word in currency_map.items():
            # Handle currency before numbers
            pattern = re.escape(symbol) + r'\s*(\d+)'
            text = re.sub(pattern, lambda m: f"{self._number_to_words(int(m.group(1)))} {word}", text)

        # Measurements
        measurement_map = {
            'kg': 'किलोग्राम',
            'km': 'किलोमीटर',
            'cm': 'सेंटीमीटर',
            'mm': 'मिलीमीटर',
            'ml': 'मिलीलीटर',
            'l': 'लीटर'
        }

        for abbrev, full_form in measurement_map.items():
            pattern = r'(\d+)\s*' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, lambda m: f"{self._number_to_words(int(m.group(1)))} {full_form}", text)

        return text


class CodeSwitchDetector:
    """Detect and handle code-switching between languages"""

    def __init__(self, primary_language: str):
        self.primary_language = primary_language
        self.script_ranges = self._load_script_ranges()

    def _load_script_ranges(self) -> Dict:
        """Unicode ranges for different scripts"""
        return {
            'devanagari': (0x0900, 0x097F),
            'tamil': (0x0B80, 0x0BFF),
            'telugu': (0x0C00, 0x0C7F),
            'bengali': (0x0980, 0x09FF),
            'gujarati': (0x0A80, 0x0AFF),
            'kannada': (0x0C80, 0x0CFF),
            'malayalam': (0x0D00, 0x0D7F),
            'gurmukhi': (0x0A00, 0x0A7F),
            'odia': (0x0B00, 0x0B7F),
            'latin': (0x0000, 0x007F)
        }

    def mark_language_boundaries(self, text: str) -> str:
        """Mark language switching boundaries in text"""
        words = text.split()
        marked_words = []
        current_lang = self.primary_language

        for word in words:
            detected_lang = self._detect_word_language(word)

            if detected_lang != current_lang:
                marked_words.append(f"<lang:{detected_lang}>")
                current_lang = detected_lang

            marked_words.append(word)

        return ' '.join(marked_words)

    def _detect_word_language(self, word: str) -> str:
        """Detect language of a single word based on script"""
        if not word:
            return self.primary_language

        # Count characters in each script
        script_counts = {}

        for char in word:
            char_code = ord(char)
            detected_script = None

            for script, (start, end) in self.script_ranges.items():
                if start <= char_code <= end:
                    detected_script = script
                    break

            if detected_script:
                script_counts[detected_script] = script_counts.get(detected_script, 0) + 1

        if not script_counts:
            return self.primary_language

        # Return most frequent script
        dominant_script = max(script_counts, key=script_counts.get)

        # Map script to language
        script_to_lang = {
            'devanagari': 'hi',
            'tamil': 'ta',
            'telugu': 'te',
            'bengali': 'bn',
            'gujarati': 'gu',
            'kannada': 'kn',
            'malayalam': 'ml',
            'gurmukhi': 'pa',
            'odia': 'or',
            'latin': 'en'
        }

        return script_to_lang.get(dominant_script, self.primary_language)


# Additional helper functions for specific language implementations
def _get_telugu_phonemes():
    """Telugu-specific phoneme inventory"""
    # Implementation for Telugu phonemes
    pass


def _get_bengali_phonemes():
    """Bengali-specific phoneme inventory"""
    # Implementation for Bengali phonemes
    pass


# Add similar functions for other languages...


def main():
    """Test the linguistic processor"""
    processor = LinguisticProcessor('hi')

    # Test text normalization
    test_text = "डॉ. राम ने १२३४ रुपये में 5 kg आम खरीदे।"
    normalized = processor.text_normalizer.normalize(test_text)
    print(f"Normalized: {normalized}")

    # Test G2P
    phonemes = processor.grapheme_to_phoneme("नमस्ते")
    print(f"Phonemes: {phonemes}")

    # Test prosody
    prosodic = processor.add_prosodic_features(phonemes, "नमस्ते!")
    print(f"Prosodic: {prosodic[:5]}")  # Show first 5


if __name__ == "__main__":
    main()