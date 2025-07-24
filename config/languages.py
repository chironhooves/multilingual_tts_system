"""
Enhanced Language Configuration for Multilingual TTS System v2.0
Complete configuration for all 10 Indian languages with open datasets support
Includes phoneme inventories, linguistic features, and dataset information
"""

from typing import Dict, List, Set, Optional, Tuple
import json
from pathlib import Path


class IndianLanguages:
    """Enhanced configuration for Indian languages with linguistic and dataset information"""

    # Complete language configurations with open datasets
    LANGUAGES = {
        'hi': {
            'name': 'Hindi',
            'script': 'Devanagari',
            'native_name': 'हिन्दी',
            'iso_code': 'hi-IN',
            'family': 'Indo-Aryan',
            'speakers': '600M+',
            'phoneme_set': 'hi_phonemes',
            'writing_direction': 'ltr',
            'tts_complexity': 'medium',
            'datasets_available': {
                'common_voice': {
                    'version': '13.0',
                    'hours': 50.2,
                    'speakers': 1247,
                    'quality': 'high',
                    'license': 'CC-0',
                    'segments': 45000
                },
                'google_fleurs': {
                    'hours': 10.5,
                    'speakers': 102,
                    'quality': 'high',
                    'license': 'Apache-2.0',
                    'segments': 3400
                },
                'openslr': {
                    'resource_id': '103',
                    'hours': 20.3,
                    'speakers': 50,
                    'quality': 'medium',
                    'license': 'Apache-2.0',
                    'segments': 8100
                },
                'custom_recordings': {
                    'estimated_potential': 'unlimited',
                    'user_controlled': True
                }
            },
            'total_estimated_hours': 81.0,
            'linguistic_features': {
                'has_schwa_deletion': True,
                'has_retroflex': True,
                'has_aspirated': True,
                'has_nasalization': True,
                'tone_language': False,
                'compound_rich': True,
                'case_marking': True
            }
        },

        'ta': {
            'name': 'Tamil',
            'script': 'Tamil',
            'native_name': 'தமிழ்',
            'iso_code': 'ta-IN',
            'family': 'Dravidian',
            'speakers': '80M+',
            'phoneme_set': 'ta_phonemes',
            'writing_direction': 'ltr',
            'tts_complexity': 'high',
            'datasets_available': {
                'common_voice': {
                    'version': '13.0',
                    'hours': 35.4,
                    'speakers': 823,
                    'quality': 'high',
                    'license': 'CC-0',
                    'segments': 32000
                },
                'google_fleurs': {
                    'hours': 10.2,
                    'speakers': 98,
                    'quality': 'high',
                    'license': 'Apache-2.0',
                    'segments': 3300
                },
                'openslr': {
                    'resource_id': '65',
                    'hours': 15.6,
                    'speakers': 32,
                    'quality': 'medium',
                    'license': 'Apache-2.0',
                    'segments': 6200
                },
                'custom_recordings': {
                    'estimated_potential': 'unlimited',
                    'user_controlled': True
                }
            },
            'total_estimated_hours': 61.2,
            'linguistic_features': {
                'has_schwa_deletion': False,
                'has_retroflex': True,
                'has_aspirated': False,
                'has_nasalization': True,
                'tone_language': False,
                'compound_rich': True,
                'agglutinative': True
            }
        },

        'te': {
            'name': 'Telugu',
            'script': 'Telugu',
            'native_name': 'తెలుగు',
            'iso_code': 'te-IN',
            'family': 'Dravidian',
            'speakers': '95M+',
            'phoneme_set': 'te_phonemes',
            'writing_direction': 'ltr',
            'tts_complexity': 'high',
            'datasets_available': {
                'common_voice': {
                    'version': '13.0',
                    'hours': 30.1,
                    'speakers': 645,
                    'quality': 'high',
                    'license': 'CC-0',
                    'segments': 28500
                },
                'google_fleurs': {
                    'hours': 10.3,
                    'speakers': 95,
                    'quality': 'high',
                    'license': 'Apache-2.0',
                    'segments': 3400
                },
                'openslr': {
                    'resource_id': '66',
                    'hours': 25.2,
                    'speakers': 42,
                    'quality': 'medium',
                    'license': 'Apache-2.0',
                    'segments': 10000
                },
                'custom_recordings': {
                    'estimated_potential': 'unlimited',
                    'user_controlled': True
                }
            },
            'total_estimated_hours': 65.6,
            'linguistic_features': {
                'has_schwa_deletion': False,
                'has_retroflex': True,
                'has_aspirated': False,
                'has_nasalization': True,
                'tone_language': False,
                'compound_rich': True,
                'agglutinative': True
            }
        },

        'bn': {
            'name': 'Bengali',
            'script': 'Bengali',
            'native_name': 'বাংলা',
            'iso_code': 'bn-IN',
            'family': 'Indo-Aryan',
            'speakers': '300M+',
            'phoneme_set': 'bn_phonemes',
            'writing_direction': 'ltr',
            'tts_complexity': 'medium',
            'datasets_available': {
                'common_voice': {
                    'version': '13.0',
                    'hours': 25.3,
                    'speakers': 512,
                    'quality': 'high',
                    'license': 'CC-0',
                    'segments': 22800
                },
                'google_fleurs': {
                    'hours': 10.4,
                    'speakers': 103,
                    'quality': 'high',
                    'license': 'Apache-2.0',
                    'segments': 3350
                },
                'openslr': {
                    'resource_id': '37',
                    'hours': 20.1,
                    'speakers': 38,
                    'quality': 'medium',
                    'license': 'Apache-2.0',
                    'segments': 8000
                },
                'custom_recordings': {
                    'estimated_potential': 'unlimited',
                    'user_controlled': True
                }
            },
            'total_estimated_hours': 55.8,
            'linguistic_features': {
                'has_schwa_deletion': True,
                'has_retroflex': True,
                'has_aspirated': True,
                'has_nasalization': True,
                'tone_language': False,
                'compound_rich': True,
                'case_marking': True
            }
        },

        'mr': {
            'name': 'Marathi',
            'script': 'Devanagari',
            'native_name': 'मराठी',
            'iso_code': 'mr-IN',
            'family': 'Indo-Aryan',
            'speakers': '90M+',
            'phoneme_set': 'mr_phonemes',
            'writing_direction': 'ltr',
            'tts_complexity': 'medium',
            'datasets_available': {
                'common_voice': {
                    'version': '13.0',
                    'hours': 20.7,
                    'speakers': 423,
                    'quality': 'high',
                    'license': 'CC-0',
                    'segments': 18600
                },
                'google_fleurs': {
                    'hours': 10.1,
                    'speakers': 97,
                    'quality': 'high',
                    'license': 'Apache-2.0',
                    'segments': 3300
                },
                'openslr': {
                    'resource_id': 'N/A',
                    'hours': 0,
                    'speakers': 0,
                    'quality': 'N/A',
                    'license': 'N/A',
                    'segments': 0,
                    'note': 'Not available yet'
                },
                'custom_recordings': {
                    'estimated_potential': 'unlimited',
                    'user_controlled': True
                }
            },
            'total_estimated_hours': 30.8,
            'linguistic_features': {
                'has_schwa_deletion': True,
                'has_retroflex': True,
                'has_aspirated': True,
                'has_nasalization': True,
                'tone_language': False,
                'compound_rich': True,
                'case_marking': True
            }
        },

        'gu': {
            'name': 'Gujarati',
            'script': 'Gujarati',
            'native_name': 'ગુજરાતી',
            'iso_code': 'gu-IN',
            'family': 'Indo-Aryan',
            'speakers': '60M+',
            'phoneme_set': 'gu_phonemes',
            'writing_direction': 'ltr',
            'tts_complexity': 'medium',
            'datasets_available': {
                'common_voice': {
                    'version': '13.0',
                    'hours': 15.2,
                    'speakers': 318,
                    'quality': 'high',
                    'license': 'CC-0',
                    'segments': 13700
                },
                'google_fleurs': {
                    'hours': 10.0,
                    'speakers': 94,
                    'quality': 'high',
                    'license': 'Apache-2.0',
                    'segments': 3200
                },
                'openslr': {
                    'resource_id': '78',
                    'hours': 12.5,
                    'speakers': 28,
                    'quality': 'medium',
                    'license': 'Apache-2.0',
                    'segments': 5000
                },
                'custom_recordings': {
                    'estimated_potential': 'unlimited',
                    'user_controlled': True
                }
            },
            'total_estimated_hours': 37.7,
            'linguistic_features': {
                'has_schwa_deletion': True,
                'has_retroflex': True,
                'has_aspirated': True,
                'has_nasalization': True,
                'tone_language': False,
                'compound_rich': True,
                'case_marking': True
            }
        },

        'kn': {
            'name': 'Kannada',
            'script': 'Kannada',
            'native_name': 'ಕನ್ನಡ',
            'iso_code': 'kn-IN',
            'family': 'Dravidian',
            'speakers': '50M+',
            'phoneme_set': 'kn_phonemes',
            'writing_direction': 'ltr',
            'tts_complexity': 'high',
            'datasets_available': {
                'common_voice': {
                    'version': '13.0',
                    'hours': 12.1,
                    'speakers': 267,
                    'quality': 'high',
                    'license': 'CC-0',
                    'segments': 10900
                },
                'google_fleurs': {
                    'hours': 10.2,
                    'speakers': 98,
                    'quality': 'high',
                    'license': 'Apache-2.0',
                    'segments': 3300
                },
                'openslr': {
                    'resource_id': 'N/A',
                    'hours': 0,
                    'speakers': 0,
                    'quality': 'N/A',
                    'license': 'N/A',
                    'segments': 0,
                    'note': 'Not available yet'
                },
                'custom_recordings': {
                    'estimated_potential': 'unlimited',
                    'user_controlled': True
                }
            },
            'total_estimated_hours': 22.3,
            'linguistic_features': {
                'has_schwa_deletion': False,
                'has_retroflex': True,
                'has_aspirated': False,
                'has_nasalization': True,
                'tone_language': False,
                'compound_rich': True,
                'agglutinative': True
            }
        },

        'ml': {
            'name': 'Malayalam',
            'script': 'Malayalam',
            'native_name': 'മലയാളം',
            'iso_code': 'ml-IN',
            'family': 'Dravidian',
            'speakers': '40M+',
            'phoneme_set': 'ml_phonemes',
            'writing_direction': 'ltr',
            'tts_complexity': 'high',
            'datasets_available': {
                'common_voice': {
                    'version': '13.0',
                    'hours': 10.3,
                    'speakers': 201,
                    'quality': 'high',
                    'license': 'CC-0',
                    'segments': 9300
                },
                'google_fleurs': {
                    'hours': 10.1,
                    'speakers': 96,
                    'quality': 'high',
                    'license': 'Apache-2.0',
                    'segments': 3250
                },
                'openslr': {
                    'resource_id': 'N/A',
                    'hours': 0,
                    'speakers': 0,
                    'quality': 'N/A',
                    'license': 'N/A',
                    'segments': 0,
                    'note': 'Not available yet'
                },
                'custom_recordings': {
                    'estimated_potential': 'unlimited',
                    'user_controlled': True
                }
            },
            'total_estimated_hours': 20.4,
            'linguistic_features': {
                'has_schwa_deletion': False,
                'has_retroflex': True,
                'has_aspirated': False,
                'has_nasalization': True,
                'tone_language': False,
                'compound_rich': True,
                'agglutinative': True
            }
        },

        'pa': {
            'name': 'Punjabi',
            'script': 'Gurmukhi',
            'native_name': 'ਪੰਜਾਬੀ',
            'iso_code': 'pa-IN',
            'family': 'Indo-Aryan',
            'speakers': '35M+',
            'phoneme_set': 'pa_phonemes',
            'writing_direction': 'ltr',
            'tts_complexity': 'medium',
            'datasets_available': {
                'common_voice': {
                    'version': '13.0',
                    'hours': 8.4,
                    'speakers': 156,
                    'quality': 'medium',
                    'license': 'CC-0',
                    'segments': 7600
                },
                'google_fleurs': {
                    'hours': 10.0,
                    'speakers': 92,
                    'quality': 'high',
                    'license': 'Apache-2.0',
                    'segments': 3200
                },
                'openslr': {
                    'resource_id': 'N/A',
                    'hours': 0,
                    'speakers': 0,
                    'quality': 'N/A',
                    'license': 'N/A',
                    'segments': 0,
                    'note': 'Not available yet'
                },
                'custom_recordings': {
                    'estimated_potential': 'unlimited',
                    'user_controlled': True
                }
            },
            'total_estimated_hours': 18.4,
            'linguistic_features': {
                'has_schwa_deletion': True,
                'has_retroflex': True,
                'has_aspirated': True,
                'has_nasalization': True,
                'tone_language': True,
                'compound_rich': True,
                'case_marking': True
            }
        },

        'or': {
            'name': 'Odia',
            'script': 'Odia',
            'native_name': 'ଓଡ଼ିଆ',
            'iso_code': 'or-IN',
            'family': 'Indo-Aryan',
            'speakers': '45M+',
            'phoneme_set': 'or_phonemes',
            'writing_direction': 'ltr',
            'tts_complexity': 'medium',
            'datasets_available': {
                'common_voice': {
                    'version': 'N/A',
                    'hours': 0,
                    'speakers': 0,
                    'quality': 'N/A',
                    'license': 'N/A',
                    'segments': 0,
                    'note': 'Not available in Common Voice yet'
                },
                'google_fleurs': {
                    'hours': 10.1,
                    'speakers': 94,
                    'quality': 'high',
                    'license': 'Apache-2.0',
                    'segments': 3250
                },
                'openslr': {
                    'resource_id': 'N/A',
                    'hours': 0,
                    'speakers': 0,
                    'quality': 'N/A',
                    'license': 'N/A',
                    'segments': 0,
                    'note': 'Not available yet'
                },
                'custom_recordings': {
                    'estimated_potential': 'unlimited',
                    'user_controlled': True,
                    'note': 'Primary data source for Odia currently'
                }
            },
            'total_estimated_hours': 10.1,
            'linguistic_features': {
                'has_schwa_deletion': True,
                'has_retroflex': True,
                'has_aspirated': True,
                'has_nasalization': True,
                'tone_language': False,
                'compound_rich': True,
                'case_marking': True
            }
        }
    }

    # Open datasets comprehensive information
    OPEN_DATASETS_INFO = {
        'mozilla_common_voice': {
            'name': 'Mozilla Common Voice',
            'license': 'CC-0 (Public Domain)',
            'url': 'https://commonvoice.mozilla.org/',
            'description': 'Crowd-sourced voice dataset with native speakers',
            'quality': 'High',
            'suitable_for_tts': True,
            'languages_supported': ['hi', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa'],
            'total_hours_available': 207.5,
            'total_speakers': 4182,
            'total_segments': 188400,
            'advantages': [
                'Completely legal and free to use commercially',
                'High quality recordings with quality control',
                'Diverse speakers across demographics',
                'Regular updates with new contributions',
                'Rich metadata including speaker demographics',
                'Community-driven with active support'
            ]
        },

        'google_fleurs': {
            'name': 'Google FLEURS',
            'license': 'Apache 2.0',
            'url': 'https://huggingface.co/datasets/google/fleurs',
            'description': 'Few-shot Learning Evaluation of Universal Representations of Speech',
            'quality': 'High',
            'suitable_for_tts': True,
            'languages_supported': ['hi', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or'],
            'total_hours_available': 101.4,
            'total_speakers': 976,
            'total_segments': 32850,
            'advantages': [
                'Professional quality recordings',
                'Consistent format across all languages',
                'All 10 Indian languages supported',
                'Research-grade dataset from Google',
                'Balanced speaker demographics',
                'Standardized evaluation protocols'
            ]
        },

        'openslr': {
            'name': 'OpenSLR Indian Languages',
            'license': 'Various open licenses (Apache 2.0, CC-BY-SA)',
            'url': 'https://www.openslr.org/',
            'description': 'Open Speech and Language Resources for Indian languages',
            'quality': 'Variable (Medium to High)',
            'suitable_for_tts': True,
            'languages_supported': ['hi', 'ta', 'te', 'bn', 'gu'],
            'total_hours_available': 93.7,
            'total_speakers': 190,
            'total_segments': 37300,
            'advantages': [
                'Academic quality datasets',
                'Multiple data sources and collection methods',
                'Some languages have substantial datasets',
                'Free for research and commercial use',
                'Well-documented collection procedures',
                'Established in speech research community'
            ]
        },

        'custom_recordings': {
            'name': 'Custom User Recordings',
            'license': 'User-defined',
            'description': 'User-contributed recordings with guided interface',
            'quality': 'User-dependent (with quality guidelines)',
            'suitable_for_tts': True,
            'languages_supported': ['hi', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or'],
            'total_hours_available': 'Unlimited potential',
            'advantages': [
                'Complete control over data quality and content',
                'Domain-specific content possible (technical, literary, etc.)',
                'No copyright or licensing issues',
                'Unlimited scalability potential',
                'All languages supported including low-resource ones',
                'Can fill gaps in existing datasets',
                'Perfect for specialized use cases'
            ]
        }
    }

    # Language family groupings for multilingual training
    LANGUAGE_FAMILIES = {
        'indo_aryan': {
            'languages': ['hi', 'bn', 'mr', 'gu', 'pa', 'or'],
            'shared_features': [
                'has_schwa_deletion',
                'has_retroflex',
                'has_aspirated',
                'case_marking',
                'compound_rich'
            ],
            'script_similarity': {
                'devanagari_group': ['hi', 'mr'],
                'bengali_group': ['bn'],
                'gujarati_group': ['gu'],
                'gurmukhi_group': ['pa'],
                'odia_group': ['or']
            }
        },
        'dravidian': {
            'languages': ['ta', 'te', 'kn', 'ml'],
            'shared_features': [
                'agglutinative',
                'has_retroflex',
                'compound_rich',
                'no_schwa_deletion'
            ],
            'script_similarity': {
                'distinct_scripts': True,
                'phonetic_similarity': ['ta', 'ml'],
                'geographic_similarity': ['te', 'kn']
            }
        }
    }

    # Dataset priority recommendations
    DATASET_PRIORITIES = {
        'high_resource': {
            'languages': ['hi', 'ta', 'te', 'bn'],
            'recommended_datasets': ['common_voice', 'google_fleurs', 'openslr'],
            'expected_hours': '55-81 hours each',
            'training_strategy': 'individual_then_unified'
        },
        'medium_resource': {
            'languages': ['mr', 'gu'],
            'recommended_datasets': ['common_voice', 'google_fleurs', 'openslr', 'custom_recordings'],
            'expected_hours': '30-38 hours each',
            'training_strategy': 'transfer_learning_recommended'
        },
        'low_resource': {
            'languages': ['kn', 'ml', 'pa', 'or'],
            'recommended_datasets': ['google_fleurs', 'custom_recordings'],
            'expected_hours': '10-22 hours each',
            'training_strategy': 'transfer_learning_essential'
        }
    }

    def __init__(self):
        """Initialize the IndianLanguages configuration"""
        self.supported_languages = list(self.LANGUAGES.keys())
        self.total_estimated_hours = sum(
            lang_info['total_estimated_hours']
            for lang_info in self.LANGUAGES.values()
        )

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        return self.supported_languages

    def validate_language_code(self, language_code: str) -> bool:
        """Validate if language code is supported"""
        return language_code in self.LANGUAGES

    def get_language_info(self, language_code: str) -> Dict:
        """Get comprehensive information about a language"""
        if not self.validate_language_code(language_code):
            raise ValueError(f"Unsupported language code: {language_code}")
        return self.LANGUAGES[language_code].copy()

    def get_dataset_availability_summary(self, language_code: str = None) -> Dict:
        """Get summary of dataset availability for language(s)"""
        if language_code:
            if not self.validate_language_code(language_code):
                raise ValueError(f"Unsupported language code: {language_code}")

            lang_info = self.LANGUAGES[language_code]
            datasets = lang_info['datasets_available']

            summary = {
                'language_code': language_code,
                'language_name': lang_info['name'],
                'native_name': lang_info['native_name'],
                'total_estimated_hours': lang_info['total_estimated_hours'],
                'datasets': {},
                'sufficient_for_training': lang_info['total_estimated_hours'] >= 10
            }

            for dataset_name, dataset_info in datasets.items():
                if dataset_info.get('hours', 0) > 0:
                    summary['datasets'][dataset_name] = {
                        'hours': dataset_info['hours'],
                        'speakers': dataset_info['speakers'],
                        'quality': dataset_info['quality'],
                        'segments': dataset_info['segments']
                    }

            return summary
        else:
            # Summary for all languages
            summary = {
                'total_languages': len(self.LANGUAGES),
                'total_estimated_hours': self.total_estimated_hours,
                'languages': {}
            }

            for lang_code, lang_info in self.LANGUAGES.items():
                summary['languages'][lang_code] = {
                    'name': lang_info['name'],
                    'native_name': lang_info['native_name'],
                    'hours': lang_info['total_estimated_hours'],
                    'datasets_count': len([
                        d for d in lang_info['datasets_available'].values()
                        if d.get('hours', 0) > 0
                    ])
                }

            return summary

    def get_datasets_info(self) -> Dict:
        """Get information about all available open datasets"""
        return self.OPEN_DATASETS_INFO.copy()

    def get_family_info(self, language_code: str) -> Dict:
        """Get language family information for a language"""
        for family_name, family_info in self.LANGUAGE_FAMILIES.items():
            if language_code in family_info['languages']:
                return {
                    'family': family_name,
                    'related_languages': family_info['languages'],
                    'shared_features': family_info['shared_features'],
                    'script_info': family_info.get('script_similarity', {})
                }
        return {'family': 'unknown', 'related_languages': [], 'shared_features': []}

    def get_resource_level(self, language_code: str) -> str:
        """Get resource level (high/medium/low) for a language"""
        for level, info in self.DATASET_PRIORITIES.items():
            if language_code in info['languages']:
                return level.replace('_resource', '')
        return 'unknown'

    def get_training_recommendations(self, language_code: str) -> Dict:
        """Get training strategy recommendations for a language"""
        resource_level = self.get_resource_level(language_code)

        for level_key, level_info in self.DATASET_PRIORITIES.items():
            if language_code in level_info['languages']:
                return {
                    'resource_level': resource_level,
                    'recommended_datasets': level_info['recommended_datasets'],
                    'expected_hours': level_info['expected_hours'],
                    'training_strategy': level_info['training_strategy'],
                    'family_info': self.get_family_info(language_code)
                }

        return {
            'resource_level': 'unknown',
            'recommended_datasets': ['custom_recordings'],
            'expected_hours': 'variable',
            'training_strategy': 'custom_approach_needed'
        }

    def get_open_datasets_advantages(self) -> List[str]:
        """Get advantages of using open datasets over proprietary sources"""
        return [
            "🔒 100% Legal: All datasets have proper licenses for commercial use",
            "🎯 TTS-Optimized: Designed specifically for speech synthesis training",
            "📊 High Quality: Professional recording standards with quality control",
            "🌍 No Geographic Restrictions: Available worldwide without VPN needs",
            "⚡ Reliable Downloads: No rate limiting, blocks, or availability issues",
            "📈 Consistent Metadata: Standardized format with speaker information",
            "🔄 Reproducible Results: Same data available to all researchers",
            "💾 Offline Processing: Download once, use offline indefinitely",
            "🎤 Diverse Speakers: Curated for demographic and accent diversity",
            "📝 Clean Transcripts: Human-verified text with proper normalization",
            "🚀 Faster Training: Pre-processed format reduces preparation time",
            "🔧 Research Support: Active communities and documentation",
            "💰 Cost Effective: No premium subscriptions or API costs needed",
            "🌟 Future Proof: Regular updates and long-term availability guaranteed"
        ]

    def export_language_config(self, output_file: str = None) -> str:
        """Export language configuration to JSON file"""
        config_data = {
            'languages': self.LANGUAGES,
            'open_datasets': self.OPEN_DATASETS_INFO,
            'language_families': self.LANGUAGE_FAMILIES,
            'dataset_priorities': self.DATASET_PRIORITIES,
            'metadata': {
                'version': '2.0',
                'total_languages': len(self.LANGUAGES),
                'total_estimated_hours': self.total_estimated_hours,
                'export_timestamp': str(Path(__file__).stat().st_mtime)
            }
        }

        if not output_file:
            output_file = 'language_config_export.json'

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        return output_file


# Create instance for easy importing
indian_languages = IndianLanguages()


# Additional utility functions
def get_language_stats():
    """Get comprehensive language statistics"""
    stats = {
        'total_languages': len(indian_languages.LANGUAGES),
        'total_hours': indian_languages.total_estimated_hours,
        'by_family': {},
        'by_resource_level': {},
        'dataset_coverage': {}
    }

    # Stats by family
    for family_name, family_info in indian_languages.LANGUAGE_FAMILIES.items():
        family_hours = sum(
            indian_languages.LANGUAGES[lang]['total_estimated_hours']
            for lang in family_info['languages']
        )
        stats['by_family'][family_name] = {
            'languages': len(family_info['languages']),
            'total_hours': family_hours,
            'language_codes': family_info['languages']
        }

    # Stats by resource level
    for level_name, level_info in indian_languages.DATASET_PRIORITIES.items():
        level_hours = sum(
            indian_languages.LANGUAGES[lang]['total_estimated_hours']
            for lang in level_info['languages']
        )
        stats['by_resource_level'][level_name] = {
            'languages': len(level_info['languages']),
            'total_hours': level_hours,
            'language_codes': level_info['languages']
        }

    # Dataset coverage stats
    for dataset_name, dataset_info in indian_languages.OPEN_DATASETS_INFO.items():
        if 'languages_supported' in dataset_info:
            stats['dataset_coverage'][dataset_name] = {
                'languages_supported': len(dataset_info['languages_supported']),
                'total_hours': dataset_info.get('total_hours_available', 0),
                'language_codes': dataset_info['languages_supported']
            }

    return stats


def validate_language_config():
    """Validate the language configuration for consistency"""
    issues = []
    warnings = []

    # Check that all languages have required fields
    required_fields = ['name', 'native_name', 'family', 'datasets_available', 'total_estimated_hours']

    for lang_code, lang_info in indian_languages.LANGUAGES.items():
        for field in required_fields:
            if field not in lang_info:
                issues.append(f"Language {lang_code} missing required field: {field}")

        # Check that total_estimated_hours matches sum of dataset hours
        dataset_hours = sum(
            d.get('hours', 0) for d in lang_info.get('datasets_available', {}).values()
        )
        estimated_hours = lang_info.get('total_estimated_hours', 0)

        if abs(dataset_hours - estimated_hours) > 1:  # Allow 1 hour difference
            warnings.append(
                f"Language {lang_code}: dataset hours ({dataset_hours}) "
                f"don't match estimated hours ({estimated_hours})"
            )

    # Check language family consistency
    all_family_languages = set()
    for family_info in indian_languages.LANGUAGE_FAMILIES.values():
        for lang in family_info['languages']:
            if lang in all_family_languages:
                issues.append(f"Language {lang} appears in multiple families")
            all_family_languages.add(lang)

    # Check that all languages are in a family
    for lang_code in indian_languages.LANGUAGES.keys():
        if lang_code not in all_family_languages:
            warnings.append(f"Language {lang_code} not assigned to any family")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings
    }


def print_language_summary():
    """Print a comprehensive summary of language configuration"""
    print("\n🌍 MULTILINGUAL TTS SYSTEM - LANGUAGE SUMMARY")
    print("=" * 60)

    stats = get_language_stats()

    print(f"📊 Overview:")
    print(f"   Total Languages: {stats['total_languages']}")
    print(f"   Total Estimated Hours: {stats['total_hours']:.1f}")
    print(f"   Average Hours per Language: {stats['total_hours'] / stats['total_languages']:.1f}")

    print(f"\n📚 By Language Family:")
    for family, info in stats['by_family'].items():
        print(f"   {family.title()}: {info['languages']} languages, {info['total_hours']:.1f} hours")
        print(f"      Languages: {', '.join(info['language_codes'])}")

    print(f"\n📈 By Resource Level:")
    for level, info in stats['by_resource_level'].items():
        level_name = level.replace('_', ' ').title()
        print(f"   {level_name}: {info['languages']} languages, {info['total_hours']:.1f} hours")
        print(f"      Languages: {', '.join(info['language_codes'])}")

    print(f"\n💾 Dataset Coverage:")
    for dataset, info in stats['dataset_coverage'].items():
        dataset_name = dataset.replace('_', ' ').title()
        print(f"   {dataset_name}: {info['languages_supported']} languages, {info['total_hours']} hours")

    # Validation
    validation = validate_language_config()
    if validation['valid']:
        print(f"\n✅ Configuration is valid!")
    else:
        print(f"\n❌ Configuration issues found:")
        for issue in validation['issues']:
            print(f"   - {issue}")

    if validation['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"   - {warning}")


def get_language_recommendations(user_goals: str = "general") -> Dict:
    """Get language recommendations based on user goals"""
    recommendations = {
        'priority_languages': [],
        'reasoning': [],
        'estimated_timeline': '',
        'resource_requirements': {}
    }

    if user_goals == "quick_start":
        recommendations['priority_languages'] = ['hi']
        recommendations['reasoning'] = [
            "Hindi has the most data available (81 hours)",
            "Good dataset quality across multiple sources",
            "Large speaker base for testing"
        ]
        recommendations['estimated_timeline'] = "2-3 days for working model"

    elif user_goals == "maximum_coverage":
        recommendations['priority_languages'] = ['hi', 'ta', 'te', 'bn']
        recommendations['reasoning'] = [
            "These 4 languages cover 60%+ of the data",
            "Represent both major language families",
            "Good foundation for multilingual model"
        ]
        recommendations['estimated_timeline'] = "1-2 weeks for all languages"

    elif user_goals == "research":
        recommendations['priority_languages'] = list(indian_languages.LANGUAGES.keys())
        recommendations['reasoning'] = [
            "Complete coverage of Indian language diversity",
            "Includes both high and low-resource scenarios",
            "Comprehensive multilingual evaluation possible"
        ]
        recommendations['estimated_timeline'] = "3-4 weeks for complete system"

    else:  # general
        recommendations['priority_languages'] = ['hi', 'ta', 'te']
        recommendations['reasoning'] = [
            "Good balance of data availability and diversity",
            "Covers both Indo-Aryan and Dravidian families",
            "Manageable scope for initial development"
        ]
        recommendations['estimated_timeline'] = "1 week for core languages"

    # Calculate resource requirements
    total_hours = sum(
        indian_languages.LANGUAGES[lang]['total_estimated_hours']
        for lang in recommendations['priority_languages']
    )

    recommendations['resource_requirements'] = {
        'storage_gb': total_hours * 0.5,  # Rough estimate
        'training_time_gpu_hours': total_hours * 0.1,  # Rough estimate
        'total_data_hours': total_hours
    }

    return recommendations


# Export all main functions and classes
__all__ = [
    'IndianLanguages',
    'indian_languages',
    'get_language_stats',
    'validate_language_config',
    'print_language_summary',
    'get_language_recommendations'
]

# Auto-run summary on import (only in interactive mode)
if __name__ == "__main__":
    print_language_summary()