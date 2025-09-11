# -*- coding: utf-8 -*-
"""
Korean IPA Backend using hangul_to_ipa library
"""

import os
import sys
from pathlib import Path
from typing import List

# Add hangul_to_ipa to path
current_dir = Path(__file__).parent.parent
hangul_to_ipa_path = current_dir / "hangul_to_ipa"
sys.path.insert(0, str(hangul_to_ipa_path))

try:
    from src.worker import convert
    HANGUL_TO_IPA_AVAILABLE = True
except ImportError as e:
    HANGUL_TO_IPA_AVAILABLE = False
    print(f"Warning: hangul_to_ipa not available: {e}")

try:
    from phonemizer.separator import Separator
    from phonemizer.punctuation import Punctuation
except ImportError:
    # Mock classes if phonemizer is not available
    class Separator:
        def __init__(self, word="_", syllable="-", phone="|"):
            self.word = word
            self.syllable = syllable
            self.phone = phone
    
    class Punctuation:
        @staticmethod
        def default_marks():
            return ".,!?;:"


class KoreanIPABackend:
    """Korean IPA backend using hangul_to_ipa library"""
    
    def __init__(
        self,
        language="ko",
        punctuation_marks=Punctuation.default_marks(),
        preserve_punctuation=True,
        with_stress=False,
        tie=False,
        language_switch="keep-flags",
        words_mismatch="ignore",
    ):
        """Initialize Korean IPA backend.
        
        Args:
            language: Language code (should be "ko" for Korean)
            punctuation_marks: Punctuation marks to preserve
            preserve_punctuation: Whether to preserve punctuation
            with_stress: Whether to include stress markers (not used for Korean)
            tie: Whether to use tie markers (not used for Korean)
            language_switch: How to handle language switches
            words_mismatch: How to handle word mismatches
        """
        if language != "ko":
            raise ValueError(f"Korean IPA backend only supports Korean (ko), got {language}")
        
        if not HANGUL_TO_IPA_AVAILABLE:
            raise ImportError("hangul_to_ipa library is not available")
        
        self.language = language
        self.punctuation_marks = punctuation_marks
        self.preserve_punctuation = preserve_punctuation
        
    def phonemize(
        self, 
        text: List[str], 
        separator: Separator, 
        strip: bool = True, 
        njobs: int = 1
    ) -> List[str]:
        """Convert Korean text to IPA phonemes.
        
        Args:
            text: List of text strings to convert
            separator: Phonemizer separator object
            strip: Whether to strip whitespace
            njobs: Number of jobs (not used, kept for compatibility)
            
        Returns:
            List of phonemized strings in IPA
        """
        if isinstance(text, str):
            text = [text]
            
        phonemized = []
        
        for line in text:
            if strip:
                line = line.strip()
            
            # Handle empty lines
            if not line:
                phonemized.append("")
                continue
                
            try:
                # Convert using hangul_to_ipa
                # Apply all phonological rules for natural pronunciation
                ipa_result = convert(
                    hangul=line,
                    rules_to_apply='pastcnhovr',  # All rules
                    convention='ipa',
                    sep=' '  # Space separator for splitting
                )
                
                # Split IPA result into individual phonemes
                if ipa_result:
                    ipa_phonemes = ipa_result.split()
                    # Join with phone separator
                    phones_str = separator.phone.join(ipa_phonemes)
                else:
                    phones_str = ""
                
                # Handle punctuation if preserve_punctuation is True
                if self.preserve_punctuation:
                    for punct in self.punctuation_marks:
                        if punct in line:
                            phones_str = phones_str + separator.phone + punct
                
                phonemized.append(phones_str)
                
            except Exception as e:
                print(f"Warning: Korean IPA conversion failed for '{line}': {e}")
                # Fallback: return original text with phone separators
                phones_str = separator.phone.join(list(line))
                phonemized.append(phones_str)
        
        return phonemized
    
    def version(self) -> str:
        """Return backend version."""
        return "hangul_to_ipa-1.0"
    
    def supported_languages(self) -> List[str]:
        """Return list of supported languages."""
        return ["ko"]


def test_korean_ipa_backend():
    """Test the Korean IPA backend"""
    
    print("=== Korean IPA Backend Test ===")
    
    if not HANGUL_TO_IPA_AVAILABLE:
        print("❌ Cannot test - hangul_to_ipa not available")
        return False
    
    # Mock separator for testing
    class MockSeparator:
        def __init__(self):
            self.phone = "|"
            self.word = "_"
            self.syllable = "-"
    
    backend = KoreanIPABackend(language="ko")
    separator = MockSeparator()
    
    test_texts = [
        "안녕하세요",
        "한국어 음성 합성",
        "사랑해요",
        "굳이?"
    ]
    
    for text in test_texts:
        try:
            result = backend.phonemize([text], separator)
            print(f"'{text}' → '{result[0]}'")
            
            # Also show direct hangul_to_ipa output for comparison
            direct_result = convert(text, convention='ipa', sep=' ')
            print(f"  (직접 변환: {direct_result})")
            
        except Exception as e:
            print(f"✗ Failed to convert '{text}': {e}")
            return False
    
    print("✓ Korean IPA backend test successful")
    return True


if __name__ == "__main__":
    test_korean_ipa_backend() 