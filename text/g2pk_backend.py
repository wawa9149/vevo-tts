# -*- coding: utf-8 -*-
# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import re
from typing import List

# Add the g2pK directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
g2pk_dir = os.path.join(os.path.dirname(current_dir), "g2pK")
sys.path.insert(0, g2pk_dir)

try:
    from g2pk import G2p
    G2PK_AVAILABLE = True
except ImportError as e:
    G2PK_AVAILABLE = False
    print(f"Warning: g2pK not available: {e}")

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


class G2pKBackend:
    """g2pK backend for Korean G2P conversion."""
    
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
        """Initialize g2pK backend.
        
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
            raise ValueError(f"g2pK backend only supports Korean (ko), got {language}")
        
        if not G2PK_AVAILABLE:
            raise ImportError("g2pK is not available. Please install required dependencies.")
        
        self.language = language
        self.punctuation_marks = punctuation_marks
        self.preserve_punctuation = preserve_punctuation
        
        # Initialize g2pK converter
        self.g2p = G2p()
        
    def phonemize(
        self, 
        text: List[str], 
        separator: Separator, 
        strip: bool = True, 
        njobs: int = 1
    ) -> List[str]:
        """Convert text to phonemes using g2pK.
        
        Args:
            text: List of text strings to convert
            separator: Phonemizer separator object
            strip: Whether to strip whitespace
            njobs: Number of jobs (not used, kept for compatibility)
            
        Returns:
            List of phonemized strings
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
                # Convert using g2pK
                phones = self.g2p(line)
                
                # Convert to individual phonemes and add separators
                phone_list = []
                for char in phones:
                    if char.strip():  # Skip empty characters
                        phone_list.append(char)
                
                # Join with phone separator
                if phone_list:
                    phones_str = separator.phone.join(phone_list)
                else:
                    phones_str = ""
                
                # Handle punctuation if preserve_punctuation is True
                if self.preserve_punctuation:
                    # Keep punctuation marks as they are
                    for punct in self.punctuation_marks:
                        if punct in line:
                            phones_str = phones_str + separator.phone + punct
                
                phonemized.append(phones_str)
                
            except Exception as e:
                print(f"Warning: g2pK conversion failed for '{line}': {e}")
                # Fallback: return original text with phone separators
                phones_str = separator.phone.join(list(line))
                phonemized.append(phones_str)
        
        return phonemized
    
    def version(self) -> str:
        """Return backend version."""
        return "g2pK-1.0"
    
    def supported_languages(self) -> List[str]:
        """Return list of supported languages."""
        return ["ko"] 