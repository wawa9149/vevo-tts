
# -*- coding: utf-8 -*-
"""
Hybrid Korean Backend: g2pK → hangul_to_ipa
"""

import sys
import os
import re
from pathlib import Path
from typing import List

# Add paths
current_dir = Path(__file__).parent.parent
g2pk_path = current_dir / "g2pK"
hangul_to_ipa_path = current_dir / "hangul_to_ipa"
sys.path.insert(0, str(g2pk_path))
sys.path.insert(0, str(hangul_to_ipa_path))

try:
    from g2pk import G2p
    from src.worker import convert
    HYBRID_AVAILABLE = True
except ImportError as e:
    HYBRID_AVAILABLE = False
    print(f"Warning: Hybrid backend not available: {e}")

try:
    from phonemizer.separator import Separator
    from phonemizer.punctuation import Punctuation
except ImportError:
    # Mock classes
    class Separator:
        def __init__(self, word="_", syllable="-", phone="|"):
            self.word = word
            self.syllable = syllable
            self.phone = phone
    
    class Punctuation:
        @staticmethod
        def default_marks():
            return ".,!?;:~"


class HybridKoreanBackend:
    """Hybrid Korean backend: g2pK pronunciation + hangul_to_ipa conversion"""
    
    def __init__(
        self,
        language="ko",
        punctuation_marks=Punctuation.default_marks(),
        preserve_punctuation=True,
        **kwargs
    ):
        if language != "ko":
            raise ValueError(f"Hybrid backend only supports Korean (ko), got {language}")
        
        if not HYBRID_AVAILABLE:
            raise ImportError("g2pK and hangul_to_ipa libraries are required")
        
        self.language = language
        self.punctuation_marks = punctuation_marks + "~"
        self.preserve_punctuation = preserve_punctuation
        
        # Initialize g2pK for pronunciation rules
        self.g2p = G2p()
        
    def phonemize(
        self, 
        text: List[str], 
        separator: Separator, 
        strip: bool = True, 
        njobs: int = 1
    ) -> List[str]:
        """Convert Korean text to IPA using g2pK → hangul_to_ipa pipeline"""
        
        if isinstance(text, str):
            text = [text]
            
        phonemized = []
        
        for line in text:
            if strip:
                line = line.strip()
            
            if not line:
                phonemized.append("")
                continue
                
            try:
                # ✅ Step 1: 단어 + 구두점 단위로 분리
                tokens = re.findall(rf"[{re.escape(self.punctuation_marks)}]|\w+", line)


                word_ipa_results = []
                for token in tokens:
                    if token in self.punctuation_marks:
                        # ✅ Step 2: 구두점은 '_'로 치환
                        word_ipa_results.append("_")
                    else:
                        # ✅ Step 3: 단어 → g2p → IPA 변환
                        word_pronunciation = self.g2p(token)
                        word_ipa = convert(
                            hangul=word_pronunciation,
                            rules_to_apply="pastcnhovr",
                            convention="ipa",
                            sep=" "
                        )
                        if word_ipa:
                            word_phonemes = word_ipa.split()
                            word_result = separator.phone.join(word_phonemes)
                            word_ipa_results.append(word_result)

                # ✅ Step 4: word & punctuation join
                phones_str = ""
                for item in word_ipa_results:
                    if item == "_":  # 구두점이 '_'로 치환된 경우
                        phones_str += separator.phone + "_"
                    else:
                        if phones_str:  # 앞에 내용 있으면 word separator
                            phones_str += separator.phone + separator.word + separator.phone
                        phones_str += item

                # ✅ Step 5: 연속된 '_' 제거 (|_|_ -> |_)
                # 연속된 |_|_|_ 패턴을 |_로 치환
                pattern = f"({re.escape(separator.phone)}_)+"
                phones_str = re.sub(pattern, f"{separator.phone}_", phones_str)

                phonemized.append(phones_str)
                
            except Exception as e:
                print(f"Warning: Hybrid conversion failed for '{line}': {e}")
                # Fallback to direct hangul_to_ipa (word by word)
                try:
                    words = line.split()
                    fallback_word_results = []
                    
                    for word in words:
                        word_ipa = convert(word, convention='ipa', sep=' ')
                        if word_ipa:
                            word_phonemes = word_ipa.split()
                            word_result = separator.phone.join(word_phonemes)
                            fallback_word_results.append(word_result)
                    
                    if fallback_word_results:
                        word_sep = separator.phone + separator.word + separator.phone
                        phones_str = word_sep.join(fallback_word_results)
                        print(f"DEBUG phones_str: {phones_str}")
                    else:
                        phones_str = ""
                    phonemized.append(phones_str)
                except:
                    # Last resort: process character by character with space handling
                    fallback_list = []
                    for char in line:
                        if char == ' ':  # Convert space to word separator
                            fallback_list.append(separator.word)
                        elif char.strip():  # Add non-empty characters
                            fallback_list.append(char)
                    phones_str = separator.phone.join(fallback_list)
                    phonemized.append(phones_str)
        
        return phonemized
    
    def version(self) -> str:
        return "hybrid-g2pk-ipa-1.0"
    
    def supported_languages(self) -> List[str]:
        return ["ko"]


def test_hybrid_backend():
    """Test the hybrid backend"""
    print("=== Hybrid Backend Test ===")
    
    if not HYBRID_AVAILABLE:
        print("❌ Cannot test - required libraries not available")
        return False
    
    class MockSeparator:
        def __init__(self):
            self.phone = "|"
            self.word = "_"
            self.syllable = "-"
    
    backend = HybridKoreanBackend(language="ko")
    separator = MockSeparator()
    
    test_texts = [
        "좋다",
        "있다", 
        "한국어",
        "안녕하세요"
    ]
    
    for text in test_texts:
        try:
            result = backend.phonemize([text], separator)
            print(f"'{text}' → '{result[0]}'")
        except Exception as e:
            print(f"✗ Failed: {e}")
            return False
    
    print("✓ Hybrid backend test successful")
    return True


if __name__ == "__main__":
    test_hybrid_backend()
