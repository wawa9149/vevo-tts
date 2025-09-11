# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from phonemizer.utils import list2str, str2list
from typing import List, Union
import os
import json
import sys
import fcntl

try:
    # Use project hybrid Korean G2P (g2pkk → jamo → IPA) when available
    from text.g2p_module import G2PModule as _KoreanG2PModule
except Exception:
    _KoreanG2PModule = None

# separator=Separator(phone=' ', word=' _ ', syllable='|'),
separator = Separator(word=" _ ", syllable="|", phone=" ")

phonemizer_zh = EspeakBackend(
    "cmn", preserve_punctuation=False, with_stress=False, language_switch="remove-flags"
)
# phonemizer_zh.separator = separator

phonemizer_en = EspeakBackend(
    "en-us",
    preserve_punctuation=False,
    with_stress=False,
    language_switch="remove-flags",
)
# phonemizer_en.separator = separator

phonemizer_ja = EspeakBackend(
    "ja", preserve_punctuation=False, with_stress=False, language_switch="remove-flags"
)
# phonemizer_ja.separator = separator

phonemizer_ko = EspeakBackend(
    "ko", preserve_punctuation=False, with_stress=False, language_switch="remove-flags"
)
# phonemizer_ko.separator = separator

phonemizer_fr = EspeakBackend(
    "fr-fr",
    preserve_punctuation=False,
    with_stress=False,
    language_switch="remove-flags",
)
# phonemizer_fr.separator = separator

phonemizer_de = EspeakBackend(
    "de", preserve_punctuation=False, with_stress=False, language_switch="remove-flags"
)
# phonemizer_de.separator = separator


lang2backend = {
    "zh": phonemizer_zh,
    "ja": phonemizer_ja,
    "en": phonemizer_en,
    "fr": phonemizer_fr,
    "ko": phonemizer_ko,
    "de": phonemizer_de,
}

TOKEN_MAP_PATH = os.path.abspath("./models/tts/maskgct/g2p/utils/mls_en.json")


def _load_token_map():
    with open(TOKEN_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_token_map(mapping: dict):
    with open(TOKEN_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)


def _assign_ids_for_missing_tokens(
    tokens: List[str], allow_dynamic_add: bool = True, pad_reserved_id: int = 1023
) -> None:
    """Ensure all tokens have IDs in the global token map.

    This function updates the on-disk map with a file lock and refreshes the in-memory map.

    Args:
        tokens: list of IPA tokens (space-split).
        allow_dynamic_add: whether to add missing tokens to the map.
        pad_reserved_id: do not allocate this id (reserved for padding).
    """
    global token
    if not allow_dynamic_add:
        return

    # Fast path: check against current in-memory map
    missing = [t for t in tokens if t and t not in token]
    if not missing:
        return

    # Lock and update on-disk map to avoid races across workers/processes
    lock_path = TOKEN_MAP_PATH + ".lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w") as lock_fp:
        fcntl.flock(lock_fp, fcntl.LOCK_EX)
        try:
            latest = _load_token_map()
            # Recompute missing against latest on-disk map
            missing_latest = [t for t in tokens if t and t not in latest]
            if missing_latest:
                # Determine next available id; keep below pad_reserved_id
                next_id = (max(latest.values()) + 1) if len(latest) > 0 else 1
                # Avoid allocating the reserved padding id
                if next_id == pad_reserved_id:
                    next_id += 1
                added = 0
                for t in missing_latest:
                    if next_id >= pad_reserved_id:
                        # Reached cap; stop adding
                        break
                    latest[t] = next_id
                    next_id += 1
                    added += 1
                if added > 0:
                    _save_token_map(latest)
                    token = latest  # refresh in-memory map
        finally:
            fcntl.flock(lock_fp, fcntl.LOCK_UN)


# Initialize global token map after function definitions
token = _load_token_map()


def phonemizer_g2p(text, language):
    # Korean: use hybrid G2P (g2pkk → jamo → IPA with phonological rules)
    if language == "ko" and _KoreanG2PModule is not None:
        ko_g2p = _KoreanG2PModule(backend="korean_hybrid", language="ko")
        phones = ko_g2p.g2p_conversion(text)  # list of IPA tokens
        phonemes = " ".join(phones)
        _assign_ids_for_missing_tokens(phones, allow_dynamic_add=True)
        token_id = [token[p] for p in phones if p in token]
        return phonemes, token_id

    # Fallback: use phonemizer/espeak for other languages (or if KO module unavailable)
    langbackend = lang2backend[language]
    phonemes = _phonemize(
        langbackend,
        text,
        separator,
        strip=True,
        njobs=1,
        prepend_text=False,
        preserve_empty_lines=False,
    )
    token_id = []
    if isinstance(phonemes, list):
        for phone in phonemes:
            phonemes_split = phone.split(" ")
            _assign_ids_for_missing_tokens(phonemes_split, allow_dynamic_add=True)
            token_id.append([token[p] for p in phonemes_split if p in token])
    else:
        phonemes_split = phonemes.split(" ")
        _assign_ids_for_missing_tokens(phonemes_split, allow_dynamic_add=True)
        token_id = [token[p] for p in phonemes_split if p in token]
    return phonemes, token_id


def _phonemize(  # pylint: disable=too-many-arguments
    backend,
    text: Union[str, List[str]],
    separator: Separator,
    strip: bool,
    njobs: int,
    prepend_text: bool,
    preserve_empty_lines: bool,
):
    """Auxiliary function to phonemize()

    Does the phonemization and returns the phonemized text. Raises a
    RuntimeError on error.

    """
    # remember the text type for output (either list or string)
    text_type = type(text)

    # force the text as a list
    text = [line.strip(os.linesep) for line in str2list(text)]

    # if preserving empty lines, note the index of each empty line
    if preserve_empty_lines:
        empty_lines = [n for n, line in enumerate(text) if not line.strip()]

    # ignore empty lines
    text = [line for line in text if line.strip()]

    if text:
        # phonemize the text
        phonemized = backend.phonemize(
            text, separator=separator, strip=strip, njobs=njobs
        )
    else:
        phonemized = []

    # if preserving empty lines, reinsert them into text and phonemized lists
    if preserve_empty_lines:
        for i in empty_lines:  # noqa
            if prepend_text:
                text.insert(i, "")
            phonemized.insert(i, "")

    # at that point, the phonemized text is a list of str. Format it as
    # expected by the parameters
    if prepend_text:
        return list(zip(text, phonemized))
    if text_type == str:
        return list2str(phonemized)
    return phonemized
