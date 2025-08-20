#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
< 참고사항 / 스크립트 개요 >

이 스크립트는 Amphion의 EmiliaDataset 구현이 요구하는 4개 캐시(pkl)를 생성합니다.
- wav_paths_cache.pkl       : MNT_PATH 기준 상대경로로 된 wav 경로 리스트
- json_paths_cache.pkl      : (문자열상 *_fixzh.json) 경로 리스트  ※ 실제로는 {prefix}.json 파일을 엽니다
- duration_cache.pkl        : 각 샘플 오디오 길이(초)
- phone_count_cache.pkl     : 각 샘플 텍스트의 한글 음절 수(공백/구두점 제거)
주의: 실제로는 g2p로 변환 후 phoneme 개수를 세야하지만 현재 코드는 단순히 text 의 개수를 세도록 되어 있습니다. 사용하시는 g2p에 따라 수정 부탁드립니다.

[ 전제 폴더 구조 (MNT_PATH = /hdd_ext/hdd2/sujin/MAGO/mago-dataset) ]
/hdd_ext/hdd2/sujin/MAGO/mago-dataset/
├─ aihub_ko/
│  ├─ dataset_large/
│  │  └─ {SPEAKER}/
│  │     ├─ wav_48000/
│  │     │  └─ {SPEAKER}_{INDEX}.wav      (예: F0003_101665.wav)
│  │     └─ script.txt                    (원문/억양 마크 포함 텍스트)
│  ├─ dataset_small/
│  │  └─ {SPEAKER}/
│  │     └─ wav_48000/ 또는 wav48000/
│  │        └─ {SPEAKER}_{INDEX}.wav      (예: F2001_000001.wav, 폴더명 표기는 혼용 가능)
│  └─ dataset_small_transcripts/          (Whisper 전사 JSON 모음; 파일별 스키마에 rel_path, text 포함)
└─ emilia_ko/
   └─ ko/
      ├─ *.wav                             (예: KO_B00002_S04731_W000001.wav)
      └─ *.json (있을 수도 있음)          (동일 stem의 보조 텍스트)

※ emilia_ko/ko_formatted/는 본 스크립트가 자동 생성하는 "규격화 스테이징 폴더"입니다.
   - 원본 ko/ 의 파일명은 끝 토큰이 'W000001'처럼 정수가 아니라 EmiliaDataset과 안 맞습니다.
   - 본 스크립트는 ko/ 를 읽어 ko_formatted/ 에 {audio_name}_{정수}.wav 형태로 링크/복사해 둡니다.
   - 같은 폴더에 {audio_name}.json (index→meta)도 생성합니다.

[ 데이터 소스별 처리 요약 ]
1) dataset_large
   - 각 스피커의 script.txt를 파싱해서 발화 ID → 텍스트를 만듭니다.
   - 텍스트 줄에서 '||', 'M', 'LH', 'HL', 'LL', 'HH' 등의 억양/구분 마크를 제거합니다.
   - wav_48000 폴더 안에 {SPEAKER}.json(실제 읽히는 파일) 저장: {index: {language, text, start, end, phone_count}}
   - phone_count = 텍스트 내 "한글 음절 수"(정규식 [가-힣]만 카운트)

2) dataset_small
   - aihub_ko/dataset_small_transcripts/ 의 Whisper JSON들을 읽어 rel_path → wav를 찾고 text를 가져옵니다.
   - duration은 JSON에 없으면 segments의 end 최댓값, 그래도 없으면 librosa로 계산합니다.
   - 스피커별 wav 폴더(wav_48000 또는 wav48000)에 {SPEAKER}.json 저장(위와 동일 스키마).
   - phone_count = 한글 음절 수.

3) emilia_ko
   - emilia_ko/ko/ 의 *.wav 파일명을 파싱하여 끝 토큰 'W000001' → 1처럼 정수 index를 추출합니다.
   - emilia_ko/ko_formatted/ 에 {audio_name}_{index}.wav 심볼릭 링크(불가 시 복사) 생성.
   - 같은 이름의 원본 json이 ko/에 있으면 text를 사용, 없으면 빈 문자열.
   - ko_formatted/ 에 {audio_name}.json 저장(위와 동일 스키마).
   - phone_count = 한글 음절 수.

[ 캐시 파일과 EmiliaDataset 연동 ]
- wav_paths_cache.pkl / json_paths_cache.pkl 에는 "MNT_PATH 기준 상대경로(앞에 / 포함)"가 저장됩니다.
  EmiliaDataset는 실제 로딩 시 self.mnt_path + rel_path 로 합칩니다.
- json_paths_cache.pkl에는 *_fixzh.json 문자열을 넣지만, EmiliaDataset가 내부에서 "_fixzh"를 제거하여
  같은 폴더의 {prefix}.json 파일을 엽니다.  → 실제 {prefix}.json 파일이 존재해야 합니다.
- 네 개 리스트는 같은 인덱스가 한 샘플을 구성합니다.

[ 환경 설정 / 실행 ]
- 코드 상단의 MNT_PATH, CACHE_PATH 값을 환경에 맞게 설정하세요.
- librosa, tqdm 필요.
- emilia 쪽에서 링크가 불가한 파일시스템이면 build_emilia(copy_instead_of_symlink=True)로 복사 사용 가능.
- 현재 파일엔 개발 편의를 위한 pdb.set_trace()가 들어가 있으니, 배포/실행 시 제거하세요.

[ 주의 사항 ]
- dataset_small의 실제 폴더명이 wav_48000과 wav48000이 혼용되어도 스크립트가 보정합니다.
- dataset_large의 발화 ID 매칭은 stem 그대로 또는 끝 인덱스를 6자리 0패딩한 키도 함께 시도합니다.
- duration 기본 샘플링은 파일 SR 그대로(librosa.load sr=None) 사용.
- phone_count는 공백/구두점/영문/숫자를 제외한 한글 음절만 카운트합니다.

이 스크립트 실행 후 CACHE_PATH 폴더에 4개 pkl이 생성되며,
EmiliaDataset(MNT_PATH/CACHE_PATH 동일 설정)에서 cache_type="path"로 바로 사용할 수 있습니다.
"""


import os, re, json, pickle, shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import librosa

import re

def count_hangul_syllables(text: str) -> int:
    # 한글 음절 블록만 카운트 (공백/구두점/영문 제외)
    return len(re.findall(r"[가-힣]", text or ""))


import pdb; pdb.set_trace()
# ===================== 경로 설정 =====================
MNT_PATH   = "/hdd_ext/hdd2/sujin/MAGO/mago-dataset"            # EmiliaDataset의 MNT_PATH와 동일
CACHE_PATH = "/hdd_ext/hdd2/sujin/MAGO/mago-dataset_cache"  # pkl 4개 저장 위치

AIHUB_ROOT          = f"{MNT_PATH}/aihub_ko"
DATASET_LARGE       = f"{AIHUB_ROOT}/dataset_large"
DATASET_SMALL       = f"{AIHUB_ROOT}/dataset_small"
SMALL_JSON_ROOT     = f"{AIHUB_ROOT}/dataset_small_transcripts"  # 👈 Whisper 전사 JSON 폴더
EMILIA_KO_ROOT      = f"{MNT_PATH}/emilia_ko/ko"
EMILIA_FMT_ROOT     = f"{MNT_PATH}/emilia_ko/ko_formatted"

SR_FOR_DURATION = None  # None이면 원본 SR 사용

# ===================== 유틸 =====================
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def rel_to_mnt(p: str) -> str:
    return p[len(MNT_PATH):] if p.startswith(MNT_PATH) else p

def load_duration(wav_path: str):
    try:
        y, sr = librosa.load(wav_path, sr=SR_FOR_DURATION)
        return float(len(y)/sr)
    except Exception as e:
        print(f"[WARN] librosa fail: {wav_path} ({e})"); return None

def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def looks_int(tok: str) -> bool:
    try: int(tok); return True
    except: return False

def extract_index_from_stem(stem: str):
    toks = stem.split("_")
    return int(toks[-1]) if looks_int(toks[-1]) else None

def strip_prosody_marks(line: str) -> str:
    s = re.sub(r"\|+", " ", line)
    s = re.sub(r"\b(M|LH|HL|LL|HH)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ===================== 1) dataset_large =====================
def parse_script_txt(script_path: Path):
    id2text = {}
    if not script_path.exists(): return id2text
    lines = script_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        header = lines[i].strip()
        m = re.match(r"^([A-Za-z0-9_]+)\s+", header)
        if m:
            utt_id = m.group(1)
            j = i + 1
            first_content = None
            while j < len(lines):
                L = lines[j].strip()
                if not L: break
                if first_content is None:
                    first_content = L
                j += 1
            if first_content:
                text = strip_prosody_marks(first_content)
                id2text[utt_id] = text
            i = j + 1
        else:
            i += 1
    return id2text

def build_large():
    wav_rel, json_rel, durs, phones = [], [], [], []
    for spk_dir in tqdm(sorted(Path(DATASET_LARGE).glob("*")), desc="dataset_large"):
        if not spk_dir.is_dir(): continue
        wavdir = spk_dir / "wav_48000"
        script = spk_dir / "script.txt"
        if not wavdir.is_dir(): continue
        speaker = spk_dir.name
        id2text = parse_script_txt(script)
        json_path_real = wavdir / f"{speaker}.json"

        meta = {}
        for wav in sorted(wavdir.glob("*.wav")):
            stem = wav.stem
            idx = extract_index_from_stem(stem)
            if idx is None: continue
            dur = load_duration(str(wav))
            if dur is None: continue

            # stem, 또는 0패딩 6자리 키로 매칭
            text = id2text.get(stem, "")
            if not text:
                toks = stem.split("_")
                head = "_".join(toks[:-1])
                text = id2text.get(f"{head}_{int(toks[-1]):06d}", "")

            meta[idx] = {
                "language": "ko",
                "text": text,
                "start": 0.0, "end": dur,
                "phone_count": count_hangul_syllables(text)
            }


        if meta:
            save_json(meta, str(json_path_real))
            for idx in sorted(meta.keys()):
                wav_file = wavdir / f"{speaker}_{idx}.wav"
                if not wav_file.exists(): continue
                wav_rel.append(rel_to_mnt(str(wav_file)))
                json_rel.append(rel_to_mnt(str(json_path_real.with_name(f"{speaker}_fixzh.json"))))
                durs.append(meta[idx]["end"])
                phones.append(meta[idx]["phone_count"])
    return wav_rel, json_rel, durs, phones

# ===================== 2) dataset_small (Whisper 전사 사용) =====================
def iter_small_transcripts():
    """dataset_small_transcripts 폴더의 모든 json을 yield."""
    root = Path(SMALL_JSON_ROOT)
    if not root.exists(): return
    for jf in root.rglob("*.json"):
        try:
            data = json.load(open(jf, "r", encoding="utf-8"))
            yield data
        except Exception as e:
            print(f"[WARN] read fail: {jf} ({e})")

def build_small_from_transcripts():
    """
    rel_path로 wav 찾기:
      wav_abs = DATASET_SMALL / rel_path
    duration:
      - json["duration"]가 있으면 사용
      - 없으면 segments의 end 최댓값
      - 그래도 안되면 librosa로 계산
    """
    wav_rel, json_rel, durs, phones = [], [], [], []
    # 스피커별로 모아 한 폴더(wav_48000)에 {SPEAKER}.json 생성
    speaker_to_meta = defaultdict(dict)

    for rec in tqdm(list(iter_small_transcripts()), desc="small_transcripts"):
        rel_path = rec.get("rel_path") or ""
        text = rec.get("text") or ""
        segments = rec.get("segments") or []
        duration = rec.get("duration")
        if duration is None and segments:
            try:
                duration = max(float(s.get("end", 0.0)) for s in segments)
            except Exception:
                duration = None

        wav_abs = Path(DATASET_SMALL) / rel_path  # ex) .../dataset_small/F2001/wav_48000/F2001_000001.wav
        if not wav_abs.exists():
            # 혹시 폴더명이 wav48000 / wav_48000 혼용이면 보정
            wav_abs2 = Path(str(wav_abs).replace("wav_48000", "wav48000"))
            if wav_abs2.exists(): wav_abs = wav_abs2
            else: 
                print(f"[MISS] {wav_abs}"); 
                continue

        if duration is None:
            duration = load_duration(str(wav_abs))
            if duration is None: 
                continue

        stem = wav_abs.stem            # F2001_000001
        idx = extract_index_from_stem(stem)
        if idx is None: 
            continue

        speaker = wav_abs.parent.parent.name  # F2001


        speaker_to_meta[speaker][idx] = {
            "language": "ko",
            "text": text,
            "start": 0.0, "end": float(duration),
            "phone_count": count_hangul_syllables(text)
        }


    # 스피커별 json 저장 & 캐시 수집
    for speaker, meta in speaker_to_meta.items():
        # 실제 wav 폴더 경로 추정 (둘 다 시도)
        wavdir = Path(DATASET_SMALL) / speaker / "wav_48000"
        if not wavdir.is_dir():
            wavdir = Path(DATASET_SMALL) / speaker / "wav48000"
        if not wavdir.is_dir(): 
            continue

        json_path_real = wavdir / f"{speaker}.json"
        save_json({int(k):v for k,v in meta.items()}, str(json_path_real))

        for idx in sorted(meta.keys()):
            wav_file = wavdir / f"{speaker}_{idx}.wav"
            if not wav_file.exists():
                # 경우에 따라 실제 파일명이 다르면 skip
                continue
            wav_rel.append(rel_to_mnt(str(wav_file)))
            json_rel.append(rel_to_mnt(str(json_path_real.with_name(f"{speaker}_fixzh.json"))))
            durs.append(meta[idx]["end"])
            phones.append(meta[idx]["phone_count"])

    return wav_rel, json_rel, durs, phones

# ===================== 3) emilia_ko 규격화 =====================
def number_from_W(token: str):
    m = re.match(r"^[Ww]0*([0-9]+)$", token)
    return int(m.group(1)) if m else None

def build_emilia(copy_instead_of_symlink=False):
    ensure_dir(EMILIA_FMT_ROOT)
    wav_rel, json_rel, durs, phones = [], [], [], []

    for wav in tqdm(sorted(Path(EMILIA_KO_ROOT).glob("*.wav")), desc="emilia_ko"):
        stem = wav.stem  # KO_B00002_S04731_W000001
        toks = stem.split("_")
        if len(toks) < 2: continue
        idx = number_from_W(toks[-1])
        if idx is None: continue
        audio_name = "_".join(toks[:-1])

        out_dir = Path(EMILIA_FMT_ROOT)
        ensure_dir(out_dir)
        new_wav = out_dir / f"{audio_name}_{idx}.wav"

        if not new_wav.exists():
            if copy_instead_of_symlink:
                shutil.copy2(str(wav), str(new_wav))
            else:
                try: os.symlink(os.path.abspath(str(wav)), str(new_wav))
                except Exception: shutil.copy2(str(wav), str(new_wav))

        # 텍스트 소스: 같은 이름의 json(있으면 사용)
        text = ""
        src_json = wav.with_suffix(".json")
        if src_json.exists():
            try:
                jd = json.load(open(src_json, "r", encoding="utf-8"))
                if isinstance(jd, dict) and "text" in jd: text = jd["text"]
                elif isinstance(jd, dict) and "0" in jd and isinstance(jd["0"], dict) and "text" in jd["0"]:
                    text = jd["0"]["text"]
            except Exception: pass

        dur = load_duration(str(wav))
        if dur is None: continue

        json_path_real = out_dir / f"{audio_name}.json"
        try:
            meta = json.load(open(json_path_real, "r", encoding="utf-8"))
            meta = {int(k): v for k, v in meta.items()}
        except Exception:
            meta = {}

        meta[idx] = {
            "language": "ko",
            "text": text,
            "start": 0.0, "end": float(dur),
            "phone_count": count_hangul_syllables(text)
        }
        


        save_json(meta, str(json_path_real))

        wav_rel.append(rel_to_mnt(str(new_wav)))
        json_rel.append(rel_to_mnt(str(json_path_real.with_name(f"{audio_name}_fixzh.json"))))
        durs.append(float(dur))
        phones.append(count_hangul_syllables(text))

    return wav_rel, json_rel, durs, phones

# ===================== 메인 =====================
def main():
    ensure_dir(CACHE_PATH)
    all_wavs, all_jsons, all_durs, all_phones = [], [], [], []

    # 1) dataset_large (script.txt 파싱)
    w, j, d, p = build_large()
    all_wavs += w; all_jsons += j; all_durs += d; all_phones += p

    # 2) dataset_small (Whisper 전사 사용)
    w, j, d, p = build_small_from_transcripts()
    all_wavs += w; all_jsons += j; all_durs += d; all_phones += p

    # 3) emilia_ko (파일명 규격화 + json)
    w, j, d, p = build_emilia(copy_instead_of_symlink=False)
    all_wavs += w; all_jsons += j; all_durs += d; all_phones += p

    # 4개 pkl 저장 (EmiliaDataset가 기대하는 이름)
    with open(os.path.join(CACHE_PATH, "wav_paths_cache.pkl"), "wb") as f:
        pickle.dump(all_wavs, f)
    with open(os.path.join(CACHE_PATH, "json_paths_cache.pkl"), "wb") as f:
        pickle.dump(all_jsons, f)
    with open(os.path.join(CACHE_PATH, "duration_cache.pkl"), "wb") as f:
        pickle.dump(all_durs, f)
    with open(os.path.join(CACHE_PATH, "phone_count_cache.pkl"), "wb") as f:
        pickle.dump(all_phones, f)

    print("✅ Done. Cache files created at:", CACHE_PATH)
    print(f"#wavs={len(all_wavs)}  #jsons(list)={len(all_jsons)}")

if __name__ == "__main__":
    main()
