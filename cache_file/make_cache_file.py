#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Amphion EmiliaDataset용 "캐시 4종(pkl)" 생성 스크립트 (최종 수정본, 주석 상세)
===============================================================================

■ 목적
  EmiliaDataset(cache_type="path")가 기대하는 4개의 pickle 캐시 파일을 생성합니다.
  - wav_paths_cache.pkl       : 각 샘플의 오디오 파일 상대경로 리스트 (MNT_PATH 기준, 앞에 / 포함)
  - json_paths_cache.pkl      : 각 샘플의 메타 JSON 상대경로 리스트 (문자열상 *_fixzh.json 로 표기)
  - duration_cache.pkl        : 각 샘플의 오디오 길이(초)
  - phone_count_cache.pkl     : 각 샘플 텍스트의 "한글 음절 수" (간이 대체; 실제는 g2p로 phoneme 개수를 세는 것이 이상적)

※ EmiliaDataset.__getitem__()는 self.mnt_path + rel_path 로 실제 파일을 읽습니다.
  → 그러므로 이 스크립트는 pkl에 "MNT_PATH 기준의 상대경로(= 선행 / 포함)"를 저장합니다.

-------------------------------------------------------------------------------
■ 상위 루트 및 하위 폴더 구성 (예시)

MNT_PATH/
├─ aihub_ko/
│  ├─ dataset_large/
│  │  └─ {SPEAKER}/
│  │     ├─ wav_48000/
│  │     │  └─ {SPEAKER}_{INDEX}.wav
│  │     │     예) F0003_101665.wav, F0003_101666.wav, ...
│  │     └─ script.txt
│  │        - 줄 구조 예:
│  │           F0003_101665
│  │           텍스트 내용 | LH | HL ... (다음 라인부터 공백라인 전까지가 텍스트)
│  │           [공백]
│  │           F0003_101666
│  │           텍스트 내용 ...
│  │           [공백]
│  │        - prosody 표기('||','M','LH','HL','LL','HH')는 제거하여 저장
│  │
│  ├─ dataset_small/
│  │  └─ {SPEAKER}/
│  │     └─ wav_48000/  
│  │        └─ {SPEAKER}_{INDEX}.wav
│  │
│  └─ dataset_small_transcripts/
│     └─ **/*.json
│        - Whisper 전사 JSON
│        - 구조 예: /dataset_small_transcripts/F2001/wav_48000/F2001_000001.json
│        - JSON 예시:
│            {
│              "rel_path": "F2001/wav_48000/F2001_000001.wav",
│              "text": "전사된 문장 ...",
│              "duration": 3.21,            ← 없을 수도 있음
│              "segments": [{"start":0.0,"end":1.1,...}, ...] ← 있을 수도 있음
│            }
│
└─ emilia_ko_unpacked/
   ├─ ko/                  ← 원본 에밀리아 ko 데이터 (mp3/json)
   │  ├─ KO_B00002_S04731_W000001.mp3
   │  ├─ KO_B00002_S04731_W000001.json (있을 수도, 없을 수도)
   │  ├─ KO_B00002_S04731_W000002.mp3
   │  └─ ...
   │
   └─ ko_formatted/        ← 본 스크립트가 생성하는 규격화 폴더
      ├─ KO_B00002_S04731_1.mp3          ← 'W000001 → 1' 처럼 번호 정규화
      ├─ KO_B00002_S04731.json           ← index → meta 딕셔너리
      └─ ...

-------------------------------------------------------------------------------
■ 처리 파이프라인 요약
1) AIHub dataset_large
   - script.txt 파싱 → 발화ID → 텍스트 매핑
   - wav_48000/*.wav 순회 → duration 계산, 텍스트 매칭
   - speaker.json 저장, 캐시 리스트 업데이트

2) AIHub dataset_small
   - dataset_small_transcripts/*.json 순회
   - rel_path로 wav 매칭, duration 추출
   - speaker.json 저장, 캐시 리스트 업데이트

3) Emilia ko (원본 ko → ko_formatted 규격화)
   - 파일명 마지막 'W000001' → 1 추출
   - ko_formatted에 심볼릭 링크(또는 복사)로 정규화 파일 생성
   - 같은 stem json 있으면 텍스트 반영
   - audio_name.json 생성, 캐시 리스트 업데이트

-------------------------------------------------------------------------------
■ JSON 저장 형식 관련
  - 기본: 딕셔너리(index→meta) 저장 (SAVE_JSON_AS_LIST=False)
  - 만약 EmiliaDataset.py가 json[index] 접근한다면:
    (1) SAVE_JSON_AS_LIST=True 로 변경, 또는
    (2) EmiliaDataset.py 내부를 json[str(index)] 접근으로 수정

-------------------------------------------------------------------------------
■ 산출물
  CACHE_PATH/
    ├─ wav_paths_cache.pkl
    ├─ json_paths_cache.pkl
    ├─ duration_cache.pkl
    └─ phone_count_cache.pkl

===============================================================================
"""


import os, re, json, pickle, shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import librosa


# ===================== (1) 필수 설정 =====================
# ── 끝에 슬래시 없이 지정 권장
MNT_PATH   = "/data/dataset-vevo/dataset-ko"         # EmiliaDataset 의 MNT_PATH와 동일하게
CACHE_PATH = "/data/dataset-vevo/dataset-ko-cache"   # 캐시(pkl) 저장 위치

# AIHub 경로들
AIHUB_ROOT      = f"{MNT_PATH}/aihub_ko"
DATASET_LARGE   = f"{AIHUB_ROOT}/dataset_large"
DATASET_SMALL   = f"{AIHUB_ROOT}/dataset_small"
SMALL_JSON_ROOT = f"{AIHUB_ROOT}/dataset_small_transcripts"

# Emilia(ko) 원본/스테이징 경로 (★★ 바뀐 구조 반영 ★★)
EMILIA_KO_ROOT  = f"{MNT_PATH}/emilia_ko"
EMILIA_FMT_ROOT = f"{MNT_PATH}/emilia_ko_formatted"

# 오디오 길이 계산 샘플레이트 (None이면 원본 SR 사용)
SR_FOR_DURATION = None


# ===================== (2) 동작 스위치 =====================
# (A) JSON을 '리스트'로 저장해야 하는 경우 True로
#     (EmiliaDataset.py가 json[index]로 접근하는 버전일 때)
# (B) 기본은 False (딕셔너리 저장). 이 경우 EmiliaDataset.py를 str(index) 키 접근으로 수정 필요
SAVE_JSON_AS_LIST = False

# 리스트 저장 시, 최대 인덱스가 너무 크면 강제로 딕셔너리로 전환하는 안전장치
LIST_HARD_CAP = 50_000

# symlink가 불가한 파일시스템(또는 윈도우)에서는 True로 → 실제 파일 복사
COPY_INSTEAD_OF_SYMLINK = False


# ===================== (3) 유틸 함수들 =====================
def ensure_dir(p: str) -> None:
    """폴더가 없으면 생성"""
    Path(p).mkdir(parents=True, exist_ok=True)

def rel_to_mnt(abs_path: str) -> str:
    """
    MNT_PATH 기준 상대경로(선행 '/' 포함)를 반환
    예) abs: /data/dataset/aihub_ko/dataset_large/F0003/wav_48000/F0003_101665.wav
        ret: /aihub_ko/dataset_large/F0003/wav_48000/F0003_101665.wav
    """
    mp = MNT_PATH
    if abs_path.startswith(mp):
        return abs_path[len(mp):]  # 앞의 '/' 유지
    return abs_path  # 예외적으로 MNT_PATH가 prefix가 아닐 경우 원문 반환

def load_duration(wav_path: str):
    """오디오 길이(초) 계산: librosa.load 실패 시 None"""
    try:
        y, sr = librosa.load(wav_path, sr=SR_FOR_DURATION)
        return float(len(y) / sr)
    except Exception as e:
        print(f"[WARN] librosa fail: {wav_path} ({e})")
        return None

def save_json_index_meta(meta_dict: dict, json_path_real: Path):
    """
    index→meta 딕셔너리를 JSON으로 저장
    - SAVE_JSON_AS_LIST=True → 리스트 형식 (json[index] 접근 호환)
    - SAVE_JSON_AS_LIST=False → 딕셔너리 형식 (json[str(index)] 접근 호환)
    """
    ensure_dir(str(json_path_real.parent))
    if SAVE_JSON_AS_LIST:
        if not meta_dict:
            obj = []
        else:
            max_idx = max(meta_dict.keys())
            if max_idx > LIST_HARD_CAP:
                print(f"[INFO] max_idx={max_idx} > {LIST_HARD_CAP}. "
                      f"리스트 저장 비효율 → 딕셔너리로 저장합니다.")
                obj = {int(k): v for k, v in meta_dict.items()}
            else:
                obj = [None] * (max_idx + 1)
                for k, v in meta_dict.items():
                    obj[int(k)] = v
    else:
        obj = {int(k): v for k, v in meta_dict.items()}

    with open(json_path_real, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def looks_int(tok: str) -> bool:
    try:
        int(tok); return True
    except:
        return False

def extract_index_from_stem(stem: str):
    """파일명 끝 토큰을 정수로 반환 (예: F0003_101665 → 101665)"""
    toks = stem.split("_")
    return int(toks[-1]) if looks_int(toks[-1]) else None

def strip_prosody_marks(line: str) -> str:
    """
    script.txt 또는 Whisper 전사에서 prosody/발음 보조 표기 제거.
    - '||', '|||', 등 파이프 → 공백
    - 'M', 'LH', 'HL', 'LL', 'HH', 'LHL' 같은 태그 제거
    - 연속 공백은 단일 공백으로 정리
    """
    # 파이프류 제거 → 공백
    s = re.sub(r"\|+", " ", line)

    # prosody 태그 제거 (단독 M, LH, HL, LL, HH, LHL 등)
    s = re.sub(r"\b(?:M|LH|HL|LL|HH|LHL)\b", "", s, flags=re.IGNORECASE)

    # 연속 공백 정리
    s = re.sub(r"\s+", " ", s).strip()
    return s


def count_hangul_syllables(text: str) -> int:
    """간이 phone_count: 한글 음절 블록 개수 세기"""
    return len(re.findall(r"[가-힣]", text or ""))


# ===================== (4) AIHub dataset_large 처리 =====================
def parse_script_txt(script_path: Path):
    """
    script.txt를 파싱해 '발화ID → 텍스트' 매핑을 만듭니다.
    - 헤더(발화ID)가 한 줄, 그 다음 한 줄 이상이 텍스트(공백라인로 구분)
    - 텍스트의 prosody 표기 제거
    """
    id2text = {}
    if not script_path.exists():
        return id2text

    lines = script_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        header = lines[i].strip()
        m = re.match(r"^([A-Za-z0-9_]+)\s+", header)  # 발화ID
        if m:
            utt_id = m.group(1)
            j = i + 1
            first_content = None
            while j < len(lines):
                L = lines[j].strip()
                if not L:
                    break
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
    """
    dataset_large의 각 스피커 폴더를 순회하여:
      - 오디오 duration 계산
      - script.txt에서 텍스트 매칭(발화ID 또는 6자리 0패딩 키)
      - wav_48000/{SPEAKER}.json 생성 (index→meta)
      - 캐시 4종 리스트에 항목 추가
    """
    wav_rel, json_rel, durs, phones = [], [], [], []

    for spk_dir in tqdm(sorted(Path(DATASET_LARGE).glob("*")), desc="dataset_large"):
        
        if not spk_dir.is_dir():
            continue

        wavdir = spk_dir / "wav_48000"
        script = spk_dir / "script.txt"
        if not wavdir.is_dir():
            continue

        speaker = spk_dir.name
        id2text = parse_script_txt(script)
        json_path_real = wavdir / f"{speaker}.json"

        # index→meta 사전
        meta = {}
        for wav in sorted(wavdir.glob("*.wav")):
            stem = wav.stem
            idx = extract_index_from_stem(stem)
            if idx is None:
                continue

            dur = load_duration(str(wav))
            if dur is None:
                continue

            # 텍스트 매칭: (1) stem 그대로 (2) 끝 인덱스를 6자리 0패딩 키로
            text = id2text.get(stem, "")
            if not text:
                toks = stem.split("_")
                head = "_".join(toks[:-1])
                text = id2text.get(f"{head}_{int(toks[-1]):06d}", "")

            meta[idx] = {
                "language": "ko",
                "text": text,
                "start": 0.0,
                "end": float(dur),
                "phone_count": count_hangul_syllables(text),
            }

        if meta:
            save_json_index_meta(meta, json_path_real)

            for idx in sorted(meta.keys()):
                wav_file = wavdir / f"{speaker}_{idx}.wav"
                if not wav_file.exists():
                    continue

                wav_rel.append(rel_to_mnt(str(wav_file)))
                # 캐시에는 *_fixzh.json로 기입하지만, 실제 파일은 {speaker}.json
                json_rel.append(rel_to_mnt(str(json_path_real.with_name(f"{speaker}_fixzh.json"))))
                durs.append(meta[idx]["end"])
                phones.append(meta[idx]["phone_count"])

    return wav_rel, json_rel, durs, phones


# ===================== (5) AIHub dataset_small 처리 =====================
def iter_small_transcripts():
    """
    dataset_small_transcripts 이하의 모든 JSON 파일을 순회하며 yield.
    JSON 스키마는 유연하나, 최소한 rel_path / text 가 있기를 기대합니다.
    """
    root = Path(SMALL_JSON_ROOT)
    if not root.exists():
        return
    for jf in root.rglob("*.json"):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            yield data
        except Exception as e:
            print(f"[WARN] read fail: {jf} ({e})")

def build_small_from_transcripts():
    """
    Whisper 전사를 이용해 dataset_small의 각 스피커별 {SPEAKER}.json(index→meta)을 생성.
    duration은 다음 우선순위로 결정:
      1) JSON의 duration 필드
      2) JSON의 segments[].end 최댓값
      3) librosa로 wav에서 직접 계산
    """
    wav_rel, json_rel, durs, phones = [], [], [], []
    speaker_to_meta = defaultdict(dict)  # speaker → (index→meta)

    for rec in tqdm(list(iter_small_transcripts()), desc="small_transcripts"):
        rel_path = rec.get("rel_path") or ""
        text = rec.get("text") or ""
        segments = rec.get("segments") or []
        duration = rec.get("duration")

        # 2순위: segments[].end 최댓값
        if duration is None and segments:
            try:
                duration = max(float(s.get("end", 0.0)) for s in segments)
            except Exception:
                duration = None

        # dataset_small 기준 상대경로를 절대경로로
        wav_abs = Path(DATASET_SMALL) / rel_path


        # 3순위: 직접 길이 계산
        if duration is None:
            duration = load_duration(str(wav_abs))
            if duration is None:
                continue

        stem = wav_abs.stem                 # 예: F2001_000001
        idx = extract_index_from_stem(stem) # 1
        if idx is None:
            continue

        speaker = Path(rel_path).parts[0]  # 예: F2001

        speaker_to_meta[speaker][idx] = {
            "language": "ko",
            "text": text,
            "start": 0.0,
            "end": float(duration),
            "phone_count": count_hangul_syllables(text),
        }

    # 스피커별 JSON 저장 및 캐시 수집
    for speaker, meta in speaker_to_meta.items():
        # 실제 wav 폴더 위치: wav_48000 또는 wav48000
        wavdir = Path(DATASET_SMALL) / speaker / "wav_48000"
        

        json_path_real = wavdir / f"{speaker}.json"
        save_json_index_meta(meta, json_path_real)

        for idx in sorted(meta.keys()):
            wav_file = wavdir / f"{speaker}_{idx:06d}.wav"
            if not wav_file.exists():
                continue

            wav_rel.append(rel_to_mnt(str(wav_file)))
            json_rel.append(rel_to_mnt(str(json_path_real.with_name(f"{speaker}_fixzh.json"))))
            durs.append(meta[idx]["end"])
            phones.append(meta[idx]["phone_count"])

    return wav_rel, json_rel, durs, phones


# ===================== (6) emilia_ko 규격화 =====================
def number_from_W(token: str):
    """마지막 토큰 'W000001' → 정수 1"""
    m = re.match(r"^[Ww]0*([0-9]+)$", token)
    return int(m.group(1)) if m else None

def build_emilia():
    """
    emilia_ko_unpacked/ko 의 mp3/wav 파일을 ko_formatted/ 로 정규화
    - 파일명: {audio_name}_{idx}.{ext} (확장자 유지)
    - 같은 stem의 원본 json이 있으면 텍스트 반영
    - ko_formatted/{audio_name}.json 생성 (index→meta)
    """
    ensure_dir(EMILIA_FMT_ROOT)
    wav_rel, json_rel, durs, phones = [], [], [], []

    # (★ wav/mp3 모두 처리 ★)
    all_files = sorted(list(Path(EMILIA_KO_ROOT).glob("*.wav")) +
                       list(Path(EMILIA_KO_ROOT).glob("*.mp3")))

    for wav in tqdm(all_files, desc="emilia_ko"):
        stem = wav.stem  # 예: KO_B00002_S04731_W000001
        toks = stem.split("_")
        if len(toks) < 2:
            continue

        idx = number_from_W(toks[-1])
        if idx is None:
            continue

        audio_name = "_".join(toks[:-1])  # 예: KO_B00002_S04731
        out_dir = Path(EMILIA_FMT_ROOT)
        ensure_dir(str(out_dir))

        # (★ 확장자 유지: wav 또는 mp3 그대로 ★)
        new_wav = out_dir / f"{audio_name}_{idx}{wav.suffix}"
        if not new_wav.exists():
            if COPY_INSTEAD_OF_SYMLINK:
                shutil.copy2(str(wav), str(new_wav))
            else:
                try:
                    os.symlink(os.path.abspath(str(wav)), str(new_wav))
                except Exception:
                    shutil.copy2(str(wav), str(new_wav))

        # 텍스트 소스: 같은 이름의 원본 json
        text = ""
        src_json = wav.with_suffix(".json")
        if src_json.exists():
            try:
                with open(src_json, "r", encoding="utf-8") as f:
                    jd = json.load(f)
                if isinstance(jd, dict) and "text" in jd:
                    text = jd["text"]
                elif isinstance(jd, dict) and "0" in jd and "text" in jd["0"]:
                    text = jd["0"]["text"]
            except Exception:
                pass

        dur = load_duration(str(wav))
        if dur is None:
            continue

        # 메타 JSON (누적 가능)
        json_path_real = out_dir / f"{audio_name}.json"
        try:
            with open(json_path_real, "r", encoding="utf-8") as f:
                old = json.load(f)
            meta = {int(k): v for k, v in (old.items() if isinstance(old, dict)
                                           else {i: v for i, v in enumerate(old) if v is not None}.items())}
        except Exception:
            meta = {}

        meta[idx] = {
            "language": "ko",
            "text": text,
            "start": 0.0,
            "end": float(dur),
            "phone_count": count_hangul_syllables(text),
        }

        save_json_index_meta(meta, json_path_real)

        wav_rel.append(rel_to_mnt(str(new_wav)))
        json_rel.append(rel_to_mnt(str(json_path_real.with_name(f"{audio_name}_fixzh.json"))))
        durs.append(float(dur))
        phones.append(count_hangul_syllables(text))

    return wav_rel, json_rel, durs, phones


# ===================== (7) 메인 =====================
def main():
    ensure_dir(CACHE_PATH)

    all_wavs, all_jsons, all_durs, all_phones = [], [], [], []
    
    # 1) AIHub large
    w, j, d, p = build_large()
    all_wavs += w; all_jsons += j; all_durs += d; all_phones += p
    print(f"[SUMMARY] dataset_large: +{len(w)}")

    # 2) AIHub small
    w, j, d, p = build_small_from_transcripts()
    all_wavs += w; all_jsons += j; all_durs += d; all_phones += p
    print(f"[SUMMARY] dataset_small: +{len(w)}")

    # 3) Emilia ko
    w, j, d, p = build_emilia()
    all_wavs += w; all_jsons += j; all_durs += d; all_phones += p
    print(f"[SUMMARY] emilia_ko: +{len(w)}")

    # === 캐시 4종 저장 ===
    with open(os.path.join(CACHE_PATH, "wav_paths_cache.pkl"), "wb") as f:
        pickle.dump(all_wavs, f)
    with open(os.path.join(CACHE_PATH, "json_paths_cache.pkl"), "wb") as f:
        pickle.dump(all_jsons, f)
    with open(os.path.join(CACHE_PATH, "duration_cache.pkl"), "wb") as f:
        pickle.dump(all_durs, f)
    with open(os.path.join(CACHE_PATH, "phone_count_cache.pkl"), "wb") as f:
        pickle.dump(all_phones, f)

    print("✅ Done. Cache files created at:", CACHE_PATH)
    print(f"#wavs={len(all_wavs)}  #jsons={len(all_jsons)} "
          f"#dur={len(all_durs)}  #phones={len(all_phones)}")

if __name__ == "__main__":
    main()
