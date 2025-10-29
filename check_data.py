# #!/usr/bin/env python3
# import subprocess
# from pathlib import Path
# from tqdm import tqdm
# import json

# # 데이터셋 루트 (원하는 경로로 수정하세요)
# DATASET_DIR = "/app/data/dataset-vevo/dataset-ko/tts-audio/Training"
# LOG_FILE = "invalid_files.log"

# def check_audio(file_path: Path):
#     """
#     ffprobe를 사용해 오디오 파일 헤더만 빠르게 검사
#     duration, sample_rate, channels 정보를 가져옴
#     """
#     cmd = [
#         "ffprobe",
#         "-v", "error",
#         "-show_entries", "format=duration:stream=sample_rate,channels",
#         "-of", "json",
#         str(file_path)
#     ]
#     try:
#         result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
#         data = json.loads(result.stdout)

#         # duration, sample_rate, channels 모두 있어야 정상으로 간주
#         duration = float(data["format"].get("duration", 0.0))
#         streams = data.get("streams", [])
#         if not streams:
#             return False, None
#         sample_rate = int(streams[0].get("sample_rate", 0))
#         channels = int(streams[0].get("channels", 0))

#         if duration <= 0 or sample_rate <= 0 or channels <= 0:
#             return False, None

#         return True, {
#             "duration": duration,
#             "sample_rate": sample_rate,
#             "channels": channels
#         }
#     except Exception:
#         return False, None

# def main():
#     dataset_path = Path(DATASET_DIR)
#     # 하위 디렉토리까지 모두 탐색
#     audio_files = list(dataset_path.rglob("*.wav")) \
#                 + list(dataset_path.rglob("*.WAV")) \
#                 + list(dataset_path.rglob("*.mp3")) \
#                 + list(dataset_path.rglob("*.MP3"))

#     invalid_files = []
#     for f in tqdm(audio_files, desc="Checking audio files (fast)"):
#         valid, meta = check_audio(f)
#         if not valid:
#             invalid_files.append(str(f))

#     if invalid_files:
#         with open(LOG_FILE, "w") as logf:
#             logf.write("\n".join(invalid_files))
#         print(f"❌ Invalid files found: {len(invalid_files)} → see {LOG_FILE}")
#     else:
#         print("✅ All audio files are valid.")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import subprocess
from pathlib import Path
from tqdm import tqdm
import json

# 데이터셋 루트 (원하는 경로로 수정하세요)
DATASET_DIR = "/app/data/dataset-vevo/dataset-ko/tts-audio/Training"
LOG_FILE = "invalid_samplerate.log"
TARGET_SR = 48000  # 원하는 샘플레이트

def check_audio(file_path: Path):
    """
    ffprobe를 사용해 오디오 파일 헤더만 빠르게 검사
    duration, sample_rate, channels 정보를 가져옴
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration:stream=sample_rate,channels",
        "-of", "json",
        str(file_path)
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        data = json.loads(result.stdout)

        # duration, sample_rate, channels 추출
        duration = float(data["format"].get("duration", 0.0))
        streams = data.get("streams", [])
        if not streams:
            return False, None
        sample_rate = int(streams[0].get("sample_rate", 0))
        channels = int(streams[0].get("channels", 0))

        return True, {
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": channels
        }
    except Exception:
        return False, None

def main():
    dataset_path = Path(DATASET_DIR)
    # 하위 디렉토리까지 모두 탐색
    audio_files = list(dataset_path.rglob("*.wav")) \
                + list(dataset_path.rglob("*.WAV")) \
                + list(dataset_path.rglob("*.mp3")) \
                + list(dataset_path.rglob("*.MP3"))

    invalid_files = []
    for f in tqdm(audio_files, desc="Checking audio files (samplerate)"):
        valid, meta = check_audio(f)
        if valid:
            if meta["sample_rate"] != TARGET_SR:
                invalid_files.append(f"{f}\t{meta['sample_rate']} Hz")
        else:
            invalid_files.append(f"{f}\tinvalid metadata")

    if invalid_files:
        with open(LOG_FILE, "w") as logf:
            logf.write("\n".join(invalid_files))
        print(f"❌ Files with wrong sample rate: {len(invalid_files)} → see {LOG_FILE}")
    else:
        print(f"✅ All audio files have {TARGET_SR} Hz sample rate.")

if __name__ == "__main__":
    main()
