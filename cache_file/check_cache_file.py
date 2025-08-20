import pickle, os, json

MNT_PATH   = "/hdd_ext/hdd2/sujin/MAGO/mago-dataset"
CACHE_PATH = "/hdd_ext/hdd2/sujin/MAGO/mago-dataset/emilia_cache"

def load(name):
    with open(os.path.join(CACHE_PATH, name), "rb") as f:
        return pickle.load(f)

wavs  = load("wav_paths_cache.pkl")
jps   = load("json_paths_cache.pkl")
durs  = load("duration_cache.pkl")
phons = load("phone_count_cache.pkl")

print(len(wavs), len(jps), len(durs), len(phons))  # 모두 동일해야 함

# 샘플 3개만 확인
for i in range(min(3, len(wavs))):
    wav_abs = MNT_PATH + wavs[i]
    json_abs_real = (MNT_PATH + jps[i]).replace("_fixzh", "")  # 실제로 열리는 파일
    print("WAV:", wav_abs)
    print("JSON(to open):", json_abs_real)
    print("dur, phone:", durs[i], phons[i])
    print("wav exists?", os.path.exists(wav_abs))
    print("json exists?", os.path.exists(json_abs_real))
    if os.path.exists(json_abs_real):
        with open(json_abs_real, "r", encoding="utf-8") as f:
            meta = json.load(f)
            # index 키가 int여야 코드에서 바로 접근 가능(우린 int로 저장)
            print("meta keys ex:", list(sorted(map(int, meta.keys())))[:5])
    print("----")
