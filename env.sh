# (non-interactive 설치)
conda install -y -c conda-forge ffmpeg

# ---------- Python base & utils ----------
# librosa==0.9.1 유지 시 numpy/numba 호환 필요: numpy 1.23.x, numba 0.56.x
pip install \
  numpy==1.23.5 numba==0.56.4 \
  setuptools ruamel.yaml tqdm colorama easydict tabulate loguru json5 Cython \
  unidecode inflect argparse g2p_en tgt librosa==0.9.1 matplotlib typeguard \
  einops omegaconf hydra-core humanfriendly pandas munch

# ---------- PyTorch stack (CUDA 12.4 wheels; CUDA 12.9 이미지에서도 OK) ----------
# 공식 cu129 휠이 없어 cu124 사용. 반드시 이 index-url을 사용하세요.
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# ---------- DL/Audio libs ----------
pip install \
  accelerate==0.24.1 transformers==4.41.2 diffusers \
  praat-parselmouth audiomentations pedalboard ffmpeg-python==0.2.0 pyworld \
  diffsptk==1.0.1 nnAudio ptwt

pip install encodec vocos speechtokenizer g2p_en descript-audio-codec
pip install torchmetrics pymcd openai-whisper frechet_audio_distance asteroid resemblyzer

# PESQ (C-extension)
pip install https://github.com/vBaiCai/python-pesq/archive/master.zip

# ⚠ fairseq는 torch 2.x와 호환 이슈가 잦습니다.
# 필요하다면 별도 환경에서 설치하거나, 프로젝트 버전에 맞춰 고정하세요.
# pip install fairseq

# lhotse
pip install "git+https://github.com/lhotse-speech/lhotse"

# (중복 방지) 최신 encodec만 별도 업데이트가 필요하면 유지
pip install -U encodec

# grapheme/phoneme
pip install phonemizer==3.2.1 pypinyin==0.48.0

# toolinsss


pip install black==24.1.1
