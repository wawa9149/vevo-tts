# Korean TTS Inference Script using trained models
import os
import torch
from datetime import datetime
from models.vc.vevo.vevo_utils import *

def korean_vevo_tts(
    src_text,
    ref_wav_path,
    timbre_ref_wav_path=None,
    output_path=None,
    ref_text=None,
    src_language="ko",
    ref_language="ko",
):
    if timbre_ref_wav_path is None:
        timbre_ref_wav_path = ref_wav_path

    gen_audio = inference_pipeline.inference_ar_and_fm(
        src_wav_path=None,
        src_text=src_text,
        style_ref_wav_path=ref_wav_path,
        timbre_ref_wav_path=timbre_ref_wav_path,
        style_ref_wav_text=ref_text,
        src_text_language=src_language,
        style_ref_wav_text_language=ref_language,
        flow_matching_steps=256,
        ar_temperature=0.4,        # 0.2 -> 0.4: 더 다양한 생성
        ar_top_k=0,
        ar_top_p=0.9,             # 0.7 -> 0.9: 더 넓은 선택
        ar_repeat_penalty=1.05,   
        ar_min_new_tokens=100,    # 50 -> 100: 충분한 길이 보장
        ar_max_length=800,        
        ar_do_sample=True,        # False -> True: 확률적 샘플링
        no_repeat_ngram_size=4,   # 8 -> 4: n-gram 반복 제한 완화
        ar_prompt_output_tokens=32,
        disable_style_text_concat=True,
    )

    assert output_path is not None
    save_audio(gen_audio, output_path=output_path)
    print(f"Generated audio saved to: {output_path}")

if __name__ == "__main__":
    # ===== Device =====
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # ===== Use locally trained models =====
    
    # Content-Style Tokenizer (fvq8192)
    content_style_tokenizer_ckpt_path = "./ckpts/vevo/fvq8192/checkpoint/epoch-0035_step-0182000_loss-13.510459/pytorch_model.bin"

    # Autoregressive Transformer (ar_synthesis)
    ar_cfg_path = "./egs/vc/AutoregressiveTransformer/ar_synthesis.json"
    ar_ckpt_path = "./ckpts/vevo/ar_synthesis/checkpoint/epoch-0074_step-0378000_loss-0.034367/pytorch_model.bin"

    # Flow Matching Transformer (fm_contentstyle)
    fmt_cfg_path = "./egs/vc/FlowMatchingTransformer/fm_contentstyle.json"
    fmt_ckpt_path = "./ckpts/vevo/fm_contentstyle/checkpoint/epoch-0060_step-0838000_loss-0.335717/pytorch_model.bin"

    # Vocoder (using HuggingFace for now)
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )
    vocoder_cfg_path = "./models/vc/vevo/config/Vocoder.json"
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    print("Loading models...")
    
    # ===== Inference Pipeline =====
    inference_pipeline = VevoInferencePipeline(
        content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
        ar_cfg_path=ar_cfg_path,
        ar_ckpt_path=ar_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )

    print("Models loaded successfully!")

    # ===== Load Korean text data =====
    with open("/app/data/vevo/src_text.txt", "r", encoding="utf-8") as f:
        src_text = f.read().strip().strip('"')
    
    with open("/app/data/vevo/ref_text.txt", "r", encoding="utf-8") as f:
        ref_text = f.read().strip().strip('"')
    # Ensure AR style-text concatenation doesn't alter content by making them identical
    
    # ref_text = src_text

    ref_wav_path = "/app/data/vevo/5f4141e29dd513131eacee2f_happy.wav"

    print(f"Source text: {src_text}")
    print(f"Reference text: {ref_text}")
    print(f"Reference audio: {ref_wav_path}")

    # ===== Generate timestamped output filename =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/app/data/vevo/output_korean_tts_{timestamp}.wav"

    # ===== Korean TTS Inference =====
    korean_vevo_tts(
        src_text=src_text,
        ref_wav_path=ref_wav_path,
        output_path=output_path,
        ref_text=ref_text,
        src_language="ko",
        ref_language="ko",
    )

    print("Korean TTS inference completed!")
