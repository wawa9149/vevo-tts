# Korean TTS Inference Script using trained models
import os
import json
import uuid
import torch
from datetime import datetime
from models.vc.vevo.vevo_utils import *

def korean_vevo_tts(
    src_text,
    ref_wav_path,
    output_path,
    ref_text,
    src_language,
    ref_language,
    flow_matching_steps,
    ar_temperature,
    ar_top_k,
    ar_top_p,
    ar_repeat_penalty,
    ar_min_new_tokens,
    ar_max_length,
    ar_do_sample,
    no_repeat_ngram_size,
    ar_prompt_output_tokens,
    disable_style_text_concat,
    timbre_ref_wav_path=None,
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
        flow_matching_steps=flow_matching_steps,
        ar_temperature=ar_temperature,
        ar_top_k=ar_top_k,
        ar_top_p=ar_top_p,
        ar_repeat_penalty=ar_repeat_penalty,
        ar_min_new_tokens=ar_min_new_tokens,
        ar_max_length=ar_max_length,
        ar_do_sample=ar_do_sample,
        no_repeat_ngram_size=no_repeat_ngram_size,
        ar_prompt_output_tokens=ar_prompt_output_tokens,
        disable_style_text_concat=disable_style_text_concat,
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
    content_style_tokenizer_ckpt_path = "/app/data/vevo/models/fvq8192/checkpoint_backup/epoch-0813_step-2190000_loss-11.957558/pytorch_model.bin"

    # Autoregressive Transformer (ar_synthesis)
    ar_cfg_path = "./models/vc/vevo/config/PhoneToVq8192.json"
    ar_ckpt_path = "/app/data/vevo/models/ar_synthesis/checkpoint/epoch-0167_step-0214000_loss-0.058975//pytorch_model.bin"

    # Flow Matching Transformer (fm_contentstyle)
    fmt_cfg_path = "./models/vc/vevo/config/Vq8192ToMels.json"
    fmt_ckpt_path = "/app/data/vevo/models/fm_contentstyle/checkpoint_backup/epoch-0060_step-0837000_loss-0.221667/pytorch_model.bin"

    # Vocoder (using HuggingFace for now)
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="/app/data/vevo/models/Vevo",
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

    # ===== Input Data =====
    with open("/home/ubuntu/tts-audio/data/inference/src_text.txt", "r", encoding="utf-8") as f:
        src_text = f.read().strip().strip('"')
    print(f"Source text: {src_text}")
    
    ref_wav_path = "/home/ubuntu/tts-audio/data/inference/5f4141e29dd513131eacee2f_happy.wav"
    with open("/home/ubuntu/tts-audio/data/inference/ref_text.txt", "r", encoding="utf-8") as f:
        ref_text = f.read().strip().strip('"')
    print(f"Reference text: {ref_text}")

    # ===== Inference Parameters =====
    inference_params = {
        "src_language": "ko",
        "ref_language": "ko",
        "flow_matching_steps": 30,
        "ar_temperature": 0.65,
        "ar_top_k": 50,
        "ar_top_p": 0.95,
        "ar_repeat_penalty": 1.1,
        "ar_min_new_tokens": 50,
        "ar_max_length": 800,
        "ar_do_sample": True,
        "no_repeat_ngram_size": 5,
        "ar_prompt_output_tokens": 32,
        "disable_style_text_concat": True,
    }

    # ===== Prepare Output Directory =====
    date_str = datetime.now().strftime("%Y%m%d")
    output_dir = f"/app/workspace/vevo-infer/{date_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    file_uuid = str(uuid.uuid4())
    output_path = os.path.join(output_dir, f"{file_uuid}.wav")
    metadata_path = os.path.join(output_dir, f"{file_uuid}.json")

    print(f"Source text: {src_text}")
    print(f"Reference audio: {ref_wav_path}")
    print(f"Output directory: {output_dir}")

    # ===== Run Inference =====
    korean_vevo_tts(
        src_text=src_text,
        ref_wav_path=ref_wav_path,
        output_path=output_path,
        ref_text=ref_text, 
        **inference_params
    )

    # ===== Save Metadata =====
    metadata = {
        "uuid": file_uuid,
        "timestamp": datetime.now().isoformat(),
        "date": date_str,
        "source_text": src_text,
        "reference_audio": ref_wav_path,
        "model_checkpoints": {
            "content_style_tokenizer": content_style_tokenizer_ckpt_path,
            "ar_transformer": ar_ckpt_path,
            "flow_matching_transformer": fmt_ckpt_path,
            "vocoder": vocoder_ckpt_path,
        },
        "model_configs": {
            "ar_config": ar_cfg_path,
            "fmt_config": fmt_cfg_path,
            "vocoder_config": vocoder_cfg_path,
        },
        "inference_parameters": inference_params
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Metadata saved to: {metadata_path}")
    print("Inference completed!")