import os
import json
import numpy as np
import torch
from tqdm import tqdm
from jiwer import wer, cer
from resemblyzer import preprocess_wav, VoiceEncoder
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torchaudio


def transcribe_korean_whisper(audio_path, processor, model):
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    inputs = processor(wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(input_features)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def compute_cer_wer_korean(cnv_folder, transcriptions_dict, processor, model):
    cer_list, wer_list = [], []
    for fname in tqdm(os.listdir(cnv_folder), desc="CER/WER"):
        if not fname.endswith(".wav") or fname not in transcriptions_dict:
            continue
        gt = transcriptions_dict[fname].strip()
        hyp = transcribe_korean_whisper(os.path.join(cnv_folder, fname), processor, model).strip()
        cer_list.append(cer(gt, hyp))
        wer_list.append(wer(gt, hyp))
    return np.mean(cer_list), np.mean(wer_list)


def compute_secs(cnv_folder, ref_folder):
    encoder = VoiceEncoder().cuda()
    cos_sims = []
    for fname in tqdm(os.listdir(cnv_folder), desc="SECS"):
        if not fname.endswith(".wav"):
            continue
        cnv_path = os.path.join(cnv_folder, fname)
        ref_path = os.path.join(ref_folder, fname)
        if not os.path.exists(ref_path):
            continue
        try:
            cnv_emb = encoder.embed_utterance(preprocess_wav(cnv_path))
            ref_emb = encoder.embed_utterance(preprocess_wav(ref_path))
            cos_sim = np.inner(cnv_emb, ref_emb) / (np.linalg.norm(cnv_emb) * np.linalg.norm(ref_emb))
            cos_sims.append(cos_sim)
        except Exception as e:
            print(f"Error in SECS for {fname}: {e}")
    return np.mean(cos_sims) if cos_sims else 0.0


def evaluate_folder_korean(cnv_folder, ref_folder, transcription_json):
    with open(transcription_json, encoding="utf-8") as f:
        transcriptions = json.load(f)

    processor = AutoProcessor.from_pretrained("SungBeom/whisper-small-ko")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("SungBeom/whisper-small-ko").to("cuda")

    print("Evaluating CER / WER with Whisper-small-ko...")
    cer_score, wer_score = compute_cer_wer_korean(cnv_folder, transcriptions, processor, model)

    print("Evaluating SECS with Resemblyzer...")
    secs_score = compute_secs(cnv_folder, ref_folder)

    print("\n Final Results:")
    print(f"CER  : {cer_score:.4f}")
    print(f"WER  : {wer_score:.4f}")
    print(f"SECS : {secs_score:.4f}")

    return cer_score, wer_score, secs_score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CER, WER, SECS on Korean VC results.")
    parser.add_argument("--converted_folder", type=str, required=True, help="Folder with converted .wav files")
    parser.add_argument("--reference_folder", type=str, required=True, help="Folder with target/reference .wav files")
    parser.add_argument("--transcription_json", type=str, required=True, help="JSON with {wav_name: ground_truth_text}")

    args = parser.parse_args()

    evaluate_folder_korean(args.converted_folder, args.reference_folder, args.transcription_json)
