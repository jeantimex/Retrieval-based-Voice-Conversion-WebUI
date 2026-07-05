#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RVC automated cover pipeline script.
This script automates downloading YouTube audio, separating vocals,
training a custom voice model, and generating a voice-converted cover.
"""

import os
import sys
import argparse
import subprocess
import shutil
import glob
import random
import json
from multiprocessing import cpu_count

# Ensure absolute paths and RVC modules are loadable
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)

from dotenv import load_dotenv
load_dotenv()

# Set environment variables for macOS (MPS / CPU)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Set TEMP env variable for UVR5 audio formatter
TEMP_DIR = os.path.join(now_dir, "TEMP")
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ["TEMP"] = TEMP_DIR

# Set up clean temporary directories
DOWNLOADS_DIR = os.path.join(now_dir, "downloads")
DATASET_DIR = os.path.join(now_dir, "dataset")

os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)


def find_yt_dlp():
    brew_path = "/opt/homebrew/bin/yt-dlp"
    if os.path.exists(brew_path):
        return brew_path
    return "yt-dlp"


def download_youtube_audio(url, name):
    print(f"\n📥 Downloading audio from YouTube: {url}...")
    temp_name = os.path.join(DOWNLOADS_DIR, name)
    
    # Remove existing files with the same name
    for f in glob.glob(f"{temp_name}.*"):
        try:
            os.remove(f)
        except Exception:
            pass
            
    cmd = [
        find_yt_dlp(),
        "-x",
        "--audio-format", "wav",
        "-o", f"{temp_name}.%(ext)s",
        url
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Find downloaded file
    target_file = f"{temp_name}.wav"
    if os.path.exists(target_file):
        print(f"✅ Downloaded successfully to: {target_file}")
        return target_file
    else:
        files = glob.glob(f"{temp_name}.*")
        if files:
            print(f"✅ Downloaded successfully to: {files[0]}")
            return files[0]
        raise FileNotFoundError(f"❌ Failed to download audio for {name}")


def separate_audio(input_file, name):
    print(f"\n🎛️ Separating vocals and instrumentals for {input_file}...")
    
    # Create clean temp input directory for UVR5
    temp_in = os.path.join(DOWNLOADS_DIR, f"temp_uvr_in_{name}")
    shutil.rmtree(temp_in, ignore_errors=True)
    os.makedirs(temp_in, exist_ok=True)
    shutil.copy(input_file, temp_in)
    
    # Create clean output directories
    vocal_out = os.path.join(DOWNLOADS_DIR, f"temp_uvr_vocals_{name}")
    ins_out = os.path.join(DOWNLOADS_DIR, f"temp_uvr_instrumentals_{name}")
    shutil.rmtree(vocal_out, ignore_errors=True)
    shutil.rmtree(ins_out, ignore_errors=True)
    os.makedirs(vocal_out, exist_ok=True)
    os.makedirs(ins_out, exist_ok=True)
    
    # Set model weights root variable
    os.environ["weight_uvr5_root"] = "assets/uvr5_weights"
    
    # Import UVR module
    from infer.modules.uvr5.modules import uvr
    
    print("Running UVR5 (this may take 1-2 minutes)...")
    # Consume generator to complete separation and print logs
    for info in uvr(
        model_name="HP5_only_main_vocal",
        inp_root=temp_in,
        save_root_vocal=vocal_out,
        paths=[],
        save_root_ins=ins_out,
        agg=10,
        format0="wav"
    ):
        print(info)
        
    vocal_files = [f for f in os.listdir(vocal_out) if f.endswith(".wav")]
    ins_files = [f for f in os.listdir(ins_out) if f.endswith(".wav")]
    
    if not vocal_files or not ins_files:
        raise RuntimeError("❌ Audio separation failed. No output files produced.")
        
    vocal_path = os.path.join(vocal_out, vocal_files[0])
    ins_path = os.path.join(ins_out, ins_files[0])
    
    print(f"✅ Separated Vocals: {vocal_path}")
    print(f"✅ Separated Instrumentals: {ins_path}")
    return vocal_path, ins_path


def run_preprocessing(dataset_dir, exp_dir, sample_rate):
    print("\n⚙️ Step 1: Preprocessing training dataset...")
    n_p = min(cpu_count(), 8)
    log_dir = os.path.join(now_dir, "logs", exp_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    with open(os.path.join(log_dir, "preprocess.log"), "w") as f:
        f.write("")
        
    cmd = [
        sys.executable,
        "infer/modules/train/preprocess.py",
        dataset_dir,
        str(sample_rate),
        str(n_p),
        log_dir,
        "False",
        "3.0"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("✅ Preprocessing complete.")


def run_feature_extraction(exp_dir, f0_method, version):
    print("\n⚙️ Step 2: Extracting pitch (F0) and speech features...")
    n_p = min(cpu_count(), 8)
    log_dir = os.path.join(now_dir, "logs", exp_dir)
    
    # 1. Pitch (F0) extraction
    f0_cmd = [
        sys.executable,
        "infer/modules/train/extract/extract_f0_print.py",
        log_dir,
        str(n_p),
        f0_method
    ]
    print(f"Executing F0: {' '.join(f0_cmd)}")
    subprocess.run(f0_cmd, check=True)
    
    # 2. HuBERT Feature extraction
    feat_cmd = [
        sys.executable,
        "infer/modules/train/extract_feature_print.py",
        "mps",  # use Metal Performance Shaders on macOS
        "1",
        "0",
        "0",
        log_dir,
        version,
        "False"
    ]
    print(f"Executing Features: {' '.join(feat_cmd)}")
    subprocess.run(feat_cmd, check=True)
    print("✅ Feature extraction complete.")


def generate_training_configs(exp_dir, sample_rate_str, version, if_f0, spk_id):
    print("\n⚙️ Generating filelist and training configuration...")
    exp_path = os.path.join(now_dir, "logs", exp_dir)
    gt_wavs_dir = os.path.join(exp_path, "0_gt_wavs")
    feature_dir = os.path.join(exp_path, "3_feature768" if version == "v2" else "3_feature256")
    
    f0_dir = os.path.join(exp_path, "2a_f0")
    f0nsf_dir = os.path.join(exp_path, "2b-f0nsf")
    
    # Find matching names
    names = set([n.split(".")[0] for n in os.listdir(gt_wavs_dir) if n.endswith(".wav")]) & \
            set([n.split(".")[0] for n in os.listdir(feature_dir) if n.endswith(".npy")])
            
    if if_f0:
        names = names & \
                set([n.split(".")[0] for n in os.listdir(f0_dir) if n.endswith(".npy") or n.endswith(".wav.npy")]) & \
                set([n.split(".")[0] for n in os.listdir(f0nsf_dir) if n.endswith(".npy") or n.endswith(".wav.npy")])
                
    if not names:
        raise ValueError("❌ No matching files found to build filelist.txt.")
        
    opt = []
    for name in names:
        if if_f0:
            opt.append(
                f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{spk_id}"
            )
        else:
            opt.append(
                f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{spk_id}"
            )
            
    # Append mute files if they exist to prevent empty batches
    mute_wav = os.path.join(now_dir, "logs", "mute", f"0_gt_wavs", f"mute{sample_rate_str}.wav")
    mute_fea = os.path.join(now_dir, "logs", "mute", "3_feature768" if version == "v2" else "3_feature256", "mute.npy")
    mute_f0 = os.path.join(now_dir, "logs", "mute", "2a_f0", "mute.wav.npy")
    mute_f0nsf = os.path.join(now_dir, "logs", "mute", "2b-f0nsf", "mute.wav.npy")
    
    if os.path.exists(mute_wav):
        for _ in range(2):
            if if_f0:
                opt.append(f"{mute_wav}|{mute_fea}|{mute_f0}|{mute_f0nsf}|{spk_id}")
            else:
                opt.append(f"{mute_wav}|{mute_fea}|{spk_id}")
                
    random.shuffle(opt)
    
    with open(os.path.join(exp_path, "filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(opt))
        
    # Copy template config.json
    if version == "v1" or sample_rate_str == "40k":
        config_folder = "v1"
    else:
        config_folder = version
        
    config_template = os.path.join(now_dir, "configs", config_folder, f"{sample_rate_str}.json")
    config_dest = os.path.join(exp_path, "config.json")
    
    if os.path.exists(config_template):
        shutil.copy(config_template, config_dest)
        print("✅ Training configs generated.")
    else:
        from configs.config import Config
        cfg = Config()
        config_key = f"{config_folder}/{sample_rate_str}.json"
        if config_key in cfg.json_config:
            with open(config_dest, "w", encoding="utf-8") as f:
                json.dump(cfg.json_config[config_key], f, indent=4)
            print("✅ Training configs generated.")
        else:
            raise FileNotFoundError(f"❌ Config template not found for {config_key}")


def run_training(exp_dir, sample_rate_str, total_epochs, batch_size, version, if_f0):
    print(f"\n🏋️ Step 3: Starting voice training for {total_epochs} epochs...")
    f0_prefix = "f0" if if_f0 else ""
    pretrained_G = os.path.join(now_dir, "assets", f"pretrained_{version}", f"{f0_prefix}G{sample_rate_str}.pth")
    pretrained_D = os.path.join(now_dir, "assets", f"pretrained_{version}", f"{f0_prefix}D{sample_rate_str}.pth")
    
    cmd = [
        sys.executable,
        "infer/modules/train/train.py",
        "-e", exp_dir,
        "-sr", sample_rate_str,
        "-f0", "1" if if_f0 else "0",
        "-bs", str(batch_size),
        "-te", str(total_epochs),
        "-se", "10",
        "-l", "1",
        "-c", "0",
        "-sw", "0",
        "-v", version
    ]
    
    if os.path.exists(pretrained_G) and os.path.exists(pretrained_D):
        cmd.extend(["-pg", pretrained_G, "-pd", pretrained_D])
    else:
        print("⚠️ Pretrained files not found. Training from scratch (might take longer).")
        
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode not in [0, 2333333]:
        raise subprocess.CalledProcessError(res.returncode, cmd)
    print("✅ Training complete.")


def run_index_training(exp_dir, version):
    print("\n⚙️ Step 4: Training Faiss feature index...")
    exp_path = os.path.join(now_dir, "logs", exp_dir)
    feature_dir = os.path.join(exp_path, "3_feature768" if version == "v2" else "3_feature256")
    
    if not os.path.exists(feature_dir):
        print("⚠️ Feature directory not found, skipping index.")
        return None
        
    listdir_res = [f for f in os.listdir(feature_dir) if f.endswith(".npy")]
    if not listdir_res:
        print("⚠️ No features found, skipping index.")
        return None
        
    import numpy as np
    import faiss
    from sklearn.cluster import MiniBatchKMeans
    
    npys = []
    for name in sorted(listdir_res):
        npys.append(np.load(os.path.join(feature_dir, name)))
        
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    
    if big_npy.shape[0] > 200000:
        print("K-means clustering to 10k centers...")
        try:
            big_npy = MiniBatchKMeans(
                n_clusters=10000,
                verbose=False,
                batch_size=256 * 4,
                compute_labels=False,
                init="random"
            ).fit(big_npy).cluster_centers_
        except Exception as e:
            print(f"⚠️ K-means failed: {e}. Using raw features.")
            
    np.save(os.path.join(exp_path, "total_fea.npy"), big_npy)
    
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    dim = 768 if version == "v2" else 256
    index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
    
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    
    trained_idx_path = os.path.join(exp_path, f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir}_{version}.index")
    faiss.write_index(index, trained_idx_path)
    
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
        
    added_idx_path = os.path.join(exp_path, f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir}_{version}.index")
    faiss.write_index(index, added_idx_path)
    
    # Copy index to weights folder
    weights_idx_path = os.path.join(now_dir, "assets", "weights", f"{exp_dir}_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir}_{version}.index")
    try:
        shutil.copy(added_idx_path, weights_idx_path)
        print(f"✅ Index copied to weights root: {weights_idx_path}")
    except Exception as e:
        print(f"⚠️ Index link failed: {e}")
        
    return weights_idx_path


def run_inference(model_name, input_wav, output_wav, index_path, pitch_shift,
                  pitch_method="pm", index_rate=0.75, protect=0.33, rms_mix_rate=0.25, filter_radius=3):
    print(f"\n🎙️ Step 5: Performing RVC voice conversion on original vocals...")
    
    # 1. Set environment variables to force CPU and single threading for inference stability on macOS
    os.environ["FORCE_CPU"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # 2. Early load Hubert model on CPU before other RVC packages import to prevent SegFault
    print("Loading Hubert model to CPU first...")
    from fairseq import checkpoint_utils
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["assets/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to("cpu").float().eval()
    print("Hubert loaded successfully!")

    from configs.config import Config
    from infer.modules.vc.modules import VC
    from scipy.io import wavfile
    
    config = Config()
    config.device = "cpu"
    config.is_half = False
    
    vc = VC(config)
    vc.hubert_model = hubert_model
    
    # RVC expects filename with .pth extension
    model_file = f"{model_name}.pth"
    vc.get_vc(model_file)
    
    info, wav_opt = vc.vc_single(
        sid=0,
        input_audio_path=input_wav,
        f0_up_key=pitch_shift,
        f0_file=None,
        f0_method=pitch_method,
        file_index=index_path,
        file_index2=None,
        index_rate=index_rate,
        filter_radius=filter_radius,
        resample_sr=0,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
    )
    
    if wav_opt[0] is None or wav_opt[1] is None:
        raise RuntimeError(f"Voice conversion failed! info: {info}")
        
    wavfile.write(output_wav, wav_opt[0], wav_opt[1])
    print(f"✅ Voice converted vocals saved to: {output_wav}")
    return output_wav


def mix_audio(vocals_path, ins_path, output_path, denoise=False):
    print(f"\n🎵 Step 6: Mixing converted vocals and original instrumentals...")
    
    vocal_input = vocals_path
    if denoise:
        print("🧹 Applying FFmpeg de-noising and noise gating to vocals...")
        temp_clean_vocals = vocals_path.replace(".wav", "_clean.wav")
        cmd_denoise = [
            "ffmpeg",
            "-i", vocals_path,
            "-af", "highpass=f=80,afftdn,agate=threshold=0.01",
            temp_clean_vocals,
            "-y"
        ]
        print(f"Executing denoise: {' '.join(cmd_denoise)}")
        subprocess.run(cmd_denoise, check=True)
        vocal_input = temp_clean_vocals

    cmd = [
        "ffmpeg",
        "-i", vocal_input,
        "-i", ins_path,
        "-filter_complex", "amix=inputs=2:duration=longest",
        output_path,
        "-y"
    ]
    print(f"Executing mix: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"🎉 RVC Cover complete! Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create RVC covers from YouTube links.")
    parser.add_argument("-s", "--song-url", required=True, help="YouTube link to the song to cover")
    parser.add_argument("-v", "--voice-url", required=True, help="YouTube link to the training voice source")
    parser.add_argument("-m", "--model-name", default="custom_voice", help="Name of the voice model")
    parser.add_argument("-p", "--pitch-shift", type=int, default=0, help="Pitch shift transposition (e.g. +12, -12)")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size for training")
    
    # Custom voice conversion parameters
    parser.add_argument("-a", "--pitch-method", default="pm", choices=["pm", "harvest", "dio", "rmvpe"],
                        help="Pitch extraction algorithm: pm (singing/fast), harvest (better/slower), rmvpe (best results), dio (fast speech) (default: pm)")
    parser.add_argument("-i", "--index-rate", type=float, default=0.75,
                        help="Feature retrieval index rate: 1.0 utilizes trained voice index fully, 0.0 uses raw speech features (default: 0.75)")
    parser.add_argument("--protect", type=float, default=0.33,
                        help="Consonant protection rate: protects voiceless consonants and breath sounds from clipping (default: 0.33)")
    parser.add_argument("--rms-mix-rate", type=float, default=0.25,
                        help="Volume envelope mix rate: 0.0 uses source vocals volume completely, 1.0 uses target voice (default: 0.25)")
    parser.add_argument("--filter-radius", type=int, default=3,
                        help="Median filter radius applied to pitch extraction results (default: 3)")
    parser.add_argument("--denoise", action="store_true",
                        help="Apply FFmpeg de-noise (afftdn), highpass filter, and noise gating (agate) to final vocals to remove RVC hiss/noise")
    
    args = parser.parse_args()
    sys.argv = sys.argv[:1]
    
    # Setup directories
    model_dataset_dir = os.path.join(DATASET_DIR, args.model_name)
    shutil.rmtree(model_dataset_dir, ignore_errors=True)
    os.makedirs(model_dataset_dir, exist_ok=True)
    
    # 1. Download audio tracks
    print("--- 📥 DOWNLOADING TRACKS ---")
    song_wav = download_youtube_audio(args.song_url, "song_input")
    voice_wav = download_youtube_audio(args.voice_url, "voice_input")
    
    # 2. Separate vocal tracks
    print("\n--- 🎛️ SEPARATING VOCALS & INSTRUMENTALS ---")
    # Separate training source audio
    voice_vocals, _ = separate_audio(voice_wav, "voice")
    # Copy voice vocals into dataset directory
    shutil.copy(voice_vocals, os.path.join(model_dataset_dir, "voice_clean.wav"))
    
    # Separate cover song audio
    song_vocals, song_instrumentals = separate_audio(song_wav, "song")
    
    # 3. Train RVC Model
    print("\n--- 🏋️ TRAINING RVC MODEL ---")
    sample_rate_str = "40k"
    sample_rate = 40000
    version = "v2"
    if_f0 = True
    
    # Execute training steps
    run_preprocessing(model_dataset_dir, args.model_name, sample_rate)
    run_feature_extraction(args.model_name, "rmvpe", version)
    generate_training_configs(args.model_name, sample_rate_str, version, if_f0, 0)
    run_training(args.model_name, sample_rate_str, args.epochs, args.batch_size, version, if_f0)
    
    # 4. Create index
    index_path = run_index_training(args.model_name, version)
    
    # 5. Inference / Voice Conversion
    print("\n--- 🎙️ APPLYING RVC VOICE CONVERSION ---")
    converted_vocals = os.path.join(DOWNLOADS_DIR, "converted_vocals.wav")
    run_inference(
        model_name=args.model_name, 
        input_wav=song_vocals, 
        output_wav=converted_vocals, 
        index_path=index_path, 
        pitch_shift=args.pitch_shift,
        pitch_method=args.pitch_method,
        index_rate=args.index_rate,
        protect=args.protect,
        rms_mix_rate=args.rms_mix_rate,
        filter_radius=args.filter_radius
    )
    
    # 6. Mix vocals & instrumentals
    print("\n--- 🎵 MIXING COVER SONG ---")
    final_output = os.path.join(DOWNLOADS_DIR, f"{args.model_name}_cover.wav")
    mix_audio(converted_vocals, song_instrumentals, final_output, denoise=args.denoise)
    
    print(f"\n✨ All done! Your finished cover is available at: {final_output}")


if __name__ == "__main__":
    main()
