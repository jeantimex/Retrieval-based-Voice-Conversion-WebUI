# Running Retrieval-based Voice Conversion (RVC) on macOS

This document provides a comprehensive guide on how to set up, configure, and run the **Retrieval-based-Voice-Conversion-WebUI** project on macOS (optimized for Apple Silicon / M-series Mac).

---

## 📋 System Requirements

Ensure you have the following system tools installed via [Homebrew](https://brew.sh/):
* **Python 3.8**: Required by RVC dependencies (like legacy `numba` and `llvmlite` versions).
* **ffmpeg**: For audio processing.
* **aria2**: Fast CLI download tool used to download models.

To install these system dependencies, run:
```bash
brew install python@3.8 ffmpeg aria2 libxml2 libxslt zlib
```

---

## 🛠️ Step-by-Step Setup

### 1. Initialize Python 3.8 Virtual Environment
Create and activate a virtual environment under the project root:
```bash
# Create venv using Python 3.8
python3.8 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### 2. Solve macOS Shared Library Compatibility (`libz` & `lxml`)
Older compiled packages like `lxml` might raise dynamic link error issues on macOS (e.g. `Library not loaded: @rpath/libz.1.dylib`). Fix this by linking Homebrew's `libz` into the local linker search path and force-rebuilding `lxml`:
```bash
# Symlink libz into Homebrew's main library folder
ln -s /opt/homebrew/opt/zlib/lib/libz.1.dylib /opt/homebrew/lib/libz.1.dylib
ln -s /opt/homebrew/opt/zlib/lib/libz.dylib /opt/homebrew/lib/libz.dylib

# Force recompile lxml pointing to Homebrew libraries
export LDFLAGS="-L/opt/homebrew/opt/libxml2/lib -L/opt/homebrew/opt/libxslt/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libxml2/include -I/opt/homebrew/opt/libxslt/include"
pip install --force-reinstall --no-cache-dir lxml
```

### 3. Downgrade Pip (Dependency Resolver Fix)
To install older dependencies (like `omegaconf==2.0.6`) which contain legacy metadata formats that modern Pip versions reject, downgrade `pip` to version `24.0`:
```bash
pip install "pip<24.1"
```

### 4. Install Requirements
Install the Python dependencies:
```bash
pip install -r requirements.txt
```
*(Note: The duplicate/problematic PyPI package `aria2` should be commented out or removed from `requirements.txt` to prevent metadata compilation issues on macOS).*

### 5. Download Pre-trained Models
Download all required model checkpoints (like `hubert_base.pt`, `rmvpe.pt`, and `uvr5_weights`) from Hugging Face:
```bash
chmod +x tools/dlmodels.sh
./tools/dlmodels.sh
```

---

## 🚀 Running the WebUI

Always run the WebUI with Apple Silicon Metal Performance Shaders (MPS) hardware acceleration enabled by setting these environment variables:

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. Configure MPS fallback and memory watermarks for macOS
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# 3. Start the application
python infer-web.py --pycmd python
```

The WebUI will initialize and start running at:
👉 **[http://localhost:7865](http://localhost:7865)**

---

## 🎙️ Automated YouTube Cover Pipeline

We have created an automated pipeline script [`cover_pipeline.py`](file:///Users/jeantimex/Workspace/github/Retrieval-based-Voice-Conversion-WebUI/cover_pipeline.py) in the root of the project directory. This script performs the following actions end-to-end:
1. Downloads the YouTube video/song audio via `yt-dlp`.
2. Automatically separates vocals & instrumentals using the high-quality UVR5 models.
3. Automatically sets up the dataset, pre-processes, extracts pitch features (F0 with RMVPE) and speech features.
4. Trains the RVC model dynamically on the vocals of your target voice.
5. Performs Faiss index training for feature retrieval.
6. Converts the song's vocals into the trained target voice (inference).
7. Mixes the new vocals back with the original background instrumentals.

### Usage
Make sure your virtual environment is active, then run the script with:
```bash
python cover_pipeline.py \
  -s "https://www.youtube.com/watch?v=SONG_LINK" \
  -v "https://www.youtube.com/watch?v=VOICE_SOURCE_LINK" \
  -m "custom_voice" \
  -p 0 \
  -e 50
```

### Parameters
* `-s` / `--song-url` (Required): YouTube link for the song to cover.
* `-v` / `--voice-url` (Required): YouTube link for the target voice to train on.
* `-m` / `--model-name` (Default: `custom_voice`): Name of the generated model.
* `-p` / `--pitch-shift` (Default: `0`): Transposes the voice (e.g. `+12` or `-12`). Use this if covering a song of the opposite gender.
* `-e` / `--epochs` (Default: `50`): Total training epochs. Higher values increase voice similarity but take longer.
* `-b` / `--batch-size` (Default: `8`): Batch size for training.
* `-a` / `--pitch-method` (Default: `pm`, choices: `["pm", "harvest", "dio", "rmvpe"]`): Pitch extraction algorithm for voice conversion. Use `harvest` for better singing quality on CPU.
* `-i` / `--index-rate` (Default: `0.75`, range `0.0` to `1.0`): Feature retrieval index rate. Higher values (e.g. `0.9` or `1.0`) make the output vocal closer to the target voice by using the trained feature index more heavily.
* `--protect` (Default: `0.33`): Consonant protection rate. Protects voiceless consonants and breath sounds. Decreasing this (e.g. to `0.25` or `0.0`) can increase target voice similarity but may introduce artifacts.
* `--rms-mix-rate` (Default: `0.25`, range `0.0` to `1.0`): Volume envelope mix rate. `0.0` uses the source vocals volume envelope completely, while `1.0` matches the target voice's volume dynamic.
* `--filter-radius` (Default: `3`): Median filtering radius applied to pitch extraction results.
* `--denoise`: Boolean flag. If specified, applies high-quality FFmpeg filters (highpass filter, `afftdn` spectral denoiser, and `agate` noise gate) to clean RVC artifacts, clicks, background hum, and noise from the finished vocals before mixing.

The finished cover song will be saved inside the [`downloads/`](file:///Users/jeantimex/Workspace/github/Retrieval-based-Voice-Conversion-WebUI/downloads) directory as `<model_name>_cover.wav`.

### macOS Optimization Details
To ensure rock-solid stability and bypass library conflicts on macOS Apple Silicon, the script incorporates the following details:
1. **Early HuBERT Loading**: Pre-loads the fairseq HuBERT model at start before importing complex compiled packages (like `faiss`, `librosa`, or `parselmouth`) to prevent C++ symbol conflicts and Segmentation Faults.
2. **Thread Pool Limiting & CPU Inference**: Runs the final inference stage on CPU and forces OpenMP/MKL thread counts to `1` (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`). This eliminates multi-threaded deadlocks and SegFaults.
3. **CPU Pitch Detection**: Uses `"pm"` (Parselmouth) for pitch extraction during voice conversion to avoid loading the heavy PyTorch RMVPE model in parallel with HuBERT, bypassing potential MPS memory allocator issues.

---

## 🔍 Troubleshooting

### 1. `Library not loaded: @rpath/libxml2.2.dylib` or `libz.1.dylib`
This occurs when the compiled Python bindings for `lxml` cannot locate Homebrew's native XML/compression libraries. Follow **Step 2** to symlink `libz` and rebuild `lxml`.

### 2. GPU Memory Instability or MPS Errors
If you run into out-of-memory crashes on Apple Silicon during training or inference, ensure you have set `export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`. This disables the strict allocation limits of PyTorch on macOS, letting it use your system's unified memory dynamically.
