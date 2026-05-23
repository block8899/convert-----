"""
U2NetP Portrait Segmentation → NCNN via pnnx CLI
Dựa trên code convert đã thành công
"""

import os
import sys
import subprocess
import urllib.request

WORK_DIR = "convert_tmp"
OUTPUT_DIR = "output"

MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
MODEL_PATH = "u2netp.onnx"

print("=" * 50)
print("U2NetP → NCNN Converter")
print("=" * 50)

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("1. Downloading U2NetP ONNX model...")
if not os.path.exists(MODEL_PATH):
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as response:
        with open(MODEL_PATH, "wb") as f:
            import shutil
            shutil.copyfileobj(response, f)
    print(f"   Done: {os.path.getsize(MODEL_PATH)/1024/1024:.1f} MB")
else:
    print(f"   Already exists: {os.path.getsize(MODEL_PATH)/1024/1024:.1f} MB")

print("2. Simplifying ONNX...")
SIMPLIFIED = os.path.join(WORK_DIR, "u2netp_simple.onnx")
subprocess.run(
    [sys.executable, "-m", "onnxsim", MODEL_PATH, SIMPLIFIED],
    check=True,
)
print(f"   Simplified: {os.path.getsize(SIMPLIFIED)/1024/1024:.1f} MB")

print("3. Converting ONNX → NCNN via pnnx CLI...")
result = subprocess.run(
    ["pnnx", "u2netp_simple.onnx", "inputshape=1,3,320,320"],
    capture_output=True,
    text=True,
    timeout=300,
    cwd=WORK_DIR,
)
print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

if result.returncode != 0:
    print(f"   WARNING: pnnx exit code {result.returncode}")

# pnnx tạo file theo tên input: u2netp_simple.ncnn.param
pf1 = os.path.join(WORK_DIR, "u2netp_simple.ncnn.param")
bf1 = os.path.join(WORK_DIR, "u2netp_simple.ncnn.bin")

# Rename output
pf_out = os.path.join(OUTPUT_DIR, "humansegv2.ncnn.param")
bf_out = os.path.join(OUTPUT_DIR, "humansegv2.ncnn.bin")

if os.path.exists(pf1) and os.path.exists(bf1):
    os.rename(pf1, pf_out)
    os.rename(bf1, bf_out)
elif os.path.exists(pf1):
    os.rename(pf1, pf_out)
    print("   WARNING: .bin not found")

print("4. Verifying...")
if os.path.exists(pf_out) and os.path.exists(bf_out):
    sp = os.path.getsize(pf_out) / 1024
    sb = os.path.getsize(bf_out) / 1024
    print(f"   humansegv2.ncnn.param: {sp:.1f} KB")
    print(f"   humansegv2.ncnn.bin:   {sb:.1f} KB")
    print(f"   Total: {(sp + sb):.0f} KB ({(sp + sb) / 1024:.1f} MB)")

    with open(pf_out, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("Input"):
            parts = line.split()
            if len(parts) >= 3:
                print(f"   Input blob: {parts[2]}")
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("7767517"):
            parts = line.split()
            if len(parts) >= 4:
                print(f"   Output blob: {parts[-1]}")
                break

    print("\n" + "=" * 50)
    print("DONE!")
    print(f"  {pf_out}")
    print(f"  {bf_out}")
    print("=" * 50)
else:
    ncnn_files = [f for f in os.listdir(WORK_DIR) if 'ncnn' in f]
    print(f"FAILED! ncnn files found: {ncnn_files}")
    print(f"All files: {os.listdir(WORK_DIR)}")
    sys.exit(1)
