"""
Convert PP-HumanSegV2 → NCNN
Approach: Clone PaddleSeg → export ONNX → simplify → NCNN
"""

import os
import subprocess
import shutil
import urllib.request
import sys

WORK_DIR = "convert_tmp"
OUTPUT_DIR = "output"
NCNN_TOOLS_URL = (
    "https://github.com/Tencent/ncnn/releases/download/20240820/"
    "ncnn-20240820-ubuntu.zip"
)


def run_cmd(cmd, cwd=None):
    print(f"    $ {cmd}")
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)


def clone_paddleseg():
    repo_dir = os.path.join(WORK_DIR, "PaddleSeg")
    if os.path.exists(repo_dir):
        print("    PaddleSeg already cloned.")
        return repo_dir

    print("[1/6] Cloning PaddleSeg repo...")
    run_cmd(
        "git clone --depth 1 https://github.com/PaddlePaddle/PaddleSeg.git",
        cwd=WORK_DIR,
    )
    return repo_dir


def export_onnx(repo_dir):
    onnx_path = os.path.join(WORK_DIR, "humansegv2.onnx")
    if os.path.exists(onnx_path):
        return onnx_path

    print("[2/6] Exporting PP-HumanSegV2 to ONNX...")

    # Write a minimal export script
    export_script = os.path.join(WORK_DIR, "export_onnx.py")
    with open(export_script, "w") as f:
        f.write('''

import paddle
from paddleseg.models import PPMobileSeg

# Build model with correct config
try:
    from paddleseg.models.backbones import STDC2
    backbone = STDC2()
except Exception:
    from paddleseg.models.backbones import STDC1
    backbone = STDC1()

model = PPMobileSeg(
    num_classes=2,
    backbone=backbone,
    align_corners=False,
)

# PaddleSeg loads pretrained weights via config, try direct download
import urllib.request
import os

weight_urls = [
    "https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/pp_humansegv2_mobile_192x192_pretrained/model.pdparams",
    "https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/pphumansegv2_mobile_192x192_pretrained/model.pdparams",
]

weights = None
for url in weight_urls:
    try:
        urllib.request.urlretrieve(url, "tmp_weights.pdparams")
        weights = paddle.load("tmp_weights.pdparams")
        print(f"Loaded weights from {url}")
        break
    except Exception:
        continue

if weights:
    model.set_state_dict(weights)
else:
    print("WARNING: No pretrained weights found, using random init")

model.eval()

# Export
input_spec = paddle.static.InputSpec(
    shape=[1, 3, 192, 192], dtype="float32", name="x"
)
paddle.onnx.export(
    model,
    "humansegv2",
    input_spec=[input_spec],
    opset_version=11,
)
print("Export done!")

# Check output
for f in os.listdir("."):
    if f.endswith(".onnx"):
        print(f"Found: {f} ({os.path.getsize(f)} bytes)")
''')

    # Install paddleseg in editable mode
    run_cmd(f"{sys.executable} -m pip install -e {repo_dir}", cwd=WORK_DIR)

    import sys
    run_cmd(f"{sys.executable} export_onnx.py", cwd=WORK_DIR)

    # Find onnx file
    for f in os.listdir(WORK_DIR):
        if f.endswith(".onnx"):
            found = os.path.join(WORK_DIR, f)
            if found != onnx_path:
                os.rename(found, onnx_path)
            print(f"    Found: {onnx_path}")
            return onnx_path

    raise RuntimeError("ONNX export produced no output file")


def simplify_onnx(onnx_path):
    sim_path = os.path.join(WORK_DIR, "humansegv2_sim.onnx")
    print("[3/6] Simplifying ONNX...")

    import onnx
    from onnxsim import simplify

    model = onnx.load(onnx_path)
    input_name = model.graph.input[0].name
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    print(f"    Input: {input_name} shape={shape}")

    model_sim, check = simplify(model, input_shapes={input_name: shape})
    assert check, "Simplify failed!"
    onnx.save(model_sim, sim_path)
    print(f"    Saved: {sim_path}")
    return sim_path


def download_ncnn_tools():
    ncnn_dir = os.path.join(WORK_DIR, "ncnn-tools")
    if os.path.exists(ncnn_dir):
        return ncnn_dir

    print("[4/6] Downloading NCNN tools...")
    zip_path = os.path.join(WORK_DIR, "ncnn.zip")
    urllib.request.urlretrieve(NCNN_TOOLS_URL, zip_path)
    shutil.unpack_archive(zip_path, WORK_DIR)

    for d in os.listdir(WORK_DIR):
        full = os.path.join(WORK_DIR, d)
        if os.path.isdir(full) and d.startswith("ncnn-"):
            os.rename(full, ncnn_dir)
            break

    for tool in ["onnx2ncnn", "ncnn-optimize"]:
        p = os.path.join(ncnn_dir, "bin", tool)
        if os.path.exists(p):
            os.chmod(p, 0o755)

    return ncnn_dir


def convert_ncnn(sim_path, ncnn_dir):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    onnx2ncnn = os.path.join(ncnn_dir, "bin", "onnx2ncnn")
    ncnn_opt = os.path.join(ncnn_dir, "bin", "ncnn-optimize")

    raw_p = os.path.join(WORK_DIR, "hseg_raw.param")
    raw_b = os.path.join(WORK_DIR, "hseg_raw.bin")

    print("[5/6] ONNX → NCNN...")
    subprocess.run([onnx2ncnn, sim_path, raw_p, raw_b], check=True)

    print("[6/6] Optimizing...")
    opt_p = os.path.join(OUTPUT_DIR, "humansegv2.param")
    opt_b = os.path.join(OUTPUT_DIR, "humansegv2.bin")
    subprocess.run([ncnn_opt, raw_p, raw_b, opt_p, opt_b], check=True)

    ps = os.path.getsize(opt_p)
    bs = os.path.getsize(opt_b)
    print(f"    humansegv2.param ({ps:,} bytes)")
    print(f"    humansegv2.bin   ({bs:,} bytes)")
    print(f"    Total: {(ps + bs) / 1024 / 1024:.2f} MB")


def main():
    print("=" * 50)
    print("PP-HumanSegV2 → NCNN")
    print("=" * 50)

    os.makedirs(WORK_DIR, exist_ok=True)
    repo_dir = clone_paddleseg()
    onnx_path = export_onnx(repo_dir)
    sim_path = simplify_onnx(onnx_path)
    ncnn_dir = download_ncnn_tools()
    convert_ncnn(sim_path, ncnn_dir)

    print("=" * 50)
    print("DONE!")
    print("=" * 50)

    shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
