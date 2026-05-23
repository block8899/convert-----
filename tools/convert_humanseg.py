"""
Convert PP-HumanSegV2 → ONNX → NCNN
"""

import os
import sys
import subprocess
import shutil
import urllib.request

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
        return repo_dir

    print("[1/6] Cloning PaddleSeg...")
    run_cmd(
        "git clone --depth 1 https://github.com/PaddlePaddle/PaddleSeg.git",
        cwd=WORK_DIR,
    )
    return repo_dir


def export_onnx(repo_dir):
    print("[2/6] Installing PaddleSeg + PaddlePaddle...")
    run_cmd(f"{sys.executable} -m pip install -e {repo_dir}", cwd=WORK_DIR)
    run_cmd(f"{sys.executable} -m pip install paddlepaddle==3.1.1", cwd=WORK_DIR)

    print("[3/6] Exporting PP-HumanSegV2 to ONNX...")

    export_script = '''
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "PaddleSeg"))

import paddle
from paddleseg.models import PPMobileSeg

# Build model
backbone = None
try:
    from paddleseg.models.backbones import STDC2
    backbone = STDC2()
except Exception:
    try:
        from paddleseg.models.backbones import STDC1
        backbone = STDC1()
    except Exception:
        pass

model = PPMobileSeg(
    num_classes=2,
    backbone=backbone,
    align_corners=False,
)

# Try download weights
import urllib.request
weight_urls = [
    "https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/pp_humansegv2_mobile_192x192_pretrained/model.pdparams",
]
weights = None
for url in weight_urls:
    try:
        urllib.request.urlretrieve(url, "tmp_weights.pdparams")
        weights = paddle.load("tmp_weights.pdparams")
        print(f"Loaded: {url}")
        break
    except Exception:
        continue

if weights:
    model.set_state_dict(weights)
else:
    print("WARNING: random init")

model.eval()

input_spec = paddle.static.InputSpec(shape=[1, 3, 192, 192], dtype="float32", name="x")
paddle.onnx.export(model, "humansegv2", input_spec=[input_spec], opset_version=11)

for f in os.listdir("."):
    if f.endswith(".onnx"):
        print(f"Output: {f} ({os.path.getsize(f)} bytes)")
'''

    script_path = os.path.join(WORK_DIR, "do_export.py")
    with open(script_path, "w") as f:
        f.write(export_script)

    run_cmd(f"{sys.executable} do_export.py", cwd=WORK_DIR)

    onnx_path = os.path.join(WORK_DIR, "humansegv2.onnx")
    for f in os.listdir(WORK_DIR):
        if f.endswith(".onnx") and "sim" not in f:
            found = os.path.join(WORK_DIR, f)
            if found != onnx_path:
                os.rename(found, onnx_path)
            print(f"    Found: {onnx_path}")
            return onnx_path

    raise RuntimeError("ONNX export produced no file")


def simplify_onnx(onnx_path):
    sim_path = os.path.join(WORK_DIR, "humansegv2_sim.onnx")
    print("[4/6] Simplifying ONNX...")

    import onnx
    from onnxsim import simplify

    model = onnx.load(onnx_path)
    name = model.graph.input[0].name
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    print(f"    Input: {name} {shape}")

    model_sim, check = simplify(model, input_shapes={name: shape})
    assert check, "Simplify failed!"
    onnx.save(model_sim, sim_path)
    print(f"    Saved: {sim_path}")
    return sim_path


def download_ncnn_tools():
    ncnn_dir = os.path.join(WORK_DIR, "ncnn-tools")
    if os.path.exists(ncnn_dir):
        return ncnn_dir

    print("[5/6] Downloading NCNN tools...")
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

    print("[6a/6] ONNX → NCNN...")
    subprocess.run([onnx2ncnn, sim_path, raw_p, raw_b], check=True)

    print("[6b/6] Optimizing...")
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
