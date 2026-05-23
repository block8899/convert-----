"""
Convert PP-HumanSegV2 Mobile → ONNX → NCNN
"""

import os
import subprocess
import shutil
import urllib.request

NCNN_TOOLS_URL = (
    "https://github.com/Tencent/ncnn/releases/download/20240820/"
    "ncnn-20240820-ubuntu.zip"
)

WORK_DIR = "convert_tmp"
OUTPUT_DIR = "output"


# ─── Step 1: Export to ONNX ────────────────────────

def export_to_onnx():
    os.makedirs(WORK_DIR, exist_ok=True)
    onnx_path = os.path.join(WORK_DIR, "humansegv2.onnx")

    print("[1/4] Exporting PP-HumanSegV2...")

    import paddle
    from paddleseg.models import PPMobileSeg

    # Dùng đúng params: ['self', 'num_classes', 'backbone', 'head_use_dw', 'align_corners', 'pretrained', 'upsample']
    model = PPMobileSeg(
        num_classes=2,
        align_corners=False,
    )

    # Load pretrained weights
    weights_path = os.path.join(WORK_DIR, "model.pdparams")
    model_url = (
        "https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/"
        "pp_humansegv2_mobile_192x192_pretrained/model.pdparams"
    )

    if not os.path.exists(weights_path):
        print("    Downloading pretrained weights...")
        urllib.request.urlretrieve(model_url, weights_path)

    state_dict = paddle.load(weights_path)
    model.set_state_dict(state_dict)
    model.eval()

    # Export to ONNX
    print("    Exporting to ONNX...")
    input_spec = paddle.static.InputSpec(
        shape=[1, 3, 192, 192], dtype="float32", name="x"
    )
    paddle.onnx.export(
        model,
        os.path.join(WORK_DIR, "humansegv2"),
        input_spec=[input_spec],
        opset_version=11,
    )

    if os.path.exists(onnx_path):
        print(f"    Saved: {onnx_path}")
        return onnx_path

    # Check if exported with different name
    for f in os.listdir(WORK_DIR):
        if f.endswith(".onnx"):
            found = os.path.join(WORK_DIR, f)
            if found != onnx_path:
                os.rename(found, onnx_path)
                print(f"    Saved: {onnx_path}")
                return onnx_path

    raise FileNotFoundError(
        f"ONNX not found. Files in {WORK_DIR}: {os.listdir(WORK_DIR)}"
    )


# ─── Step 2: Simplify ONNX ─────────────────────────

def simplify_onnx(onnx_path):
    sim_path = os.path.join(WORK_DIR, "humansegv2_sim.onnx")
    print("[2/4] Simplifying ONNX...")

    import onnx
    from onnxsim import simplify

    model = onnx.load(onnx_path)

    # Detect input name
    input_name = model.graph.input[0].name
    print(f"    Input name: {input_name}")

    model_sim, check = simplify(
        model,
        input_shapes={input_name: [1, 3, 192, 192]}
    )
    assert check, "ONNX simplify failed!"
    onnx.save(model_sim, sim_path)

    print(f"    Inputs:  {[i.name for i in model_sim.graph.input]}")
    print(f"    Outputs: {[o.name for o in model_sim.graph.output]}")
    print(f"    Saved:   {sim_path}")
    return sim_path


# ─── Step 3: Download NCNN tools ───────────────────

def download_ncnn_tools():
    ncnn_dir = os.path.join(WORK_DIR, "ncnn-tools")
    if os.path.exists(ncnn_dir):
        print("[3/4] NCNN tools already exist.")
        return ncnn_dir

    print("[3/4] Downloading NCNN tools...")
    zip_path = os.path.join(WORK_DIR, "ncnn.zip")
    urllib.request.urlretrieve(NCNN_TOOLS_URL, zip_path)
    shutil.unpack_archive(zip_path, WORK_DIR)

    extracted = None
    for d in os.listdir(WORK_DIR):
        full = os.path.join(WORK_DIR, d)
        if os.path.isdir(full) and d.startswith("ncnn-"):
            extracted = full
            break

    os.rename(extracted, ncnn_dir)
    for tool in ["onnx2ncnn", "ncnn-optimize"]:
        tool_path = os.path.join(ncnn_dir, "bin", tool)
        if os.path.exists(tool_path):
            os.chmod(tool_path, 0o755)

    print(f"    Tools at: {ncnn_dir}/bin/")
    return ncnn_dir


# ─── Step 4: ONNX → NCNN ──────────────────────────

def onnx_to_ncnn(sim_path, ncnn_tools_dir):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    onnx2ncnn = os.path.join(ncnn_tools_dir, "bin", "onnx2ncnn")
    ncnn_opt = os.path.join(ncnn_tools_dir, "bin", "ncnn-optimize")

    raw_param = os.path.join(WORK_DIR, "humansegv2_raw.param")
    raw_bin = os.path.join(WORK_DIR, "humansegv2_raw.bin")

    print("[4a/4] Converting ONNX → NCNN...")
    subprocess.run([onnx2ncnn, sim_path, raw_param, raw_bin], check=True)

    print("[4b/4] Optimizing NCNN model...")
    opt_param = os.path.join(OUTPUT_DIR, "humansegv2.param")
    opt_bin = os.path.join(OUTPUT_DIR, "humansegv2.bin")
    subprocess.run(
        [ncnn_opt, raw_param, raw_bin, opt_param, opt_bin],
        check=True,
    )

    p_size = os.path.getsize(opt_param)
    b_size = os.path.getsize(opt_bin)
    print(f"    Output: humansegv2.param ({p_size:,} bytes)")
    print(f"    Output: humansegv2.bin ({b_size:,} bytes)")
    print(f"    Total:  {(p_size + b_size) / 1024 / 1024:.2f} MB")


# ─── Main ──────────────────────────────────────────

def main():
    print("=" * 50)
    print("PP-HumanSegV2 → NCNN Converter")
    print("=" * 50)

    onnx_path = export_to_onnx()
    sim_path = simplify_onnx(onnx_path)
    ncnn_tools_dir = download_ncnn_tools()
    onnx_to_ncnn(sim_path, ncnn_tools_dir)

    print()
    print("=" * 50)
    print("DONE! Files in output/:")
    print("  - humansegv2.param")
    print("  - humansegv2.bin")
    print("=" * 50)

    shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
