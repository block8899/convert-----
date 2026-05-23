"""
Convert Person Segmentation ONNX → NCNN
Dùng model ONNX có sẵn, bỏ qua PaddlePaddle
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

# ONNX model URLs — thử lần lượt
ONNX_URLS = [
    # PP-HumanSeg ONNX từ PaddleSeg export sẵn
    (
        "https://github.com/PaddlePaddle/PaddleSeg/raw/"
        "release/2.9/contrib/PP-HumanSeg/inference_models/"
        "pp_humansegv2_mobile_192x192_inference_model_with_softmax/"
        "model.onnx"
    ),
    # MODNet — người segmentation nhẹ (~2MB)
    (
        "https://github.com/ZHKKKe/MODNet/raw/"
        "master/pretrained/modnet_webcam_portrait_matting.ckpt"
    ),
    # PP-HumanSeg V1 ONNX
    (
        "https://github.com/PaddlePaddle/PaddleSeg/raw/"
        "release/2.7/contrib/PP-HumanSeg/models/"
        "pp_humanseg_mobile_192x192_pretrained/"
        "model.onnx"
    ),
]


def download_onnx():
    os.makedirs(WORK_DIR, exist_ok=True)
    onnx_path = os.path.join(WORK_DIR, "humansegv2.onnx")

    print("[1/3] Downloading person segmentation ONNX model...")

    for i, url in enumerate(ONNX_URLS):
        try:
            print(f"    Trying source {i+1}...")
            urllib.request.urlretrieve(url, onnx_path)
            size = os.path.getsize(onnx_path)
            print(f"    Downloaded! ({size:,} bytes)")
            return onnx_path
        except Exception as e:
            print(f"    Failed: {e}")
            continue

    raise RuntimeError(
        "All ONNX sources failed. "
        "Please manually place humansegv2.onnx in convert_tmp/"
    )


def simplify_onnx(onnx_path):
    sim_path = os.path.join(WORK_DIR, "humansegv2_sim.onnx")
    print("[2/3] Simplifying ONNX...")

    import onnx
    from onnxsim import simplify

    model = onnx.load(onnx_path)
    input_name = model.graph.input[0].name
    input_shape = [
        dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim
    ]
    print(f"    Input: {input_name} {input_shape}")

    # Auto detect shape
    if len(input_shape) == 4:
        shapes = {input_name: input_shape}
    else:
        shapes = {input_name: [1, 3, 192, 192]}

    model_sim, check = simplify(model, input_shapes=shapes)
    assert check, "ONNX simplify failed!"
    onnx.save(model_sim, sim_path)

    print(f"    Inputs:  {[i.name for i in model_sim.graph.input]}")
    print(f"    Outputs: {[o.name for o in model_sim.graph.output]}")
    print(f"    Saved:   {sim_path}")
    return sim_path


def download_ncnn_tools():
    ncnn_dir = os.path.join(WORK_DIR, "ncnn-tools")
    if os.path.exists(ncnn_dir):
        return ncnn_dir

    print("[3/3] Downloading NCNN tools...")
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


def convert(sim_path, ncnn_tools_dir):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    onnx2ncnn = os.path.join(ncnn_tools_dir, "bin", "onnx2ncnn")
    ncnn_opt = os.path.join(ncnn_tools_dir, "bin", "ncnn-optimize")

    raw_p = os.path.join(WORK_DIR, "humansegv2_raw.param")
    raw_b = os.path.join(WORK_DIR, "humansegv2_raw.bin")

    print("[4/3] ONNX → NCNN...")
    subprocess.run([onnx2ncnn, sim_path, raw_p, raw_b], check=True)

    print("[5/3] Optimizing...")
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
    print("Person Segmentation → NCNN Converter")
    print("=" * 50)

    onnx_path = download_onnx()
    sim_path = simplify_onnx(onnx_path)
    ncnn_dir = download_ncnn_tools()
    convert(sim_path, ncnn_dir)

    print()
    print("=" * 50)
    print("DONE!")
    print("  output/humansegv2.param")
    print("  output/humansegv2.bin")
    print("=" * 50)

    shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
