import os
import sys
import subprocess
import shutil


def find_tool(name):
    path = shutil.which(name)
    if path:
        return path
    print(f"ERROR: {name} not found in PATH!")
    sys.exit(1)


def run_cmd(cmd, timeout=300):
    ret = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if ret.stdout:
        lines = ret.stdout.strip().split('\n')
        for line in lines[-20:]:
            print(f"   {line}")
    if ret.stderr:
        lines = ret.stderr.strip().split('\n')
        for line in lines[-20:]:
            print(f"   [err] {line}")
    return ret


def main():
    print("=== AnimeGANv3 ONNX -> NCNN ===\n")

    onnx2ncnn = find_tool("onnx2ncnn")
    ncnnoptimize = find_tool("ncnnoptimize")

    # In version
    ret = subprocess.run([onnx2ncnn], capture_output=True, text=True)
    ver = ret.stderr.strip().split('\n')[0] if ret.stderr else "unknown"
    print(f"onnx2ncnn: {onnx2ncnn} ({ver})")
    print(f"ncnnoptimize: {ncnnoptimize}")

    onnx_file = "AnimeGANv3_PortraitSketch_25.onnx"
    if not os.path.exists(onnx_file):
        print(f"MISSING: {onnx_file}")
        sys.exit(1)

    print(f"Input: {os.path.getsize(onnx_file) / 1024 / 1024:.1f} MB")

    # 1. Simplify ONNX
    print("\n1. Simplifying ONNX...")
    sim_file = "animegan_sim.onnx"
    ret = run_cmd([sys.executable, "-m", "onnxsim", onnx_file, sim_file])
    if ret.returncode != 0 or not os.path.exists(sim_file):
        print("   Simplify failed, using original")
        shutil.copy(onnx_file, sim_file)
    else:
        print(f"   OK: {os.path.getsize(sim_file) / 1024 / 1024:.1f} MB")

    # 2. ONNX -> NCNN FP32
    print("\n2. Converting to FP32...")
    fp32_param = "animegan_fp32.param"
    fp32_bin = "animegan_fp32.bin"

    for f in [fp32_param, fp32_bin]:
        if os.path.exists(f):
            os.remove(f)

    ret = run_cmd([onnx2ncnn, sim_file, fp32_param, fp32_bin])

    # Kiểm tra output hợp lệ
    if not os.path.exists(fp32_param) or not os.path.exists(fp32_bin):
        print("   FAILED: onnx2ncnn did not create output files!")
        sys.exit(1)

    fp32_size = os.path.getsize(fp32_bin)
    if fp32_size == 0:
        print("   FAILED: .bin file is 0 bytes — onnx2ncnn conversion failed!")
        print("   The model may have unsupported ops for this ncnn version.")
        sys.exit(1)

    print(f"   FP32 OK: param={os.path.getsize(fp32_param) / 1024:.1f} KB, "
          f"bin={fp32_size / 1024 / 1024:.1f} MB")

    # 3. FP32 -> FP16
    print("\n3. Converting to FP16...")
    fp16_param = "animegan_fp16.param"
    fp16_bin = "animegan_fp16.bin"

    for f in [fp16_param, fp16_bin]:
        if os.path.exists(f):
            os.remove(f)

    ret = run_cmd([ncnnoptimize, fp32_param, fp32_bin, fp16_param, fp16_bin, "65536"])

    if os.path.exists(fp16_param) and os.path.exists(fp16_bin) and os.path.getsize(fp16_bin) > 0:
        fp16_size = os.path.getsize(fp16_bin)
        print(f"   FP16 OK: param={os.path.getsize(fp16_param) / 1024:.1f} KB, "
              f"bin={fp16_size / 1024 / 1024:.1f} MB")
        print(f"   Size reduction: {(1 - fp16_size / fp32_size) * 100:.1f}%")
    else:
        print("   FP16 failed, skipping")
        fp16_param = None
        fp16_bin = None

    # 4. Copy to output
    os.makedirs("output", exist_ok=True)
    shutil.copy(fp32_param, "output/animegan.param")
    shutil.copy(fp32_bin, "output/animegan.bin")

    if fp16_param and fp16_bin:
        shutil.copy(fp16_param, "output/animegan_fp16.param")
        shutil.copy(fp16_bin, "output/animegan_fp16.bin")

    # 5. Verify
    print("\n=== Output ===")
    for f in sorted(os.listdir("output")):
        fpath = os.path.join("output", f)
        size = os.path.getsize(fpath)
        if size > 1024 * 1024:
            print(f"  {f}: {size / 1024 / 1024:.1f} MB")
        else:
            print(f"  {f}: {size / 1024:.1f} KB")

    print("\nAnimeGANv3 OK!")


if __name__ == "__main__":
    main()
