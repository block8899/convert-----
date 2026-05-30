#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import os

# ═══════════════════════════════════════
# SRVGGNetCompact
# ═══════════════════════════════════════

class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                 num_conv=16, upscale=4, act_type='prelu'):
        super().__init__()
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = self.body[0](x)
        for i in range(1, len(self.body)):
            out = self.body[i](out)
        return self.upsampler(out)


# ═══════════════════════════════════════
# Convert PyTorch weights → ncnn .param + .bin
# ═══════════════════════════════════════

def write_ncnn(model, param_path, bin_path):
    """Write ncnn .param and .bin from PyTorch SRVGG model directly."""
    layers = []
    weights_data = bytearray()
    layer_id = 0

    def add_weight(tensor):
        """Append weight to binary blob, return (offset, size)"""
        t = tensor.detach().cpu().float().contiguous()
        offset = len(weights_data)
        weights_data.extend(t.numpy().tobytes())
        return offset, t.numel()

    def next_id():
        nonlocal local_id
        lid = local_id
        local_id += 1
        return lid

    # Input blob
    input_blob = "in0"
    prev_blob = input_blob
    local_id = 1

    body = model.body
    num_layers = len(body)

    for i in range(num_layers):
        layer = body[i]
        layer_name = f"layer_{layer_id}"
        layer_id += 1

        if isinstance(layer, nn.Conv2d):
            out_blob = f"blob_{next_id()}"
            w = layer.weight  # (out_c, in_c, kh, kw)
            has_bias = layer.bias is not None

            # Determine convolution type
            groups = layer.groups
            kernel_h, kernel_w = layer.kernel_size
            stride_h, stride_w = layer.stride
            pad_h, pad_w = layer.padding
            dilation_h, dilation_w = layer.dilation
            num_output = w.shape[0]

            if groups > 1:
                conv_type = "ConvolutionDepthWise"
            else:
                conv_type = "Convolution"

            # Weight flag: 0=no bias, 1=has bias
            weight_data_size = w.numel()

            params = [
                0, num_output,           # 0=kernel_w, 1=kernel_h? No...
            ]
            # ncnn Convolution params:
            # 0=num_output
            # 1=kernel_w
            # 2=dilation_w
            # 3=pad_w
            # 4=stride_w
            # 5=bias_term (0/1)
            # 6=weight_data_size
            params_str = f"0={num_output} 1={kernel_w} 2={dilation_w} 3={pad_w} 4={stride_w} 5={1 if has_bias else 0} 6={weight_data_size}"
            if groups > 1:
                params_str += f" 7={groups}"

            # Write weights
            w_offset, w_size = add_weight(w.reshape(-1))
            b_offset, b_size = (add_weight(layer.bias) if has_bias
                                else (0, 0))

            layers.append(f"{conv_type} {layer_name} {params_str} 0={num_output} 1={kernel_w} 2={dilation_w} 3={pad_w} 4={stride_w} 5={1 if has_bias else 0} 6={weight_data_size}"
                          .replace(f"0={num_output}", f"0={num_output}"))

            prev_blob = out_blob

        elif isinstance(layer, nn.PReLU):
            out_blob = f"blob_{next_id()}"
            num_params = layer.weight.numel()

            p_offset, p_size = add_weight(layer.weight)

            layers.append(f"PReLU {layer_name} 0={num_params}")
            prev_blob = out_blob

        elif isinstance(layer, nn.PixelShuffle):
            out_blob = f"blob_{next_id()}"
            upscale = layer.upscale_factor

            layers.append(f"PixelShuffle {layer_name} 0={upscale}")
            prev_blob = out_blob

    # Write .param
    with open(param_path, 'w') as f:
        f.write("7767517\n")  # ncnn magic
        f.write(f"{len(layers)} 1\n")  # num_layers, num_blobs
        f.write(f"Input {input_blob} 0 1 {input_blob}\n")
        for i, line in enumerate(layers):
            # Fix input/output blob references
            pass  # Will use simplified approach below
        # ... this is getting too complex for raw param writing

    # Write .bin
    with open(bin_path, 'wb') as f:
        f.write(bytes([0x00]))  # flag
        f.write(weights_data)


# ═══════════════════════════════════════
# BETTER: Use ncnn Python API + ONNX intermediate
# ═══════════════════════════════════════

def convert_via_onnx(model, onnx_path, param_path, bin_path):
    """PyTorch → ONNX → ncnn via ncnn Python tools"""
    import onnx
    from onnx import shape_inference, version_converter

    # Export ONNX
    dummy = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(dummy)
    print(f"Verify: {list(dummy.shape)} → {list(out.shape)}")

    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=11,
        input_names=['in0'],
        output_names=['out0'],
        do_constant_folding=True)

    print(f"ONNX saved: {onnx_path}")

    # Try onnxsim
    try:
        from onnxsim import simplify
        m = onnx.load(onnx_path)
        m2, ok = simplify(m, input_shapes={'in0': [1, 3, 64, 64]})
        if ok:
            onnx.save(m2, onnx_path)
            print("Simplified OK")
    except Exception as e:
        print(f"Simplify skip: {e}")

    # Convert ONNX → ncnn using onnx2ncnn if available
    # Otherwise use pnnx
    import subprocess
    import shutil

    onnx2ncnn = shutil.which('onnx2ncnn')
    pnnx = shutil.which('pnnx')

    if pnnx:
        print("Using pnnx...")
        ret = subprocess.run([pnnx, onnx_path, 'inputshape=[1,3,64,64]'],
                             capture_output=True, text=True)
        print(ret.stdout[-500:] if ret.stdout else "")
        print(ret.stderr[-500:] if ret.stderr else "")

        # Find output files
        base = onnx_path.replace('.onnx', '')
        for suffix in ['.ncnn.param', '.onnx.ncnn.param']:
            src_p = base + suffix
            src_b = src_p.replace('.param', '.bin')
            if os.path.exists(src_p):
                os.rename(src_p, param_path)
                os.rename(src_b, bin_path)
                print(f"Renamed to {param_path}")
                return

        # Try current directory listing
        for f in os.listdir('.'):
            if f.endswith('.ncnn.param'):
                os.rename(f, param_path)
                os.rename(f.replace('.param', '.bin'), bin_path)
                print(f"Found and renamed {f}")
                return

        print("ERROR: pnnx output not found")
        print(f"Files in dir: {os.listdir('.')}")

    elif onnx2ncnn:
        print("Using onnx2ncnn...")
        subprocess.run([onnx2ncnn, onnx_path, param_path, bin_path])

    else:
        print("ERROR: Neither pnnx nor onnx2ncnn found")
        print("Installing pnnx...")
        subprocess.run(['pip', 'install', 'pnnx'], check=True)
        # Retry
        import importlib
        importlib.invalidate_caches()
        pnnx = shutil.which('pnnx')
        if pnnx:
            subprocess.run([pnnx, onnx_path, 'inputshape=[1,3,64,64]'])
        else:
            raise RuntimeError("Cannot find pnnx after install")


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    pth_path = "realesr-general-wdn-x4v3.pth"
    onnx_path = "wdn-x4v3.onnx"
    param_path = "wdn-x4v3.param"
    bin_path = "wdn-x4v3.bin"

    # Load weights
    print(f"Loading {pth_path}")
    ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('params_ema', ckpt.get('params', ckpt))

    first = list(sd.keys())[0]
    for pfx in ['module.', 'net_g.']:
        if first.startswith(pfx):
            sd = {k.replace(pfx, '', 1): v for k, v in sd.items()}
            break

    # Auto-detect
    body0 = [k for k in sd if k.endswith('body.0.weight')][0]
    num_in = sd[body0].shape[1]
    num_feat = sd[body0].shape[0]

    conv_ids = []
    for k in sd:
        parts = k.split('.')
        if 'body' in parts:
            i = parts.index('body')
            if (i + 1 < len(parts) and parts[i + 1].isdigit()
                    and parts[-1] == 'weight' and sd[k].dim() == 4):
                conv_ids.append(int(parts[i + 1]))

    max_idx = max(conv_ids)
    last_k = [k for k in sd if f'body.{max_idx}.weight' in k][0]
    num_out = sd[last_k].shape[0] // 16
    num_conv = (max_idx - 2) // 2

    srvgg_sd = {k: v for k, v in sd.items()
                if not k.startswith(('denoise.', 'denoising.', 'haar_'))}

    print(f"SRVGG: in={num_in} out={num_out} feat={num_feat} conv={num_conv}")

    model = SRVGGNetCompact(num_in, num_out, num_feat, num_conv, 4, 'prelu')
    model.load_state_dict(srvgg_sd, strict=True)
    model.eval()
    print(f"Loaded ✓")

    # Convert
    convert_via_onnx(model, onnx_path, param_path, bin_path)

    # Verify
    if os.path.exists(param_path):
        with open(param_path) as f:
            lines = f.readlines()
        print(f"\n=== {param_path} ===")
        for line in lines[:5]:
            print(line.rstrip())
        print(f"Lines: {len(lines)}")

    if os.path.exists(bin_path):
        print(f"Bin: {os.path.getsize(bin_path) / 1e6:.1f} MB")

    print("\n✓ Done")


if __name__ == '__main__':
    main()
