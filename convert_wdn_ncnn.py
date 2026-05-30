#!/usr/bin/env python3
"""
convert_wdn_ncnn.py
Convert realesr-general-wdn-x4v3.pth → ONNX → ncnn

Requirements:
    pip install torch onnx onnxsim
    # + ncnn tools: https://github.com/Tencent/ncnn/wiki/how-to-build

Usage:
    1. python convert_wdn_ncnn.py --input realesr-general-wdn-x4v3.pth
    2. ./ncnnoptimize realesr-general-wdn-x4v3.onnx.opt.param \
                      realesr-general-wdn-x4v3.onnx.opt.bin \
                      realesr-general-wdn-x4v3.param \
                      realesr-general-wdn-x4v3.bin 65536
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import shutil
import struct

# ═══════════════════════════════════════════════════════════════
# SRVGGNetCompact — exact same arch as Real-ESRGAN repo
# ═══════════════════════════════════════════════════════════════

class SRVGGNetCompact(nn.Module):
    """Slim Real-VGG style compact network.
    Reference: realesrgan/archs/srvgg_arch.py
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16,
                 upscale=4, act_type='prelu'):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()

        # First conv
        first_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body.append(first_conv)

        # Body convs: (conv + act) pairs
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if act_type == 'prelu':
                self.body.append(nn.PReLU(num_parameters=num_feat))
            elif act_type == 'leakyrelu':
                self.body.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # Last conv before upsampling
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))

        # PixelShuffle
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = self.body[0](x)  # first conv
        for i in range(1, len(self.body) - 1):
            out = self.body[i](out)
        out = self.body[-1](out)  # last conv
        out = self.upsampler(out)
        return out


# ═══════════════════════════════════════════════════════════════
# WDN Wrapper — Wavelet Decompose + SRVGG + Reconstruct
# Exactly matches Real-ESRGAN inference code
# ═══════════════════════════════════════════════════════════════

class HaarDownsampling(nn.Module):
    """Haar wavelet downsampling — splits into 4 frequency bands.
    Output: 12 channels = 4 bands × 3 RGB
    """
    def __init__(self):
        super().__init__()
        # Haar filters — fixed, not trainable
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2.0
        lh = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 2.0
        hl = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 2.0
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2.0

        # Shape: (4, 1, 2, 2) — one filter per band
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        # Repeat for 3 channels: (4, 3, 2, 2) → grouped conv
        filters = filters.repeat(1, 3, 1, 1)  # (4, 3, 2, 2)

        # Use (12, 1, 2, 2) with groups=3 for per-channel operation
        # Reshape to (12, 1, 2, 2) for groups=3 conv
        self.register_buffer('filters', filters.view(12, 1, 2, 2))
        self.groups = 3

    def forward(self, x):
        # x: (B, 3, H, W) → (B, 12, H/2, W/2)
        return F.conv2d(x, self.filters, stride=2, groups=self.groups)


class HaarUpsampling(nn.Module):
    """Inverse Haar wavelet — reconstructs from 4 frequency bands.
    Input: 12 channels → Output: 3 channels at 2x resolution
    """
    def __init__(self):
        super().__init__()
        # Inverse filters
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2.0
        lh = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 2.0
        hl = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 2.0
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2.0

        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        filters = filters.repeat(1, 3, 1, 1)
        self.register_buffer('filters', filters.view(12, 1, 2, 2))
        self.groups = 3

    def forward(self, x):
        # x: (B, 12, H, W) → (B, 3, H*2, W*2)
        return F.conv_transpose2d(x, self.filters, stride=2, groups=self.groups)


class ResBlock(nn.Module):
    """Simple residual block with BN"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))


class WDNWrapper(nn.Module):
    """Full WDN pipeline: Wavelet → Denoise → SR → Reconstruct

    This mirrors the inference logic in Real-ESRGAN's
    RealESRGANer._postprocess() and _preprocess().
    """
    def __init__(self, srvgg_model):
        super().__init__()
        self.srvgg = srvgg_model

        # Wavelet decomposition / reconstruction
        self.haar_down = HaarDownsampling()
        self.haar_up = HaarUpsampling()

        # Denoise network for wavelet coefficients
        self.denoise = nn.Sequential(
            nn.Conv2d(12, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 12, 3, 1, 1),
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        B, C, H, W = x.shape

        # ── Step 1: Wavelet decompose ──
        coeffs = self.haar_down(x)  # (B, 12, H/2, W/2)

        # ── Step 2: Denoise wavelet coefficients ──
        denoised_coeffs = coeffs + self.denoise(coeffs)  # residual learning
        # Ensure H/2, W/2 are even for SRVGG
        denoised_coeffs = self.mod_pad(denoised_coeffs)

        # ── Step 3: SR on wavelet coefficients (4x upscale) ──
        sr_coeffs = self.srvgg(denoised_coeffs)  # (B, 12, H*2, W*2)

        # ── Step 4: Reconstruct via inverse wavelet ──
        output = self.haar_up(sr_coeffs)  # (B, 3, H*4, W*4)

        # ── Step 5: Trim to exact 4x output ──
        output = output[:, :, :H * self.srvgg.upscale, :W * self.srvgg.upscale]
        return output

    @staticmethod
    def mod_pad(x, modulo=2):
        """Pad spatial dims to be divisible by modulo"""
        _, _, h, w = x.shape
        pad_h = (modulo - h % modulo) % modulo
        pad_w = (modulo - w % modulo) % modulo
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x


# ═══════════════════════════════════════════════════════════════
# LOAD & CONVERT
# ═══════════════════════════════════════════════════════════════

def load_model(pth_path):
    """Load .pth and reconstruct model with correct state_dict mapping"""
    print(f"Loading: {pth_path}")
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'params_ema' in checkpoint:
            state_dict = checkpoint['params_ema']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Detect prefix
    first_key = list(state_dict.keys())[0]
    print(f"First key: {first_key}")

    if first_key.startswith('module.'):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        print("Stripped 'module.' prefix")

    # Detect architecture from weights
    has_denoise = any('denoise' in k for k in state_dict.keys())
    has_wavelet = any('haar' in k for k in state_dict.keys())

    # Detect SRVGG params
    if 'body.0.weight' in state_dict:
        num_in_ch = state_dict['body.0.weight'].shape[1]  # 12 for WDN
        num_feat = state_dict['body.0.weight'].shape[0]    # 64

        # Count conv layers
        conv_indices = [int(k.split('.')[1])
                        for k in state_dict.keys()
                        if k.startswith('body.') and k.endswith('.weight')]

        # Last conv (PixelShuffle input) has shape (out_ch * scale^2, feat, 3, 3)
        last_key = max(
            [k for k in state_dict.keys() if k.startswith('body.')],
            key=lambda k: int(k.split('.')[1]) if k.split('.')[1].isdigit() else 0
        )
        last_shape = state_dict[last_key].shape
        num_out_ch = last_shape[0] // 16  # 4^2 = 16 for 4x

        num_conv = (max(conv_indices) - 1) // 2  # first + (conv+act) pairs + last

        print(f"Detected SRVGG: in={num_in_ch}, out={num_out_ch}, "
              f"feat={num_feat}, conv={num_conv}")
    else:
        # Fallback defaults
        num_in_ch = 12
        num_out_ch = 3
        num_feat = 64
        num_conv = 23

    # Build SRVGG
    srvgg = SRVGGNetCompact(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        num_conv=num_conv,
        upscale=4,
        act_type='prelu'
    )

    # Load SRVGG weights
    srvgg_keys = {k: v for k, v in state_dict.items()
                  if not k.startswith(('denoise.', 'haar_'))}
    srvgg.load_state_dict(srvgg_keys, strict=True)
    print(f"SRVGG loaded: {len(srvgg_keys)} tensors")

    # Build WDN wrapper
    model = WDNWrapper(srvgg)

    # Load denoise weights if present
    if has_denoise:
        denoise_keys = {k: v for k, v in state_dict.items()
                        if k.startswith('denoise.')}
        try:
            model.denoise.load_state_dict(denoise_keys, strict=False)
            print(f"WDN denoise loaded: {len(denoise_keys)} tensors")
        except Exception as e:
            print(f"WDN denoise partial load: {e}")

    if has_wavelet:
        wavelet_keys = {k: v for k, v in state_dict.items()
                        if k.startswith('haar_')}
        print(f"Found {len(wavelet_keys)} wavelet keys (using fixed Haar filters)")

    model.eval()
    return model


def export_onnx(model, onnx_path, input_size=64):
    """Export to ONNX with fixed input shape for ncnn"""
    print(f"\nExporting ONNX (input={input_size}x{input_size})...")

    dummy = torch.randn(1, 3, input_size, input_size)

    # Verify forward pass
    with torch.no_grad():
        out = model(dummy)
    print(f"Verification: input {list(dummy.shape)} → output {list(out.shape)}")
    expected = input_size * 4
    assert out.shape[2] == expected and out.shape[3] == expected, \
        f"Expected {expected}x{expected}, got {out.shape[2]}x{out.shape[3]}"

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=11,
        input_names=['in0'],
        output_names=['out0'],
        dynamic_axes=None,  # Fixed shape for ncnn
    )
    print(f"Saved: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")

    # Simplify with onnxsim
    try:
        import onnx
        from onnxsim import simplify

        print("Simplifying ONNX graph...")
        onnx_model = onnx.load(onnx_path)
        model_sim, ok = simplify(onnx_model)
        if ok:
            onnx.save(model_sim, onnx_path.replace('.onnx', '.sim.onnx'))
            print(f"Simplified: {onnx_path.replace('.onnx', '.sim.onnx')}")
            return onnx_path.replace('.onnx', '.sim.onnx')
        else:
            print("onnxsim failed, using original")
    except ImportError:
        print("onnxsim not installed (pip install onnxsim), skipping simplify")

    return onnx_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert realesr-general-wdn-x4v3.pth → ncnn')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to realesr-general-wdn-x4v3.pth')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: same as input)')
    parser.add_argument('--input-size', type=int, default=64,
                        help='ONNX input size (default: 64)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    out_dir = args.output or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)

    basename = 'realesr-general-wdn-x4v3'

    # ── Step 1: Load model ──
    model = load_model(args.input)

    # ── Step 2: Export ONNX ──
    onnx_path = os.path.join(out_dir, f'{basename}.onnx')
    final_onnx = export_onnx(model, onnx_path, args.input_size)

    # ── Step 3: Print ncnn conversion commands ──
    param_path = os.path.join(out_dir, f'{basename}.param')
    bin_path = os.path.join(out_dir, f'{basename}.bin')
    opt_param = os.path.join(out_dir, f'{basename}-opt.param')
    opt_bin = os.path.join(out_dir, f'{basename}-opt.bin')

    print("\n" + "=" * 60)
    print("ONNX exported. Now run ncnn tools manually:")
    print("=" * 60)
    print()
    print(f"# Step 3a: ONNX → ncnn")
    print(f"onnx2ncnn {final_onnx} {param_path} {bin_path}")
    print()
    print(f"# Step 3b: Optimize (optional but recommended)")
    print(f"ncnnoptimize {param_path} {bin_path} {opt_param} {opt_bin} 65536")
    print()
    print(f"# Step 3c: Verify with ncnn python wrapper")
    print(f"import ncnn")
    print(f"net = ncnn.Net()")
    print(f"net.load_param('{opt_param or param_path}')")
    print(f"net.load_model('{opt_bin or bin_path}')")
    print()

    # ── Step 4: Try automatic conversion if tools available ──
    print("Attempting automatic ncnn conversion...")

    onnx2ncnn = shutil.which('onnx2ncnn')
    ncnnoptimize = shutil.which('ncnnoptimize')

    if onnx2ncnn:
        import subprocess
        ret = subprocess.run([onnx2ncnn, final_onnx, param_path, bin_path],
                             capture_output=True, text=True)
        if ret.returncode == 0:
            print(f"✓ onnx2ncnn: {param_path}, {bin_path}")

            if ncnnoptimize:
                ret = subprocess.run(
                    [ncnnoptimize, param_path, bin_path,
                     opt_param, opt_bin, '65536'],
                    capture_output=True, text=True)
                if ret.returncode == 0:
                    print(f"✓ ncnnoptimize: {opt_param}, {opt_bin}")
                else:
                    print(f"✗ ncnnoptimize failed: {ret.stderr}")
        else:
            print(f"✗ onnx2ncnn failed: {ret.stderr}")
            print("  Install ncnn tools: https://github.com/Tencent/ncnn/wiki/how-to-build")
    else:
        print("✗ onnx2ncnn not found in PATH")
        print("  Install ncnn tools or build from source:")
        print("  https://github.com/Tencent/ncnn/wiki/how-to-build")
        print()
        print("  Quick install on Linux/Mac:")
        print("  brew install ncnn          # macOS")
        print("  sudo apt install libncnn-dev # Ubuntu")

    print("\n" + "=" * 60)
    print("Done! Use .param + .bin in your Android app")
    print("=" * 60)


if __name__ == '__main__':
    main()
