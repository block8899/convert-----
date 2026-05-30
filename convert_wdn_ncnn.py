#!/usr/bin/env python3
"""
Convert realesr-general-wdn-x4v3.pth → ONNX
Handles the WDN wavelet denoise + SRVGG pipeline correctly.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys


# ═══════════════════════════════════════════════════════════════
# ARCHITECTURES — exact match with Real-ESRGAN repo
# ═══════════════════════════════════════════════════════════════

class SRVGGNetCompact(nn.Module):
    """Exact copy from basicsr/archs/srvgg_arch.py"""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                 num_conv=16, upscale=4, act_type='prelu'):
        super().__init__()
        self.in_nc = num_in_ch
        self.out_nc = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = self.body[0](x)
        for i in range(1, len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        return out


class UNet(nn.Module):
    """Exact copy from basicsr/archs/denoising_arch.py (WaveletResUNet)"""

    def __init__(self, in_nc=12, out_nc=12, nc=[64, 128, 256, 512]):
        super().__init__()
        self.m_head = nn.Conv2d(in_nc, nc[0], 3, 1, 1, bias=False)

        # downsample
        self.m_down1 = nn.Sequential(
            nn.Conv2d(nc[0], nc[1], 2, 2, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.m_down2 = nn.Sequential(
            nn.Conv2d(nc[1], nc[2], 2, 2, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.m_down3 = nn.Sequential(
            nn.Conv2d(nc[2], nc[3], 2, 2, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # body
        self.m_body = nn.Sequential(
            nn.Conv2d(nc[3], nc[3], 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nc[3], nc[3], 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nc[3], nc[3], 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nc[3], nc[3], 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # upsample
        self.m_up3 = nn.Sequential(
            nn.Conv2d(nc[3] * 2, nc[3], 1, 1, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nc[3], nc[3] * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2))
        self.m_up2 = nn.Sequential(
            nn.Conv2d(nc[2] * 2, nc[2], 1, 1, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nc[2], nc[2] * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2))
        self.m_up1 = nn.Sequential(
            nn.Conv2d(nc[1] * 2, nc[1], 1, 1, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nc[1], nc[1] * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2))

        # tail
        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1, bias=False)

    def forward(self, x0):
        head = self.m_head(x0)
        # down
        x1 = self.m_down1(head)
        x2 = self.m_down2(x1)
        x3 = self.m_down3(x2)
        # body
        x3 = self.m_body(x3) + x3
        # up + skip
        x3 = self.m_up3(torch.cat([x3, x2], dim=1))
        x2 = self.m_up2(torch.cat([x3, x1], dim=1))
        x1 = self.m_up1(torch.cat([x2, head], dim=1))
        # tail
        out = self.m_tail(x1) + x0
        return out


class WaveletHaarDownsampling(nn.Module):
    """Haar wavelet — train time (learnable). Inference: fix weights."""

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 4.0
        lh = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 4.0
        hl = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 4.0
        hh = torch.tensor([[1, -1], [-1, -1]], dtype=torch.float32) / 4.0

        # (4, 1, 2, 2) for each color channel
        self.register_buffer(
            'weight_LL', ll.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))
        self.register_buffer(
            'weight_LH', lh.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))
        self.register_buffer(
            'weight_HL', hl.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))
        self.register_buffer(
            'weight_HH', hh.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))

    def forward(self, x):
        return torch.cat([
            F.conv2d(x, self.weight_LL, bias=None, stride=2, groups=3),
            F.conv2d(x, self.weight_LH, bias=None, stride=2, groups=3),
            F.conv2d(x, self.weight_HL, bias=None, stride=2, groups=3),
            F.conv2d(x, self.weight_HH, bias=None, stride=2, groups=3),
        ], dim=1)  # (B, 12, H/2, W/2)


# ═══════════════════════════════════════════════════════════════
# FULL WDN MODEL — wrap everything into single forward()
# This is what gets exported to ONNX
# ═══════════════════════════════════════════════════════════════

class FullWDNModel(nn.Module):
    """
    Complete WDN pipeline in one module:
    x (B,3,H,W) → Haar_down → UNet denoise → SRVGG → output (B,3,H*4,W*4)
    """

    def __init__(self, srvgg, unet, haar):
        super().__init__()
        self.srvgg = srvgg
        self.unet = unet
        self.haar = haar

    @staticmethod
    def _mod2(x):
        """Pad to even spatial dims"""
        _, _, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, pad_h, pad_w

    def forward(self, x):
        B, C, H, W = x.shape

        # Step 1: Pad input to even dims for Haar
        x_pad, pad_h, pad_w = self._mod2(x)

        # Step 2: Wavelet decompose  → (B, 12, H/2, W/2)
        coeffs = self.haar(x_pad)
        h_half = coeffs.shape[2]
        w_half = coeffs.shape[3]

        # Step 3: UNet denoise in wavelet domain
        coeffs_dn = self.unet(coeffs)

        # Step 4: SRVGG 4x on wavelet coefficients
        # Input: (B, 12, H/2, W/2) → Output: (B, 3, H*2, W*2)
        # PixelShuffle 4x: 12→3, 2x spatial
        sr_out = self.srvgg(coeffs_dn)  # (B, 3, H*2, W*2)

        # Step 5: Trim to exact expected size
        expected_h = (H + pad_h) * 2
        expected_w = (W + pad_w) * 2
        sr_out = sr_out[:, :, :expected_h, :expected_w]

        return sr_out


# ═══════════════════════════════════════════════════════════════
# LOAD STATE DICT — robust with auto-detection
# ═══════════════════════════════════════════════════════════════

def load_wdn_model(pth_path):
    print(f"Loading: {pth_path}")

    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)

    # Extract state_dict
    if isinstance(checkpoint, dict):
        for key in ['params_ema', 'params', 'state_dict']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"  Found state_dict under '{key}'")
                break
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip prefix
    first_key = list(state_dict.keys())[0]
    prefix = ''
    if first_key.startswith('module.'):
        prefix = 'module.'
    elif first_key.startswith('net_g.'):
        prefix = 'net_g.'

    if prefix:
        state_dict = {k.replace(prefix, '', 1): v
                      for k, v in state_dict.items()}
        print(f"  Stripped prefix: '{prefix}'")

    # ── Detect SRVGG config ──
    srvgg_state = {}
    unet_state = {}

    for k, v in state_dict.items():
        if k.startswith('denoise.') or k.startswith('denoising.'):
            clean_k = k.replace('denoising.', 'denoise.')
            if clean_k.startswith('denoise.'):
                unet_state[clean_k[len('denoise.'):]] = v
        elif not k.startswith('haar_'):
            srvgg_state[k] = v

    # Detect from weights
    first_conv_key = 'body.0.weight'
    if first_conv_key not in srvgg_state:
        # Try with different prefixes
        for k in srvgg_state:
            if 'body.0.weight' in k:
                first_conv_key = k
                break

    num_in_ch = srvgg_state[first_conv_key].shape[1]   # 12 for WDN
    num_feat = srvgg_state[first_conv_key].shape[0]     # 64

    # Count conv layers
    conv_ids = []
    prelu_keys = []
    for k in srvgg_state:
        parts = k.split('.')
        if len(parts) >= 3 and parts[0] == 'body':
            idx = int(parts[1])
            if parts[2] == 'weight' and len(srvgg_state[k].shape) == 4:
                conv_ids.append(idx)
            if parts[2] == 'weight' and len(srvgg_state[k].shape) == 1:
                prelu_keys.append(idx)

    # Last conv is the biggest index
    max_idx = max(conv_ids)
    last_shape = srvgg_state[f'body.{max_idx}.weight'].shape
    num_out_ch = last_shape[0] // 16  # PixelShuffle 4x → /4²

    # num_conv = (max_idx - 1 - 1) / 2 → body: [0, 1_act, 2_conv, 3_act, ..., max, upsampler]
    # indices: 0=first_conv, 1=first_act, 2..(max-1)=conv+act pairs, max=last_conv
    num_conv = (max_idx - 2) // 2

    print(f"  SRVGG: in={num_in_ch}, out={num_out_ch}, "
          f"feat={num_feat}, conv={num_conv}")
    print(f"  UNet weights: {len(unet_state)} tensors")

    # ── Build models ──
    srvgg = SRVGGNetCompact(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        num_conv=num_conv,
        upscale=4,
        act_type='prelu'
    )
    srvgg.load_state_dict(srvgg_state, strict=True)
    print(f"  SRVGG loaded ✓ ({sum(p.numel() for p in srvgg.parameters())} params)")

    unet = UNet(in_nc=12, out_nc=12, nc=[64, 128, 256, 512])
    if unet_state:
        unet.load_state_dict(unet_state, strict=True)
        print(f"  UNet loaded ✓ ({sum(p.numel() for p in unet.parameters())} params)")
    else:
        print("  WARNING: No UNet weights found, using random init")

    haar = WaveletHaarDownsampling()

    # ── Combine ──
    model = FullWDNModel(srvgg, unet, haar)
    model.eval()

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    return model


# ═══════════════════════════════════════════════════════════════
# EXPORT ONNX
# ═══════════════════════════════════════════════════════════════

def export_onnx(model, output_path, input_size=64):
    print(f"\nExporting ONNX (input=1x3x{input_size}x{input_size})...")

    dummy = torch.randn(1, 3, input_size, input_size)

    # Verify forward pass
    with torch.no_grad():
        out = model(dummy)
    print(f"  Verification: {list(dummy.shape)} → {list(out.shape)}")
    print(f"  Expected: [1, 3, {input_size*2}, {input_size*2}]")

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=11,
        input_names=['in0'],
        output_names=['out0'],
        dynamic_axes=None,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")

    # ── Simplify with onnxsim ──
    sim_path = output_path.replace('.onnx', '.sim.onnx')
    try:
        import onnx
        from onnxsim import simplify

        print("  Simplifying with onnxsim...")
        onnx_model = onnx.load(output_path)
        model_sim, ok = simplify(
            onnx_model,
            input_shapes={'in0': [1, 3, input_size, input_size]}
        )
        if ok:
            onnx.save(model_sim, sim_path)
            print(f"  Simplified: {sim_path} ({os.path.getsize(sim_path)/1e6:.1f} MB)")
            return sim_path
        else:
            print("  onnxsim returned False, using original")
    except ImportError:
        print("  onnxsim not installed, skipping")

    return output_path


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='.pth file path')
    parser.add_argument('--output', required=True, help='.onnx output path')
    parser.add_argument('--input-size', type=int, default=64)
    args = parser.parse_args()

    model = load_wdn_model(args.input)
    onnx_path = export_onnx(model, args.output, args.input_size)

    print(f"\n✓ ONNX saved: {onnx_path}")
    print("Next step: onnx2ncnn + ncnnoptimize (handled by GitHub Actions)")


if __name__ == '__main__':
    main()
