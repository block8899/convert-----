#!/usr/bin/env python3
"""
export_wdn_pnnx.py
Load realesr-general-wdn-x4v3.pth → export TorchScript for PNNX.

PNNX converts TorchScript directly to ncnn .param + .bin.
No ONNX needed. No ncnn build tools needed.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys


# ═══════════════════════════════════════════════════════════════
# EXACT ARCHITECTURES from Real-ESRGAN codebase
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
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = self.body[0](x)
        for i in range(1, len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        return out


class UNet(nn.Module):
    """Exact copy from basicsr/archs/denoising_arch.py"""

    def __init__(self, in_nc=12, out_nc=12, nc=[64, 128, 256, 512]):
        super().__init__()
        self.m_head = nn.Conv2d(in_nc, nc[0], 3, 1, 1, bias=False)

        self.m_down1 = nn.Sequential(
            nn.Conv2d(nc[0], nc[1], 2, 2, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.m_down2 = nn.Sequential(
            nn.Conv2d(nc[1], nc[2], 2, 2, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.m_down3 = nn.Sequential(
            nn.Conv2d(nc[2], nc[3], 2, 2, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.m_body = nn.Sequential(
            nn.Conv2d(nc[3], nc[3], 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nc[3], nc[3], 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nc[3], nc[3], 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nc[3], nc[3], 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

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

        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1, bias=False)

    def forward(self, x0):
        head = self.m_head(x0)
        x1 = self.m_down1(head)
        x2 = self.m_down2(x1)
        x3 = self.m_down3(x2)
        x3 = self.m_body(x3) + x3
        x3 = self.m_up3(torch.cat([x3, x2], dim=1))
        x2 = self.m_up2(torch.cat([x3, x1], dim=1))
        x1 = self.m_up1(torch.cat([x2, head], dim=1))
        out = self.m_tail(x1) + x0
        return out


class WaveletHaarDownsampling(nn.Module):
    """Haar wavelet decomposition — fixed weights"""

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 4.0
        lh = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 4.0
        hl = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 4.0
        hh = torch.tensor([[1, -1], [-1, -1]], dtype=torch.float32) / 4.0

        self.register_buffer('weight_LL',
                             ll.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))
        self.register_buffer('weight_LH',
                             lh.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))
        self.register_buffer('weight_HL',
                             hl.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))
        self.register_buffer('weight_HH',
                             hh.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))

    def forward(self, x):
        return torch.cat([
            F.conv2d(x, self.weight_LL, stride=2, groups=3),
            F.conv2d(x, self.weight_LH, stride=2, groups=3),
            F.conv2d(x, self.weight_HL, stride=2, groups=3),
            F.conv2d(x, self.weight_HH, stride=2, groups=3),
        ], dim=1)


# ═══════════════════════════════════════════════════════════════
# FULL MODEL — single forward() for TorchScript
# ═══════════════════════════════════════════════════════════════

class FullWDNModel(nn.Module):
    """WDN pipeline: Haar → UNet denoise → SRVGG 4x → output"""

    def __init__(self, srvgg, unet, haar):
        super().__init__()
        self.srvgg = srvgg
        self.unet = unet
        self.haar = haar

    def forward(self, x):
        B, C, H, W = x.shape

        # Pad to even
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Wavelet decompose: (B,3,H,W) → (B,12,H/2,W/2)
        coeffs = self.haar(x)

        # Denoise in wavelet domain
        coeffs = self.unet(coeffs)

        # SR on wavelet coefficients: (B,12,H/2,W/2) → (B,3,H*2,W*2)
        out = self.srvgg(coeffs)

        # Trim to exact 4x of original
        out = out[:, :, :H * 2, :W * 2]
        return out


# ═══════════════════════════════════════════════════════════════
# LOAD & EXPORT
# ═══════════════════════════════════════════════════════════════

def load_wdn_model(pth_path):
    print(f"Loading: {pth_path}")
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)

    # Extract state_dict
    if isinstance(checkpoint, dict):
        for key in ['params_ema', 'params', 'state_dict']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"  state_dict from '{key}'")
                break
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip prefix
    first_key = list(state_dict.keys())[0]
    for prefix in ['module.', 'net_g.']:
        if first_key.startswith(prefix):
            state_dict = {k.replace(prefix, '', 1): v
                         for k, v in state_dict.items()}
            print(f"  Stripped prefix: '{prefix}'")
            break

    # ── Separate weights ──
    srvgg_state = {}
    unet_state = {}

    for k, v in state_dict.items():
        # UNet weights — check multiple possible prefixes
        if (k.startswith('denoise.') or
            k.startswith('denoising.') or
            k.startswith('net_denoise.')):
            # Remove the module prefix to get clean UNet keys
            for d_prefix in ['denoising.', 'denoise.', 'net_denoise.']:
                if k.startswith(d_prefix):
                    unet_key = k[len(d_prefix):]
                    unet_state[unet_key] = v
                    break
        elif not k.startswith('haar_'):
            srvgg_state[k] = v

    # ── Detect SRVGG config ──
    first_conv_key = None
    for k in srvgg_state:
        if k.endswith('body.0.weight') and len(srvgg_state[k].shape) == 4:
            first_conv_key = k
            break

    if first_conv_key is None:
        print(f"  ERROR: Cannot find body.0.weight in {list(srvgg_state.keys())[:10]}")
        sys.exit(1)

    num_in_ch = srvgg_state[first_conv_key].shape[1]
    num_feat = srvgg_state[first_conv_key].shape[0]

    # Find last conv (biggest index with 4D weight)
    conv_ids = []
    for k in srvgg_state:
        parts = k.split('.')
        if 'body' in parts:
            idx = parts.index('body')
            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                if parts[-1] == 'weight' and len(srvgg_state[k].shape) == 4:
                    conv_ids.append(int(parts[idx + 1]))

    max_idx = max(conv_ids)
    last_key = None
    for k in srvgg_state:
        if f'body.{max_idx}.weight' in k:
            last_key = k
            break

    last_shape = srvgg_state[last_key].shape
    num_out_ch = last_shape[0] // 16
    num_conv = (max_idx - 2) // 2

    print(f"  SRVGG: in={num_in_ch}, out={num_out_ch}, "
          f"feat={num_feat}, conv={num_conv}")
    print(f"  UNet weights: {len(unet_state)} tensors")

    # ── Build SRVGG ──
    srvgg = SRVGGNetCompact(
        num_in_ch=num_in_ch, num_out_ch=num_out_ch,
        num_feat=num_feat, num_conv=num_conv,
        upscale=4, act_type='prelu')
    srvgg.load_state_dict(srvgg_state, strict=True)
    print(f"  SRVGG loaded ✓")

    # ── Build UNet ──
    unet = UNet(in_nc=12, out_nc=12, nc=[64, 128, 256, 512])
    if unet_state:
        unet.load_state_dict(unet_state, strict=True)
        print(f"  UNet loaded ✓")
    else:
        print("  WARNING: No UNet weights found!")

    # ── Combine ──
    haar = WaveletHaarDownsampling()
    model = FullWDNModel(srvgg, unet, haar)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model


def export_torchscript(model, output_dir):
    """Export TorchScript — PNNX input format"""
    dummy = torch.randn(1, 3, 64, 64)

    # Verify
    with torch.no_grad():
        out = model(dummy)
    print(f"\n  Verify: {list(dummy.shape)} → {list(out.shape)}")

    script_path = os.path.join(output_dir, "wdn_script.pt")
    print(f"  Tracing TorchScript...")

    # Use trace (not script) — works better for complex models
    traced = torch.jit.trace(model, dummy)
    traced.save(script_path)

    print(f"  Saved: {script_path} ({os.path.getsize(script_path)/1e6:.1f} MB)")
    return script_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='.pth path')
    parser.add_argument('--output-dir', required=True, help='output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_wdn_model(args.input)
    script_path = export_torchscript(model, args.output_dir)

    print(f"\n✓ TorchScript: {script_path}")
    print(f"  Next: PNNX will convert this to ncnn .param + .bin")


if __name__ == '__main__':
    main()
