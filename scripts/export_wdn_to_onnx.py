#!/usr/bin/env python3
"""
Convert realesr-general-wdn-x4v3.pth → ONNX.
Handles WDN (wavelet denoise + SRVGG) architecture.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys


# ═══════════════════════════════════════════════════════
# SRVGGNetCompact — exact from basicsr/archs/srvgg_arch.py
# ═══════════════════════════════════════════════════════

class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                 num_conv=16, upscale=4, act_type='prelu'):
        super().__init__()
        self.in_nc = num_in_ch
        self.out_nc = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        if act_type == 'prelu':
            self.body.append(nn.PReLU(num_parameters=num_feat))
        elif act_type == 'leakyrelu':
            self.body.append(nn.LeakyReLU(0.1, inplace=True))
        else:
            self.body.append(nn.ReLU(inplace=True))

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if act_type == 'prelu':
                self.body.append(nn.PReLU(num_parameters=num_feat))
            elif act_type == 'leakyrelu':
                self.body.append(nn.LeakyReLU(0.1, inplace=True))
            else:
                self.body.append(nn.ReLU(inplace=True))

        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = self.body[0](x)
        for i in range(1, len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        return out


# ═══════════════════════════════════════════════════════
# UNet denoise — exact from basicsr/archs/denoising_arch.py
# ═══════════════════════════════════════════════════════

class UNet(nn.Module):
    def __init__(self, in_nc=12, out_nc=12, nc=[64, 128, 256, 512]):
        super().__init__()
        self.m_head = nn.Conv2d(in_nc, nc[0], 3, 1, 1, bias=False)

        self.m_down1 = nn.Sequential(
            nn.Conv2d(nc[0], nc[1], 2, 2, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True))
        self.m_down2 = nn.Sequential(
            nn.Conv2d(nc[1], nc[2], 2, 2, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True))
        self.m_down3 = nn.Sequential(
            nn.Conv2d(nc[2], nc[3], 2, 2, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True))

        self.m_body = nn.Sequential(
            nn.Conv2d(nc[3], nc[3], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc[3], nc[3], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc[3], nc[3], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc[3], nc[3], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True))

        self.m_up3 = nn.Sequential(
            nn.Conv2d(nc[3] * 2, nc[3], 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc[3], nc[3] * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2))
        self.m_up2 = nn.Sequential(
            nn.Conv2d(nc[2] * 2, nc[2], 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc[2], nc[2] * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2))
        self.m_up1 = nn.Sequential(
            nn.Conv2d(nc[1] * 2, nc[1], 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
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


# ═══════════════════════════════════════════════════════
# Haar Wavelet — fixed weights
# ═══════════════════════════════════════════════════════

class WaveletHaarDownsampling(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 4.0
        lh = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 4.0
        hl = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 4.0
        hh = torch.tensor([[1, -1], [-1, -1]], dtype=torch.float32) / 4.0

        self.register_buffer('weight_LL', ll.view(1,1,2,2).repeat(3,1,1,1))
        self.register_buffer('weight_LH', lh.view(1,1,2,2).repeat(3,1,1,1))
        self.register_buffer('weight_HL', hl.view(1,1,2,2).repeat(3,1,1,1))
        self.register_buffer('weight_HH', hh.view(1,1,2,2).repeat(3,1,1,1))

    def forward(self, x):
        return torch.cat([
            F.conv2d(x, self.weight_LL, stride=2, groups=3),
            F.conv2d(x, self.weight_LH, stride=2, groups=3),
            F.conv2d(x, self.weight_HL, stride=2, groups=3),
            F.conv2d(x, self.weight_HH, stride=2, groups=3),
        ], dim=1)


# ═══════════════════════════════════════════════════════
# Full WDN — single forward() for ONNX export
# ═══════════════════════════════════════════════════════

class FullWDNModel(nn.Module):
    def __init__(self, srvgg, unet, haar):
        super().__init__()
        self.srvgg = srvgg
        self.unet = unet
        self.haar = haar

    def forward(self, x):
        B, C, H, W = x.shape

        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        coeffs = self.haar(x)          # (B, 12, H/2, W/2)
        coeffs = self.unet(coeffs)     # denoise
        out = self.srvgg(coeffs)       # (B, 3, H*2, W*2)
        out = out[:, :, :H * 2, :W * 2]
        return out


# ═══════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════

def load_wdn_model(pth_path):
    print(f"Loading: {pth_path}")
    ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)

    if isinstance(ckpt, dict):
        for k in ['params_ema', 'params', 'state_dict']:
            if k in ckpt:
                sd = ckpt[k]
                print(f"  Found under '{k}'")
                break
        else:
            sd = ckpt
    else:
        sd = ckpt

    # Strip prefix
    first = list(sd.keys())[0]
    for pfx in ['module.', 'net_g.']:
        if first.startswith(pfx):
            sd = {k.replace(pfx, '', 1): v for k, v in sd.items()}
            print(f"  Stripped '{pfx}'")
            break

    # Separate SRVGG vs UNet weights
    srvgg_sd = {}
    unet_sd = {}
    for k, v in sd.items():
        if k.startswith('haar_'):
            continue
        is_unet = False
        for dp in ['denoising.', 'denoise.', 'net_denoise.']:
            if k.startswith(dp):
                unet_sd[k[len(dp):]] = v
                is_unet = True
                break
        if not is_unet:
            srvgg_sd[k] = v

    # Detect SRVGG config
    body_key = [k for k in srvgg_sd if k.endswith('body.0.weight')][0]
    num_in = srvgg_sd[body_key].shape[1]
    num_feat = srvgg_sd[body_key].shape[0]

    conv_ids = []
    for k in srvgg_sd:
        parts = k.split('.')
        if 'body' in parts:
            i = parts.index('body')
            if (i+1 < len(parts) and parts[i+1].isdigit()
                    and parts[-1] == 'weight'
                    and srvgg_sd[k].dim() == 4):
                conv_ids.append(int(parts[i+1]))

    max_idx = max(conv_ids)
    last_k = [k for k in srvgg_sd if f'body.{max_idx}.weight' in k][0]
    num_out = srvgg_sd[last_k].shape[0] // 16
    num_conv = (max_idx - 2) // 2

    print(f"  SRVGG: in={num_in} out={num_out} feat={num_feat} conv={num_conv}")
    print(f"  UNet:  {len(unet_sd)} tensors")

    srvgg = SRVGGNetCompact(num_in, num_out, num_feat, num_conv, 4, 'prelu')
    srvgg.load_state_dict(srvgg_sd, strict=True)
    print("  SRVGG ✓")

    unet = UNet(12, 12, [64, 128, 256, 512])
    unet.load_state_dict(unet_sd, strict=True)
    print("  UNet  ✓")

    haar = WaveletHaarDownsampling()
    model = FullWDNModel(srvgg, unet, haar)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ═══════════════════════════════════════════════════════
# EXPORT ONNX
# ═══════════════════════════════════════════════════════

def export_onnx(model, out_path, sz=64):
    print(f"\nExporting ONNX (1x3x{sz}x{sz})...")
    dummy = torch.randn(1, 3, sz, sz)

    with torch.no_grad():
        out = model(dummy)
    print(f"  {list(dummy.shape)} → {list(out.shape)}")

    torch.onnx.export(
        model, dummy, out_path,
        opset_version=11,
        input_names=['in0'],
        output_names=['out0'],
        dynamic_axes=None,
        do_constant_folding=True)
    print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")

    # Simplify
    sim_path = out_path.replace('.onnx', '.sim.onnx')
    try:
        import onnx
        from onnxsim import simplify
        print("  Simplifying...")
        m = onnx.load(out_path)
        m2, ok = simplify(m, input_shapes={'in0': [1,3,sz,sz]})
        if ok:
            onnx.save(m2, sim_path)
            print(f"  Simplified: {sim_path}")
            return sim_path
    except Exception as e:
        print(f"  onnxsim skip: {e}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    model = load_wdn_model(args.input)
    export_onnx(model, args.output)
    print("\n✓ Done. Next: onnx2ncnn")


if __name__ == '__main__':
    main()
