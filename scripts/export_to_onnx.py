#!/usr/bin/env python3
import torch
import torch.nn as nn
import pnnx
import os
import gc

# ═══════════════════════════════════════
# SRVGGNetCompact — exact from basicsr
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
# LOAD
# ═══════════════════════════════════════

PTH = "wdn-x4v3.pth"

print("1. Loading model...")
ckpt = torch.load(PTH, map_location="cpu", weights_only=False)
sd = ckpt.get("params_ema", ckpt.get("params", ckpt))

first = list(sd.keys())[0]
for pfx in ["module.", "net_g."]:
    if first.startswith(pfx):
        sd = {k.replace(pfx, "", 1): v for k, v in sd.items()}
        break

# Auto-detect config
body0 = [k for k in sd if k.endswith("body.0.weight")][0]
num_in = sd[body0].shape[1]
num_feat = sd[body0].shape[0]

conv_ids = []
for k in sd:
    parts = k.split(".")
    if "body" in parts:
        i = parts.index("body")
        if (i + 1 < len(parts) and parts[i + 1].isdigit()
                and parts[-1] == "weight" and sd[k].dim() == 4):
            conv_ids.append(int(parts[i + 1]))

max_idx = max(conv_ids)
last_k = [k for k in sd if f"body.{max_idx}.weight" in k][0]
num_out = sd[last_k].shape[0] // 16
num_conv = (max_idx - 2) // 2

srvgg_sd = {k: v for k, v in sd.items()
            if not k.startswith(("denoise.", "denoising.", "haar_"))}

print(f"   SRVGG: in={num_in} out={num_out} feat={num_feat} conv={num_conv}")

model = SRVGGNetCompact(num_in, num_out, num_feat, num_conv, 4, "prelu")
model.load_state_dict(srvgg_sd, strict=True)
model.eval()

torch.set_grad_enabled(False)

# ═══════════════════════════════════════
# CONVERT via pnnx.export()
# ═══════════════════════════════════════

print("2. Converting via PNNX...")
dummy = torch.randn(1, 3, 64, 64)

# Verify shape first
with torch.no_grad():
    out = model(dummy)
print(f"   Verify: {list(dummy.shape)} -> {list(out.shape)}")

pnnx.export(model, "wdn-x4v3", inputs=dummy)
print("   PNNX export done!")

# Cleanup
del model, dummy
gc.collect()

# ═══════════════════════════════════════
# VERIFY
# ═══════════════════════════════════════

param_f = "wdn-x4v3.ncnn.param"
bin_f = "wdn-x4v3.ncnn.bin"

if os.path.exists(param_f) and os.path.exists(bin_f):
    print(f"\n3. Output:")
    print(f"   {param_f}: {os.path.getsize(param_f)/1024:.1f} KB")
    print(f"   {bin_f}: {os.path.getsize(bin_f)/1024/1024:.1f} MB")

    with open(param_f) as f:
        lines = f.readlines()
    print(f"   Layers: {len(lines) - 2}")
    print(f"\n   First 5 lines:")
    for line in lines[:5]:
        print(f"   {line.rstrip()}")

    print("\n✓ NCNN conversion OK!")
else:
    print(f"\n✗ FAILED! Files: {os.listdir('.')}")
    exit(1)
