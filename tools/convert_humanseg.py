"""
U2Net Portrait Segmentation → NCNN via PNNX
Approach: PyTorch pretrained → PNNX → NCNN (giống DnCNN)
"""

import torch
import torch.nn as nn
import pnnx
import os
import sys
import gc
import urllib.request

WORK_DIR = "convert_tmp"
OUTPUT_DIR = "output"

# ═══════════════════════════════════════════════════
# U2-Net Architecture (full)
# Source: https://github.com/xuebinqin/U-2-Net
# ═══════════════════════════════════════════════════

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super().__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def _upsample(self, x, size):
        return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx5d = self.rebnconv5d(torch.cat((self._upsample(hx6d, hx5.shape[2:]), hx5), 1))
        hx4d = self.rebnconv4d(torch.cat((self._upsample(hx5d, hx4.shape[2:]), hx4), 1))
        hx3d = self.rebnconv3d(torch.cat((self._upsample(hx4d, hx3.shape[2:]), hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((self._upsample(hx3d, hx2.shape[2:]), hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((self._upsample(hx2d, hx1.shape[2:]), hx1), 1))
        return hx1d + hxin


class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def _upsample(self, x, size):
        return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx4d = self.rebnconv4d(torch.cat((self._upsample(hx5, hx4.shape[2:]), hx4), 1))
        hx3d = self.rebnconv3d(torch.cat((self._upsample(hx4d, hx3.shape[2:]), hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((self._upsample(hx3d, hx2.shape[2:]), hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((self._upsample(hx2d, hx1.shape[2:]), hx1), 1))
        return hx1d + hxin


class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def _upsample(self, x, size):
        return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx3d = self.rebnconv3d(torch.cat((self._upsample(hx4, hx3.shape[2:]), hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((self._upsample(hx3d, hx2.shape[2:]), hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((self._upsample(hx2d, hx1.shape[2:]), hx1), 1))
        return hx1d + hxin


class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def _upsample(self, x, size):
        return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx2d = self.rebnconv2d(torch.cat((self._upsample(hx3, hx2.shape[2:]), hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((self._upsample(hx2d, hx1.shape[2:]), hx1), 1))
        return hx1d + hxin


class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx4d = self.rebnconv4d(torch.cat((hx4, hx3), 1))
        hx3d = self.rebnconv3d(torch.cat((hx4d, hx2), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx1), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hxin), 1))
        return hx1d + hxin


class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)

    def _upsample(self, x, size):
        return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)

    def forward(self, x):
        hx = x
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        hx6 = self.stage6(hx)
        hx5d = self.stage5d(torch.cat((self._upsample(hx6, hx5.shape[2:]), hx5), 1))
        hx4d = self.stage4d(torch.cat((self._upsample(hx5d, hx4.shape[2:]), hx4), 1))
        hx3d = self.stage3d(torch.cat((self._upsample(hx4d, hx3.shape[2:]), hx3), 1))
        hx2d = self.stage2d(torch.cat((self._upsample(hx3d, hx2.shape[2:]), hx2), 1))
        hx1d = self.stage1d(torch.cat((self._upsample(hx2d, hx1.shape[2:]), hx1), 1))

        d1 = self.side1(hx1d)
        d2 = self._upsample(self.side2(hx2d), d1.shape[2:])
        d3 = self._upsample(self.side3(hx3d), d1.shape[2:])
        d4 = self._upsample(self.side4(hx4d), d1.shape[2:])
        d5 = self._upsample(self.side5(hx5d), d1.shape[2:])
        d6 = self._upsample(self.side6(hx6), d1.shape[2:])

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        return torch.sigmoid(d0)


# ═══════════════════════════════════════════════════
# Main — identical pattern to DnCNN converter
# ═══════════════════════════════════════════════════

WEIGHT_URL = "https://github.com/xuebinqin/U-2-Net/raw/master/saved_models/u2net/u2net.pth"
WEIGHT_PATH = "u2net.pth"

print("=" * 50)
print("U2Net Portrait → NCNN Converter")
print("=" * 50)

print("1. Creating U2NET model...")
model = U2NET(in_ch=3, out_ch=1)
model.eval()
torch.set_grad_enabled(False)

params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {params:,}")

print("2. Downloading pretrained weights...")
if not os.path.exists(WEIGHT_PATH):
    print("   Downloading...")
    req = urllib.request.Request(WEIGHT_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as response:
        with open(WEIGHT_PATH, "wb") as f:
            shutil.copyfileobj(response, f)
    print(f"   Done: {os.path.getsize(WEIGHT_PATH)/1024/1024:.1f} MB")
else:
    print(f"   Already exists: {os.path.getsize(WEIGHT_PATH)/1024/1024:.1f} MB")

print("   Loading weights...")
ckpt = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)

if isinstance(ckpt, dict):
    if 'params' in ckpt:
        state_dict = ckpt['params']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
else:
    state_dict = ckpt

# Handle 'module.' prefix
new_state_dict = {}
for k, v in state_dict.items():
    name = k.replace('module.', '')
    new_state_dict[name] = v

model.load_state_dict(new_state_dict, strict=True)
print("   Weights loaded OK!")

print("3. Converting to NCNN via PNNX...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dùng input size nhỏ hơn để convert nhanh, PNNX sẽ handle dynamic size
dummy = torch.randn(1, 3, 192, 192)

try:
    pnnx.export(model, "u2net", inputs=dummy)
    print("   Done!")
except Exception as e:
    print(f"   PNNX failed: {e}")
    sys.exit(1)

del model, dummy
gc.collect()

print("4. Verifying + renaming...")
pf = "u2net.ncnn.param"
bf = "u2net.ncnn.bin"

if os.path.exists(pf) and os.path.exists(bf):
    sp = os.path.getsize(pf) / 1024
    sb = os.path.getsize(bf) / 1024
    print(f"   {pf}: {sp:.1f} KB")
    print(f"   {bf}: {sb:.1f} KB")
    print(f"   Total: {(sp + sb) / 1024:.1f} MB")

    # Rename to output
    dst_p = os.path.join(OUTPUT_DIR, "humansegv2.ncnn.param")
    dst_b = os.path.join(OUTPUT_DIR, "humansegv2.ncnn.bin")
    shutil.move(pf, dst_p)
    shutil.move(bf, dst_b)

    # Verify param content
    with open(dst_p, "r") as f:
        content = f.read()
        if "Input" in content:
            print("   Param has Input layer ✓")
        # Find output blob
        for line in reversed(content.strip().split('\n')):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith("7767517"):
                parts = line.split()
                if len(parts) >= 4:
                    print(f"   Output blob: {parts[-1]}")
                    break

    print("U2NET → NCNN OK!")
else:
    print("FAILED!")
    print(f"Files: {os.listdir('.')}")
    sys.exit(1)

print("=" * 50)
print("DONE!")
print(f"  {dst_p}")
print(f"  {dst_b}")
print("=" * 50)
