#!/usr/bin/env python3
import argparse, torch, torch.nn as nn, os

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    print(f"Loading {args.input}")
    ckpt = torch.load(args.input, map_location='cpu', weights_only=False)
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
            if i+1 < len(parts) and parts[i+1].isdigit() and parts[-1] == 'weight' and sd[k].dim() == 4:
                conv_ids.append(int(parts[i+1]))
    max_idx = max(conv_ids)
    last_k = [k for k in sd if f'body.{max_idx}.weight' in k][0]
    num_out = sd[last_k].shape[0] // 16
    num_conv = (max_idx - 2) // 2

    # SRVGG keys only
    srvgg_sd = {k: v for k, v in sd.items() if not k.startswith(('denoise.','denoising.','haar_'))}

    print(f"SRVGG: in={num_in} out={num_out} feat={num_feat} conv={num_conv}")

    model = SRVGGNetCompact(num_in, num_out, num_feat, num_conv, 4, 'prelu')
    model.load_state_dict(srvgg_sd, strict=True)
    model.eval()

    dummy = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(dummy)
    print(f"Verify: {list(dummy.shape)} → {list(out.shape)}")

    torch.onnx.export(model, dummy, args.output, opset_version=11,
                      input_names=['in0'], output_names=['out0'],
                      do_constant_folding=True)
    print(f"Saved: {args.output} ({os.path.getsize(args.output)/1e6:.1f} MB)")

    # Simplify
    try:
        import onnx
        from onnxsim import simplify
        m = onnx.load(args.output)
        m2, ok = simplify(m, input_shapes={'in0': [1,3,64,64]})
        if ok:
            onnx.save(m2, args.output)
            print("Simplified OK")
    except Exception as e:
        print(f"Simplify skip: {e}")

    print("Done")

if __name__ == '__main__':
    main()
