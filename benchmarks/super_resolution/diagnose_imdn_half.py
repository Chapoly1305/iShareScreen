"""Preserve the pure-FP16 failure evidence using unmodified official forward code."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from isharescreen.enhancement.models import Upscaler


def main():
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4)
    adapter=Upscaler("imdn",device="cuda",precision="fp32")
    native=adapter.model.half()
    observations=[]
    def inspect(module,inputs):
        if observations:return
        x=inputs[0]
        summed=x.sum(3,keepdim=True).sum(2,keepdim=True)
        reference=x.float().sum(3,keepdim=True).sum(2,keepdim=True)
        observations.append({"activation_shape":list(x.shape),"input_all_finite":bool(torch.isfinite(x).all()),
                             "fp16_spatial_sum_all_finite":bool(torch.isfinite(summed).all()),
                             "fp32_spatial_sum_all_finite":bool(torch.isfinite(reference).all()),
                             "fp32_spatial_sum_max_abs":float(reference.abs().max())})
    handles=[m.register_forward_pre_hook(inspect) for m in native.modules() if m.__class__.__name__=="CCALayer"]
    rgb=np.array(Image.open(args.input).convert("RGB"))
    x=torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).to("cuda",dtype=torch.float16)/255
    with torch.inference_mode():out=native(x)
    for handle in handles:handle.remove()
    print(json.dumps({"checkpoint":adapter.provenance,"pure_half_diagnostic":True,"attention":observations,
                      "output_elements":out.numel(),"output_nonfinite":int((~torch.isfinite(out)).sum())},indent=2))


if __name__=="__main__":main()
