"""Compare adapter pixels with official preprocessing/quantization helpers."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch

from isharescreen.enhancement.models import Upscaler, project_home


def original_function(path,name):
    # Import just the original helper unchanged; old upstream module-level
    # imports reference deprecated skimage APIs unrelated to inference.
    tree=ast.parse(path.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.Module(body=[function],type_ignores=[])
    namespace={"np":np,"torch":torch}
    exec(compile(module,str(path),"exec"),namespace)
    return namespace[name]


def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cpu");args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(0)
    rgb=np.random.default_rng(0).integers(0,256,(24,32,3),dtype=np.uint8)
    vendor=project_home()/"vendor"
    evidence=[]
    for key in ["edsr","imdn","safmn"]:
        adapter=Upscaler(key,device=args.device)
        out,_=adapter(rgb)
        if key=="edsr":
            tensor=torch.from_numpy(rgb.transpose(2,0,1).copy()).float().unsqueeze(0)
        elif key=="imdn":
            tensor=torch.from_numpy((rgb/255.0).transpose(2,0,1).copy()).float().unsqueeze(0)
        else:
            tensor=original_function(vendor/"SAFMN/basicsr/utils/img_util.py","uint2tensor4")(rgb)
        with torch.no_grad():native=adapter.model(tensor.to(adapter.device))
        if key=="edsr":
            native=original_function(vendor/"EDSR-PyTorch/src/utility.py","quantize")(native,255)
            expected=native[0].byte().permute(1,2,0).cpu().numpy()
        elif key=="imdn":
            expected=original_function(vendor/"IMDN/utils.py","tensor2np")(native[0].cpu())
        else:
            expected=original_function(vendor/"SAFMN/basicsr/utils/img_util.py","tensor2uint")(native)
        difference=np.abs(out.astype(int)-expected.astype(int))
        row={"model":key,"device":adapter.device,"max_abs_pixel_error":int(difference.max()),
             "different_channels":int(np.count_nonzero(difference)),"shape":list(out.shape),**adapter.provenance}
        if difference.max()>1:raise AssertionError(row)
        evidence.append(row)
    print(json.dumps(evidence,indent=2))


if __name__=="__main__":main()
