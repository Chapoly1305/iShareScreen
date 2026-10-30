"""Same uniform-window RGB SSIM as skimage defaults, using exact integral sums.

All integer RGB moments and integral sums fit exactly in float64 (< 2**53).
Only valid 7x7 windows are averaged, matching skimage's three-pixel crop and
sample-covariance correction. This does not approximate or resize the images.
"""
import math

import torch


def quality(reference, output):
    assert reference.shape == output.shape
    with torch.inference_mode():
        x=torch.from_numpy(reference.copy()).to('cuda:0',dtype=torch.float64).permute(2,0,1)[None]
        y=torch.from_numpy(output.copy()).to('cuda:0',dtype=torch.float64).permute(2,0,1)[None]
        moments=torch.cat((x,y,x*x,y*y,x*y),dim=1)
        integral=torch.nn.functional.pad(moments.cumsum(-1).cumsum(-2),(1,0,1,0))
        means=(integral[...,7:,7:]-integral[...,:-7,7:]-integral[...,7:,:-7]+integral[...,:-7,:-7])/49
        ux,uy,xx,yy,xy=means.split(3,dim=1)
        vx=(xx-ux*ux)*49/48;vy=(yy-uy*uy)*49/48;cov=(xy-ux*uy)*49/48
        ssim=(((2*ux*uy+6.5025)*(2*cov+58.5225))/((ux*ux+uy*uy+6.5025)*(vx+vy+58.5225))).mean()
        diff=x-y
        def psnr(error):
            mse=float(error.square().mean().item())
            return 10*math.log10(255**2/mse) if mse else None
        rgb_psnr=psnr(diff)
        matrix=torch.tensor([[.2126,.7152,.0722],[-.114572,-.385428,.5],[.5,-.454153,-.045847]],dtype=torch.float32,device=x.device)
        color=diff[0].permute(1,2,0).float() @ matrix.T
        return dict(psnr=rgb_psnr,ssim=float(ssim.item()),y_psnr=psnr(color[...,0]),
                    chroma_psnr=psnr(color[...,1:]),pixel_metric_backend='cuda_float64_integral_ssim')
