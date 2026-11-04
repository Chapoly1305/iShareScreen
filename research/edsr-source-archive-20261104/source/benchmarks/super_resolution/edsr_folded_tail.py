"""Algebraically lower EDSR's linear x2 upsampling tail into LR convolutions.

No learning, pruning, or checkpoint changes. Floating-point reassociation still
needs the deployment quality gate. Borders retain the original operations:
naively composing padded convolutions changes the outermost HR pixel.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FoldedTail(nn.Module):
    def __init__(self, original):
        super().__init__()
        up, last = original[0][0], original[1]
        assert isinstance(original[0][1], nn.PixelShuffle) and original[0][1].upscale_factor == 2
        assert len(original[0]) == 2 and len(original) == 2
        assert up.kernel_size == last.kernel_size == (3, 3)
        assert up.stride == last.stride == (1, 1)
        assert up.padding == last.padding == (1, 1)
        assert up.out_channels == 4 * last.in_channels
        self.up, self.last = up, last
        dtype = up.weight.dtype
        u, v = up.weight.detach().double(), last.weight.detach().double()
        ub, vb = up.bias.detach().double(), last.bias.detach().double()
        c, d, o = up.in_channels, last.in_channels, last.out_channels
        w = torch.zeros(o*4, c, 5, 5, dtype=torch.float64, device=u.device)
        b = torch.zeros(o*4, dtype=torch.float64, device=u.device)
        for py in range(2):
            for px in range(2):
                phase = py*2 + px
                b[phase::4] = vb
                for ky in range(3):
                    for kx in range(3):
                        ly, iy = divmod(py+ky-1, 2)
                        lx, ix = divmod(px+kx-1, 2)
                        channels = torch.arange(d, device=u.device)*4 + iy*2 + ix
                        w[phase::4, :, ly+1:ly+4, lx+1:lx+4] += torch.einsum("od,dcij->ocij",v[:,:,ky,kx],u[channels])
                        b[phase::4] += v[:,:,ky,kx] @ ub[channels]
        self.folded = nn.Conv2d(c, o*4, 5, padding=2, device=u.device, dtype=dtype)
        with torch.no_grad():
            self.folded.weight.copy_(w.to(dtype));self.folded.bias.copy_(b.to(dtype))
        self.requires_grad_(False)

    def border(self, x, padding):
        # Compute only the LR edge row/column; its two HR phases suffice for
        # the outermost HR output. Other axes retain the original zero padding.
        u = F.conv2d(F.pad(x,padding),self.up.weight,self.up.bias)
        return self.last(F.pixel_shuffle(u,2))

    def forward(self, x):
        interior = F.pixel_shuffle(self.folded(x),2)
        top = self.border(x[:,:,:2,:],(1,1,1,0))[:,:,:1,:]
        bottom = self.border(x[:,:,-2:,:],(1,1,0,1))[:,:,-1:,:]
        left = self.border(x[:,:,:,:2],(1,0,1,1))[:,:,1:-1,:1]
        right = self.border(x[:,:,:,-2:],(0,1,1,1))[:,:,1:-1,-1:]
        middle = torch.cat((left,interior[:,:,1:-1,1:-1],right),dim=3)
        return torch.cat((top,middle,bottom),dim=2)


def selftest():
    import json
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    records=[]
    for dtype in (torch.float64,torch.float32):
        original=nn.Sequential(nn.Sequential(nn.Conv2d(64,256,3,padding=1,dtype=dtype),nn.PixelShuffle(2)),
                               nn.Conv2d(64,3,3,padding=1,dtype=dtype)).eval()
        folded=FoldedTail(original)
        for h,w in ((2,2),(3,5),(9,13),(18,32)):
            for kind in ("random","constant","corner_impulse"):
                x=torch.randn(1,64,h,w,dtype=dtype)
                if kind=="constant":x.fill_(1)
                if kind=="corner_impulse":x.zero_();x[:,:,0,0]=1;x[:,:,-1,-1]=-1
                with torch.no_grad():
                    expected=original(x);actual=folded(x)
                error=float((actual-expected).abs().max())
                torch.testing.assert_close(actual,expected,atol=2e-12 if dtype==torch.float64 else 3e-6,rtol=1e-10 if dtype==torch.float64 else 3e-5)
                records.append(dict(dtype=str(dtype),shape=[h,w],input=kind,max_abs_error=error))
    print(json.dumps(dict(passed=True,records=records),indent=2))


if __name__=="__main__":selftest()
