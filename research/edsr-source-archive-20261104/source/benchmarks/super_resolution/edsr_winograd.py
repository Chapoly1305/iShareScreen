"""Experimental fused F(2x2,3x3) Winograd for EDSR's 64-channel convolutions.

Input/weights/output transforms are fused into one kernel. No transformed
feature map is materialized in global memory. Input and output are FP16 NHWC
storage, represented as PyTorch channels-last NCHW tensors. Accumulation and
the transforms use FP32 before the tensor-core dot's FP16 operands.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def transform_pair(i):
    return tl.where(i==0,0,1),tl.where(i==3,3,2),tl.where(i==2,-1.,1.),tl.where((i==0)|(i==3),-1.,1.)


@triton.jit
def load_patch(X,ys,xs,channels,tiles_valid,SRC_Y0,
               SRC_H:tl.constexpr,GLOBAL_H:tl.constexpr,W:tl.constexpr):
    sy=ys-SRC_Y0
    valid=tiles_valid & (ys>=0) & (ys<GLOBAL_H) & (sy>=0) & (sy<SRC_H) & (xs>=0) & (xs<W)
    return tl.load(X+(sy[:,None]*W+xs[:,None])*64+channels[None,:],valid[:,None],0,cache_modifier=".ca").to(tl.float32)


@triton.jit
def store_output(Y,Skip,Bias,values,ys,xs,channels,tiles_valid,DST_Y0,
                 GLOBAL_H:tl.constexpr,W:tl.constexpr,RELU:tl.constexpr,RESIDUAL:tl.constexpr):
    values=values+tl.load(Bias+channels)[None,:]
    valid_global=(ys>=0)&(ys<GLOBAL_H)
    if RESIDUAL:
        values+=tl.load(Skip+(ys[:,None]*W+xs[:,None])*64+channels[None,:],(tiles_valid & valid_global)[:,None],0).to(tl.float32)
    if RELU:values=tl.maximum(values,0.)
    values=tl.where(valid_global[:,None],values,0.)
    tl.store(Y+((ys-DST_Y0)[:,None]*W+xs[:,None])*64+channels[None,:],values,tiles_valid[:,None])


@triton.jit
def winograd_kernel(X,U,Bias,Skip,Y,SRC_Y0,DST_Y0,OUT_Y0,
                    SRC_H:tl.constexpr,GLOBAL_H:tl.constexpr,W:tl.constexpr,OUT_H:tl.constexpr,
                    RELU:tl.constexpr,RESIDUAL:tl.constexpr,BLOCK_M:tl.constexpr,UNROLL:tl.constexpr):
    tiles=tl.program_id(0)*BLOCK_M+tl.arange(0,BLOCK_M)
    valid=tiles<(OUT_H//2)*(W//2)
    py=OUT_Y0+2*(tiles//(W//2))-1
    px=2*(tiles%(W//2))-1
    channels=tl.arange(0,64)
    v00=tl.full((BLOCK_M,64),0.,tl.float32)
    v01=tl.full((BLOCK_M,64),0.,tl.float32)
    v10=tl.full((BLOCK_M,64),0.,tl.float32)
    v11=tl.full((BLOCK_M,64),0.,tl.float32)
    # Keep only one transformed product live. Fully unrolling all sixteen
    # products lets CSE retain too many input patches and causes spilling.
    for transform in tl.range(0,16,loop_unroll_factor=UNROLL):
        a=transform//4;b=transform%4
        r0,r1,sr0,sr1=transform_pair(a)
        c0,c1,sc0,sc1=transform_pair(b)
        d00=load_patch(X,py+r0,px+c0,channels,valid,SRC_Y0,SRC_H,GLOBAL_H,W)
        d01=load_patch(X,py+r0,px+c1,channels,valid,SRC_Y0,SRC_H,GLOBAL_H,W)
        d10=load_patch(X,py+r1,px+c0,channels,valid,SRC_Y0,SRC_H,GLOBAL_H,W)
        d11=load_patch(X,py+r1,px+c1,channels,valid,SRC_Y0,SRC_H,GLOBAL_H,W)
        transformed=((d00*sr0+d10*sr1)*sc0+(d01*sr0+d11*sr1)*sc1).to(tl.float16)
        weights=tl.load(U+transform*4096+channels[:,None]*64+channels[None,:],cache_modifier=".cg")
        product=tl.dot(transformed,weights,allow_tf32=False)
        if (a<3)&(b<3):v00+=product
        if (a<3)&(b>0):
            if b==1:v01+=product
            else:v01-=product
        if (a>0)&(b<3):
            if a==1:v10+=product
            else:v10-=product
        if (a>0)&(b>0):
            if ((a==1)&(b==1))|((a>1)&(b>1)):v11+=product
            else:v11-=product
    store_output(Y,Skip,Bias,v00,py+1,px+1,channels,valid,DST_Y0,GLOBAL_H,W,RELU,RESIDUAL)
    store_output(Y,Skip,Bias,v01,py+1,px+2,channels,valid,DST_Y0,GLOBAL_H,W,RELU,RESIDUAL)
    store_output(Y,Skip,Bias,v10,py+2,px+1,channels,valid,DST_Y0,GLOBAL_H,W,RELU,RESIDUAL)
    store_output(Y,Skip,Bias,v11,py+2,px+2,channels,valid,DST_Y0,GLOBAL_H,W,RELU,RESIDUAL)


def transform_weights(weights):
    assert weights.shape==(64,64,3,3)
    matrix=torch.tensor([[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]],dtype=torch.float64,device=weights.device)
    return torch.einsum("ah,ochw,bw->abco",matrix,weights.double(),matrix).contiguous()


def winograd(x,u,bias,out,skip=None,*,relu=False,block_m=16,warps=4,unroll=1,
             src_y0=0,dst_y0=0,out_y0=0,out_h=None,global_h=None):
    assert x.ndim==4 and x.shape[:2]==(1,64) and x.dtype==torch.float16
    assert x.is_contiguous(memory_format=torch.channels_last) and out.is_contiguous(memory_format=torch.channels_last)
    h=x.shape[2] if global_h is None else global_h
    oh=out.shape[2] if out_h is None else out_h;w=x.shape[3]
    assert oh%2==0 and w%2==0
    return winograd_kernel[(triton.cdiv(oh*w//4,block_m),)](x,u,bias,x if skip is None else skip,out,
        src_y0,dst_y0,out_y0,x.shape[2],h,w,oh,relu,skip is not None,block_m,unroll,
        num_warps=warps,num_stages=1,enable_fp_fusion=False)
