"""Experimental F(4x4,3x3) convolution with streamed inverse transforms.

All transforms stay inside the CUDA kernel. Six vertically transformed input
vectors and four partial inverse-transform vectors are reused per transform
row, limiting live registers. This module is not a selected inference backend.
"""
import torch
import triton
import triton.language as tl

from benchmarks.super_resolution.edsr_winograd import load_patch,store_output


@triton.jit
def bt_vertical(X,ys,xs,channels,valid,a,SRC_Y0,SRC_H:tl.constexpr,H:tl.constexpr,W:tl.constexpr):
    if a==0:
        value=4*load_patch(X,ys,xs,channels,valid,SRC_Y0,SRC_H,H,W)-5*load_patch(X,ys+2,xs,channels,valid,SRC_Y0,SRC_H,H,W)+load_patch(X,ys+4,xs,channels,valid,SRC_Y0,SRC_H,H,W)
    elif a==5:
        value=4*load_patch(X,ys+1,xs,channels,valid,SRC_Y0,SRC_H,H,W)-5*load_patch(X,ys+3,xs,channels,valid,SRC_Y0,SRC_H,H,W)+load_patch(X,ys+5,xs,channels,valid,SRC_Y0,SRC_H,H,W)
    else:
        c1=tl.where(a==1,-4.,tl.where(a==2,4.,tl.where(a==3,-2.,2.)))
        c2=tl.where(a<=2,-4.,-1.)
        c3=tl.where(a==1,1.,tl.where(a==2,-1.,tl.where(a==3,2.,-2.)))
        value=c1*load_patch(X,ys+1,xs,channels,valid,SRC_Y0,SRC_H,H,W)+c2*load_patch(X,ys+2,xs,channels,valid,SRC_Y0,SRC_H,H,W)+c3*load_patch(X,ys+3,xs,channels,valid,SRC_Y0,SRC_H,H,W)+load_patch(X,ys+4,xs,channels,valid,SRC_Y0,SRC_H,H,W)
    return value


@triton.jit
def bt_horizontal(r0,r1,r2,r3,r4,r5,b):
    if b==0:value=4*r0-5*r2+r4
    elif b==5:value=4*r1-5*r3+r5
    else:
        c1=tl.where(b==1,-4.,tl.where(b==2,4.,tl.where(b==3,-2.,2.)))
        c2=tl.where(b<=2,-4.,-1.)
        c3=tl.where(b==1,1.,tl.where(b==2,-1.,tl.where(b==3,2.,-2.)))
        value=c1*r1+c2*r2+c3*r3+r4
    return value


@triton.jit
def at_coeff(i):
    c0=tl.where(i<5,1.,0.)
    c1=tl.where(i==1,1.,tl.where(i==2,-1.,tl.where(i==3,2.,tl.where(i==4,-2.,0.))))
    c2=tl.where((i==1)|(i==2),1.,tl.where((i==3)|(i==4),4.,0.))
    c3=tl.where((i==1)|(i==5),1.,tl.where(i==2,-1.,tl.where(i==3,8.,tl.where(i==4,-8.,0.))))
    return c0,c1,c2,c3


@triton.jit
def winograd4_kernel(X,U,Bias,Skip,Y,SRC_Y0,DST_Y0,OUT_Y0,
    SRC_H:tl.constexpr,H:tl.constexpr,W:tl.constexpr,OUT_H:tl.constexpr,
    RELU:tl.constexpr,RESIDUAL:tl.constexpr,M:tl.constexpr,N:tl.constexpr,UNROLL:tl.constexpr):
    tiles=tl.program_id(0)*M+tl.arange(0,M)
    valid=tiles<tl.cdiv(OUT_H,4)*tl.cdiv(W,4)
    py=OUT_Y0+4*(tiles//tl.cdiv(W,4))-1;px=4*(tiles%tl.cdiv(W,4))-1
    ic=tl.arange(0,64);oc=tl.program_id(1)*N+tl.arange(0,N)
    v00=tl.full((M,N),0.,tl.float32);v01=tl.full((M,N),0.,tl.float32)
    v02=tl.full((M,N),0.,tl.float32);v03=tl.full((M,N),0.,tl.float32)
    v10=tl.full((M,N),0.,tl.float32);v11=tl.full((M,N),0.,tl.float32)
    v12=tl.full((M,N),0.,tl.float32);v13=tl.full((M,N),0.,tl.float32)
    v20=tl.full((M,N),0.,tl.float32);v21=tl.full((M,N),0.,tl.float32)
    v22=tl.full((M,N),0.,tl.float32);v23=tl.full((M,N),0.,tl.float32)
    v30=tl.full((M,N),0.,tl.float32);v31=tl.full((M,N),0.,tl.float32)
    v32=tl.full((M,N),0.,tl.float32);v33=tl.full((M,N),0.,tl.float32)
    for a in tl.range(0,6,loop_unroll_factor=1):
        r0=bt_vertical(X,py,px,ic,valid,a,SRC_Y0,SRC_H,H,W)
        r1=bt_vertical(X,py,px+1,ic,valid,a,SRC_Y0,SRC_H,H,W)
        r2=bt_vertical(X,py,px+2,ic,valid,a,SRC_Y0,SRC_H,H,W)
        r3=bt_vertical(X,py,px+3,ic,valid,a,SRC_Y0,SRC_H,H,W)
        r4=bt_vertical(X,py,px+4,ic,valid,a,SRC_Y0,SRC_H,H,W)
        r5=bt_vertical(X,py,px+5,ic,valid,a,SRC_Y0,SRC_H,H,W)
        t0=tl.full((M,N),0.,tl.float32);t1=tl.full((M,N),0.,tl.float32)
        t2=tl.full((M,N),0.,tl.float32);t3=tl.full((M,N),0.,tl.float32)
        for b in tl.range(0,6,loop_unroll_factor=UNROLL):
            transformed=(bt_horizontal(r0,r1,r2,r3,r4,r5,b)*(1/16)).to(tl.float16)
            weights=tl.load(U+(a*6+b)*4096+ic[:,None]*64+oc[None,:],cache_modifier=".cg")
            product=tl.dot(transformed,weights,allow_tf32=False)
            c0,c1,c2,c3=at_coeff(b)
            t0+=c0*product;t1+=c1*product;t2+=c2*product;t3+=c3*product
        c0,c1,c2,c3=at_coeff(a)
        v00+=c0*t0;v01+=c0*t1;v02+=c0*t2;v03+=c0*t3
        v10+=c1*t0;v11+=c1*t1;v12+=c1*t2;v13+=c1*t3
        v20+=c2*t0;v21+=c2*t1;v22+=c2*t2;v23+=c2*t3
        v30+=c3*t0;v31+=c3*t1;v32+=c3*t2;v33+=c3*t3
    for j in tl.static_range(4):
        if j==0:p0=v00;p1=v10;p2=v20;p3=v30
        elif j==1:p0=v01;p1=v11;p2=v21;p3=v31
        elif j==2:p0=v02;p1=v12;p2=v22;p3=v32
        else:p0=v03;p1=v13;p2=v23;p3=v33
        vx=valid&(px+1+j<W)
        store_output(Y,Skip,Bias,p0,py+1,px+1+j,oc,vx&(py+1-OUT_Y0<OUT_H),DST_Y0,H,W,RELU,RESIDUAL)
        store_output(Y,Skip,Bias,p1,py+2,px+1+j,oc,vx&(py+2-OUT_Y0<OUT_H),DST_Y0,H,W,RELU,RESIDUAL)
        store_output(Y,Skip,Bias,p2,py+3,px+1+j,oc,vx&(py+3-OUT_Y0<OUT_H),DST_Y0,H,W,RELU,RESIDUAL)
        store_output(Y,Skip,Bias,p3,py+4,px+1+j,oc,vx&(py+4-OUT_Y0<OUT_H),DST_Y0,H,W,RELU,RESIDUAL)


def transform_weights4(weights):
    assert weights.shape==(64,64,3,3)
    g=torch.tensor([[.25,0,0],[-1/6,-1/6,-1/6],[-1/6,1/6,-1/6],
        [1/24,1/12,1/6],[1/24,-1/12,1/6],[0,0,1]],dtype=torch.float64,device=weights.device)
    return (torch.einsum("ah,ochw,bw->abco",g,weights.double(),g)*16).contiguous()


def winograd4(x,u,bias,out,skip=None,*,relu=False,block_m=16,block_n=32,warps=8,unroll=1,
              src_y0=0,dst_y0=0,out_y0=0,out_h=None,global_h=None):
    assert x.ndim==4 and x.shape[:2]==(1,64) and x.dtype==torch.float16
    assert x.is_contiguous(memory_format=torch.channels_last) and out.is_contiguous(memory_format=torch.channels_last)
    h=x.shape[2] if global_h is None else global_h;oh=out.shape[2] if out_h is None else out_h;w=x.shape[3]
    grid=(triton.cdiv(triton.cdiv(oh,4)*triton.cdiv(w,4),block_m),64//block_n)
    return winograd4_kernel[grid](x,u,bias,x if skip is None else skip,out,src_y0,dst_y0,out_y0,
        x.shape[2],h,w,oh,relu,skip is not None,block_m,block_n,unroll,num_warps=warps,num_stages=1)
