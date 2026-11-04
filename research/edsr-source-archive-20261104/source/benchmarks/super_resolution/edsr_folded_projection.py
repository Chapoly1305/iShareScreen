"""Compose EDSR's final linear projection and upsampling tail.

The 7x7 LR convolutions replace full-frame 64-channel intermediates. Four HR
edge pixels retain the original operations to preserve finite-image padding.
This is algebraic equivalence, not FP16 bit equivalence; deployment quality
must still be checked against the unchanged FP32 and AMP references.
"""
import copy
import json

import torch
from torch import nn
from torch.nn import functional as F

from benchmarks.super_resolution.edsr_folded_tail import FoldedTail


def compose(outer, inner, include_outer_bias=True):
    """Compose cross-correlation kernels in FP64, before destination rounding."""
    a,b=outer.weight.detach().double(),inner.weight.detach().double()
    ka,kb=a.shape[-1],b.shape[-1]
    assert a.shape[1]==b.shape[0] and a.shape[-2]==ka and b.shape[-2]==kb
    w=torch.zeros(a.shape[0],b.shape[1],ka+kb-1,ka+kb-1,dtype=torch.float64,device=a.device)
    for y in range(ka):
        for x in range(ka):
            w[:,:,y:y+kb,x:x+kb]+=torch.einsum('od,dcij->ocij',a[:,:,y,x],b)
    bias=a.sum(dim=(2,3)) @ inner.bias.detach().double()
    if include_outer_bias:bias+=outer.bias.detach().double()
    result=nn.Conv2d(b.shape[1],a.shape[0],ka+kb-1,padding=(ka+kb-2)//2,
                     dtype=outer.weight.dtype,device=outer.weight.device)
    with torch.no_grad():result.weight.copy_(w);result.bias.copy_(bias)
    return result


class FoldedProjectionTail(nn.Module):
    def __init__(self,project,tail,head=None):
        super().__init__()
        assert project.kernel_size==(3,3) and project.padding==(1,1)
        self.project=copy.deepcopy(project)
        self.up=copy.deepcopy(tail.up);self.last=copy.deepcopy(tail.last)
        self.folded_project=compose(tail.folded,project)
        self.fold_head=head is not None
        if self.fold_head:
            assert head.kernel_size==(3,3) and head.padding==(1,1)
            self.folded_head=compose(tail.folded,head,include_outer_bias=False)
        else:
            self.folded_head=copy.deepcopy(tail.folded)
            with torch.no_grad():self.folded_head.bias.zero_()
        self.requires_grad_(False)

    def border(self,x,head,padding):
        projected=F.conv2d(F.pad(x,padding),self.project.weight,self.project.bias)
        return self.last(F.pixel_shuffle(self.up(projected+head),2))

    def forward(self,features,head,head_input):
        # Fixed production shape is 720x1280. At least five LR pixels per axis
        # are required for these four-pixel HR edge strips.
        core=self.folded_project(features)+self.folded_head(head_input if self.fold_head else head)
        interior=F.pixel_shuffle(core,2)
        top=self.border(features[:,:,:5,:],head[:,:,:4,:],(1,1,1,0))[:,:,:4,:]
        bottom=self.border(features[:,:,-5:,:],head[:,:,-4:,:],(1,1,0,1))[:,:,-4:,:]
        left=self.border(features[:,:,:,:5],head[:,:,:,:4],(1,0,1,1))[:,:,4:-4,:4]
        right=self.border(features[:,:,:,-5:],head[:,:,:,-4:],(0,1,1,1))[:,:,4:-4,-4:]
        middle=torch.cat((left,interior[:,:,4:-4,4:-4],right),dim=3)
        return torch.cat((top,middle,bottom),dim=2)


def replace_tail(model,fold_head,export_path):
    """Replace only the tail subgraph in the frozen, balanced INT8 ONNX."""
    import onnx
    from onnx import numpy_helper
    data={w.name:numpy_helper.to_array(w).copy()for w in model.graph.initializer}
    def conv(prefix):
        weight=torch.from_numpy(data[prefix+'.weight'])
        layer=nn.Conv2d(weight.shape[1],weight.shape[0],weight.shape[2],padding=weight.shape[2]//2,dtype=weight.dtype)
        with torch.no_grad():
            layer.weight.copy_(weight);layer.bias.copy_(torch.from_numpy(data[prefix+'.bias']))
        return layer
    project=conv('model.body.16');head=conv('model.head.0')
    original=nn.Sequential(nn.Sequential(conv('model.tail.up'),nn.PixelShuffle(2)),conv('model.tail.last'))
    tail=FoldedTail(original)
    # Preserve the exact rounded coefficients already present in the source
    # ONNX; do not silently reconstruct its folded tail with different rounding.
    with torch.no_grad():
        tail.folded.weight.copy_(torch.from_numpy(data['model.tail.folded.weight']))
        tail.folded.bias.copy_(torch.from_numpy(data['model.tail.folded.bias']))
    module=FoldedProjectionTail(project,tail,head if fold_head else None).eval()
    shapes=((1,64,8,8),(1,64,8,8),(1,3,8,8))
    names=['features','head','head_input']
    torch.onnx.export(module,tuple(torch.zeros(s,dtype=torch.float16)for s in shapes),str(export_path),
        opset_version=17,input_names=names,output_names=['tail_output'],dynamo=False,do_constant_folding=True,
        dynamic_axes={**{k:{2:'h',3:'w'}for k in names},'tail_output':{2:'oh',3:'ow'}})
    graph=onnx.load(export_path).graph
    inputs=dict(features='/model/body/body.15/Add_output_0',head='/model/head/head.0/Conv_output_0',
                head_input='/model/sub_mean/Conv_output_0')
    def rename(name):return inputs.get(name,'projection_tail/'+name) if name else name
    inserted=[]
    for old in graph.node:
        node=copy.deepcopy(old);node.name='projection_tail/'+node.name
        for i,name in enumerate(node.input):node.input[i]=rename(name)
        for i,name in enumerate(node.output):node.output[i]=rename(name)
        inserted.append(node)
    for old in graph.initializer:
        item=copy.deepcopy(old);item.name=rename(item.name);model.graph.initializer.append(item)
    nodes=[];found=False
    for old in model.graph.node:
        node=copy.deepcopy(old)
        if node.name=='/model/add_mean/Conv':
            assert node.input[0]=='/model/tail/Concat_5_output_0'
            nodes.extend(inserted);node.input[0]=rename('tail_output');found=True
        nodes.append(node)
    assert found
    needed={'output'};kept=[]
    for node in reversed(nodes):
        if needed.intersection(node.output):kept.append(node);needed.update(node.input)
    kept.reverse();initializers=[copy.deepcopy(w)for w in model.graph.initializer if w.name in needed]
    del model.graph.node[:];model.graph.node.extend(kept)
    del model.graph.initializer[:];model.graph.initializer.extend(initializers)
    onnx.checker.check_model(model)
    return dict(fold_head=fold_head,interior_kernel=7,border_hr_pixels=4,
                source='Exact FP16 coefficients from the hash-bound source ONNX; FP64 composition then FP16 storage.')


@torch.inference_mode()
def selftest():
    torch.set_num_threads(1);torch.manual_seed(1729);records=[]
    for dtype in (torch.float64,torch.float32):
        project=nn.Conv2d(64,64,3,padding=1,dtype=dtype)
        head=nn.Conv2d(3,64,3,padding=1,dtype=dtype)
        original=nn.Sequential(nn.Sequential(nn.Conv2d(64,256,3,padding=1,dtype=dtype),nn.PixelShuffle(2)),
                               nn.Conv2d(64,3,3,padding=1,dtype=dtype))
        tail=FoldedTail(original)
        for fold_head in (False,True):
            combined=FoldedProjectionTail(project,tail,head if fold_head else None)
            for h,w in ((5,5),(7,11),(16,24)):
                for kind in ('random','constant','corner_impulse'):
                    x=torch.randn(1,64,h,w,dtype=dtype);rgb=torch.randn(1,3,h,w,dtype=dtype)
                    if kind=='constant':x.fill_(1);rgb.fill_(1)
                    if kind=='corner_impulse':
                        x.zero_();rgb.zero_();x[:,:,0,0]=1;x[:,:,-1,-1]=-1;rgb[:,:,0,0]=-1;rgb[:,:,-1,-1]=1
                    skip=head(rgb);expected=original(project(x)+skip);actual=combined(x,skip,rgb)
                    error=float((expected-actual).abs().max())
                    torch.testing.assert_close(actual,expected,atol=2e-12 if dtype==torch.float64 else 3e-6,
                                               rtol=1e-10 if dtype==torch.float64 else 3e-5)
                    records.append(dict(dtype=str(dtype),fold_head=fold_head,shape=[h,w],kind=kind,max_abs_error=error))
    print(json.dumps(dict(passed=True,records=records),indent=2))


if __name__=='__main__':selftest()
