"""Fuse EDSR residual convolutions and quantizers as in-graph AOT operators."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import tensorrt as trt

from benchmarks.super_resolution import edsr_int8_plugin
from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save


def quantize_terminal_block(model,calibration):
    """Extend the same train-only quantization recipe to residual block 15."""
    assert calibration['complete'] and calibration['calibration']['split']=='train'
    assert calibration['checkpoint']['sha256']=='2c923e9ba9541cb85006684c0072868b2d71354d665cdd85cd2b0baac6faaabb'
    data={w.name:numpy_helper.to_array(w).copy() for w in model.graph.initializer}
    maxima=calibration['calibration']['maxima'];nodes=[];changed=[]
    for original in model.graph.node:
        node=copy.deepcopy(original)
        name=node.input[1].removeprefix('model.').removesuffix('.weight') if node.op_type=='Conv' else ''
        if name in ('body.15.body.0','body.15.body.2'):
            weight=data[node.input[1]].astype(np.float32);activation=np.asarray(maxima[name],dtype=np.float32)
            magnitude=np.maximum(np.abs(weight).max(axis=(0,2,3)),1e-12)
            balance=np.maximum(activation,1e-12)**.75/magnitude**.25
            balanced_weight=weight*balance.reshape(1,64,1,1)
            prefix='i8_'+name
            model.graph.initializer.extend([
                numpy_helper.from_array((1/balance).reshape(1,64,1,1),prefix+'_inverse'),
                numpy_helper.from_array(balanced_weight,prefix+'_balanced_weight')])
            scales=[np.asarray(np.max(activation/balance)/127,dtype=np.float32),
                    np.max(np.abs(balanced_weight),axis=(1,2,3))/127]
            nodes.extend([
                helper.make_node('Cast',[node.input[0]],[prefix+'_float'],to=TensorProto.FLOAT,name=prefix+'_input_float'),
                helper.make_node('Mul',[prefix+'_float',prefix+'_inverse'],[prefix+'_balanced_input'],name=prefix+'_balance')])
            for index,operand in enumerate((prefix+'_balanced_input',prefix+'_balanced_weight')):
                tag=prefix+('_activation' if index==0 else '_weight');scale=np.asarray(scales[index],dtype=np.float32)
                assert np.all(scale>0)
                model.graph.initializer.extend([numpy_helper.from_array(scale,tag+'_scale'),
                    numpy_helper.from_array(np.zeros(scale.shape,dtype=np.int8),tag+'_zero')])
                attrs={} if index==0 else dict(axis=0)
                nodes.extend([
                    helper.make_node('QuantizeLinear',[operand,tag+'_scale',tag+'_zero'],[tag+'_q'],name=tag+'_Q',**attrs),
                    helper.make_node('DequantizeLinear',[tag+'_q',tag+'_scale',tag+'_zero'],[tag+'_dq'],name=tag+'_DQ',**attrs),
                    helper.make_node('Cast',[tag+'_dq'],[tag+'_half'],to=TensorProto.FLOAT16,name=tag+'_output_half')])
                node.input[index]=tag+'_half'
            changed.append(name)
        nodes.append(node)
    assert changed==['body.15.body.0','body.15.body.2']
    del model.graph.node[:];model.graph.node.extend(nodes)
    return changed


def transform(model,shape=(720,1280),block_count=15):
    data = {w.name:numpy_helper.to_array(w).copy() for w in model.graph.initializer}
    for i in range(block_count-1):
        constant = next(n for n in model.graph.node if n.name == f"/model/body/body.{i}/Constant")
        assert np.all(numpy_helper.to_array(next(a.t for a in constant.attribute if a.name == "value")) == 1)
    nodes = []; replaced = []
    for old in model.graph.node:
        node = copy.deepcopy(old)
        # The new plugin already produces the following block's quantized input.
        if any(node.name in (f"i8_body.{i}.body.0_input_float", f"i8_body.{i}.body.0_balance",
                             f"i8_body.{i}.body.0_activation_Q") for i in range(1,block_count)):
            continue
        indices = [i for i in range(block_count-1) if node.name == f"/model/body/body.{i}/Add"]
        if not indices:
            nodes.append(node); continue
        i = indices[0]; prefix = f"i8_body.{i}.body.2"; nxt = f"i8_body.{i+1}.body.0"
        scale = data[prefix + "_weight_scale"]
        balanced = data[prefix + "_balanced_weight"]
        q = np.clip(np.rint(balanced / scale[:,None,None,None]), -127, 127).astype(np.int8)
        arrays = dict(weight=np.ascontiguousarray(q.transpose(2,3,1,0).reshape(576,64)),
                      alpha=(scale * data[prefix + "_activation_scale"]).astype(np.float32),
                      bias=data[f"model.body.{i}.body.2.bias"].astype(np.float32),
                      factor=(data[nxt + "_inverse"].reshape(64) / data[nxt + "_activation_scale"]).astype(np.float32))
        names = {k:f"dual_{i}_{k}" for k in arrays}
        for k,v in arrays.items(): model.graph.initializer.append(numpy_helper.from_array(v,names[k]))
        skip = "/model/head/head.0/Conv_output_0" if i == 0 else f"/model/body/body.{i-1}/Add_output_0"
        nodes.append(helper.make_node("edsr_dual", [prefix + "_activation_q",names["weight"],skip,
            names["alpha"],names["bias"],names["factor"]],
            [node.output[0],nxt + "_activation_q"], name=f"dual_conv_block_{i}", domain="iss",plugin_namespace="iss",
            height=shape[0],width=shape[1]))
        replaced.append(i)
    assert replaced == list(range(block_count-1))
    # Prune replaced conv2/DQ/bias nodes and their initializers. Preserve every
    # remaining original node; the terminal quantized block is fused separately.
    needed = {"output"}; kept = []
    for node in reversed(nodes):
        if needed.intersection(node.output):
            kept.append(node); needed.update(node.input)
    kept.reverse()
    initializers = [copy.deepcopy(w) for w in model.graph.initializer if w.name in needed]
    del model.graph.node[:]; model.graph.node.extend(kept)
    del model.graph.initializer[:]; model.graph.initializer.extend(initializers)
    model.opset_import.append(helper.make_opsetid("iss",1))
    onnx.checker.check_model(model)
    return replaced


def fuse_entry(model,shape=(720,1280)):
    from benchmarks.super_resolution import edsr_entry_plugin
    data={w.name:numpy_helper.to_array(w).copy() for w in model.graph.initializer}
    prefix="i8_body.0.body.0"
    factor=(data[prefix+"_inverse"].reshape(64)/data[prefix+"_activation_scale"]).astype(np.float32)
    model.graph.initializer.append(numpy_helper.from_array(factor,"entry_factor"))
    nodes=[];changed=[]
    for original in model.graph.node:
        n=copy.deepcopy(original)
        if n.name==prefix+"_activation_Q":
            n=helper.make_node("edsr_entry",["/model/head/head.0/Conv_output_0","entry_factor"],
                ["entry_skip",n.output[0]],name="fused_head_entry",domain="iss",plugin_namespace="iss",
                height=shape[0],width=shape[1])
            changed.append("quantizer")
        if n.name=="dual_conv_block_0":
            n.input[2]="entry_skip";changed.append("skip")
        nodes.append(n)
    assert changed==["quantizer","skip"]
    needed={"output"};kept=[]
    for node in reversed(nodes):
        if needed.intersection(node.output):kept.append(node);needed.update(node.input)
    kept.reverse()
    initializers=[copy.deepcopy(w) for w in model.graph.initializer if w.name in needed]
    del model.graph.node[:];model.graph.node.extend(kept)
    del model.graph.initializer[:];model.graph.initializer.extend(initializers)
    onnx.checker.check_model(model)
    return edsr_entry_plugin


def fold_conv1_output_basis(model,protect_native_block=False,block_count=15):
    """Expose native INT8 conv1->ReLU->Q fusion across a plugin boundary.

    Fold the positive per-channel input balance of conv2 into conv1's output
    DQ scales and bias. Keep conv1's logical dequantized operands in FP32 so
    Cast-to-FP16 nodes do not interrupt TensorRT's quantization pattern.
    INT8 weight codes/scales at quantization and all residual skips are kept.
    This reassociates rounding and needs the unchanged image-quality gate.
    """
    data={w.name:numpy_helper.to_array(w).copy() for w in model.graph.initializer}
    changes=[]
    count=block_count-1 if protect_native_block else block_count
    for i in range(count):
        first=f"i8_body.{i}.body.0";second=f"i8_body.{i}.body.2"
        inv=data[second+"_inverse"].reshape(64)
        scale_name=first+"_output_balanced_scale";bias_name=first+"_output_balanced_bias"
        model.graph.initializer.extend([
            numpy_helper.from_array((data[first+"_weight_scale"]*inv).astype(np.float32),scale_name),
            numpy_helper.from_array((data[f"model.body.{i}.body.0.bias"].astype(np.float32)*inv).astype(np.float32),bias_name)])
        for node in model.graph.node:
            if node.name==first+"_weight_DQ":node.input[1]=scale_name
            if node.name==f"/model/body/body.{i}/body/body.0/Conv":
                node.input[0]=first+"_activation_dq";node.input[1]=first+"_weight_dq";node.input[2]=bias_name
                changes.append(i)
            if node.name==second+"_activation_Q":
                node.input[0]=f"/model/body/body.{i}/body/body.1/Relu_output_0"
    assert changes==list(range(count))
    return changes


def fuse_last_residual(model,shape=(720,1280),block_count=15):
    from benchmarks.super_resolution import edsr_final_plugin
    data={w.name:numpy_helper.to_array(w).copy() for w in model.graph.initializer}
    index=block_count-1
    prefix=f"i8_body.{index}.body.2";scale=data[prefix+"_weight_scale"]
    q=np.clip(np.rint(data[prefix+"_balanced_weight"]/scale[:,None,None,None]),-127,127).astype(np.int8)
    arrays=dict(weight=np.ascontiguousarray(q.transpose(2,3,1,0).reshape(576,64)),
        alpha=(scale*data[prefix+"_activation_scale"]).astype(np.float32),
        bias=data[f"model.body.{index}.body.2.bias"].astype(np.float32))
    for key,val in arrays.items():model.graph.initializer.append(numpy_helper.from_array(val,"final_"+key))
    nodes=[];changed=0
    for original in model.graph.node:
        n=copy.deepcopy(original)
        if n.name==f"/model/body/body.{index}/Add":
            n=helper.make_node("edsr_final",[prefix+"_activation_q","final_weight",
                f"/model/body/body.{index-1}/Add_output_0","final_alpha","final_bias"],list(n.output),
                name="fused_final_residual",domain="iss",plugin_namespace="iss",height=shape[0],width=shape[1]);changed+=1
        nodes.append(n)
    assert changed==1
    needed={"output"};kept=[]
    for node in reversed(nodes):
        if needed.intersection(node.output):kept.append(node);needed.update(node.input)
    kept.reverse();initializers=[copy.deepcopy(w) for w in model.graph.initializer if w.name in needed]
    del model.graph.node[:];model.graph.node.extend(kept)
    del model.graph.initializer[:];model.graph.initializer.extend(initializers)
    onnx.checker.check_model(model)
    return edsr_final_plugin


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--entry",action="store_true")
    p.add_argument("--conv1-fp32",action="store_true")
    p.add_argument("--protect-native-block",action="store_true")
    p.add_argument("--last-plugin",action="store_true")
    p.add_argument("--activation-margin",type=float,default=1.)
    p.add_argument("--height",type=int,default=720);p.add_argument("--width",type=int,default=1280)
    p.add_argument("--quantize-terminal-block",action="store_true")
    p.add_argument("--calibration",type=Path)
    p.add_argument("--fold-projection-tail",action="store_true")
    p.add_argument("--fold-head-branch",action="store_true")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);dest=a.output/"BUILD.json"
    if dest.exists():raise FileExistsError(dest)
    result=dict(complete=False,started=time.time(),gpu=guard(),scope=__doc__,trt=trt.__version__,
        source_onnx_sha256=digest(a.onnx),script_sha256=digest(Path(__file__)),
        plugin_source_sha256=digest(Path(edsr_int8_plugin.__file__)),
        kernel_source_sha256=digest(Path(__file__).with_name("profile_edsr_int8_epilogue.py")))
    save(dest,result)
    try:
        assert digest(a.onnx)=="122026964213a1b60eeea2a702a00915dea4eef98813e22d63a21f8b1e31068c"
        model=onnx.load(a.onnx)
        assert not a.fold_head_branch or a.fold_projection_tail
        if a.fold_projection_tail:
            from benchmarks.super_resolution import edsr_folded_projection
            result['projection_tail_source_sha256']=digest(Path(edsr_folded_projection.__file__))
            subgraph=a.output/'projection-tail.onnx'
            result['projection_tail']=edsr_folded_projection.replace_tail(model,a.fold_head_branch,subgraph)
            result['projection_tail_onnx_sha256']=digest(subgraph)
        block_count=16 if a.quantize_terminal_block else 15
        result['quantized_residual_blocks']=block_count
        if a.quantize_terminal_block:
            assert a.calibration is not None
            result['calibration_sha256']=digest(a.calibration)
            result['extended_quantized_layers']=quantize_terminal_block(model,json.loads(a.calibration.read_text()))
        shape=(a.height,a.width)
        assert 1<=a.height<=720 and 1<=a.width<=1280
        result["shape"]=list(shape)
        assert a.activation_margin>=1
        result["activation_margin"]=a.activation_margin
        if a.activation_margin!=1:
            count=0
            for item in model.graph.initializer:
                if item.name.endswith("_activation_scale"):
                    value=(numpy_helper.to_array(item)*np.float32(a.activation_margin)).astype(np.float32)
                    item.CopyFrom(numpy_helper.from_array(value,item.name));count+=1
            assert count==2*block_count
        result["conv1_fp32"]=a.conv1_fp32
        result["protect_native_block"]=a.protect_native_block
        if a.conv1_fp32:result["folded_conv1_blocks"]=fold_conv1_output_basis(model,a.protect_native_block,block_count)
        result["replaced_blocks"]=transform(model,shape,block_count)
        result["fused_entry"]=a.entry
        if a.entry:
            entry_plugin=fuse_entry(model,shape)
            result["entry_plugin_sha256"]=digest(Path(entry_plugin.__file__))
        result["last_plugin"]=a.last_plugin
        if a.last_plugin:
            final_plugin=fuse_last_residual(model,shape,block_count)
            result["final_plugin_sha256"]=digest(Path(final_plugin.__file__))
        path=a.output/"edsr.onnx";onnx.save(model,path);result["onnx_sha256"]=digest(path)
        kernel=edsr_int8_plugin.compile_kernel(*shape)
        (a.output/"dual.ptx").write_text(kernel.asm["ptx"])
        result["kernel"]=dict(name=kernel.metadata.name,registers=getattr(kernel,"n_regs",None),spills=getattr(kernel,"n_spills",None),
            shared_bytes=kernel.metadata.shared,ptx_sha256=digest(a.output/"dual.ptx"))
        save(dest,result)
        logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger)
        net=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.PREFER_AOT_PYTHON_PLUGINS))
        parser=trt.OnnxParser(net,logger)
        if not parser.parse(model.SerializeToString()):
            raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
        net.get_input(0).shape=(1,3,*shape)
        config=builder.create_builder_config();config.clear_flag(trt.BuilderFlag.TF32)
        config.builder_optimization_level=5;config.avg_timing_iterations=8
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,4<<30)
        config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        t0=time.time();blob=builder.build_serialized_network(net,config)
        if blob is None:raise RuntimeError("AOT residual plugin build failed")
        path=a.output/"edsr.engine";path.write_bytes(bytes(blob))
        result.update(complete=True,finished=time.time(),build_s=time.time()-t0,engine_sha256=digest(path))
        save(dest,result);print("built AOT plugin",result["build_s"],flush=True)
    except BaseException as exc:
        result.update(error=repr(exc),failed=time.time());save(dest,result);raise


if __name__=="__main__":main()
