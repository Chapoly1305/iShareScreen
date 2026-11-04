"""Build a train-calibrated, channel-balanced INT8 EDSR with FP16 residuals."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import tensorrt as trt

from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx",type=Path,required=True);p.add_argument("--calibration",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--alpha",type=float,default=.75)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);dest=a.output/"BUILD.json"
    if dest.exists():raise FileExistsError(dest)
    cal=json.loads(a.calibration.read_text())
    assert cal["complete"] and cal["calibration"]["split"]=="train" and cal["protect_terminal_block"]
    assert cal["modes"]["int8_alpha_"+str(a.alpha)]["passed"]
    result=dict(complete=False,started=time.time(),gpu=guard(),trt=trt.__version__,scope=__doc__,
        source_onnx_sha256=digest(a.onnx),calibration_sha256=digest(a.calibration),
        script_sha256=digest(Path(__file__)),alpha=a.alpha,quantized_layers=[])
    save(dest,result)
    try:
        model=onnx.load(a.onnx)
        weights={w.name:numpy_helper.to_array(w) for w in model.graph.initializer}
        maxima=cal["calibration"]["maxima"];nodes=[]
        for node in model.graph.node:
            name=node.input[1].removeprefix("model.").removesuffix(".weight") if node.op_type=="Conv" else ""
            if name in maxima and not name.startswith("body.15."):
                weight=weights[node.input[1]].astype(np.float32)
                activation=np.asarray(maxima[name],dtype=np.float32)
                magnitude=np.maximum(np.abs(weight).max(axis=(0,2,3)),1e-12)
                balance=np.maximum(activation,1e-12)**a.alpha/magnitude**(1-a.alpha)
                balanced_weight=weight*balance.reshape(1,64,1,1)
                # Quantizer scales and channel factors stay FP32. All graph
                # convolution operands still use the original FP16 contract.
                scales=[np.array(np.max(activation/balance)/127,dtype=np.float32),
                        np.max(np.abs(balanced_weight),axis=(1,2,3))/127]
                prefix="i8_"+name
                model.graph.initializer.extend([
                    numpy_helper.from_array((1/balance).reshape(1,64,1,1),prefix+"_inverse"),
                    numpy_helper.from_array(balanced_weight,prefix+"_balanced_weight")])
                nodes.extend([
                    helper.make_node("Cast",[node.input[0]],[prefix+"_float"],to=TensorProto.FLOAT,name=prefix+"_input_float"),
                    helper.make_node("Mul",[prefix+"_float",prefix+"_inverse"],[prefix+"_balanced_input"],name=prefix+"_balance")])
                for i,operand in enumerate([prefix+"_balanced_input",prefix+"_balanced_weight"]):
                    tag=prefix+("_activation" if i==0 else "_weight")
                    scale=np.asarray(scales[i],dtype=np.float32);assert np.all(scale>0)
                    model.graph.initializer.extend([numpy_helper.from_array(scale,tag+"_scale"),
                        numpy_helper.from_array(np.zeros(scale.shape,dtype=np.int8),tag+"_zero")])
                    attrs={} if i==0 else dict(axis=0)
                    nodes.extend([
                        helper.make_node("QuantizeLinear",[operand,tag+"_scale",tag+"_zero"],[tag+"_q"],name=tag+"_Q",**attrs),
                        helper.make_node("DequantizeLinear",[tag+"_q",tag+"_scale",tag+"_zero"],[tag+"_dq"],name=tag+"_DQ",**attrs),
                        helper.make_node("Cast",[tag+"_dq"],[tag+"_half"],name=tag+"_output_half",to=TensorProto.FLOAT16)])
                    node.input[i]=tag+"_half"
                result["quantized_layers"].append(name)
            nodes.append(node)
        assert len(result["quantized_layers"])==30
        del model.graph.node[:];model.graph.node.extend(nodes)
        onnx.checker.check_model(model);path=a.output/"edsr.onnx";onnx.save(model,path)
        result["onnx_sha256"]=digest(path);save(dest,result)
        logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger)
        net=builder.create_network(0);parser=trt.OnnxParser(net,logger)
        if not parser.parse(path.read_bytes()):raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
        net.get_input(0).shape=(1,3,720,1280)
        config=builder.create_builder_config();config.clear_flag(trt.BuilderFlag.TF32)
        config.builder_optimization_level=5;config.avg_timing_iterations=8
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,4<<30)
        config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        t0=time.time();blob=builder.build_serialized_network(net,config)
        if blob is None:raise RuntimeError("Balanced INT8 engine build failed")
        path=a.output/"edsr.engine";path.write_bytes(bytes(blob))
        result.update(complete=True,finished=time.time(),build_s=time.time()-t0,engine_sha256=digest(path))
        save(dest,result);print(json.dumps(result,indent=2),flush=True)
    except BaseException as exc:
        result.update(error=repr(exc),failed=time.time());save(dest,result);raise


if __name__=="__main__":main()
