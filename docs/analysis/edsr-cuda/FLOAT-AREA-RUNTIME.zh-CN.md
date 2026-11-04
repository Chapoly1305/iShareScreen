# 浮点 Area 已接入可选 CUDA 运行时：2026-11-04

根据缩放与实际渲染对比，将可选 CUDA EDSR 运行时的 1080p 输出改为：**在浮点域做 Area 缩放，最后才舍入、裁剪到 [0,255] 并打包 RGB8/RGBA8。** 原生模型输出仍为 1440p，输出消费者继续接收 1080p。此改动不重建 TensorRT 引擎，也不改变模型精度分配。

修改位置是 [pack_area](../../../src/isharescreen/enhancement/cuda_kernels.py) 及 [运行时调用](../../../src/isharescreen/enhancement/cuda_surface.py)。内核通过编译期参数 `QUANTIZE_FIRST=False` 选择新路径，仍是一个融合内核，不增加额外缩放 pass。内核旧特化默认保留给历史比较脚本；运行时显式选择新路径。已更新像素验证和流水线基准的独立 CPU 参考，避免用旧量化顺序验证新输出。

这属于可选 GPU-surface 组件的实现更新，**并未把它接入桌面窗口，也未新增 CUDA/WGPU 纹理互操作**。前述 CPU 上传路径的约 7.7–13.2 ms 不用于这次全 GPU 流水线的延迟结论。

## 实际 NVDEC 输入：72 帧验证

使用同一冻结引擎、24 个 Mac 开发验证视频、每视频 3 帧。通过原生 NVDEC 回调运行完整 GPU 路径，输入归一化与独立参考一致；新输出的 RGB 与 OpenCV 浮点 Area 参考逐像素一致，alpha 恒为 255。

所有参考使用相同的实际 NVDEC RGB 输入：优化引擎的新输出、同一引擎的旧输出，以及同权重 PyTorch FP32 / AMP。FP32 / AMP 均采用新浮点 Area 后处理，TF32 关闭。

| 方案 | 平均 RGB PSNR，dB |
| --- | ---: |
| 优化引擎 + 旧缩放 | 31.491511 |
| 优化引擎 + 新浮点 Area | **31.916647** |
| 同权重 FP32 + 浮点 Area | 31.922571 |
| 同权重 AMP + 浮点 Area | 31.920832 |

新缩放相对旧缩放：平均 **+0.425136 dB**，64 帧改善、8 帧下降；最差帧 −0.059566 dB，最佳帧 +1.597624 dB。与上一轮保存 RGB 输入的 +0.425711 dB 属于不同输入路径的结果，分别保留。

| 新优化输出的对照 | 平均 ΔPSNR，dB | 最差帧 ΔPSNR，dB | 质量门槛 |
| --- | ---: | ---: | --- |
| FP32，同输入同后处理 | −0.005924 | −0.025861 | 通过 |
| AMP，同输入同后处理 | −0.004186 | −0.023498 | 通过 |

门槛仍为平均损失 ≤0.05 dB、最差帧损失 ≤0.10 dB。没有把后处理带来的 PSNR 增益当作量化无损的证明。这里是开发验证，未新增独立 test 评价或真实阅读实验。

## 正确性、性能及证据

- 72 帧输入/输出独立参考检查全部通过。
- GPU 生命周期检查通过：持有输出、映射复用、缓存淘汰、部分启动失败、取消、错误 context 与延迟消费者等；原始布尔检查项保留在 [quality.json](evidence/float-area-runtime/quality.json)。
- 现有 `tests/test_cuda_surface.py` 的 6 项 CPU 检查通过；修改文件语法与差异空白检查通过。
- 全 GPU 流水线使用原来的完整 HEVC AU 提交 → GPU 消费者完成及映射释放边界，固定 30 fps 到达，分别运行两次各 1,000 个计时样本；原始数据见 [第一次](evidence/float-area-runtime/latency-first.json) 与 [重复](evidence/float-area-runtime/latency-repeat.json)。不包括 UDP、窗口渲染或屏幕显示，也不由不同阶段的 P95 相加得到总延迟。

| 同机同参数流水线 | 第一次 P95，ms | 重复 P95，ms |
| --- | ---: | ---: |
| 新浮点 Area | 10.579296 | 10.421074 |
| 随后补测的原始旧版本 | 10.607647 | 10.362829 |

新版本 P50 分别为 10.309498 / 10.255175 ms。首轮新旧版本都比各自重复轮更慢；新旧 P95 范围重叠，且逐轮高低次序不一致，不能声称稳定加速，也不能将早期 10.352 / 10.370 ms 与本次首轮差异全部归因于缩放。对照顺序是新、新、旧、旧，每次为独立进程，未做随机交错；这是工程回归，不是精确延迟差异的统计显著性试验。

旧版本的 [第一次](evidence/float-area-runtime/latency-legacy-first.json) 与 [重复](evidence/float-area-runtime/latency-legacy-repeat.json) 连同原始样本单独保留。计时 JSON 的 `quality_scope` 沿用早期解码验证背景摘要；本次新执行的质量检查以 `quality.json` 的 72 帧及计时文件的预热像素检查为准，没有重新执行早期 1,200 帧解码器验证。

在 lab / Alienware / RTX 4090（UUID `GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a`）的独立代码副本验证，执行源码哈希与本地改动一致，保留修改前源码。依赖是项目已有的 `.venv/optional/tensorrt11` 和 `.venv/optional/pynvcodec`；首次遗漏后者的依赖检查日志另存，正式结果使用完整依赖路径。

本地完整原始目录：`benchmarks/super_resolution/experiments/float-area-runtime-20261104-v1/`；远端目录：`/home/chen/iShareScreen/research/float-area-runtime-20261104-v1/`。旧实验结果与新实验分别保存，旧的 10.352 / 10.370 ms 记录不被覆盖。

[证据索引与哈希清单](evidence/float-area-runtime/README.md) 记录全部结果和执行源码。两种版本的验证副本保存在远端 `overlay` 与 `legacy_overlay`；GPU 任务结束后复查无 compute 进程。

## 复现

在已授权且无其他 compute 负载的 lab 主机、包含本次修改的项目目录执行。需已有冻结输入、已接受引擎和对应的原生解码库，输出路径必须不存在。

```bash
source dev-environment.sh
export PYTHONPATH="$PWD/.venv/optional/pynvcodec:$PWD/.venv/optional/tensorrt11:$PYTHONPATH"
python -m benchmarks.super_resolution.edsr_cuda.validate \
  --engine research/edsr-packed-state-20261104/engine-byte-cache/edsr.engine \
  --engine-sha256 74f418fb7ad0cd03ae25d6ff701fead1414a6f180e334d13e9281263007dde74 \
  --validation benchmarks/super_resolution/experiments/validation-baselines-2026-10-29 \
  --native-library benchmarks/super_resolution/artifacts/edsr-cuda-cleanup-20261104/nvdec.so \
  --reference-checkpoint benchmarks/super_resolution/experiments/codescan-720p-600k-420-v1/models/edsr/best.pt \
  --expected-gpu-uuid GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a \
  --output research/float-area-repeat/quality.json
```

流水线计时沿用 [运行指南](../../../benchmarks/super_resolution/edsr_cuda/README.md) 中的命令。`--reference-checkpoint` 是新增的可选质量检查参数：提供后强制校验冻结权重 SHA256，并对同输入的 FP32 / AMP 执行质量门槛；不提供时仅做运行时等价与生命周期检查及新旧缩放的 PSNR 对比。
