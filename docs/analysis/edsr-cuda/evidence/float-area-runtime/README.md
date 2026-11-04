# 浮点 Area 运行时采用与回归证据

2026-11-04，lab RTX 4090。见 [采用报告](../../FLOAT-AREA-RUNTIME.zh-CN.md)。

| 文件 | 内容 |
| --- | --- |
| [quality.json](quality.json) | 72 帧实际 NVDEC 输入/输出等价、新旧缩放 PSNR、同输入同后处理 FP32/AMP 质量门槛及生命周期检查 |
| [latency-first.json](latency-first.json)、[latency-repeat.json](latency-repeat.json) | 新浮点 Area，两轮各 1,000 帧，30 fps，完整 AU 至消费者完成/映射释放 |
| [latency-legacy-first.json](latency-legacy-first.json)、[latency-legacy-repeat.json](latency-legacy-repeat.json) | 修改前原始版本的同机同参数对照，两轮各 1,000 帧 |
| [MANIFEST.json](MANIFEST.json) | 结果、原始日志、修改前及修改后源码的路径、大小与 SHA256 |

JSON 为原始文件的完整副本，没有删除逐帧计时。质量参考使用新的浮点 Area 协议；模型与引擎身份不变。数据不代表独立 test 或真实屏幕延迟，不与此前经过 CPU 上传的离屏渲染实验混用。

本地完整目录为项目下 `benchmarks/super_resolution/experiments/float-area-runtime-20261104-v1/`，包含 `before`、`patches` 及日志。该原始工件目录受忽略规则管理；本目录保留可随项目收录的全部核心证据。
