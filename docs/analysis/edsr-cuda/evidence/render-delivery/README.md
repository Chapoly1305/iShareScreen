# 渲染交付路径实验数据

实验日期：2026-11-04。见 [审稿式实验报告](../../RENDER-DELIVERY.zh-CN.md)。

| 文件 | 内容 |
| --- | --- |
| [comparison.json](comparison.json) | 72 帧 × 3 路径的 216 条质量记录、13,200 个计时样本、6,000 组绘制 GPU ticks、配置和源文件身份；仅省略 OCR 原文 |
| [comparison.png](comparison.png) | 质量、GPU 转换/绘制及现有 CPU 上传路径耗时图 |
| [text-crops.png](text-crops.png) | 固定首个视频的原生像素文字裁剪 |
| [MANIFEST.json](MANIFEST.json) | 本目录证据及本地原始工件的大小和 SHA256 |

完整原始目录：项目下 `benchmarks/super_resolution/experiments/render-delivery-20261104-v1/`。
其 `full` 保留全部结果与 32 张原始裁剪；`source` 保留执行源码和渲染器快照；
`smoke` 与 `smoke-native-wait` 保留首次失败和修正后通过的单帧检查，`full.log` 保留正式日志。
原始目录受实验工件忽略规则管理；本目录数据可以独立重算汇总及分位数。

直接交付路径使用真实渲染器的 RGBA8 + 双线性采样；最终目标为 1080p。
墙钟耗时包含明确的 CPU 下载/上传接口，不是零拷贝，也不包含模型推理或屏幕呈现。
