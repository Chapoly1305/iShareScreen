# 1080p 缩放实验数据索引

实验日期：2026-11-04。完整定义、结果解释与复现命令见 [实验报告](../../RESIZE.zh-CN.md)。

| 文件 | 保存内容 |
| --- | --- |
| [comparison.json](comparison.json) | 72 帧 × 5 方法的 360 条质量记录、OCR 计数、10,000 个普通提交计时样本、配置与源文件哈希 |
| [graph-timing.json](graph-timing.json) | 5 方法 × 3 轮的 15,000 个 CUDA Graph 计时样本及配置 |
| [comparison.png](comparison.png) | 平均质量变化与独立缩放耗时图 |
| [text-crops.png](text-crops.png) | 固定首个采样视频的文字裁剪对照 |
| [MANIFEST.json](MANIFEST.json) | 上述文件、实验脚本和本地原始结果的大小与 SHA256 |

完整 OCR 原文、48 张原始裁剪与原始 JSON 保存在项目本地
`benchmarks/super_resolution/experiments/resize-comparison-20261104-v1/`。
该目录属于忽略的大体积实验工件，不随普通 Git 提交收录；本目录保留可独立重新计算质量汇总和耗时分位数的数据。

本结果使用固定 EDSR 引擎，比较后处理算法，没有替换运行时。
