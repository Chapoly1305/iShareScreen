# H100 与 lab 项目同步（2026-10-29）

H100 `/home/jchen73/iShareScreen` 为当前代码和数据划分的权威来源，包含未提交修改。
lab 的工作副本为 `chen@lab:/home/chen/iShareScreen`，GPU 为 RTX 4090，UUID
`GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a`。

## 归档位置

- H100 旧实验：`/home/jchen73/iShareScreen-archives/before-retrain-2026-10-29_192818/`。
- lab 覆盖前的旧项目：`/home/chen/iShareScreen-archives/before-h100-sync-20261029-193649/`。
- H100 `benchmarks/super_resolution/artifacts` 是访问外部旧归档的兼容链接，不能用于新输出。

归档为同一磁盘上的迁移，不是异机备份。历史远端实验副本没有清除。

## 当前副本

代码（含 Git 元数据和未提交文件）、train/validation/test、官方模型缓存、LPIPS 权重和
OCR 字典已同步。lab 的数据链接已展开为本机文件，运行时不依赖 H100 归档路径。
原始 MOV 已由用户删除，保留的 24 段完整 RGB8 参考视频和 4,155 对图像仍可使用。

没有复制 `.venv`、旧训练权重、历史归档、临时迁移文件或机器私有设置。
lab `.venv` 使用本机 Python 3.12，单独安装 torch 2.6.0+cu124 / torchvision 0.21.0+cu124。
`source dev-environment.sh` 自动识别本机项目目录。每轮实验另存依赖清单和模型身份。

## 本轮评测

`benchmarks/super_resolution/experiments/validation-baselines-2026-10-29/` 保存冻结协议、
编码记录和评测结果。H100 与 lab CPU 分担互斥的视频编码任务；GPU 推理只用 lab 的 4090。
7 个官方 ×2 模型和无模型对照，4 个分辨率、2 个码率、2 个色度采样，共 16 个组合。
所有视频完整重编码；每段固定 3 个匹配时间戳作模型画质评估。

后续同步代码时应排除运行中的 experiments、lab `.venv` 和本机缓存，避免覆盖新结果。
数据以各 split 的 `manifest.json` 为入口，不能把历史 artifacts 的旧训练划分当作当前划分。
