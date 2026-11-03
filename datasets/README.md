# 数据划分（2026-10-29）

原始文件和历史实验保持原样；以下目录通过符号链接访问数据，不复制大型视频或图像。

| 目录 | 内容 | 用途 |
| --- | --- | --- |
| `train/` 的来源子目录 | 1,637 对：1,037 DocLayNet PDF、400 代码、100 表格、100 终端 | 后续训练 |
| `validation/mac/` | 24 段处理后录屏、4,155 对图像 | 用户指定的主要验证集，整个 Mac 录屏会话归在一起 |
| `validation/views/` | 原来的 36 个代理验证视图和 36 个配置验证视图 | 辅助公共验证；各自用途保存在 `public-manifest.json` |
| `test/views/` | 已有的 72 个独立测试视图 | 保持历史测试身份，用于回归评估 |

每个主目录的 `manifest.json` 保存所属样本和来源。图像路径相对于该 manifest 所在目录，LR/HR 成对保留。主要图像验证清单使用已有审核后的 4,155 对。

训练集按真实来源组织，每个来源目录均有自己的 `manifest.json` 和 `views/`：

| 来源目录 | 样本数 | 来源 |
| --- | ---: | --- |
| `train/doclaynet/` | 1,037 | DocLayNet 官方训练 PDF |
| `train/matter_sdk/` | 390 | connectedhomeip / Matter SDK 源码 |
| `train/isharescreen/` | 10 | 本项目源码 |
| `train/synthetic_table/` | 100 | 旧语料程序生成的数值表格 |
| `train/synthetic_terminal/` | 100 | 旧语料程序生成的终端文字 |

顶层 train 清单汇总全部来源，子目录清单只列该来源；两种清单中的图像路径均相对于各自清单目录。已下载但未入选的其他公开数据不混入这 1,637 对。

用户确认于 2026-10-29 删除了 `cs682_dataset/` 中的原始 MOV，不再提供原始录屏入口。H100 的已处理录屏和图像对通过历史 artifacts 兼容路径访问项目外归档；lab 同步副本将这些链接展开成本机数据文件。同源帧和衍生版本不进入新的 train/test。公共数据沿用已有来源组划分，并检查来源组互斥和训练集与 Mac 图像的已有 HR 文件/像素哈希无完全重复；尚未证明 Mac 画面中的文档内容与公共训练语料完全无重合。

Mac 数据过去参与过部分模型训练，因此不是这些历史模型的未见验证集。已有 72 个测试视图也已经评估过，后续针对其结果调整方案时，需要另建未见测试集。

这些目录和清单是新的组织入口，未修改历史冻结数据和训练程序。它们不是原训练器的 packed-array manifest；后续训练需显式读取新清单或生成新的 pack，不能继续沿用包含 Mac 数据的旧训练划分。

重建目录链接：在项目根目录激活 `dev-environment.sh` 后运行 `python datasets/prepare_splits.py`。已有正确链接可以重复使用，不覆盖其他文件。

## CodeSCAN 专项文字素材

原始 12,000 个样本重新按 80% / 10% / 10% 划分；每个样本的全部 7 个文件保持在一起。按跨语言的规范化仓库名分组，任何相同文件 SHA-256、解码后图片像素、规范化完整源码或整目录哈希都会合并分组。因此不是随机拆分图片。

| 目录 | 样本数 | 比例 |
| --- | ---: | ---: |
| `train/codescan/samples/` | 9,600 | 80% |
| `validation/codescan/samples/` | 1,200 | 10% |
| `test/codescan/samples/` | 1,200 | 10% |

每个来源目录的 `manifest.json` 列出它独占的样本入口；总索引为 `codescan-splits.json`。12,000 张彩色截图没有完全相同的解码像素。跨集合的样本路径、实际来源目录、仓库、任意样本文件哈希、图片像素哈希、规范化完整源码和整目录哈希交集均必须为 0；复核时重新读取全部 84,000 个样本文件。重复的文件名如 `img.png` 是角色名，允许出现在不同的唯一目录中；部分相同辅助文件或黑白图片会留在同一集合，不声称每个辅助文件在整个语料中都唯一。

原始 ZIP、完整解压数据及官方划分保存在 `sources/codescan/`，不放入任一训练/验证/测试入口；原先训练目录中的整包链接已移除。完整审计见 `sources/codescan/split-v1/AUDIT.json`，冻结方案见 `PLAN.json`。精确哈希检查不能证明不存在近似片段或语义相似的代码。

这些是原始截图，`bw.png` 是黑白版本，不能当作低清/高清配对。旧的顶层 1,637 对训练清单保持原样，CodeSCAN 使用单独的来源清单；尚未生成新的 720p / 600 Kbps / 4:2:0 训练对，2026-10-30 用户已解除训练暂停；新 codec 配对和训练记录独立保存在 benchmarks/super_resolution/experiments/codescan-720p-600k-420-v1/。Mac 录屏继续保留为主要 validation。冻结清单中的暂停状态是冻结时记录；当前授权以 benchmarks/super_resolution/training-hold.json 为准。

复现：激活项目环境后，依次运行 `python datasets/split_codescan.py inventory`、`split`、`verify`；冻结目录已存在时只运行 `verify`，脚本拒绝覆盖已有划分。

## DVR 录屏节选（2026-11-01）

素材筛选记录位于 `sources/dvr-intake-2026-11-01/`。首批从 8 个较高码率来源选出 29 段，每段 20 秒，覆盖代码、终端、中英文网页、PDF、幻灯片、图表和窗口变化。

独立片段放在 `sources/dvr-clips-v1/`，打开其中的 `index.html` 可按场景筛选并播放 MP4 预览。`references/` 是供后续生成输入和评估使用的无损 FFV1 文件，保留源视频的解码像素、尺寸和帧率；`previews/` 仅供浏览。来源本身已压缩，参考版不会增加损失，也不能消除原有损失。`manifest.json` 和 `clips.csv` 记录源文件哈希、精确帧范围与片段哈希，`AUDIT.json` 和 `BOUNDARY-AUDIT.json` 保存逐帧像素及独立时间定位核验。

这批仅完成素材节选，尚未加入 train/validation/test，也没有为此启动训练。后续应按录像会话和文档/项目内容分组，再检查与现有验证、测试内容的重合；同一原视频的多个片段不能随机分到不同集合。原始 `DVR/`、CodeSCAN 冻结划分和当前 Mac validation 保留。

新增 `2026-10-31_12-07-43.mov` 的检查记录与 6 段候选浏览预览独立保存在 `sources/dvr-intake-2026-11-01-addendum/`，打开 `index.html` 浏览，详见 `REVIEW.md`。候选清单记录原文件哈希和帧范围，供后续生成参考片段使用；当前未分配数据集。
