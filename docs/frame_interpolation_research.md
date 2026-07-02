# 视频插帧 / 帧率转换技术调研

## 状态

- **性质**:内部技术调研笔记,用于评估是否在 iShareScreen 客户端引入插帧(frame interpolation / frame-rate conversion, FRC)以提升远程桌面流畅度。
- **日期**:2026-07-01
- **方法**:多源联网检索 + 对抗式核查(抓取 24 个来源、抽取 101 条论断、验证 25 条、确认 24 条、否决 1 条)。
- **置信度标注**:文中每节末尾标注证据强度。凡属**推断**或**证据弱**的,均显式标出——不要当成已验证结论直接拍板。

## 摘要

实时插帧分三大阵营:

1. **纯经典**(运动估计 + 运动补偿,即 MEMC 那一套):**AMD AMF FRC**、**AMD AFMF**、电视 MEMC 芯片。
2. **纯机器学习**(神经网络端到端):**RIFE / FILM / DAIN** 等模型,以及 **Intel VPL 的 AI 帧插值**。
3. **游戏引擎内 / 需引擎数据**:**AMD FSR3 Frame Generation** —— 强制要求应用提供 motion vector + depth,**无法只靠解码视频运行**。

对"只有解码画面、没有引擎 buffer"的远程桌面/屏幕共享场景:

- **经典光流类(AMF FRC / AFMF)天然适配**:只吃相邻两解码帧,不需要 motion vector / depth / UI mask。这就是最初推荐 AMF FRC 的依据,调研**证实了这个方向判断**。
- **ML 类(RIFE / Intel VPL AI)也只吃两帧**,质量通常更高(尤其大运动),但算力、延迟、工程集成成本更高。
- **除 FSR FG 外,几乎所有方案本质都是"插值"**:需要下一帧 N+1,因此都引入约一帧延迟(@60fps ≈ +16.7ms)。这对交互式远程桌面是绕不开的成本。
- **最大未知:屏幕内容(锐利文字、突现/消失的 UI、移动光标、大片静止区)上的 artifact 表现,缺一手实测证据,属推断,必须实机验证。**

---

## 1. AMD AMF FRC(Advanced Media Framework — Frame Rate Conversion)

- **阵营**:**纯经典**。内部用**分层运动搜索(hierarchical motion search)+ 运动补偿插值**生成中间帧,**不含神经网络**。
- **输入需求**:只要**相邻两解码帧**(以 D3D11/12 texture 包装的 `AMFSurface`),**不需要** motion vector / depth / UI mask。
- **插值 vs 外插**:**插值**——需要 N+1 帧 → 引入约一帧延迟。
- **远程桌面契合度**:**最高**。输入形态就是"两张解码画面",是 media pipeline API 而非游戏渲染 API。
- **集成现实(iShareScreen 特定)**:AMF 要 D3D11/12 surface,而本项目客户端是 Python + wgpu,wgpu 独占 D3D12 device 且不易暴露原生句柄 → 需 C/C++ 扩展做 D3D↔wgpu 互操作,是这条路线的主要工程成本。仅 Windows + AMD。
- **证据强度**:高(一手来源:[AMF FRC API 文档](https://github.com/GPUOpen-LibrariesAndSDKs/AMF/blob/master/amf/doc/AMF_FRC_API.md)、[AMF 仓库](https://github.com/GPUOpen-LibrariesAndSDKs/AMF)、[GPUOpen AMF](https://gpuopen.com/advanced-media-framework/))。

## 2. AMD AFMF(Fluid Motion Frames)

- **阵营**:**纯经典**——驱动层、引擎外的纯光流插值,做法类似电视 MEMC。
- **输入需求**:不需要引擎数据,驱动直接对 present 的画面做插帧。
- **插值 vs 外插**:插值。
- **产品化限制**:它是**驱动控制面功能**(用户在 Radeon 驱动里开),**不是能稳定嵌入你产品的应用级 SDK**。不适合作为 iShareScreen 的可靠依赖。
- **时效性警告**:有报道 **AFMF3 可能加入 ML 增强**,故"纯光流"定性仅针对当前 AFMF 2.x 代;后续版本可能开始利用某种运动/内容分析而非纯双帧光流。
- **证据强度**:中-高(一手:[AMD AFMF 官方页](https://www.amd.com/en/products/software/adrenalin/afmf.html);演进方向来自二手:[igorslab AFMF3](https://www.igorslab.de/en/afmf-3-amds-next-frame-generation-attack-on-nvidias-dlss-with-fsr-redstone-in-the-bag/)、[XDA](https://www.xda-developers.com/amd-fluid-motion-frames/))。

## 3. AMD FSR3 / FidelityFX Frame Generation

- **阵营**:**游戏引擎内,需引擎数据**。
- **输入需求**:**强制要求应用提供渲染分辨率的 motion vector 与 depth**;通过 swapchain proxy 处理插帧负载与 frame pacing。
- **远程桌面契合度**:**直接排除**——远程桌面拿不到 MV/depth。
- **重要更正**:调研中"FSR FG 是 ML + 经典 MV reprojection 混合合成中间帧"这条论断被 **0-3 否决**。因此本报告**仅确认它需要 MV+depth**,**不对其内部是否为 ML 下结论**。
- **证据强度**:中(需 MV+depth 已证实:[数字趋势 FSR3 解析](https://www.digitaltrends.com/computing/amd-fsr-3-explained/)、[GPUOpen FSR FrameGen](https://gpuopen.com/amd-fsr-framegeneration/);内部机制未确认)。

## 4. Intel VPL / oneVPL 帧插值(对早期假设的修正)

- **阵营**:**纯 ML** —— Intel 新的帧插值走了 **AI 神经网络**路线,接口 `mfxExtVPPAIFrameInterpolation`(不是经典 VPP FRC 光流)。
- **输入需求**:相邻两帧,不需要引擎数据。
- **插值 vs 外插**:插值。
- **状态警告**:在 VPL 2.12.0 仍标 **experimental API**,可用性/稳定性要打问号。
- **注意**:这修正了"Intel = 经典 VPP FRC"的早期假设。如果想跨厂商且质量优先,Intel 端其实和 RIFE 一样属 ML 阵营。
- **证据强度**:中-高(一手:[Intel VPL AI 插帧博客](https://community.intel.com/t5/Blogs/Tech-Innovation/Tools/Intel-VPL-Unveils-AI-Powered-Video-Frame-Interpolation/post/1620366)、[VPL VPP structs 文档](https://intel.github.io/libvpl/latest/API_ref/VPL_structs_vpp.html))。

## 5. ML 插帧模型(RIFE / FILM / DAIN)

- **阵营**:**纯神经网络**。
- **机制**:CNN 端到端估计**中间光流**(如 RIFE 的 IFNet),再 **warp(扭曲对齐)+ 融合(fusion)** 生成中间帧。
- **输入需求**:只要相邻两帧,不需要引擎数据。
- **插值 vs 外插**:插值 → 一帧延迟 + 推理耗时。
- **算力/速度**:代价大;**RIFE 专为实时设计,比 DAIN 快 4–27×**(此倍数为作者自报基准,非独立验证)。质量通常高于经典光流,尤其大运动。
- **DAIN/SVFI 说明**:DAIN 仅作为 RIFE 的速度对照点被证实(比 RIFE 慢);其"光流 + depth-aware + 上下文"内部细节,以及 SVFI(基于 RIFE 等模型的批处理工具)**未获独立一手验证**。
- **证据强度**:高(一手:[RIFE arXiv](https://arxiv.org/abs/2011.06294)、[Practical-RIFE](https://github.com/hzwer/Practical-RIFE)、[Google FILM](https://research.google/blog/large-motion-frame-interpolation/)、[FILM arXiv](https://arxiv.org/abs/2202.04901))。

## 6. 电视/显示器 MEMC(经典对照)

- **阵营**:**纯经典**——运动估计(Motion Estimation)+ 运动补偿(Motion Compensation)。AFMF / AMF FRC 本质是同一套思路。
- **对我们的借鉴**:电视 MEMC 靠**检测文字/字幕/OSD 区域并局部关闭插帧**来压制 artifact——这套"内容自适应门控"策略对远程桌面(大量文本)有直接借鉴意义。
- **证据强度**:**低**。本轮存活的论断中**没有直接引用显示器 MEMC 规格的一手来源**,该点主要经由 AFMF/AMF FRC 的"类似电视做法"与经典运动补偿文献间接支撑。**当作行业背景,而非本报告新验证的结论。**
- 相关来源(二手/间接):[mpv MEMC 说明](https://deepwiki.com/hooke007/mpv_PlayKit/6.2-frame-interpolation-(memc))、[XGIMI 插帧 artifact 科普](https://us.xgimi.com/blogs/projectors-101/why-does-motion-interpolation-create-artifacts-during-fast-action-scenes)。

---

## 7. 横向对比

| 方案 | 阵营 | 需引擎 MV/depth | 插值/外插 | 只靠两帧 | 远程桌面契合 |
|---|---|---|---|---|---|
| AMD AMF FRC | 纯经典 | 否 | 插值(+1 帧) | 是 | ✅ 最高 |
| AMD AFMF | 纯经典(驱动层) | 否 | 插值 | 是 | ⚠️ 原理合适但只是驱动开关,非 SDK |
| AMD FSR3 FG | 引擎内 | **是** | 插值 | 否 | ❌ 排除 |
| Intel VPL AI 插帧 | 纯 ML(experimental) | 否 | 插值 | 是 | ✅ 输入合适,成熟度存疑 |
| RIFE / FILM / DAIN | 纯 ML | 否 | 插值 | 是 | ✅ 输入合适,算力/集成重 |
| 电视 MEMC | 纯经典 | 否 | 插值 | 是 | (对照,不可直接集成) |

### 经典光流 vs ML,针对远程桌面后处理

| 维度 | 经典光流(AMF FRC / AFMF) | ML(RIFE / Intel VPL AI) |
|---|---|---|
| 输入契合(只要两帧) | ✅ 已证实 | ✅ 已证实 |
| 画质(尤其大运动) | 中 | 高 |
| 算力 | 低-中 | 高 |
| 延迟 | 一帧 | 一帧 + 推理耗时 |
| 集成难度 | 低-中(AMF 需 D3D↔wgpu interop) | 高(模型/运行时/interop) |
| 成熟度 | AMF FRC 稳定;AFMF 是驱动开关 | RIFE 成熟需自集成;Intel AI 仍 experimental |
| 平台覆盖 | AMF=Win/AMD;AFMF=Win/AMD 驱动 | RIFE 跨平台;Intel AI=Intel |

## 8. 关键共性:延迟

**除 FSR FG(被归为外插 / 无额外延迟框架)外,上述方案本质都是"插值",都要下一帧 N+1,因此都引入约一帧延迟(@60fps ≈ +16.7ms)。** 无论走经典还是 ML,这个成本都存在。对交互式远程桌面,这是需要实测能否接受的核心指标。

外插(extrapolation)可不压帧,但会在遮挡边缘(disocclusion)产生 artifact,是另一种取舍。

## 9. 结论(对 iShareScreen)

1. **方向判断被证实**:若要做,经典光流类的 **AMD AMF FRC** 最贴合(纯经典、两帧输入、无需引擎 buffer、成熟);但在本项目要付 D3D↔wgpu 原生互操作的成本,且仅 Win/AMD。
2. **一个修正**:Intel 端不是经典 VPP FRC,而是 **AI 神经网络插帧(experimental)**。跨厂商且质量优先时,ML 路线(RIFE 或 Intel AI)更统一,但集成成本高。
3. **无论哪条路都吃一帧延迟**,且**屏幕内容 artifact 是最大未知,必须用真实远程桌面画面实测**。
4. **推荐先做 wgpu-compute 跨厂商原型**(见 [frame_interpolation_prototype_plan.md](frame_interpolation_prototype_plan.md)):在真实桌面流上量 artifact + 延迟,再决定是否为 AMD 投入 AMF 原生集成。

## 10. 待解问题(需后续一手验证)

- 电视/显示器 MEMC 芯片(MediaTek、Pixelworks 等)处理文字/字幕/光标等屏幕内容时的 artifact 抑制策略(文字/OSD 区域检测并禁用插帧)公开技术资料 —— 对远程桌面有直接借鉴。
- RIFE 与 DAIN/其他 ML 模型在远程桌面低延迟场景的端到端延迟(编解码 + 插帧)对比。
- 真实远程桌面内容(大片静止 + 文本 + 移动光标)上,经典光流类与 AI 类的实测 artifact 类型与画质差异 —— 目前无一手基准。
- AFMF3 引入 ML 增强后,"驱动层纯光流"定性是否改变。

## 11. 主要来源

一手来源:
- [AMF FRC API 文档](https://github.com/GPUOpen-LibrariesAndSDKs/AMF/blob/master/amf/doc/AMF_FRC_API.md)
- [AMD AFMF 官方页](https://www.amd.com/en/products/software/adrenalin/afmf.html)
- [GPUOpen — Advanced Media Framework](https://gpuopen.com/advanced-media-framework/) / [AMF 仓库](https://github.com/GPUOpen-LibrariesAndSDKs/AMF)
- [GPUOpen — FSR Frame Generation](https://gpuopen.com/amd-fsr-framegeneration/)
- [Intel VPL — AI-Powered Video Frame Interpolation](https://community.intel.com/t5/Blogs/Tech-Innovation/Tools/Intel-VPL-Unveils-AI-Powered-Video-Frame-Interpolation/post/1620366) / [VPL VPP structs](https://intel.github.io/libvpl/latest/API_ref/VPL_structs_vpp.html)
- [RIFE (arXiv 2011.06294)](https://arxiv.org/abs/2011.06294) / [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
- [Google FILM](https://research.google/blog/large-motion-frame-interpolation/) / [FILM (arXiv 2202.04901)](https://arxiv.org/abs/2202.04901)

二手/科普:
- [数字趋势 — FSR 3 explained](https://www.digitaltrends.com/computing/amd-fsr-3-explained/)
- [igorslab — AFMF3](https://www.igorslab.de/en/afmf-3-amds-next-frame-generation-attack-on-nvidias-dlss-with-fsr-redstone-in-the-bag/)
- [XDA — AMD Fluid Motion Frames](https://www.xda-developers.com/amd-fluid-motion-frames/)
- [Blur Busters — interpolation/extrapolation/reprojection](https://blurbusters.com/frame-generation-essentials-interpolation-extrapolation-and-reprojection/)
- [mpv PlayKit — MEMC](https://deepwiki.com/hooke007/mpv_PlayKit/6.2-frame-interpolation-(memc))
- [RIFE vs FILM 对比](https://apatero.com/blog/rife-vs-film-video-frame-interpolation-comparison-2025)
