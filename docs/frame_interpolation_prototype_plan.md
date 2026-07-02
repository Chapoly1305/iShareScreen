# wgpu-compute 插帧原型 — 技术方案

## 状态

- **性质**:实现方案草案。目标是在**不引入任何原生依赖**的前提下,用 wgpu compute 在客户端做插帧原型,用来**验证价值(artifact)与延迟**,再决定是否为 AMD 投入 AMF 原生集成。
- **日期**:2026-07-01
- **背景**:见 [frame_interpolation_research.md](frame_interpolation_research.md)。关键结论:插帧本质是"插值"→ 吃约一帧延迟;远程桌面 artifact(文本/光标/静止区)是最大未知,必须实机验证。

## 1. 目标与非目标

**目标**
- 跨厂商(AMD/Intel/NVIDIA/Apple 都能跑),纯 WGSL compute,无 C/C++ 扩展、无 D3D↔wgpu 互操作。
- 在**真实远程桌面流**上量化:① 增加的端到端延迟;② 屏幕内容 artifact 类型与严重度;③ compute 耗时占用。
- 默认关闭,`ISS_FRC` 开关控制,方便 A/B。

**非目标(原型阶段)**
- 不追求 AMF/RIFE 级画质。
- 不做外插(extrapolation);先只做插值,把延迟成本摆到台面上测。
- 不改编码器/网络侧;纯客户端后处理。

## 2. 约束回顾(来自调研 + 代码库)

1. **+1 帧延迟不可避免**:要生成 mid(N, N+1) 必须先拿到 N+1 → 全流延后一个源帧(@60fps ≈ +16.7ms)。这是要测的核心代价。
2. **只在有刷新率余量时有意义**:源 ≤60fps。60Hz 屏没有余量(无法插入);**原型只在 120Hz+ 显示器上产出合成帧**。
3. **光标已是独立 overlay**([gpu.py](../src/isharescreen/frontend/desktop/gpu.py) 的 draw() 里视频之后才画光标),插帧器只处理视频,**天然不会插到光标** → 无光标 ghosting。这是本项目的先天优势,方案必须保持"插帧在光标 overlay 之前"。
4. **呈现必须在主线程**(Cocoa/wgpu),compute 也在主线程 device 上跑,无跨线程问题。
5. **内容自适应是刚需**:远程桌面大量静止 + 锐利文本,盲插会在文字边缘出鬼影。电视 MEMC 靠"文字/OSD 区域检测并局部禁用插帧"压制,我们要借鉴。

## 3. 当前管线与插入点

当前(事件驱动循环,见 [app.py](../src/isharescreen/frontend/desktop/app.py) 主循环 + [gpu.py](../src/isharescreen/frontend/desktop/gpu.py) `Renderer.draw`):

```
tile NALU → 解码(worker 线程)→ TileFrame(CPU YUV)
  → upload_tile() 写入 canvas Y/U/V 纹理(wgpu)
  → draw(): YUV→RGB(BT.709 shader)→ 光标 overlay → present(vsync)
```

原型改造后:

```
upload_tile() 同上
  → [新] 把当前整帧 YUV→RGB 渲染到离屏 RGBA 目标(复用现有 YUV→RGB shader)
  → [新] 存入 2 帧历史环(frame N-1, N)
  → [新] 当 N+1 到达:compute 生成 mid;present 时序为 N →(下个 vsync)mid →(再下个)N+1
  → blit 选中的帧到 swapchain → 光标 overlay → present
```

**为什么先转 RGBA**:插帧在 RGBA 上做,避免同时处理 planar / NV24 biplanar 两套 YUV 布局(见 gpu.py 的两个 WGSL 变体),把 FRC 和色彩转换解耦,原型更简单。代价是多一个离屏 pass + 显存,原型阶段可接受。

## 4. 分阶段实现

### P0 — 脚手架(插帧=直通,先证明不回归)
- 加 `ISS_FRC` 开关(默认 off)。
- 离屏 RGBA render target + YUV→RGB pass(复用现有 shader,输出到 texture 而非 swapchain)。
- 2~3 帧历史环(RGBA 纹理),快照策略:在一整组 tile 上传完(`any_fresh` 且构成一帧)后快照为一帧。
- 合成呈现调度器:能在两真实帧之间插入额外的 vsync present(此时 interpolator 输出 = 直接复制邻帧,即先不真插)。
- 打点:每帧记录 源帧间隔、compute 耗时、合成帧计数、端到端延迟估计。
- **验收**:开 `ISS_FRC` 后画面与关闭时一致(仅多了直通合成帧),无崩溃、无色彩/对齐回归。

### P1 — 线性混合(baseline,验证时序与延迟)
- interpolator = `mid = 0.5*A + 0.5*B`(compute 或 fragment)。
- **目的**:先把"压一帧 + 合成帧节奏"跑通,实测延迟手感;混合会在运动区出现双影(可预期),但足以验证管线与延迟成本。
- **验收**:120Hz 屏上,拖窗口/滚动时能看到"更顺但有双影";量出实际增加的延迟。

### P2 — 经典块运动估计 + 运动补偿(真正画质)
- compute shader 做块匹配(block matching)双向运动估计 → 运动补偿 warp → 融合(遮挡处回退到混合)。这是"经典光流类"的自研简版,对标 AMF FRC / MEMC 的思路。
- 分层搜索(粗到细)控制算力。
- **验收**:运动区双影明显减少;量 compute 耗时是否在 120Hz 预算(<~4ms)内。

### P3 — 内容自适应门控(远程桌面关键)
- **静止区跳过**:帧差为零的块直接复制,不插(省算力 + 避免静止文本抖动)。
- **文本/高频边缘保护**:检测高梯度/锐利边缘块,降权或禁用插帧(借鉴电视 MEMC 的 OSD 检测)。
- **全局运动门控**:整帧几乎无运动时,完全不产合成帧(纯静态桌面零开销、零 artifact)。
- **验收**:静态文本页面无抖动/无鬼影;仅在真正运动内容上插帧。

## 5. 延迟与呈现时序

- **hold-one-frame**:收到 N 时不立即当"最新可呈现",而是保留到 N+1 到达,期间在 vsync 上依次呈现 N、mid、N+1。这就是那约一帧延迟的来源,必须在打点里如实量出。
- **与现有事件驱动循环配合**:真实帧到达仍由 `post_empty_event` 唤醒;合成帧则由一个"下一个 vsync 到期"的定时 present 触发(present 本身走 Fifo vsync 定拍)。
- **降级/关断**:显示器刷新率 ≤ 源帧率、或 compute 超预算、或门控判定无运动时,**跳过合成、回到直通**——保证插帧从不让基础体验变差。
- **可选对照**:后续可加一个"外插模式"分支(不压帧,拿运动矢量把 N 外推),对比延迟与 artifact 取舍。原型不先做。

## 6. 配置与打点

- `ISS_FRC=1` 开启(默认关)。
- `ISS_FRC_MODE=blend|memc`(P1/P2 切换)。
- `ISS_FRC_TARGET_HZ`(合成目标刷新率;默认取显示器刷新率,且必须 > 源帧率才生效)。
- 日志:源 fps、呈现 fps、合成帧比例、compute ms、估计端到端新增延迟。放在 present 附近(现循环 `force_draw()` 之后)。

## 7. 验证协议(决定要不要上 AMF 的依据)

在真实远程桌面流上,分场景实测并记录:
1. **静态文本页面**:是否出现抖动/鬼影(门控是否生效)。
2. **平滑滚动 / 拖窗口**:流畅度提升是否可感、运动区 artifact 是否可接受。
3. **视频播放窗口**:插帧收益最大场景,画质对比。
4. **交互延迟**:开/关插帧下的操作跟手度(鼠标已解耦,主要看内容响应延迟)。
5. **算力**:compute ms 是否在目标刷新率预算内;GPU 占用/功耗变化。

**决策门**:若 P2/P3 在真实内容上"运动收益明显 + artifact 可控 + 延迟可接受",再评估 AMD AMF 原生路径(仅为 Win/AMD 拿额外质量/能效);否则说明该场景不值得,止步于 wgpu 方案或直接放弃。

## 8. 风险

- **文本/光标区 artifact**(最大未知,P3 门控针对性缓解;需实测)。
- **延迟不可接受**(交互场景;P1 就要量,早止损)。
- **算力超预算**(低端 GPU 上 P2 可能吃不消 → 门控 + 降级)。
- **tile 时间戳不一致**:4 个 tile 可能跨时间戳,快照整帧时轻微撕裂;原型接受,必要时按 tile 时间戳对齐快照。
- **显存**:离屏 RGBA + 多帧历史;高分辨率下注意占用。

## 9. 涉及文件(预估改动点)

- [src/isharescreen/frontend/desktop/gpu.py](../src/isharescreen/frontend/desktop/gpu.py) — 离屏 RGBA target、YUV→RGB 复用、interpolate compute pass、blit、历史环。
- [src/isharescreen/frontend/desktop/app.py](../src/isharescreen/frontend/desktop/app.py) — 合成帧呈现调度、`ISS_FRC*` 开关、打点。
- (只读参考)[src/isharescreen/proxy/media/tiles.py](../src/isharescreen/proxy/media/tiles.py) — TileFrame 布局。
