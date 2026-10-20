# 启动媒体协商：字段、能力和取值约束

分析日期：2026-10-20。范围为 iShareScreen 发出的 `0x1c` MediaStreamConfiguration，以及其 plist → zlib → `VCMediaNegotiationBlob`，重点为桌面视频和带宽协商。

后续实现已增加 `MediaOfferConfig` 配置对象并接入 Session，见 [配置用法](../media-offer-config.md)。下文“当前代码”和错误注释审计记录的是重构前的分析快照；默认消息值仍保留，字段命名和注释已在新模型中纠正。

二进制基线：macOS **15.7.5 / 24G624** 的 `AVConference.arm64e`，SHA-256 为 `e9b5018706e0390d09fa08d8fec2ab0b80b0e55558094b025acd2aa28bdeb94a`。本机运行系统为 15.7.9 / 24G830；本报告没有把本机版本当作该静态样本。反编译、关键汇编和参数常量保存在 [evidence JSON](media-negotiation-24G624-evidence.json)。

证据边界：已核对当前 offer 构造代码、实际构造出的 protobuf 字节和上述二进制的序列化/解析/配置链；**没有进行本次服务端连接、码率扫描或编码器实测**。下文的“编码范围”“被消费的位”“配置默认值”“实际工作范围”分开列出。

## 1. 结论

- 可以在启动 offer 中声明**适用于特定工作模式/连接类型的带宽上限**，字段是顶层 `F9[].F2 = maxBandwidth`。它不是固定码率、CRF 或 QP 请求。
- 对此样本的 `RemoteDesktopScreenSharing`，`operatingMode=9` → arbiter mode `5`，`connectionType=7`；对应 **`F9[].F3 = 0x40000`，原生名 `mode_LowLatencyScreen_Wired`**。这里的 Wired 是协商配置名；不能仅根据机器实际使用 Wi-Fi 就改另一组配置。
- 本项目目前已发送该项，值为 `60_000_000`。可以保持 selector 不变，只改变这项上限来设计后续实验；当前没有用户可用的码率参数。
- 此 profile 的 `minBandwidth()` 返回 `20_000_000`，`maxBandwidth()` 返回 `60_000_000`。它们是**本地流配置值**，不是已经证明的请求合法区间。请求低于 20 Mbps 可能形成 `txMinBitrate > txMaxBitrate`，后续处理和真实输出待测。
- 视频分辨率、30/60 fps 能力集合、codec payload bank、像素格式、HDR、分片数量、RTCP feedback、LTRP、FEC/RTX 等也有协商字段；服务端通常取双方能力的交集，声明值不等于最终输出。

## 2. 外层结构

```text
加密的 RFB 控制通道
└─ 0x1c MediaStreamConfiguration
   ├─ 固定头：size / version / flags / 各 offer 长度 / UUID
   ├─ audio：两个 46-byte SRTP key blob + audio_offer bplist
   └─ video1：两个 46-byte SRTP key blob + video_offer bplist
      ├─ avcMediaStreamNegotiatorMode = 7
      ├─ avcMediaStreamNegotiatorMediaBlob = zlib(protobuf MediaBlob)
      ├─ avcMediaStreamOptionCallID = UUID 字符串
      └─ avcMediaStreamOptionRemoteEndpointInfo = 独立的 endpoint-info 数据
```

当前代码只填一个视频 offer；协议还留有 video2 长度和流块。音频 mode 为 `8`。mode 是流用途选择，不是码率档位。

当前 `build_0x1c()`：头部 version 为 `3`，`message_size=audio_len+video_len+0xd8`，总字节数为 `message_size+4`，长度字段为 BE `u16`，所以两段 offer 长度之和需不超过 `65_319`。SRTP key blob 为 `32-byte key + 14-byte salt`。

`+0x06` flags 的字节序存在旧 RFC 与当前实现的冲突；当前代码按 **BE u32** 写入，并记录了 IPC/NDR 交换导致的运行时依据。不要按旧 RFC 的 LE 描述直接修改它。常用位：`1` 为视频1的 60 fps 能力、`2` 为视频2的 60 fps 能力、`4` 为不把光标烧进视频。它们不是任意 FPS 数值。

## 3. MediaBlob 顶层字段

下面是字段号/类型复原，不是完整的 proto2 required/optional 声明。长度编码字段使用 protobuf wire type 2；整数、enum、bool 使用 varint/wire type 0。

| 字段 | 原生名称 | 类型/用途 | 当前发送 |
|---|---|---|---|
| F1 | allowDynamicMaxBitrate | bool，动态最大码率能力 | 1 |
| F2 | allowsContentsChangeWithAspectPreservation | bool，内容宽高比变化能力 | 1 |
| F3 | audioSettings | 音频子消息 | audio offer 中出现 |
| F4 | videoSettings | 视频/相机配置子消息 | 当前桌面 offer 不填 |
| F5 | screenSettings | 桌面配置，类型同 VideoSettings | video offer 中出现 |
| F6 | userAgent | string | iShareScreen 版本 |
| F7 | basebandCodec | string | 不填 |
| F8 | basebandCodecSampleRate | uint32 | 0 |
| F9 | bandwidthSettings | repeated BandwidthSettings | 10 项硬编码列表 |
| F10 | captionsSettings | 子消息 | 不填 |
| F11 | multiwayAudioStreams | repeated 子消息 | 不填 |
| F12 | momentsSettings | 子消息 | 不填 |
| F13 | ntpTime | uint64 | 时间戳 |
| F14 | blobVersion | uint32 | 2 |
| F15 | multiwayVideoStreams | repeated 子消息 | 不填 |
| F16 | mediaControlInfoVersion | uint32 | 0 |
| F17 | faceTimeSettings | 子消息 | 不填 |
| F18 | accessNetworkType | uint32 | 1 |

**桌面流是 F5 screenSettings，不是顶层 F4 videoSettings。** 两者使用相同的 VideoSettings 类型，容易混淆。其他业务的字段存在，并不表示 Screen Sharing 支持把它们作为独立功能申请。

## 4. 带宽配置：结构和合法值

```text
MediaBlob.F9[] = BandwidthSettings {
    F1 configuration:          int32 enum/bitmap
    F2 maxBandwidth:           uint32，bps
    F3 configurationExtension: int32 enum/bitmap（可缺省）
}
```

`writeTo:` @ `0x1b7b7a944` 依次以字段号 1、2、3 写 configuration、maxBandwidth、configurationExtension。`ReadFrom` @ `0x1b7b7a5ec` 将对应 varint 保存到这三个 32-bit 成员；没有发现码率数值的应用级上下限检查。

| 字段 | 规范编码值域 | 此消费者实际使用 | 注意 |
|---|---|---|---|
| F1 configuration | int32，`-2^31..2^31-1`；应用中采用非负位图 | 低 12 位 `0x0001..0x0800`，允许按位 OR，已消费掩码 `0x0fff` | 0 本身不产生配置；高位枚举名称存在，但此消费者从 F3 取扩展位 |
| F2 maxBandwidth | `0..4_294_967_295`，uint32 | 保存并用于配置比较 | 0 不代表已确认的“无限制”；正数也不保证可用 |
| F3 configurationExtension | int32，`-2^31..2^31-1`；应用中采用非负位图 | `0x1000..0x40000`，允许按位 OR，已消费掩码 `0x7f000` | 低 12 位在此消费者中不产生配置；更高未知位未被消费 |

原生 enum 的字符串转换器把 `0x7ffff` 命名为 `mode_SUPPORTED`，但消费者把低位和扩展位分别从 F1/F3 读取。因此不能把“enum 名称存在”当成“放在哪个字段都生效”。对同一个 mode/connection 不要放入互相冲突的多个上限；查找函数按匹配配置返回值，未建立重复配置的稳定优先级承诺。

### 4.1 selector 位表

| 字段 | 位 | 原生名称 |
|---|---:|---|
| F1 | 0x0001 | mode_Default_Wifi |
| F1 | 0x0002 | mode_Default_LTE |
| F1 | 0x0004 | mode_FaceTime_Wifi |
| F1 | 0x0008 | mode_FaceTime_LTE |
| F1 | 0x0010 | mode_Screen_Wifi |
| F1 | 0x0020 | mode_Screen_LTE |
| F1 | 0x0040 | mode_FaceTime_3G |
| F1 | 0x0080 | mode_Screen_3G |
| F1 | 0x0100 | mode_Default_3G |
| F1 | 0x0200 | mode_FaceTime_2G |
| F1 | 0x0400 | mode_Screen_2G |
| F1 | 0x0800 | mode_Default_2G |
| F3 | 0x1000 | mode_iPadCompanion_AWDL |
| F3 | 0x2000 | mode_iPadCompanion_USB |
| F3 | 0x4000 | mode_FaceTime_5G |
| F3 | 0x8000 | mode_RemoteCamera_AWDL |
| F3 | 0x10000 | mode_RemoteCamera_USB |
| F3 | 0x20000 | mode_MultiwayScreen_Wifi |
| F3 | **0x40000** | **mode_LowLatencyScreen_Wired** |

`newBandwidthConfigurations` @ `0x1b7f568e0` 还将 F3 的 `0x1000/0x2000` 分别展开到 arbiter mode 4 和 5 的 AWDL/USB 配置。F3 的 `0x40000` 明确构造 mode 5、connection 7。

### 4.2 当前桌面流应修改的条目

```text
F9 entry {
    F1 = 0
    F2 = 30_000_000       // 示例：声明 30 Mbps 上限；尚未实测
    F3 = 0x40000         // LowLatencyScreen_Wired
}
```

当前这项是 `(0, 60_000_000, 262144)`。保持其他模式的条目不变，只改目标条目，能避免把不同业务的参数混为一个全局码率。

普通、非 foveation、非特殊 HDR transport 路径大致为：

```text
negotiated_cap = min(local_matching_bandwidth, remote_matching_bandwidth)
txMaxBitrate   = min(profile.maxBandwidth, negotiated_cap)
txMinBitrate   = profile.minBandwidth
```

证据：`setupVideoStreamConfiguration:errorReason:` @ `0x1b7f2c3bc`，先读对端匹配配置，在 `0x1b7f2c998` 写 `setTxMaxBitrate:`，在 `0x1b7f2c9b4` 写 `setTxMinBitrate:`。同样设置接收侧的 max/min。

此 profile 默认 min/max 为 **20 / 60 Mbps**。因此 30、40、60 Mbps 是可用于首轮验证的声明值；这不表示已经测得 20–60 Mbps 为连续合法区间。低于 20 Mbps 时 min/max 的冲突、高于 60 Mbps 时默认上限的裁剪，以及实际输出码率，都要在目标系统上验证。

特殊分支：`foveationIsSupported` 可令候选上限取 60 Mbps；transport type 2 且 HDR mode 3 的分支可令候选上限取 80 Mbps，然后仍经过 profile 上限裁剪。这也是不能把上面的简化公式套到所有模式的原因。

### 4.3 当前代码中的错误注释

- `_AUDIO_F9_TIERS` 实际构造顶层通用 BandwidthSettings，并非音频专属。
- F1 不是 codec kind，F3 不是 buffer cap；例如 `4074 = 0x0fea` 是组合位图，不是“header marker”。
- `(0,75_000_000,0x80000)` 和 `(0,100_000_000,0x100000)` 的 selector **不在本样本消费者识别的位内**。不能据此声称它们启用了 75/100 Mbps HP 档位。
- 当前列表中的 4100、6500、299 等 F2 数值也不能按注释解释成 CELT/SILK codec 标记；需要保留实际字段含义，按匹配的 mode/network 判断相关性。

本次仅记录分析，没有修改这些常量或运行逻辑。

## 5. 桌面视频可声明的字段

路径为 `MediaBlob.F5 = screenSettings`。

| 子字段 | 名称 | 编码范围/识别值 | 协商意义和限制 |
|---|---|---|---|
| F1 | rtpSSRC | uint32 | 会话 RTP/RTCP 身份；不是质量参数 |
| F2 | allowRTCPFB | bool `0/1` | 反馈能力，需双方支持 |
| F3 | videoPayloadCollections | repeated 子消息 | 可提供多个 codec payload bank/规则 |
| F4 | customVideoWidth | uint32 | 并无已证明的完整工作区间；桌面尺寸实际由服务端/显示配置决定 |
| F5 | customVideoHeight | uint32 | 同上；不能用 uint32 全范围作为可运行分辨率 |
| F6 | tilesPerFrame | uint32 | 原生协商取双方较小值；项目现有路径使用 1 或 4；未证明其他正数可正常编码 |
| F7 | ltrpEnabled | bool `0/1` | 双方能力；本项目仅 HEVC 有完整 ACK 支持 |
| F8 | pixelFormats | uint32 位图，识别 `0x01..0x20`，组合掩码 `0x3f` | 取双方交集；双方非零但无交集会失败 |
| F9 | hdrModesSupported | uint32 位图，识别 `0x01/02/04/08`，组合掩码 `0x0f` | 取双方交集，仍受像素格式及硬件限制 |
| F10 | fecEnabled | bool `0/1` | 原生消费者还检查 accessNetworkType；不能承诺设置 1 就启用 |
| F11 | rtxEnabled | bool `0/1` | 原生消费者还检查 accessNetworkType |
| F12 | blackFrameOnClearScreenEnabled | bool `0/1` | 清屏黑帧能力，受本地设置限制 |
| F13 | foveationSupported | bool `0/1` | 双方能力及特殊编码路径；当前项目未填 |

bool 的规范发送值是 0/1；不要把解析器可能容忍的其他 varint 数当成额外功能。整数可编码也不代表服务端认可。当前分辨率请求走 `0x1d`，并声明 3840×2160 的 backing 上限；这是当前实现/显示配置的边界，不是 F5.F4/F5 的 protobuf 位宽限制。

### 5.1 像素格式位图

| 位 | 原生 FourCC | 说明 |
|---:|---|---|
| 0x01 | 420f | 8-bit 4:2:0，full range |
| 0x02 | 420v | 8-bit 4:2:0，video range |
| 0x04 | x420 | 10-bit 4:2:0 |
| 0x08 | 444f | 8-bit 4:4:4，full range |
| 0x10 | 444v | 8-bit 4:4:4，video range |
| 0x20 | xf44 | 10-bit 4:4:4 |

HDR 字段各位分别对应原生 HDR mode 0、1、2、3。这里不依据数字猜测 PQ/HLG 名称。`isPixelFormatValid:hdrMode:` 检查：上述 8-bit 格式配 mode 0；x420 配 mode 1/2；xf44 配 mode 1/3。具体接收/解码链仍须支持对应格式。

### 5.2 codec 和帧率/尺寸规则

```text
screenSettings.F3[] = VideoPayloadSettings {
    F1 payload:              uint32
    F2 videoRuleCollections: repeated VideoRuleCollection
    F3 featureString:        string
    F4 parameterSet:         uint32
}

VideoRuleCollection {
    F1 transport:           enum { 1=WiFi, 2=Cellular }
    F2 operation:           enum { 1=Encode, 2=Decode }
    F3 formats:             uint32 bitmap
    F4 preferredFormat:     uint32 bitmap
    F5 formatsExt1:         uint32 bitmap
    F6 preferredFormatExt1: uint32 bitmap
}
```

以上是能力/偏好集合；`formats` 不是宽、高或 FPS 的十进制数。`preferredFormat` 也不是 CRF/QP。项目目前 payload bank 用 `123/100`，并有经项目既有运行记录确认的输出映射反转，不能只按变量名判定 HEVC/H.264。`featureString` 含 LTR、CABAC 等能力 token，未在此次分析中建立所有 token 的可用值域。

本样本桌面分支识别的 `formats` 位如下。括号内是 FPS；这些是可声明的规则，不保证宿主实际采用对应尺寸。custom dimensions 和显示配置仍可能改变最终尺寸。

| 尺寸 | 30 fps 位 | 60 fps 位 |
|---|---:|---:|
| 1024×768 | 0x000001 | 0x000002 |
| 1136×640 | 0x000004 | 0x000008 |
| 1334×750 | 0x000010 | 0x000020 |
| 1664×1248 | 0x000040 | 0x000080 |
| 1920×1080 | 0x000100 | 0x000200 |
| 2208×1242 | 0x000400 | 0x000800 |
| 2048×1536 | 0x001000 | 0x002000 |
| 1280×720 | 0x004000 | 0x008000 |
| 1920×896 | 0x010000 | 0x020000 |
| 2732×2048 | — | 0x040000 |
| 2388×1668 | — | 0x080000 |
| 2224×1668 | — | 0x100000 |
| 3072×1728 | — | 0x200000 |
| 3840×2160 | — | 0x400000 |

桌面分支识别 `formats & 0x7fffff`；其高位及 formatsExt1 进入未支持位提示路径，不能照搬相机规则的扩展值。项目当前 `50115=0xc3c3` 声明 1024×768、1664×1248、1920×1080、1280×720 的 30/60 fps 组合，`preferredFormat=0`。本次未找到“任意填写 24/120/144 fps”的直接字段。

## 6. 必须纠正的音频字段解释

当前源码/RFC 把音频 `MediaBlob.F3.F4` 称为 requested audio bitrate / preferredMediaBitRate。**此解释与本次静态基线不一致，不应继续作为已证实的字段语义。**

原生 `VCMediaNegotiationBlobAudioSettings.writeTo:` @ `0x1b7b891bc` 的字段为：F1 rtpSSRC、F2 audioUnitModel、F3 supportFlags、**F4 payloadFlags**、F5 secondaryFlags、F6 useSBR。前五项为 uint32，F6 为 bool。

`setupAudioStreamConfiguration:errorReason:` @ `0x1b7f2c048` 从**本地 negotiator settings**取 `preferredMediaBitRate`；桌面系统音频的 getter @ `0x1b7e42694` 返回 `320_000`。因此先前“F4 可按音频码率选择 tier”的说法需要撤回。已有改 F4 能开/关音频的观察与 payload capability flags 变化也相容，不能用该观察单独证明码率语义。

本次没有进一步复原音频所有 payloadFlags 位，亦未改 `--no-audio` 的现有行为。

## 7. 后续验证边界

要证明可用码率范围，应保持分辨率、codec、HDR、tiles 和播放内容固定，只扫描匹配的 `F9.F2`，依次检查 offer 字节、server answer、服务端最终 txMin/txMax 配置和持续的 SRTP 视频有效载荷码率。建议先用 20/30/40/60 Mbps，再单独验证 5/10 Mbps 和 75/100 Mbps 的裁剪或异常行为。内容静止时实际码率可以远低于上限，因此不能仅用静态桌面的收包速率证明请求生效。

当前完成的是字段/消费者静态分析与本地构造字节核对；未把它计作服务端接受、真实码率控制或完整有效值域验证。
