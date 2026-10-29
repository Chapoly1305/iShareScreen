# 配置启动媒体协商

## CLI precedence and hardware preset example

Explicit `--codec` / `--decoder` flags override inherited environment settings.
Omitting a flag preserves the inherited value; explicitly passing `auto` clears
that override and uses the frontend's normal automatic selection.

For an RX 6800 XT using the existing H.264 path:

```sh
iss --host mac.local -u me --frontend desktop --codec avc --decoder libav-avc420
```

`libav-avc420` is the decoder registry ID. `d3d11va` is a hardware backend
chosen internally on Windows, not a valid `--decoder` name. Hardware may still
fall back to software if initialization fails. A JSON preset loader is not yet
implemented; a future preset for this path should use `codec: avc`,
`chroma: "420"`, and `decoder: libav-avc420`.

Pixel format offers must be nonempty. AVC offers must include an 8-bit 4:2:0
format; HEVC 4:2:0 remains available through the low-level offer configuration
but advertising it does not prove the host or selected decoder supports it.

`src/isharescreen/proxy/protocol/media_config.py` 将启动 offer 的已复原字段保存为可编辑的 Python dataclass。配置经 `create_offers()` 序列化为 plist/zlib/protobuf，也可以直接交给 `SessionConfig.media_offer`。每次 offer 的 SSRC、时间戳和 UUID 仍由构造器生成。

## 常用配置

```python
from isharescreen.proxy.protocol.media_config import MediaOfferConfig, PixelFormat
from isharescreen.proxy.session import Session, SessionConfig

media = MediaOfferConfig()
media.set_video_bitrate(30_000_000)       # 声明桌面流带宽上限：30 Mbps
media.video.tiles_per_frame = 1
media.video.ltrp_enabled = False
media.video.allow_rtcp_feedback = True  # 与 LTRP 独立设置
media.video.pixel_formats = PixelFormat.YUV420_8_FULL
media.video.fec_enabled = False
media.video.rtx_enabled = False
media.validate()

session = Session(SessionConfig(
    host="mac.local",
    username="user",
    password="...",
    audio=True,
    media_offer=media,
))
# session.connect()  # 使用项目原有连接生命周期
```

`MediaOfferConfig()` 不读取环境变量；`MediaOfferConfig.from_environment()` 显式继承现有 `ISS_VIDEO_CODEC`、`ISS_TILES_PER_FRAME`、`ISS_LTRP` 和 `ISS_AUDIO_F4`。未传 `media_offer` 的现有 Session 继续从这些变量构造配置。

Session 在构造时复制配置并验证，后续首次握手、重连和布局重新协商使用该副本。修改原始 `media` 对象不会更改已经创建的 Session；要使用新值，创建新 Session。这是启动配置接口，不是在线码率控制接口。

## 对象和可设置的值

| 对象/属性 | 设置内容 |
|---|---|
| `media.video` / `VideoSettings` | codec、RTCP、分片数、LTRP、像素格式、HDR、FEC、RTX、清屏黑帧、foveation、自定义宽高 |
| `media.video.payloads` / `VideoPayloadSettings` | payload bank、feature string、parameter set、规则列表 |
| `payload.rules` / `VideoRuleCollection` | transport、encode/decode、尺寸/FPS 能力及偏好位图 |
| `media.bandwidth` / `BandwidthSettings` | 每种模式/网络的带宽上限及 selector |
| `media.audio` / `AudioSettings` | audio unit model、support/payload/secondary flags、SBR |
| `media.allow_dynamic_max_bitrate` | 动态最大码率能力布尔值 |
| `media.allows_contents_change_with_aspect_preservation` | 内容宽高比变化能力布尔值 |
| `media.baseband_codec_sample_rate` | 顶层 F8 数值 |
| `media.blob_version` | 顶层 F14，默认 2 |
| `media.media_control_info_version` | 顶层 F16，默认 0 |
| `media.access_network_type` | 顶层 F18，默认 1 |

`video.codec` 可用 `"both"`、`"hevc"`、`"avc"`。`tiles_per_frame=None` 自动选择 HEVC/both 为 4、AVC 为 1；LTRP 为 `None` 时随 codec 自动选择。显式为 AVC 开启 LTRP 会报错，因为客户端没有相应的 ACK 路径。RTCP 为 `None` 时保持原有随 LTRP 选择的默认行为，可以显式独立设为 `True/False`。

FEC、RTX、foveation 等可选字段中，`None` 表示不发送字段，`False` 表示明确发送 0。关闭 LTRP 会同时从 feature string 中删除 `LTR` token。

## 声明帧率和像素格式

```python
from isharescreen.proxy.protocol.media_config import HDRMode, ScreenFormat

for payload in media.video.payloads:
    for rule in payload.rules:
        rule.formats = ScreenFormat.HD1080_30 | ScreenFormat.HD1080_60
        rule.preferred_format = ScreenFormat.HD1080_60

media.video.hdr_modes = HDRMode.SDR
```

这里声明的是能力/偏好集合，不是强制 FPS。`ScreenFormat` 保存了本次静态分析复原的 23 个桌面格式位；`preferred_format` 必须是 `formats` 的子集。虚拟显示尺寸仍使用现有 `SessionConfig.advertise` / `0x1d` 路径。HDR 的显示配置开关仍是现有 `SessionConfig.hdr`，这里的 HDR 位图单独描述媒体能力。

`PixelFormat` 和 `HDRMode` 使用命名位，不必再手写 63、1 等常量。名称和全部规则见 [字段分析](analysis/media-negotiation-24G624.md)。

## 配置指定模式的带宽

```python
from isharescreen.proxy.protocol.media_config import BandwidthMode, BandwidthSettings

entry = BandwidthSettings.for_mode(
    BandwidthMode.LOW_LATENCY_SCREEN_WIRED,
    30_000_000,
)
# entry.configuration == 0
# entry.configuration_extension == 0x40000
```

常用的 `media.set_video_bitrate(bps)` 会替换匹配桌面模式的上限，保留其他模式的值，并处理组合 selector/重复匹配，避免产生冲突的桌面上限。它不强制固定码率。

`for_mode()` 接受已知命名位并自动拆分 F1/F3；直接构造 `BandwidthSettings(...)` 可保存旧消息中的未识别 selector。默认列表保留旧值，包括既有的 75/100 Mbps 条目；这不表示其 selector 已证实有效。

码率 setter 接受 `1..4_294_967_295` 的整数。这只是字段编码范围。静态样本桌面 profile 默认 min/max 为 20/60 Mbps；设置更低或更高的值不会由客户端伪装成“已经支持”，服务端效果需要实测。

## 单独构造消息

```python
from isharescreen.proxy.protocol.offers import create_offers

video_plist, audio_plist = create_offers(config=media, audio_enabled=False)
```

音频开关保留现有 `SessionConfig.audio` / `audio_enabled` 行为。`media.audio.payload_flags` 为 `None` 时选择已有开/关掩码；显式数值会覆盖该选择，与旧 `ISS_AUDIO_F4` 的优先级一致。该字段是能力 flags，不是音频码率。

验证会拒绝越界整数、非布尔开关、未知像素/HDR 位、未知桌面格式位及不匹配的 codec/LTRP。它不证明目标系统会接受或执行所有编码合法的配置。

## 验证依据

重构前保存了 40 个确定性 MediaBlob 的 SHA-256，覆盖 codec、LTRP、分片环境变量、音频开关及音频覆写；默认序列化与这些基准逐字节对应。相关回归和 Session 接入测试在 `tests/test_media_offer_config.py`。协议字段来源和限制见 [24G624 分析](analysis/media-negotiation-24G624.md)。


### Independent codec and chroma selection

The connection page exposes codec (Auto, HEVC, H.264) and chroma sampling
(Auto, 4:4:4, 4:2:0) separately. The CLI equivalent is `--codec hevc --chroma 420`.
`--chroma` accepts `auto`, `444`, or `420`; omitted values inherit
`ISS_VIDEO_CHROMA`, while explicit `auto` clears that override.

Explicit chroma choices restrict the media offer to the corresponding 8-bit
pixel formats. Auto retains the existing format bitmap. Auto codec plus 4:4:4
selects HEVC. H.264 4:4:4 and browser 4:4:4 are rejected because those paths
are not supported by this client. HEVC 4:2:0 is an advertised preference;
remote-host negotiation and hardware decoding still require live validation.
The current HEVC decoder registry retains its conservative 4:4:4 capability
probes, so requesting 4:2:0 does not guarantee selection of a different GPU.
