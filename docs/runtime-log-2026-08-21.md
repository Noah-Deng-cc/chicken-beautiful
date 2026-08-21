# 运行验证日志（2026-08-21）

## 硬件枚举

- 树莓派 `100.126.205.69` 通过管理员 SSH 可访问；项目用户已加入 `audio`、`video`、`i2c`、`gpio` 组。
- CSI 摄像头 OV5647：`/dev/video0`，`picamera2` 可打开并采集 `640x480` 帧。
- USB 音频：PCM2902 `USB PnP Sound Device` 提供单声道采集；M83 `USB Audio` 提供播放。
- USB 麦克风硬件采样率仅声明 `44100`/`48000 Hz`，因此树莓派配置使用 `44100 Hz`，不能使用模板的 `16000 Hz`。
- I2C 节点 `/dev/i2c-0`、`/dev/i2c-2`、`/dev/i2c-10`、`/dev/i2c-11` 和 GPIO 节点已存在；未枚举到串口或 CO2/热成像传感器。

## 软件与链路

- `sounddevice`、`vosk` 已安装到项目用户环境，Vosk 普通话小模型加载成功（模型目录约 66 MB）。模型未提交 Git。
- 麦克风实测连续返回 3200 字节 PCM 块；Vosk 2 秒监听在无讲话时安全返回空结果。
- `espeak-ng` 已安装；`espeak-ng -d hw:1,0 -v cmn` 实测退出码为 0，M83 播放设备可用。
- 同济 `chicken-beauty` MCP 真实调用成功，输入包含 `face_emotion=sad` 和用户语音文本并返回文本。
- Agent 返回文本直接交给 `SystemSpeechOutput` 时，实测中文播报成功；约 180 字节回复耗时约 22 秒，因此树莓派配置将 TTS 超时设为 60 秒。

## 自动化回归

- 本地全量测试：`817 passed`。
- 树莓派此前全量测试：`816 passed, 1 skipped`；跳过项是缺少本地私密 `request.md` fixture 的安全扫描。
- 新增树莓派硬件配置模板：`config/settings.pi.example.yaml`，不包含 API 密钥或模型二进制。

## 尚待现场确认

- 需要在麦克风前说普通话，确认 Vosk 返回非空文本；当前无人讲话测试只能证明采集和识别器加载正常。
- 需要观察实际扬声器音量和持续运行温度/FPS；M83 播放命令已经通过退出码验收。
- 热成像和 CO2 设备仍未连接，启用对应驱动前需确认型号、总线地址和校准参数。

## 全链路复测补充

- 树莓派 `self-test` 中摄像头成功完成 libcamera 配置；由于现场画面无人脸，视觉探测返回空读数，CLI 当前将该情况显示为 `vision: failed`。这表示“无脸”被当作自检失败，不能解释为摄像头故障。
- 摄像头专项复测得到 `(480, 640, 3)` 帧；Haar 检测未发现人脸并返回 `EmotionReading=None`，符合防止背景误分类的设计。
- 同济 Agent → `SystemSpeechOutput` → M83 全链路复测成功，Agent 文本存在，TTS 返回 `True`，一次测试耗时约 16.4 秒。
- 复测期间 USB 集线器日志出现 `disabled by hub (EMI?)`；M83 自动重枚举后仍可播放，但 PCM2902 麦克风设备已断开，PortAudio 当前只看到 M83 输出。因此必须重新插拔/更换供电稳定的 USB Hub 后，才能进行真人语音识别验收。
