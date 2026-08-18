# Dormitory Edge Assistant

面向 Raspberry Pi Zero 2 W 的宿舍边缘服务：采集七类人脸情绪、热成像温度和 CO2 浓度，处理本地提醒，并可通过麦克风与外部智能体对话。情绪和温度信息仅作为环境观察，不构成医学诊断。

## 当前范围

- 开发机：七类情绪 YOLO 训练与 ONNX/NCNN 导出入口。
- 树莓派：仅加载导出的模型进行低分辨率 CPU 推理；不安装 PyTorch 或 Ultralytics 训练栈。
- 运行时：可替换的视觉、热成像、CO2、语音、智能体、提醒和 JSONL 记录组件。
- 自动化：默认模拟模式，不需要摄像头、声卡、GPIO 或公网。

真实训练、真实智能体契约和树莓派硬件验收尚未完成，不能由模拟测试替代。详见 [架构说明](docs/architecture.md) 与 [硬件清单](docs/hardware-checklist.md)。

## 本地开发

要求 Python 3.10+。创建隔离环境并安装依赖：

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Windows PowerShell 使用 `.venv\Scripts\python.exe` 替代 `.venv/bin/python`。复制 `config/settings.example.yaml` 为被忽略的 `config/settings.yaml` 后再写入本机路径和设备配置。

```bash
python main.py check-config --config config/settings.example.yaml
python main.py self-test --config config/settings.example.yaml
python main.py run --config config/settings.example.yaml --mock
python -m pytest -q
```

`run --mock` 是当前允许的交互式常驻运行方式；使用 Ctrl+C 停止。`check-config` 不访问硬件，`self-test` 会创建已启用组件并返回明确的失败退出码。

## 模型训练与导出

训练只在具有 NVIDIA GPU 的开发机上执行。先确认数据集许可证、路径、训练/验证划分和标签顺序完全一致：`angry`、`disgusted`、`fearful`、`happy`、`neutral`、`sad`、`surprised`。

```bash
python training/train.py --model /path/to/base.pt --dataset training/emotions.yaml --imgsz 320 --device 0
python training/export.py --weights /path/to/best.pt --format onnx --imgsz 320 --device cpu
```

不要提交数据集、权重或运行目录。实训完成时记录随机种子、软件版本、每类 precision/recall/F1、混淆矩阵、输入尺寸、导出格式和 SHA-256 校验和。仅在确认数据许可后执行训练；当前仓库不包含数据或模型指标。

## 树莓派部署

目标为 64 位 Raspberry Pi OS Lite 的 Zero 2 W。先安全传输已导出的模型，再复制并填写 `config/settings.pi.yaml`，将模型输入限制在 320 或更低，并使用低采样率。安装前运行：

```bash
sudo deploy/install_pi.sh --check-only
```

通过检查后，审阅脚本和 [硬件清单](docs/hardware-checklist.md)，再执行安装。脚本创建非 root 服务账户、受限环境文件和 systemd 单元。完成硬件、密钥和安全验收后才可启动真实服务；不得将已安装服务视为已验收服务。

## 安全与隐私

- 凭据只放在权限为 `0640` 的 `/etc/dorm-assistant/dorm-assistant.env` 或本机秘密设施中；环境模板仅使用占位符。
- 使用 SSH 密钥认证并关闭口令认证；如曾暴露口令或 API 密钥，立即轮换。
- 不提交 `.env`、本机配置、模型、原始音频、图像、日志、训练数据或会话数据。
- JSONL 默认不保存对话文字、原始音频或原始图像。开启前应获得适当授权并定义保留期限。

## 已知限制

- 真实传感器型号、接线和校准参数尚待确认。
- 外部智能体真实 API 契约尚待使用轮换后的最小权限密钥验证。
- 尚未在树莓派上完成连续两小时、内存、温度和端到端延迟验收。
