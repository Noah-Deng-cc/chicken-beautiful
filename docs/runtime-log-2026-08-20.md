# 运行验证日志（2026-08-20）

## 本地数据回归

| 项目 | 结果 |
| --- | --- |
| 全量 `pytest` | `817 passed` |
| `git diff --check` | 通过 |
| 七类 logits fixture | `surprised`，置信度超过 `0.30` |
| 人脸裁剪注入测试 | 通过，分类器只收到裁剪区域 |

## 树莓派实测

- 节点：Tailscale `100.126.205.69`，项目提交 `83b2132`。
- 摄像头：OV5647 + `picamera2`，帧尺寸 `480x640x3`。
- Haar cascade：成功加载；公开 Lena 样例检测到 `169x169` 人脸区域。
- 人脸区域情绪推理：`neutral`，约 `0.889`。
- 实时画面无人脸时返回空读数，未将背景送入情绪分类器。
- `emotion.onnx`：OpenCV DNN 可加载，输出 `(1, 7)` logits；已执行 softmax 后处理。
- MCP Agent：此前已完成真实调用并返回文本；本轮未重复发送对话，避免产生无必要的远端请求。

## 未完成项

- 树莓派账户无 sudo 权限，`pytest` 和音频依赖尚未安装。
- 未检测到麦克风/声卡，因此 `语音识别 -> Agent -> TTS` 仍不能在硬件上验收。
- 模型标签顺序来自当前七类 FER 约定，替换模型前必须重新核对模型卡和输出映射。
- 当前 Haar cascade 只选最大人脸；多人场景、侧脸和弱光场景仍需专门评估。

## 管理员远程复测

- 通过 Tailscale `100.126.205.69` 以管理员账号 SSH 登录成功，并使用 `sudo -u dengjingwen bash -lc` 在项目用户环境执行测试。
- 树莓派已从 `83b2132` 快进同步到 `2750827`；GitHub `origin/main` 已包含该提交。
- 远端回归测试：`816 passed, 1 failed`。唯一失败是安全策略测试无条件读取被 `.gitignore` 排除的本地 `request.md`；显式排除这一条测试后，项目测试为 `816 passed, 1 deselected`。
- CSI 摄像头已成功打开并采集三帧 `640x480` 图像；当前镜头未检测到人脸，管道返回空读数，未发生背景误分类。
- MCP `chicken-beauty` 已再次真实调用成功，输入包含 `face_emotion=sad` 和用户文本，服务返回中文文本。
- 设备检查仍未发现 ALSA 采集或播放声卡（`arecord -l`、`aplay -l` 均为 `no soundcards found`），因此语音输入和 TTS 尚不能在树莓派硬件上验收。
