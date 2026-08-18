# 同济智能体 API 接入记录

## 已确认的接口

接口基址：

`https://agent.tongji.edu.cn/api/proxy/api/v1/`

所有请求使用以下请求头：

```http
Apikey: <真实 API Key>
Content-Type: application/json
```

应用 ID/AppID 不能直接替代 `Apikey`。`AppKey` 是请求体中的历史字段，文档标记为废弃；如服务端要求，可令其与真实 API Key 相同，但不得把密钥提交到仓库。

## 创建会话

```bash
curl --location "$TONGJI_API_BASE/create_conversation" \
  --header "Apikey: $TONGJI_API_KEY" \
  --header "Content-Type: application/json" \
  --data '{
    "UserID": "<user-id>",
    "Inputs": {
      "face_emotion": "neutral",
      "user_speech": "你好"
    },
    "ConversationName": "情绪陪伴会话"
  }'
```

成功响应中的 `Conversation.AppConversationID` 是后续聊天请求使用的会话 ID。`UserID` 应由应用稳定生成，长度限制为 1 到 20 个字符。

## 更新会话

```bash
curl --location "$TONGJI_API_BASE/update_conversation" \
  --header "Apikey: $TONGJI_API_KEY" \
  --header "Content-Type: application/json" \
  --data '{
    "AppConversationID": "<conversation-id>",
    "UserID": "<user-id>",
    "ConversationName": "情绪陪伴会话",
    "Inputs": {
      "face_emotion": "sad",
      "user_speech": "我今天有点累"
    }
  }'
```

当前提供的资料没有给出 `ChatQuery` 的路径、请求字段和响应字段；在此契约补齐前，不应把现有 `src/agent/tongji.py` 的暂定 `Authorization/query/data.answer` 映射当作已验证实现。

## 项目运行时变量

树莓派视觉管道输出情绪标签，语音管道输出普通话文本，应用将它们映射为：

| 运行时数据 | 智能体变量 |
| --- | --- |
| 情绪枚举值，如 `happy`、`sad`、`angry`、`neutral` | `face_emotion` |
| ASR 转写文本 | `user_speech` |
| 应用侧稳定用户标识 | `UserID` |

智能体回复应直接作为语音播报文本，保持单段、简短、口语化。原始图像、音频和 API 凭据不发送给智能体，也不写入日志。

## 智能体提示词

```text
# 角色
你是一款面向青年学生的陪伴型对话机器人，特色是能够结合人脸情绪与用户语音文本综合判断用户内心情绪，并根据情绪动态调整回复语气，与用户进行对话聊天。

## 目标
1. 结合传入的面部情绪标签{{face_emotion}}与用户说话内容{{user_speech}}生成贴合当下情绪的口语化简短回复；
2. 识别用户口是心非、反话讽刺，不单纯按照文字表面意思应答；
3. 全程保持对话连贯自然，回复语气过渡自然，避免语气突兀切换。

## 技能和流程说明
1. 优先读取面部情绪{{face_emotion}}作为核心判断依据，面部真实情绪优先级高于文字字面表达；
2. 根据情绪标签匹配对应沟通语气；
3. 识别语句中的讽刺、阴阳怪气的表述，结合真实情绪修正回复逻辑；
4. 保持对话上下文平缓过渡，不出现情绪风格剧烈跳转。

## 输出格式
回复简洁口语化，单段短句，不使用复杂书面语，不长篇大论，符合日常生活交际用语习惯。

## 限制
- happy：轻快活泼，积极回应开心点，多用轻松短句；sad：温柔安抚，优先共情，避免过度说教；angry：包容温和，先认同感受，不进一步激怒；neutral：中立平和；unknown：温和中立。
- 文字平静但面部为 sad/angry 时，主动关心和安抚隐藏的负面情绪。
- 反讽或阴阳怪气必须结合 face_emotion 解读，不按字面迎合。
- 不得生硬套模板，不得无视面部情绪，不输出消极、全盘说教类内容。
- 对过分消极、过激或违法言论必须指正，不得迎合。
- 前后语气保持连贯，不剧烈突变。
```

## 本次验证

使用最小测试请求访问 `create_conversation`，服务返回 `HTTP 403`，错误为 `Unauthorized: invalid token`。这证明接口地址可达，但当前提供的 AppID 不是有效的 `Apikey`。真实 API Key 应通过本机环境变量或受限 EnvironmentFile 注入，不能写入配置模板、代码或 Git。

## MCP 配置

MCP 配置模板位于 `config/mcp.example.json`，工具 schema 位于 `config/mcp.tools.json`，本机密钥位于被忽略的 `config/mcp.local.json`。MCP URL 使用 `api_key` 查询参数，不能把密钥写入提交文件。

验证记录：MCP `initialize`、`tools/list` 和 `tools/call` 均返回 `HTTP 200`。服务端为 `HiAgent-MCP-Server`，协议版本为 `2024-11-05`，工具标记为 `stream_only`。`chicken-beauty` 已成功返回流式文本，说明当前密钥、端点和工具调用链路可用。

## 项目内调用

运行配置将 `agent.driver` 设为 `tongji_mcp`，并通过 `agent.api_key_env` 注入密钥。客户端实现位于 `src/agent/mcp.py`，会自动执行 MCP 初始化和工具发现，维护 `Mcp-Session-Id`，再调用 `chicken-beauty`：

```python
from src.agent import TongjiMcpAgentClient

reply = client.reply(
    "我今天有点累",
    {"emotion": {"dominant": "sad"}},
    None,
)
print(reply.text)
```

`Query` 中会包含 `face_emotion` 和 `user_speech`，上下文中符合工具 schema 的 `files` 列表会映射为 `Files`；不会上传原始音频或图像内容。
