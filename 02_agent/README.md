# L2 · Agent 层

**一句话**：把用户的一句话，变成一串"调工具 → 看结果 → 再想 → 再调"的循环，直到任务完成。

## 为什么需要 Agent

L3 的模型本质上只能做一件事：**给它一段文本，它 predict 下一个 token**。它不能自己读文件、写文件、调 API。

那 ChatGPT 是怎么"帮我做了个 Todo 网站"的？答案是**在模型外面套一个循环**：

```
  ┌──────────────────────────────────────┐
  │                                      │
  │  1. 把"用户问题 + 已有对话"送进模型  │
  │  2. 模型输出：要么是文字（讲给用户） │
  │              要么是"调用某个工具"    │
  │  3. 如果是工具调用：执行它，把结果  │
  │     append 回对话                    │
  │  4. 回到 1                           │
  │                                      │
  └──────────────────────────────────────┘
              ↑ 这就是 Agent
```

这个循环是 **L2 唯一做的事**。工具本身（写文件、跑 shell）可以很简单——难的是让模型在"该调工具"时调、"该停"时停。现代 LLM（Claude / GPT）原生支持 **tool use**：在 system prompt 里告诉它工具的 JSON schema，它会返回结构化的 `tool_use` 块，我们执行完把 `tool_result` 塞回去。

## 目录

```
02_agent/
├── agent.py        # Plan-Act-Observe 循环，~120 行
├── tools.py        # write_file / run_shell 两个工具实现，~60 行
└── README.md
```

## 怎么跑

```bash
export ANTHROPIC_API_KEY=sk-...
python agent.py "帮我做一个 Todo 网站"
```

你会看到类似：

```
[assistant] 好的，我帮你做一个 Todo 网站。我会写一个 index.html 加一个 Flask 后端...
[tool_use] write_file(path="index.html", content_preview="<!doctype html>...")
[tool_result] wrote 2134 bytes to generated/index.html
[tool_use] write_file(path="server.py", content_preview="from flask import Flask...")
[tool_result] wrote 612 bytes to generated/server.py
[tool_use] run_shell(cmd="pip install flask")
[tool_result] exit 0, 1.2s
[assistant] 完成。打开 generated/index.html 就能用了。
```

然后 `generated/` 下会有 Todo 网站的文件。

## 和其他层的接口

- **往上（L1 App）**：`run_agent(query) -> Iterator[Event]`。L1 把事件流式传给浏览器。
- **往下（L3 Model）**：每个循环迭代调用一次 `client.messages.create(...)`。这个调用就是"往 L3 发一个 HTTP 请求，让它跑一次 transformer forward，把 token stream 回来"。

## 这一层的"最小"在哪里

- 只有 2 个工具（`write_file`、`run_shell`）。真实 Agent（Claude Code、Cursor）有十几个工具（Read、Edit、Grep、Bash、WebFetch…），但循环结构完全相同。
- 没有 planning 独立步骤：现代 LLM 直接在思考中规划，不需要单独一个 "planner"。
- 没有长期记忆 / RAG / subagents。这些都是在这个最小循环上加层。

## 自己做一个 Agent，最容易踩的坑

1. **停不下来**：模型一直 tool_use 不肯 text-reply。解法：在 system prompt 里明确告知"任务完成后直接回复总结，不要再 tool_use"，以及设一个 `max_iterations` 兜底。
2. **写进错误的路径**：把工具限制在一个 `work_dir` 里，拼路径前 `Path.resolve()` 校验。本 demo 里 `tools.py::_safe_path` 干这件事。
3. **shell 命令阻塞**：别忘了超时。
