# MCP 主 Agent 接入示例

本项目的 MCP Server 使用 stdio 作为默认传输。主 Agent/Host 启动时，工作目录应指向
项目根目录，并使用已经安装本项目依赖的 Python 解释器。

```json
{
  "mcpServers": {
    "paper-research-agent": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "paper_agent.mcp_server"],
      "cwd": "/absolute/path/to/paper-research-agent",
      "env": {
        "PAPER_AGENT_ENV": "demo",
        "PAPER_AGENT_MAX_CONCURRENCY": "4"
      }
    }
  }
}
```

Windows 将 `command` 改为 `.venv\\Scripts\\python.exe`。真实组员适配器完成前必须保持
`PAPER_AGENT_ENV=demo`；生产分支应在 `bootstrap.py` 中明确装配，不能在 MCP 工具函数内
临时创建或切换组件。

主 Agent 推荐先调用 `get_pipeline_contract` 检查版本，再调用 `research_papers`。正常演示
只暴露粗粒度工具，避免逐篇 Markdown 在 Host 与 MCP Server 之间反复传输。

