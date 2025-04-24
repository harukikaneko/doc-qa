# Document QA System with Vector Similarity Search

DuckDB-VSS と Python を使用した文書質問応答システム。plamo-embedding-1b を利用して、
MCP（Mean Contextualized Pooling）方式で文書検索を行います。

## インストール

```bash
uv venv
uv pip install -e .
```

## 使用方法

### ドキュメントのインデックス化

```bash
docqa index --dir /path/to/documents --model plamo
```

### API（例）

```bash
curl -X POST "http://127.0.0.1:8000/mcp/query" \
    -H "Content-Type: application/json" \
    -d '{"text":""}'
```

### setup

```json
{
  "mcpServers": {
    "qa-doc-mcp": {
      "command": "/path/to/uv", // 絶対パスが推奨
      "args": ["--directory", "/path/to/doc-qa/doc_qa_vss", "run", "server.py"]
    }
  }
}
```
