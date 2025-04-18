# Document QA System with Vector Similarity Search

DuckDB-VSS と Python を使用した文書質問応答システム。plamo-embedding-1b や Ruri-v3 などの埋め込みモデルを利用して、
MCP（Mean Contextualized Pooling）方式で文書検索を行います。

## インストール

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## 使用方法

### ドキュメントのインデックス化

```bash
docqa index --dir /path/to/documents --model plamo
```

### 対話型質問応答

```bash
docqa interact --model plamo --use-llm
```

### 単一の質問に回答

```bash
docqa ask "あなたの質問" --model plamo --use-llm
```
