"""
MCP Server for Document QA System
"""

import sys
import logging
import argparse
from pydantic import BaseModel
from fastmcp import FastMCP

mcp = FastMCP("Doc QA VSS", "doc_qa_vss")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# doc_qa_vssモジュールのインポートを試みる
try:
    from doc_qa_vss.pipeline import DocumentQASystem
except ImportError as e:
    logger.error(f"モジュールのインポートに失敗しました: {e}")
    logger.info("パッケージのインストール状態を確認してください")
    sys.exit(1)

class Question(BaseModel):
    text: str

class MCPResponse(BaseModel):
    question: str
    mcp_context: str
    sources: list

# コマンドライン引数の処理
parser = argparse.ArgumentParser(description="MCPサーバーを起動")
parser.add_argument("--model", default="plamo", help="使用するモデル名")
parser.add_argument("--db", default="docstore.db", help="ベクトルDBへのパス")
parser.add_argument("--host", default="127.0.0.1", help="ホストアドレス")
parser.add_argument("--port", type=int, default=8000, help="ポート番号")

args = parser.parse_args()

@mcp.tool()
async def get_document(question: Question) -> MCPResponse:
    # 質問処理
    result = qa_system.answer_question_mcp(question.text)
    return {
        "question": result["question"],
        "mcp_context": result["mcp_context"],
        "sources": result["sources"]
    }

# エントリーポイント
if __name__ == "__main__":
    # QAシステムの初期化
    try:
        logger.info(f"QAシステムを初期化しています（モデル: {args.model}, DB: {args.db}）")
        qa_system = DocumentQASystem.setup(
            model_name=args.model,
            db_path=args.db, 
            use_llm=False  # MCPではLLMは使用しない
        )
        logger.info("QAシステムの初期化が完了しました")
    except Exception as e:
        logger.error(f"QAシステムの初期化に失敗しました: {e}")
        sys.exit(1)
    
    print("Starting MCP server in stdio mode")
    mcp.run(transport="stdio")