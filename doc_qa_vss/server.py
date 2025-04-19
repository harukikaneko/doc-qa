"""
MCP Server for Document QA System
"""

import atexit
from contextlib import asynccontextmanager
from contextvars import Context
from dataclasses import dataclass
import sys
import logging
import argparse
from typing import AsyncIterator
from doc_qa_vss.pipeline import setup_system
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP

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
parser.add_argument("--db", default="../docstore.db", help="ベクトルDBへのパス")

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
        db, embedder = setup_system(args.model, args.db)
        qa_system = DocumentQASystem.setup(
            embedder==embedder,
            db=db, 
        )
        logger.info("QAシステムの初期化が完了しました")

        def cleanup():
                logger.info("サーバーをシャットダウンしています。DBコネクションを閉じます。")
                if db:
                    db.close()
                logger.info("DBコネクションを閉じました。")
            
        # プログラム終了時に実行される関数を登録
        atexit.register(cleanup)
        
    except Exception as e:
        logger.error(f"QAシステムの初期化に失敗しました: {e}")
        sys.exit(1)
    
    print("Starting MCP server in stdio mode")
    mcp.run(transport="stdio")