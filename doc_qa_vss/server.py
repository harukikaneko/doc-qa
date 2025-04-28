"""
MCP Server for Document QA System
"""

import os
import sys
import atexit
import logging
from typing import Optional, Dict, Any, Tuple

from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

# 設定の外部化
from dotenv import load_dotenv
load_dotenv()  # .envファイルから環境変数を読み込む

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 必須モジュールのインポート
try:
    from doc_qa_vss.db.vector_db import VectorDatabase
    from doc_qa_vss.document.processor import index_document_content
    from doc_qa_vss.models.embedding import BaseEmbedding
    from doc_qa_vss.pipeline import setup_system, DocumentQASystem
except ImportError as e:
    logger.error(f"モジュールのインポートに失敗しました: {e}")
    logger.info("パッケージのインストール状態を確認してください")
    sys.exit(1)

# 設定を環境変数から読み込む
MODEL_NAME = os.getenv("MODEL_NAME", "plamo")
DB_PATH = os.getenv("DB_PATH", "../docstore.db")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "hotchpotch/japanese-bge-reranker-v2-m3-v1")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "2048"))

class QASystemSingleton:
    _instance: Optional[Tuple[VectorDatabase, BaseEmbedding, DocumentQASystem]] = None
    
    @classmethod
    async def get_instance(cls) -> Tuple[VectorDatabase, BaseEmbedding, DocumentQASystem]:
        """シングルトンインスタンスを取得
        
        リクエストごとに再接続するため、既存のインスタンスを強制的に再作成
        """
        # 毎回新しいインスタンスを作成
        if cls._instance is not None:
            # 既存の接続をクローズ
            db, _, _ = cls._instance
            logger.info("既存のデータベース接続をクローズしています")
            await db.close()
            cls._instance = None
        
        logger.info("QAシステムを初期化しています")
        try:
            db, embedder = setup_system(MODEL_NAME, DB_PATH)
            qa_system = DocumentQASystem.setup(
                embedder=embedder,
                db=db,
            )
            cls._instance = (db, embedder, qa_system)
            logger.info("QAシステムの初期化が完了しました")
            
            # 終了時にリソースをクリーンアップ
            atexit.register(cls.cleanup)
        except Exception as e:
            logger.error(f"QAシステムの初期化に失敗しました: {str(e)}")
            raise RuntimeError(f"QAシステムの初期化に失敗: {str(e)}")
                
        return cls._instance
    
    @classmethod
    def cleanup(cls):
        """リソースのクリーンアップ"""
        if cls._instance:
            db, _, _ = cls._instance
            logger.info("データベース接続をクローズしています")
            db.close()
            cls._instance = None

# リクエスト/レスポンスモデル
class Question(BaseModel):
    text: str = Field(..., description="ユーザーからの質問文")

class Document(BaseModel):
    title: str = Field(..., description="ドキュメントのタイトル")
    content: str = Field(..., description="ドキュメントの内容")

class MCPResponse(BaseModel):
    status: str = Field("success", description="処理結果のステータス")
    data: Dict[str, Any] = Field({}, description="レスポンスデータ")
    error: Optional[str] = Field(None, description="エラーメッセージ（存在する場合）")

# MCP設定
mcp = FastMCP("Doc QA VSS")

# MCPツール
@mcp.tool()
async def get_answer(question: Question) -> MCPResponse:
    """質問に対する回答を取得（ベクトル検索）"""
    try:
        db, _, qa_system = await QASystemSingleton.get_instance()
        # 質問処理
        logger.info(f"質問を処理しています: {question.text}")
        result = await qa_system.answer_question_mcp(question.text)
        
        # 処理完了後にDBクローズ
        logger.info("質問処理完了後、データベース接続をクローズします")
        await db.close()
        
        return MCPResponse(
            status="success",
            data={
                "question": result["question"],
                "mcp_context": result["mcp_context"],
                "sources": result["sources"]
            }
        )
    except Exception as e:
        logger.error(f"質問処理中にエラーが発生しました: {str(e)}")
        # エラー時にもクローズを試みる
        if 'db' in locals() and db:
            try:
                await db.close()
                logger.info("エラー発生後、データベース接続をクローズしました")
            except Exception as close_error:
                logger.error(f"クローズ中にエラーが発生: {str(close_error)}")
        
        return MCPResponse(
            status="error",
            error=f"質問処理に失敗しました: {str(e)}"
        )

@mcp.tool()
async def create_document(document: Document) -> MCPResponse:
    """新しいドキュメントをインデックス化"""
    try:
        db, embedder, _ = await QASystemSingleton.get_instance()
        logger.info(f"ドキュメントをインデックス化中: {document.title}")
        doc_id = index_document_content(
            document.content, 
            db, 
            embedder, 
            metadata={"title": document.title}
        )

        # 処理完了後にDBクローズ
        logger.info("ドキュメントインデックス化完了後、データベース接続をクローズします")
        await db.close()
        
        return MCPResponse(
            status="success",
            data={
                "doc_id": doc_id,
                "title": document.title,
                "message": f"ドキュメント '{document.title}' のインデックス化が完了しました"
            }
        )
    except Exception as e:
        logger.error(f"ドキュメントのインデックス化に失敗しました: {str(e)}")
        # エラー時にもクローズを試みる
        if 'db' in locals() and db:
            try:
                await db.close()
                logger.info("エラー発生後、データベース接続をクローズしました")
            except Exception as close_error:
                logger.error(f"クローズ中にエラーが発生: {str(close_error)}")
        
        return MCPResponse(
            status="error",
            error=f"ドキュメント '{document.title}' のインデックス化に失敗しました: {str(e)}"
        )

# エントリーポイント
if __name__ == "__main__":
    print("Starting MCP server in stdio mode")
    mcp.run(transport="stdio")