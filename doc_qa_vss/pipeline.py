import logging
from typing import List, Dict, Any, Tuple, Optional
import time

from doc_qa_vss.db.vector_db import VectorDatabase
from doc_qa_vss.models.embedding import get_embedder, BaseEmbedding

logger = logging.getLogger(__name__)

def setup_system(model_name: str, db_path: str) -> Tuple[VectorDatabase, BaseEmbedding]:
    """
    システムのセットアップ：データベースと埋め込みモデル
    
    Args:
        model_name: 埋め込みモデル名 ('plamo')
        db_path: データベースファイルパス
    
    Returns:
        データベースと埋め込みモデルのタプル
    """
    # データベースの初期化
    db = VectorDatabase(db_path)
    
    # 埋め込みモデルの初期化
    embedder = get_embedder(model_name)
    
    return db, embedder

class DocumentQASystem:
    """ドキュメント質問応答システム"""
    
    def __init__(self, db: VectorDatabase, embedder: BaseEmbedding):
        """
        初期化
        
        Args:
            db: ベクトルデータベース
            embedder: 埋め込みモデル
        """
        self.db = db
        self.embedder = embedder
    
    @classmethod
    def setup(cls, embedder: BaseEmbedding, db: VectorDatabase):
        """
        システムのセットアップと初期化
        
        Args:
            embedder: 埋め込みモデル
            db: データベース
        
        Returns:
            初期化されたDocumentQASystemインスタンス
        """

        return cls(db, embedder)

    
    def retrieve_documents(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        質問に関連するドキュメントを検索
        
        Args:
            question: 検索クエリ
            top_k: 返す結果の最大数
        
        Returns:
            関連ドキュメントのリスト
        """
        # 質問をエンベディング化
        query_embedding = self.embedder.embed_query(question)
        
        # 類似ドキュメントを検索
        results = self.db.search_similar(query_embedding, top_k)
        
        return results
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        検索結果から文脈を形成
        
        Args:
            retrieved_docs: 検索結果ドキュメントのリスト
        
        Returns:
            整形された文脈文字列
        """
        context = ""
        for i, doc in enumerate(retrieved_docs):
            metadata = doc["metadata"]
            filename = metadata.get("filename", "不明")
            page = metadata.get("page", "不明")
            
            context += f"文書{i+1} (ファイル: {filename}, ページ: {page}):\n{doc['content']}\n\n"
        
        return context
    
    
    async def answer_question_mcp(self, question: str) -> Dict[str, Any]:
        """
        MCP形式で質問に回答
        
        Args:
            question: 質問
        
        Returns:
            MCP形式の回答
        """
        start_time = time.time()
        logger.info(f"MCP質問処理中: '{question}'")
        
        # 関連ドキュメントを検索
        retrieved_docs = self.retrieve_documents(question)
        
        if not retrieved_docs:
            logger.warning("関連するドキュメントが見つかりませんでした。")
            return {
                "question": question,
                "mcp_context": "関連するドキュメントが見つかりませんでした。",
                "sources": [],
                "elapsed_time": time.time() - start_time
            }
        
        # MCP用のフォーマットで文脈を整形
        mcp_context = self._format_mcp_context(retrieved_docs)
        
        # ソース情報の整形
        sources = []
        for doc in retrieved_docs:
            sources.append({
                "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                "similarity": doc["similarity"],
                "page": doc["metadata"].get("page", "N/A"),
                "filename": doc["metadata"].get("filename", "N/A")
            })
        
        elapsed_time = time.time() - start_time
        logger.info(f"MCP質問処理完了。経過時間: {elapsed_time:.2f}秒")
        
        return {
            "question": question,
            "mcp_context": mcp_context,
            "sources": sources,
            "elapsed_time": elapsed_time
        }

    def _format_mcp_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        検索結果からMCP用のコンテキストを形成
        
        Args:
            retrieved_docs: 検索結果ドキュメントのリスト
        
        Returns:
            MCP用に整形された文脈文字列
        """
        mcp_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc["metadata"]
            filename = metadata.get("filename", "不明")
            page = metadata.get("page", "不明")
            
            # MCPフォーマットでドキュメントを追加
            mcp_parts.append(f"[出典{i}] ファイル: {filename}, ページ: {page}\n{doc['content']}")
        
        return "\n\n".join(mcp_parts)