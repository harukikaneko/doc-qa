import logging
import time
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import CrossEncoder
from lindera_py import Segmenter, Tokenizer, load_dictionary

from doc_qa_vss.db.vector_db import VectorDatabase
from doc_qa_vss.models.embedding import get_embedder, BaseEmbedding

logger = logging.getLogger(__name__)

def setup_system(model_name: str = "plamo",
    db_path: str = ":memory:", 
    embedding_dim: int = 2048,
    reranker_model: str = "hotchpotch/japanese-bge-reranker-v2-m3-v1"
) -> Tuple[VectorDatabase, BaseEmbedding]:
    """
    システムのセットアップ：データベースと埋め込みモデル
    
    Args:
        model_name: 埋め込みモデル名
        db_path: データベースファイルパス
        embedding_dim: 埋め込みベクトルの次元数
        reranker_model: リランカーモデル名
    
    Returns:
        データベースと埋め込みモデルのタプル
    """

    # デバイスの設定
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # データベースの初期化
    db = VectorDatabase(
        db_path=db_path,
        embedding_dim=embedding_dim,
        reranker_model=reranker_model,
        device=device
    )
    
    # 埋め込みモデルの初期化
    embedder = get_embedder(model_name)
    
    return db, embedder

class DocumentQASystem:
    """ドキュメント質問応答システム"""
    
    def __init__(self, db: VectorDatabase, embedder: BaseEmbedding):
        """
        初期化
        
        Args:
            db: ハイブリッド検索データベース
            embedder: 埋め込みモデル
        """
        self.db = db
        self.embedder = embedder
    
    @classmethod
    def setup(cls, db: VectorDatabase, embedder: BaseEmbedding):
        """
        システムのセットアップと初期化
        
        Args:
            db: データベース
            embedder: 埋め込みモデル
        
        Returns:
            初期化されたHybridSearchSystemインスタンス
        """
        return cls(db, embedder)
    
    def index_document(self, content: str, metadata: Dict[str, Any] = None) -> int:
        """
        ドキュメントをインデックス化
        
        Args:
            content: ドキュメントのテキスト内容
            metadata: 追加のメタデータ（オプション）
            
        Returns:
            インデックス化されたドキュメントのID
        """
        if metadata is None:
            metadata = {}
            
        # テキストをベクトル埋め込み
        with torch.inference_mode():
            embedding = self.embedder.embed_text(content)
            
        # データベースに追加
        doc_id = self.db.insert_document(
            content=content,
            embedding=embedding,
            metadata=metadata
        )
        
        return doc_id
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        ハイブリッド検索を実行
        
        Args:
            query: 検索クエリ
            top_k: 返す結果の最大数
        
        Returns:
            検索結果のリスト
        """
        # クエリをベクトル埋め込み
        with torch.inference_mode():
            query_embedding = self.embedder.embed_query(query)
            
        # ハイブリッド検索を実行
        results = self.db.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            limit=top_k
        )
        
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
            title = metadata.get("title", "不明")
            source = metadata.get("source", "不明")
            
            fts_score = doc.get("fts_score", "N/A")
            vss_distance = doc.get("vss_distance", "N/A")
            rerank_score = doc.get("rerank_score", "N/A")
            
            context += f"文書{i+1} (タイトル: {title}, ソース: {source}):\n"
            context += f"スコア情報: FTS={fts_score:.4f if isinstance(fts_score, float) else fts_score}, "
            context += f"VSS={vss_distance:.4f if isinstance(vss_distance, float) else vss_distance}, "
            context += f"Rerank={rerank_score:.4f if isinstance(rerank_score, float) else rerank_score}\n"
            context += f"{doc['content']}\n\n"
        
        return context
    
    async def answer_question_mcp(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        MCP形式でクエリの検索結果を返す
        
        Args:
            question: 検索クエリ
            top_k: 返す結果の最大数
        
        Returns:
            MCP形式の回答
        """
        start_time = time.time()
        logger.info(f"ハイブリッド検索MCP処理中: '{question}'")
        
        # ハイブリッド検索
        retrieved_docs = self.hybrid_search(question, top_k=top_k)
        
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
                "title": doc["metadata"].get("title", "不明"),
                "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                "fts_score": doc.get("fts_score", "N/A"),
                "vss_distance": doc.get("vss_distance", "N/A"),
                "rerank_score": doc.get("rerank_score", "N/A"),
                "source": doc["metadata"].get("source", "N/A")
            })
        
        elapsed_time = time.time() - start_time
        logger.info(f"ハイブリッド検索MCP処理完了。経過時間: {elapsed_time:.2f}秒")
        
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
            title = metadata.get("title", "不明")
            source = metadata.get("source", "不明")
            
            # スコア情報
            score_info = []
            if "fts_score" in doc:
                score_info.append(f"FTS: {doc['fts_score']:.4f}")
            if "vss_distance" in doc:
                score_info.append(f"VSS: {doc['vss_distance']:.4f}")
            if "rerank_score" in doc:
                score_info.append(f"Rerank: {doc['rerank_score']:.4f}")
            
            score_text = ", ".join(score_info)
            
            # MCPフォーマットでドキュメントを追加
            mcp_parts.append(f"[出典{i}] タイトル: {title}, ソース: {source}, スコア: {score_text}\n{doc['content']}")
        
        return "\n\n".join(mcp_parts)