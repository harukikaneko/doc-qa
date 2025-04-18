import duckdb
import json
import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class VectorDatabase:
    """DuckDB-VSSを使ったベクトルデータベース管理クラス"""
    
    def __init__(self, db_path: str = "docstore.db"):
        """
        ベクトルデータベースを初期化
        
        Args:
            db_path: DuckDBデータベースファイルのパス
        """
        self.db_path = db_path
        self.conn = self._setup_database()
    
    def _setup_database(self) -> duckdb.DuckDBPyConnection:
        """DuckDBとVSS拡張をセットアップ"""
        logger.info(f"データベース {self.db_path} を開いています...")
        conn = duckdb.connect(self.db_path)
        
        # VSS拡張のインストールと読み込み
        try:
            conn.execute("INSTALL vss;")
            conn.execute("LOAD vss;")
        except Exception as e:
            logger.warning(f"VSS拡張のインストール中にエラーが発生しました: {e}")
            logger.info("既にインストールされている可能性があります。続行します。")
        
        # ドキュメントとエンベディングを保存するテーブルを作成
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                content TEXT,
                metadata JSON
            );
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                document_id INTEGER,
                embedding FLOAT VECTOR,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );
        """)
        
        # インデックスを確認または作成
        try:
            conn.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_idx 
                ON embeddings USING vss (embedding)
            """)
            logger.info("VSS インデックスが正常に作成または確認されました。")
        except Exception as e:
            logger.error(f"VSS インデックスの作成中にエラーが発生しました: {e}")
        
        return conn
    
    def clear_database(self):
        """データベースをクリア"""
        logger.info("データベースをクリアしています...")
        self.conn.execute("DELETE FROM embeddings")
        self.conn.execute("DELETE FROM documents")
        logger.info("データベースがクリアされました。")
    
    def get_document_count(self) -> int:
        """ドキュメント数を取得"""
        result = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return result[0] if result else 0
    
    def insert_document(self, content: str, metadata: Dict[str, Any], embedding: np.ndarray) -> int:
        """
        ドキュメントとそのエンベディングをデータベースに挿入
        
        Args:
            content: ドキュメントのテキスト内容
            metadata: ドキュメントに関するメタデータ
            embedding: ドキュメントのベクトル表現
        
        Returns:
            挿入されたドキュメントのID
        """
        # 次のIDを取得
        result = self.conn.execute("SELECT COALESCE(MAX(id) + 1, 0) FROM documents").fetchone()
        doc_id = result[0] if result else 0
        
        # メタデータをJSON形式に変換
        metadata_json = json.dumps(metadata)
        
        # ドキュメントを挿入
        self.conn.execute(
            "INSERT INTO documents (id, content, metadata) VALUES (?, ?, ?)",
            [doc_id, content, metadata_json]
        )
        
        # エンベディングを挿入
        self.conn.execute(
            "INSERT INTO embeddings (id, document_id, embedding) VALUES (?, ?, ?)",
            [doc_id, doc_id, embedding]
        )
        
        return doc_id
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        クエリに類似したドキュメントを検索
        
        Args:
            query_embedding: 検索クエリのベクトル表現
            top_k: 返す結果の最大数
        
        Returns:
            類似ドキュメントのリスト（内容、類似度、メタデータを含む）
        """
        results = self.conn.execute("""
            SELECT 
                d.id,
                d.content, 
                vss.cosine_similarity(e.embedding, ?) as similarity, 
                d.metadata
            FROM documents d
            JOIN embeddings e ON d.id = e.document_id
            ORDER BY similarity DESC
            LIMIT ?
        """, [query_embedding, top_k]).fetchall()
        
        formatted_results = []
        for doc_id, content, similarity, metadata_json in results:
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                logger.warning(f"ドキュメント {doc_id} のメタデータを解析できませんでした")
                metadata = {}
            
            formatted_results.append({
                "id": doc_id,
                "content": content,
                "similarity": similarity,
                "metadata": metadata
            })
        
        return formatted_results
    
    def close(self):
        """データベース接続を閉じる"""
        if self.conn:
            self.conn.close()
            logger.info("データベース接続が閉じられました。")