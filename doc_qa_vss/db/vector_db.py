import duckdb
import json
import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class VectorDatabase:
    """DuckDB-VSS（HNSW）を使ったベクトルデータベース管理クラス"""

    def __init__(self, db_path: str = "docstore.db", embedding_dim: int = 768):
        """
        ベクトルデータベースを初期化
        
        Args:
            db_path: DuckDBデータベースファイルのパス
            embedding_dim: 埋め込みベクトルの次元数
        """
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.conn = self._setup_database()

    def _setup_database(self) -> duckdb.DuckDBPyConnection:
        """DuckDBとVSS拡張（HNSW）のセットアップ"""
        logger.info(f"データベース {self.db_path} をオープン...")
        conn = duckdb.connect(self.db_path)

        # VSS拡張インストール＆ロード
        try:
            conn.execute("INSTALL vss;")
        except Exception:
            pass  # 既にインストール済みの可能性あり
        conn.execute("LOAD vss;")
        # VSS 拡張は実験的機能であるため、実験的な永続化を有効にする
        conn.execute("SET hnsw_enable_experimental_persistence = true;")

        logger.info("VSS拡張(HNSW)がロードされました。")

        # ドキュメントテーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id       INTEGER PRIMARY KEY,
                content  TEXT,
                metadata JSON
            );
        """)
        logger.info("documents テーブルが確認されました。")

        # 埋め込みテーブル（固定長配列を使用）
        # ※ 既に FLOAT[] で作成してしまっている場合は、マイグレーションが必要です。
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                id            INTEGER PRIMARY KEY,
                document_id   INTEGER REFERENCES documents(id),
                embedding     FLOAT[{self.embedding_dim}]
            );
        """)
        logger.info("embeddings テーブルが固定長 FLOAT[] 型で確認されました。")

        # HNSW インデックス（コサイン距離）
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS embeddings_idx
            ON embeddings
            USING HNSW (embedding)
            WITH (metric = 'cosine');
        """)
        logger.info("HNSW インデックス (cosine) が確認されました。")

        return conn

    def clear_database(self):
        """データベースをクリア"""
        logger.info("データベースをクリアします...")
        self.conn.execute("DELETE FROM embeddings;")
        self.conn.execute("DELETE FROM documents;")
        logger.info("クリア完了。")

    def get_document_count(self) -> int:
        """ドキュメント数を取得"""
        result = self.conn.execute("SELECT COUNT(*) FROM documents;").fetchone()
        return result[0] or 0

    def insert_document(self, content: str, metadata: Dict[str, Any], embedding: np.ndarray) -> int:
        """
        ドキュメントと埋め込みを挿入

        Args:
            content: テキスト
            metadata: メタデータ
            embedding: numpy.ndarray（長さ must == embedding_dim）

        Returns:
            doc_id
        """
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"embedding の長さが {self.embedding_dim} ではありません (got {embedding.shape[0]})")

        # 次のID取得
        result = self.conn.execute("SELECT COALESCE(MAX(id), -1) + 1 FROM documents;").fetchone()
        doc_id = result[0]

        # ドキュメント挿入
        self.conn.execute(
            "INSERT INTO documents (id, content, metadata) VALUES (?, ?, ?);",
            [doc_id, content, json.dumps(metadata)]
        )
        # 埋め込み挿入
        self.conn.execute(
            "INSERT INTO embeddings (id, document_id, embedding) VALUES (?, ?, ?);",
            [doc_id, doc_id, embedding.tolist()]
        )

        return doc_id

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        類似ドキュメント検索

        Args:
            query_embedding: numpy.ndarray
            top_k: 返却件数

        Returns:
            リスト(dict{id, content, similarity, metadata})
        """
        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"query_embedding の長さが {self.embedding_dim} ではありません")
        
        rows = self.conn.execute(f"""
            SELECT 
                d.id,
                d.content,
                array_cosine_distance(e.embedding, ?::FLOAT[{self.embedding_dim}]) AS similarity,
                d.metadata
            FROM embeddings e
            JOIN documents d ON d.id = e.document_id
            ORDER BY similarity
            LIMIT ?;
        """, [query_embedding.tolist(), top_k]).fetchall()

        results = []
        for doc_id, content, sim, meta_json in rows:
            try:
                meta = json.loads(meta_json)
            except json.JSONDecodeError:
                meta = {}
            results.append({
                "id": doc_id,
                "content": content,
                "similarity": sim,
                "metadata": meta
            })
        return results

    def close(self):
        """接続をクローズ"""
        self.conn.close()
        logger.info("接続を閉じました。")
