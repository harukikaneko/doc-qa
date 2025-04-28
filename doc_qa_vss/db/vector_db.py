import os
import duckdb
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional

from sentence_transformers import CrossEncoder
from lindera_py import Segmenter, Tokenizer, load_dictionary
import torch

logger = logging.getLogger(__name__)

class VectorDatabase:
    """DuckDB-VSS（HNSW）を使ったベクトルデータベース管理クラス"""

    def __init__(self, 
        db_path: str = "docstore.db", 
        embedding_dim: int = 2048,
        reranker_model: str = "hotchpotch/japanese-bge-reranker-v2-m3-v1",
        device: str = None
    ):
        """
        データベースを初期化
        
        Args:
            db_path: DuckDBデータベースのパス
            embedding_dim: 埋め込みベクトルの次元数
            reranker_model: リランキングに使用するCrossEncoderモデル名
            device: 使用するデバイス（"cuda"/"cpu"、Noneの場合は自動判定）
        """
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        
        # デバイスの設定
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 形態素解析器を初期化
        self.dictionary = load_dictionary("ipadic")
        self.segmenter = Segmenter("normal", self.dictionary)
        self.tokenizer = Tokenizer(self.segmenter)
        
        # リランカーモデルの初期化
        self.reranker = CrossEncoder(reranker_model, max_length=512, device=self.device)

        # データベース接続を設定
        self.conn = self._setup_database()
        
        logger.info(f"ハイブリッド検索データベースが初期化されました（デバイス: {self.device}）")

    def _setup_database(self) -> duckdb.DuckDBPyConnection:
        """DuckDBとVSS拡張（HNSW）のセットアップ"""
        logger.info(f"データベース {self.db_path} をオープン...")
        conn = duckdb.connect(self.db_path)

        # doc-qaディレクトリ内に秘密情報と拡張機能のディレクトリを設定
        base_dir = "../"  # ベースディレクトリ

        # 必要なディレクトリパスを構築
        # https://github.com/duckdb/duckdb/issues/12837
        secrets_dir = os.path.join(base_dir, "secrets")
        extension_dir = os.path.join(base_dir, "extensions")

        # 絶対パスに変換（DuckDBが相対パスを正しく解決できない場合のため）
        secrets_dir = os.path.abspath(secrets_dir)
        extension_dir = os.path.abspath(extension_dir)

        # ディレクトリの設定
        conn.execute(f"SET secret_directory='{secrets_dir}';")
        conn.execute(f"SET extension_directory='{extension_dir}';")

        # VSS拡張インストール＆ロード
        conn.install_extension("vss")
        conn.load_extension("vss")
        
        # FTS拡張機能をロード
        conn.install_extension("fts")
        conn.load_extension("fts")
        
        conn.execute("LOAD fts;")
        
        logger.info("VSS拡張とFTS拡張がロードされました。")
        
        # ID用のシーケンス作成
        conn.execute("CREATE SEQUENCE IF NOT EXISTS id_sequence START 1;")
        
        # テーブル作成
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER DEFAULT nextval('id_sequence') PRIMARY KEY,
                content VARCHAR,
                content_v FLOAT[{self.embedding_dim}],
                content_t VARCHAR,
                metadata JSON
            );
        """)
        
        logger.info("documents テーブルが作成されました。")

        # FTSインデックスを作成
        conn.execute("""
            PRAGMA create_fts_index(
                'documents',       -- インデックスを作成するテーブル
                'id',              -- ドキュメント識別子
                'content_t',       -- インデックス化する列
                stemmer = 'none',  -- 日本語ではステミングを無効化
                stopwords = 'none', -- 英語のストップワードを無効化
                ignore = '',
                lower = false,
                strip_accents = false,
                overwrite = 1       -- 既存のインデックスを上書き
            );
        """)
        
        return conn
    
    def tokenize_text(self, text: str) -> str:
        """
        テキストを形態素解析して空白区切りのトークン列に変換
        
        Args:
            text: 分割する日本語テキスト
            
        Returns:
            空白区切りのトークン文字列
        """
        tokens = self.tokenizer.tokenize(text)
        return " ".join(t.text for t in tokens)
    
    def insert_document(
        self, 
        content: str, 
        embedding: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        ドキュメントを追加
        
        Args:
            content: ドキュメントのテキスト内容
            embedding: ドキュメントのベクトル埋め込み
            metadata: 追加のメタデータ（オプション）
            
        Returns:
            追加されたドキュメントのID
        """
        if metadata is None:
            metadata = {}
            
        # メタデータをJSON文字列に変換
        metadata_json = json.dumps(metadata)
        
        # トークン化したテキストを準備
        content_t = self.tokenize_text(content)

        # 次のID取得
        result = self.conn.execute("SELECT COALESCE(MAX(id), -1) + 1 FROM documents;").fetchone()
        doc_id = result[0]

        # ドキュメント挿入
        self.conn.execute(
            """
            INSERT INTO documents (content, content_v, content_t, metadata) 
            VALUES (?, ?, ?, ?)
            RETURNING id
            """,
            [content, embedding.tolist(), content_t, metadata_json]
        )
        
        logger.info(f"ドキュメント ID: {doc_id} を追加しました")
        return doc_id

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
    
    def search_fulltext(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        全文検索を実行
        
        Args:
            query: 検索クエリ
            limit: 返す結果の最大数
            
        Returns:
            検索結果のリスト
        """
        # クエリをトークン化
        query_tokens = self.tokenize_text(query)
        
        if not query_tokens:
            logger.warning(f"検索クエリ '{query}' から有効なトークンが抽出できませんでした")
            return []
        
        # BM25スコアリングを使用して検索
        result = self.conn.execute(f"""
            SELECT 
                id,
                content,
                metadata,
                fts_main_documents.match_bm25(id, '{query_tokens}') AS score
            FROM documents
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT {top_k}
        """).fetchall()
        
        formatted_results = []
        for doc_id, content, meta_json, score in result:
            try:
                metadata = json.loads(meta_json)
            except (json.JSONDecodeError, TypeError):
                metadata = {}
                
            formatted_results.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata,
                "score": score
            })
        
        logger.info(f"全文検索クエリ '{query}' に対して {len(formatted_results)} 件の結果が見つかりました")
        return formatted_results

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        類似ドキュメント検索

        Args:
            query_embedding: numpy.ndarray
            top_k: 返却件数

        Returns:
            リスト(dict{id, content, similarity, metadata})
        """
        
        rows = self.conn.execute(f"""
            SELECT 
                id,
                content,
                metadata,
                array_cosine_distance(content_v, ?::FLOAT[{self.embedding_dim}]) AS similarity,
            FROM documents
            ORDER BY similarity
            LIMIT ?;
        """, [query_embedding, top_k]).fetchall()

        results = []
        for doc_id, content, meta_json, sim in rows:
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

        logger.info(f"vss検索クエリ に対して {len(results)} 件の結果が見つかりました")
        return results
    
    def hybrid_search(
        self, 
        query: str, 
        query_embedding: np.ndarray, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        ハイブリッド検索（全文検索＋ベクトル検索＋リランキング）を実行
        
        Args:
            query: テキストクエリ
            query_embedding: クエリのベクトル埋め込み
            limit: 返す結果の最大数
            
        Returns:
            リランキングされた検索結果のリスト
        """
        # 全文検索を実行
        fts_results = self.search_fulltext(query, top_k=limit)
        
        # ベクトル検索を実行
        vss_results = self.search_similar(query_embedding, top_k=limit)
        
        # 結果をマージ
        merged_results = self._merge_results(fts_results, vss_results)
        
        # リランキング用のPassageとIDをマッピング
        passages = {result["content"]: result for result in merged_results}
        contents = list(passages.keys())
        
        if not contents:
            logger.warning("検索結果が見つかりませんでした")
            return []
        
        # CrossEncoderでリランキング
        with torch.inference_mode():
            rerank_scores = self.reranker.predict([(query, content) for content in contents])
        
        # スコア高い順にソート
        reranked_results = []
        for content, score in zip(contents, rerank_scores):
            result = passages[content]
            result["rerank_score"] = float(score)  # numpyの型からPythonのfloatに変換
            reranked_results.append(result)
        
        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # 上位N件に絞る
        final_results = reranked_results[:limit]
        logger.info(f"ハイブリッド検索で {len(final_results)} 件の結果が見つかりました")
        
        return final_results
    
    def _merge_results(
        self, 
        fts_results: List[Dict[str, Any]], 
        vss_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        全文検索結果とベクトル検索結果をマージ
        
        Args:
            fts_results: 全文検索結果のリスト
            vss_results: ベクトル検索結果のリスト
            
        Returns:
            マージされた結果のリスト
        """
        # IDをキーにした辞書を作成
        merged_dict = {}
        
        # 全文検索結果を辞書に追加
        for result in fts_results:
            doc_id = result["id"]
            if doc_id not in merged_dict:
                merged_dict[doc_id] = result.copy()
                merged_dict[doc_id]["fts_score"] = result["score"]
                # scoreキーを削除
                if "score" in merged_dict[doc_id]:
                    del merged_dict[doc_id]["score"]
            else:
                merged_dict[doc_id]["fts_score"] = result["score"]
        
        # ベクトル検索結果を辞書に追加
        for result in vss_results:
            doc_id = result["id"]
            if doc_id not in merged_dict:
                merged_dict[doc_id] = result.copy()
                merged_dict[doc_id]["vss_similarity"] = result["similarity"]
                # distanceキーを削除
                if "similarity" in merged_dict[doc_id]:
                    del merged_dict[doc_id]["similarity"]
            else:
                merged_dict[doc_id]["vss_similarity"] = result["similarity"]
        
        # 辞書の値のリストを返す
        return list(merged_dict.values())

    async def close(self):
        """接続をクローズ"""
        self.conn.close()
        logger.info("接続を閉じました。")
