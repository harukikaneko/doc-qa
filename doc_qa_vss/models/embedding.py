import torch
import numpy as np
from typing import List
import logging
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class BaseEmbedding:
    """埋め込みモデルの基底クラス"""

    def __init__(self, model_name: str, use_trust_remote_code: bool = False):
        self.model_name = model_name
        logger.info(f"モデル {model_name} を読み込み中...")
        # リモートコードを含むモデルの場合に信頼フラグを設定
        kwargs = {"trust_remote_code": True} if use_trust_remote_code else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        logger.info(f"モデルを {self.device} にロードしました。")

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        raise NotImplementedError

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class PlamoEmbedding(BaseEmbedding):
    """pfnet/plamo-embedding-1bモデルを使用した埋め込み生成"""

    def __init__(self, model_name: str = "pfnet/plamo-embedding-1b"):
        # trust_remote_code=True を渡す
        super().__init__(model_name, use_trust_remote_code=True)

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = []
        for text in texts:
            if len(text) > 10000:
                logger.warning(f"テキストが長すぎます ({len(text)} 文字)。結果が切り詰められる可能性があります。")
            # 公式メソッド encode_query / encode_document を利用
            with torch.no_grad():
                # 文書と検索クエリを自動判定しない場合はすべて document として扱う例
                emb = self.model.encode_document([text], self.tokenizer).cpu().numpy()[0]
            # L2 正規化
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return embeddings


def get_embedder(model_name: str = "plamo") -> BaseEmbedding:
    """モデル名に基づいて適切な埋め込みモデルを返す"""
    if model_name.lower() == "plamo":
        return PlamoEmbedding()
    else:
        raise ValueError(f"不明なモデル名: {model_name}。'plamo' または 'ruri' を使用してください。")
