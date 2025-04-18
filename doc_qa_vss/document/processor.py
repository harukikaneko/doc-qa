import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

from langchain.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from doc_qa_vss.db.vector_db import VectorDatabase
from doc_qa_vss.models.embedding import BaseEmbedding

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """ドキュメントの読み込みと分割を行うクラス"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        初期化
        
        Args:
            chunk_size: テキスト分割のチャンクサイズ
            chunk_overlap: チャンク間のオーバーラップ文字数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "、", " ", ""]
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        ファイルタイプに応じたローダーを使用してドキュメントを読み込む
        
        Args:
            file_path: 読み込むファイルのパス
        
        Returns:
            ドキュメントのリスト
        """
        file_path = str(file_path)  # Pathオブジェクトの場合は文字列に変換
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.pdf':
                logger.info(f"PDFファイルを読み込み中: {file_path}")
                loader = PyPDFLoader(file_path)
            elif file_ext in ['.docx', '.doc']:
                logger.info(f"Wordファイルを読み込み中: {file_path}")
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_ext == '.txt':
                logger.info(f"テキストファイルを読み込み中: {file_path}")
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_ext in ['.csv']:
                logger.info(f"CSVファイルを読み込み中: {file_path}")
                loader = CSVLoader(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                logger.info(f"Excelファイルを読み込み中: {file_path}")
                loader = UnstructuredExcelLoader(file_path)
            else:
                logger.warning(f"未サポートのファイル形式: {file_ext}")
                return []
            
            documents = loader.load()
            logger.info(f"{len(documents)} ドキュメントが読み込まれました")
            return documents
            
        except Exception as e:
            logger.error(f"ファイル読み込み中にエラーが発生しました ({file_path}): {str(e)}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        ドキュメントを適切なサイズのチャンクに分割
        
        Args:
            documents: 分割するドキュメントのリスト
        
        Returns:
            分割されたドキュメントのリスト
        """
        try:
            splits = self.text_splitter.split_documents(documents)
            logger.info(f"ドキュメントが {len(splits)} チャンクに分割されました")
            return splits
        except Exception as e:
            logger.error(f"ドキュメント分割中にエラーが発生しました: {str(e)}")
            return documents  # エラーが発生した場合は元のドキュメントを返す
    
    def process_file(self, file_path: str) -> List[Document]:
        """
        ファイルを読み込んで適切なサイズに分割
        
        Args:
            file_path: 処理するファイルのパス
        
        Returns:
            処理されたドキュメントチャンクのリスト
        """
        # ファイル名をメタデータとして取得
        filename = os.path.basename(file_path)
        
        # ドキュメントを読み込む
        documents = self.load_document(file_path)
        
        # ファイル名をメタデータに追加
        for doc in documents:
            doc.metadata["filename"] = filename
        
        # 分割
        if documents:
            chunks = self.split_documents(documents)
            return chunks
        
        return []


def index_documents_directory(directory_path: str, db: VectorDatabase, embedder: BaseEmbedding, 
                             file_extensions: List[str] = ['.pdf', '.docx', '.txt', '.csv', '.xlsx']):
    """
    ディレクトリ内のすべてのサポートされているファイルをインデックス化
    
    Args:
        directory_path: インデックス化するドキュメントのディレクトリ
        db: ベクトルデータベース
        embedder: 埋め込みモデル
        file_extensions: インデックス化するファイル拡張子のリスト
    
    Returns:
        インデックス化されたドキュメントの数
    """
    logger.info(f"{directory_path} 内のドキュメントをインデックス化中...")
    
    processor = DocumentProcessor()
    directory = Path(directory_path)
    
    if not directory.exists() or not directory.is_dir():
        logger.error(f"ディレクトリが存在しません: {directory_path}")
        return 0
    
    # サポートされているファイルをスキャン
    files = []
    for ext in file_extensions:
        files.extend(list(directory.glob(f"**/*{ext}")))
    
    if not files:
        logger.warning(f"サポートされているファイルが見つかりませんでした: {file_extensions}")
        return 0
    
    logger.info(f"{len(files)} ファイルが見つかりました")
    
    total_chunks = 0
    for file_path in tqdm(files):
        try:
            # ファイルを処理
            chunks = processor.process_file(str(file_path))
            
            if not chunks:
                logger.warning(f"ファイルからチャンクを抽出できませんでした: {file_path}")
                continue
            
            # 各チャンクをインデックス化
            for chunk in chunks:
                # エンベディングを生成
                embedding = embedder.embed_text(chunk.page_content)
                
                # データベースに保存
                db.insert_document(
                    content=chunk.page_content,
                    metadata=chunk.metadata,
                    embedding=embedding
                )
                
                total_chunks += 1
            
            logger.info(f"{file_path} のインデックス化完了: {len(chunks)} チャンク")
            
        except Exception as e:
            logger.error(f"ファイルのインデックス化エラー {file_path}: {str(e)}")
    
    logger.info(f"全ドキュメントのインデックス化完了。合計 {total_chunks} チャンク")
    return total_chunks