import logging
import os
from markitdown import MarkItDown
from pathlib import Path
from tqdm import tqdm
from anthropic import Anthropic
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter

from doc_qa_vss.db.vector_db import VectorDatabase
from doc_qa_vss.models.embedding import BaseEmbedding

logger = logging.getLogger(__name__)

load_dotenv()

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
    
    def process_markdown(self, markdown_text: str, metadata: dict = None) -> list:
        """
        Markdownテキストを処理してチャンクに分割
        
        Args:
            markdown_text: 処理するMarkdownテキスト
            metadata: チャンクに追加するメタデータ
        
        Returns:
            チャンクのリスト
        """
        if metadata is None:
            metadata = {}
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""]
        )
        
        chunks = []
        texts = text_splitter.split_text(markdown_text)
        
        for i, text in enumerate(texts):
            # 新しいメタデータを作成（元のメタデータをコピー）
            chunk_metadata = metadata.copy()
            # チャンク番号を追加
            chunk_metadata["chunk"] = i
            
            # Document オブジェクトを作成（langchainの Document クラスを想定）
            from langchain.docstore.document import Document
            chunk = Document(page_content=text, metadata=chunk_metadata)
            chunks.append(chunk)
        
        return chunks


def index_documents_directory(directory_path: str, db: VectorDatabase, embedder: BaseEmbedding) -> int:
    """
    ディレクトリ内のすべてのmarkitdownでサポートされているファイルをインデックス化
    
    Args:
        directory_path: インデックス化するドキュメントのディレクトリ
        db: ベクトルデータベース
        embedder: 埋め込みモデル
    
    Returns:
        インデックス化されたドキュメントの数
    """
    logger = logging.getLogger(__name__)
    logger.info(f"{directory_path} 内のドキュメントをインデックス化中...")
    
    processor = DocumentProcessor()
    directory = Path(directory_path)
    
    if not directory.exists() or not directory.is_dir():
        logger.error(f"ディレクトリが存在しません: {directory_path}")
        return 0
    
    # すべてのファイルを取得 (markitdownが処理できるかは後でチェック)
    files = list(directory.glob("**/*.*"))
    
    if not files:
        logger.warning(f"ディレクトリ内にファイルが見つかりませんでした")
        return 0
    
    logger.info(f"{len(files)} ファイルが見つかりました。markitdownで処理可能なファイルをインデックス化します")
    
    total_chunks = 0
    for file_path in tqdm(files):
        try:
            # markitdownを使用してファイルをMarkdownに変換
            # markitdownがサポートしていないファイルタイプの場合は例外が発生するので、
            # その場合は次のファイルに進む
            md_content = convert_to_markdown(str(file_path))
            
            # Markdownコンテンツから処理チャンクを取得
            chunks = processor.process_markdown(md_content, metadata={"source": str(file_path)})
            
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
            # markitdownがサポートしていないファイルタイプや処理エラーの場合
            logger.debug(f"ファイル {file_path} はスキップされました: {str(e)}")
            continue
    
    logger.info(f"全ドキュメントのインデックス化完了。合計 {total_chunks} チャンク")
    return total_chunks


def index_document_content(content: str, db: VectorDatabase, embedder: BaseEmbedding, metadata: dict = None) -> int:
    """
    任意のコンテンツをインデックス化
    
    Args:
        content: インデックス化するコンテンツ
        db: ベクトルデータベース
        embedder: 埋め込みモデル
        metadata: メタデータ（オプション）
    
    Returns:
        インデックス化されたドキュメtントのID
    """
    logger = logging.getLogger(__name__)
    logger.info("コンテンツをインデックス化中...")

    # エンベディングを生成
    embedding = embedder.embed_text(content)
    
    # データベースに保存
    doc_id = db.insert_document(
        content=content,
        metadata=metadata,
        embedding=embedding
    )
    
    logger.info(f"インデックス化完了")
    return doc_id

def convert_to_markdown(file_path: str) -> str:
    """
    任意のファイルタイプをMarkdownに変換
    
    Args:
        file_path: 変換するファイルのパス
    
    Returns:
        Markdown形式のテキスト
    """
    logger = logging.getLogger(__name__)
    logger.info(f"ファイルをMarkdownに変換中: {file_path}")
    # markitdownを使用してファイルをMarkdownに変換
    try:
        if is_image_file(file_path):
            client = Anthropic(api_key=os.getenv("OPENAI_API_KEY", ""))
            md = MarkItDown(llm_client=client, llm_model="gpt-4o")
            result = md.convert(file_path)
            logger.info(result)
            return result.text_content
        else:
            md = MarkItDown()
            result = md.convert(file_path)
            return result.text_content
    except Exception as e:
        logger.error(f"Markdownへの変換エラー {file_path}: {str(e)}")
        # エラーの場合は空の文字列を返す代わりに例外を再発生させる
        raise

def is_image_file(file_path: str) -> bool:
    """
    ファイルが画像かどうかを判定
    
    Args:
        file_path: 判定するファイルのパス
    
    Returns:
        画像ファイルの場合はTrue、それ以外はFalse
    """
    # 一般的な画像ファイルの拡張子リスト
    image_extensions = ['.jpg', '.jpeg', '.png']
    _, ext = os.path.splitext(file_path.lower())
    return ext in image_extensions