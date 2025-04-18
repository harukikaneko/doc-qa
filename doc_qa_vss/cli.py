import argparse
import logging
import sys

from doc_qa_vss.pipeline import setup_system
from doc_qa_vss.document.processor import index_documents_directory

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("doc_qa_vss.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="ドキュメント質問応答システム")
    subparsers = parser.add_subparsers(dest="command", help="コマンド")
    
    # indexコマンド
    index_parser = subparsers.add_parser("index", help="ドキュメントをインデックス化")
    index_parser.add_argument("--dir", "-d", required=True, help="インデックス化するドキュメントディレクトリ")
    index_parser.add_argument("--model", "-m", choices=["plamo", "ruri"], default="plamo", 
                             help="使用する埋め込みモデル (plamo または ruri)")
    index_parser.add_argument("--db", default="docstore.db", help="DuckDBデータベースのパス")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.command:
        print("コマンドを指定してください。詳細は --help を参照してください。")
        sys.exit(1)
    
    if args.command == "index":
        # システム設定
        conn, embedder = setup_system(args.model, args.db)
        
        # ドキュメントのインデックス化
        index_documents_directory(args.dir, conn, embedder)
        

if __name__ == "__main__":
    main()
