import argparse
import logging
import sys
from pathlib import Path

from doc_qa_vss.pipeline import DocumentQASystem, setup_system
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
    
    # interactコマンド
    interact_parser = subparsers.add_parser("interact", help="対話型質問応答セッション")
    interact_parser.add_argument("--model", "-m", choices=["plamo", "ruri"], default="plamo", 
                                help="使用する埋め込みモデル (plamo または ruri)")
    interact_parser.add_argument("--use-llm", "-l", action="store_true", help="LLMを使って回答を生成")
    interact_parser.add_argument("--llm-model", default="matsuo-lab/ELYZA-japanese-Llama-2-7b-fast",
                               help="使用するLLMモデル名")
    interact_parser.add_argument("--db", default="docstore.db", help="DuckDBデータベースのパス")
    
    # askコマンド
    ask_parser = subparsers.add_parser("ask", help="単一の質問に回答")
    ask_parser.add_argument("question", help="質問")
    ask_parser.add_argument("--model", "-m", choices=["plamo", "ruri"], default="plamo", 
                           help="使用する埋め込みモデル (plamo または ruri)")
    ask_parser.add_argument("--use-llm", "-l", action="store_true", help="LLMを使って回答を生成")
    ask_parser.add_argument("--llm-model", default="matsuo-lab/ELYZA-japanese-Llama-2-7b-fast",
                          help="使用するLLMモデル名")
    ask_parser.add_argument("--db", default="docstore.db", help="DuckDBデータベースのパス")
    
    return parser.parse_args()

def interactive_session(qa_system):
    """対話型質問応答セッション"""
    print("\n=== ドキュメント質問応答システム ===")
    print("終了するには 'exit' または 'quit' と入力してください\n")
    
    while True:
        question = input("\n質問を入力してください: ")
        if question.lower() in ["exit", "quit"]:
            break
        
        print("\n回答を生成中...\n")
        
        try:
            result = qa_system.answer_question(question)
            
            # 結果の表示
            print(f"\n回答: {result['answer']}\n")
            print("参照ソース:")
            for i, source in enumerate(result['sources']):
                print(f"ソース {i+1} (類似度: {source['similarity']:.4f}, ページ: {source['page']})")
                print(f"ファイル: {source.get('filename', 'N/A')}")
                print(f"内容: {source['content']}\n")
        except Exception as e:
            logger.error(f"質問処理エラー: {str(e)}")
            print(f"エラーが発生しました: {str(e)}")

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
        
    elif args.command == "interact":
        # システム設定とQAシステムの初期化
        qa_system = DocumentQASystem.setup(
            model_name=args.model,
            db_path=args.db, 
            use_llm=args.use_llm,
            llm_model=args.llm_model if args.use_llm else None
        )
        
        # 対話型セッションの開始
        interactive_session(qa_system)
        
    elif args.command == "ask":
        # システム設定とQAシステムの初期化
        qa_system = DocumentQASystem.setup(
            model_name=args.model,
            db_path=args.db, 
            use_llm=args.use_llm,
            llm_model=args.llm_model if args.use_llm else None
        )
        
        # 質問への回答
        try:
            result = qa_system.answer_question(args.question)
            
            # 結果の表示
            print(f"\n回答: {result['answer']}\n")
            print("参照ソース:")
            for i, source in enumerate(result['sources']):
                print(f"ソース {i+1} (類似度: {source['similarity']:.4f}, ページ: {source['page']})")
                print(f"ファイル: {source.get('filename', 'N/A')}")
                print(f"内容: {source['content']}\n")
        except Exception as e:
            logger.error(f"質問処理エラー: {str(e)}")
            print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()
