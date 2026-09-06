"""命令行接入脚本：把本地文档目录写入知识库。

用法：
  python scripts/ingest.py docs/                 # 递归导入 docs 目录
  python scripts/ingest.py path/to/file.pdf      # 导入单个文件
  python scripts/ingest.py --source ref.pdf      # 指定来源名（去重用）
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.chain import RAGChain
from rag.vector_store import get_store_manager


def main() -> None:
    parser = argparse.ArgumentParser(description="文档接入脚本")
    parser.add_argument("path", help="文件或目录路径")
    parser.add_argument("--source", help="源名（文件名，用于去重）")
    args = parser.parse_args()

    chain = RAGChain(manager=get_store_manager())
    if os.path.isdir(args.path):
        info = chain.ingest_files_from_dir(args.path)
        print(f"已导入 {info['sources']}，共 {info['chunks']} 个片段")
    else:
        if not os.path.exists(args.path):
            print(f"文件不存在: {args.path}")
            raise SystemExit(1)
        info = chain.ingest_file(args.path, doc_id=args.source)
        print(f"已导入 {info['source']}，共 {info['chunks']} 个片段")


if __name__ == "__main__":
    main()