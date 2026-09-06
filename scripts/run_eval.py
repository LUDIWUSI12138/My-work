"""自动化评估与 A/B 对比脚本。

用法：
  python scripts/run_eval.py                  # 跑一次评估，打印指标
  python scripts/run_eval.py --ab             # A/B：对比 few-shot 开关 / 不同 TOP_K
  python scripts/run_eval.py --output out.json # 保存明细到文件
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.chain import RAGChain
from rag.vector_store import get_store_manager
from evaluation.evalset import EVAL_SET
from evaluation.metrics import compute_metrics


def run_question(chain: RAGChain, q: str, use_few_shot: bool = True) -> str:
    try:
        result = chain.query(q, include_few_shot=use_few_shot)
        return result.answer
    except Exception as e:  # noqa: BLE001
        return f"[错误] {e}"


def run_all(chain: RAGChain, use_few_shot: bool = True) -> list:
    results = []
    for item in EVAL_SET:
        answer = run_question(chain, item["question"], use_few_shot)
        row = {**item, "answer": answer, "baseline_answer": item.get("reference", "")}
        results.append(row)
    return results


def pretty(metrics: dict) -> str:
    return (
        f"样本数={metrics['total']}\n"
        f"  回答准确率(命中关键词) = {metrics['answer_accuracy']}\n"
        f"  拒答正确率(unanswerable) = {metrics['refusal_rate']}\n"
        f"  幻觉率 = {metrics['hallucination_rate']}\n"
        f"  明细: factual={metrics['factual_count']} relevant={metrics['relevant_count']} "
        f"unanswerable={metrics['unanswerable_count']} refuse_ok={metrics['refuse_ok_count']} "
        f"fabricated={metrics['fabricated_count']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 系统评估")
    parser.add_argument("--ab", action="store_true", help="执行 A/B 对比")
    parser.add_argument("--output", type=str, default="", help="保存明细 JSON 到文件")
    parser.add_argument("--topk-a", type=int, default=3)
    parser.add_argument("--topk-b", type=int, default=6)
    args = parser.parse_args()

    chain = RAGChain(manager=get_store_manager())

    if args.ab:
        print("=== A/B：Few-Shot 开 / 关 ===")
        print("[A] few_shot=True")
        mA = compute_metrics(run_all(chain, use_few_shot=True))
        print(pretty(mA))
        print("[B] few_shot=False")
        mB = compute_metrics(run_all(chain, use_few_shot=False))
        print(pretty(mB))

        print("\n=== A/B：TOP_K 灵敏性 ===")
        for k in (args.topk_a, args.topk_b):
            score_docs = chain.retriever.retrieve("试用期多久？", k=k)
            print(f"[TOP_K={k}] 检索到 {len(score_docs)} 个相关片段")
        raise SystemExit(0)

    results = run_all(chain)
    metrics = compute_metrics(results)
    print(pretty(metrics))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)
        print(f"\n明细已保存: {args.output}")


if __name__ == "__main__":
    main()