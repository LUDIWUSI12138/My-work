"""评估工具：人工标注 + 指标计算（准确率 / 相关性 / 拒答能力 / 幻觉概率）。"""
from __future__ import annotations

from typing import Dict, List


def compute_metrics(results: List[Dict]) -> Dict[str, float]:
    """输入 results，每项包含:
    - question, answer, keywords, category, expected
    输出汇总指标。
    """
    n = len(results)
    if n == 0:
        return {}

    def hit_keywords(answer: str) -> bool:
        ans = answer or ""
        # 命中过半关键词视为语义对齐
        hits = sum(1 for kw in item["keywords"] if kw in ans)
        return hits >= max(1, len(item["keywords"]) // 2)

    # 相关率：factual 类且命中关键词
    factual = [r for r in results if r["category"] == "factual"]
    relevant = sum(1 for r in factual if hit_keywords(r)) if factual else 0

    # 拒答正确率：unanswerable 类应包含拒答信号
    unans = [r for r in results if r["category"] == "unanswerable"]
    refuse_ok = 0
    refuse_sigs = ("未提及", "资料", "没有", "未覆盖", "无法", "不包含")
    for r in unans:
        ans = r.get("answer", "") or ""
        if any(sig in ans for sig in refuse_sigs):
            refuse_ok += 1

    # 幻觉率量化：unanswerable 类若 AI 强行给出具体数值/事实则为疑似幻觉
    def looks_fabricated(r: Dict, ans: str) -> bool:
        if r["category"] != "unanswerable":
            return False
        # 包含具体承诺数字或明确结论，却无拒答信号 → 疑似幻觉
        has_number = any(ch.isdigit() for ch in ans)
        no_refusal = not any(sig in ans for sig in refuse_sigs)
        return has_number and no_refusal

    fabrication = sum(1 for r in results if looks_fabricated(r, r.get("answer", "")))

    return {
        "total": n,
        "answer_accuracy": round(relevant / len(factual), 4) if factual else 0.0,
        "refusal_rate": round(refuse_ok / len(unans), 4) if unans else 1.0,
        "hallucination_rate": round(fabrication / len(unans), 4) if unans else 0.0,
        "factual_count": len(factual),
        "unanswerable_count": len(unans),
        "relevant_count": relevant,
        "refuse_ok_count": refuse_ok,
        "fabricated_count": fabrication,
    }


def human_review_display(row: Dict) -> str:
    return (
        f"Q: {row['question']}\n"
        f"A: {row.get('answer', '')[:200]}\n"
        f"类别: {row['category']} | 关键词: {row['keywords']}"
    )