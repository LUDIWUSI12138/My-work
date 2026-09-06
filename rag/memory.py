"""长期记忆：跨会话记住用户画像、偏好与最近交流，增强陪伴感。

存储为 data/memory/{user_id}.json，包含：
- profile: 用户画像键值（名字、偏好等，可运行时沉淀）
- notes:  助手沉淀的观察/备注（如情绪、关注点）
- history: 最近的问答交互（滚动保留，用于跨会话延续话题）
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self, memory_dir: str | None = None) -> None:
        self.memory_dir = memory_dir or settings.memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)

    def _path(self, user_id: str) -> str:
        safe = "".join(c for c in user_id if c.isalnum() or c in "-_") or "default"
        return os.path.join(self.memory_dir, f"{safe}.json")

    def load(self, user_id: str) -> Dict[str, Any]:
        p = self._path(user_id)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data.setdefault("profile", {})
                data.setdefault("notes", [])
                data.setdefault("history", [])
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("读取记忆失败 %s: %s", p, e)
        return {"profile": {}, "notes": [], "history": []}

    def _save(self, user_id: str, data: Dict[str, Any]) -> None:
        p = self._path(user_id)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ---- 基础写操作 ----
    def remember(self, user_id: str, key: str, value: Any) -> None:
        data = self.load(user_id)
        data["profile"][key] = value
        self._save(user_id, data)

    def add_note(self, user_id: str, note: str, max_notes: int = 20) -> None:
        data = self.load(user_id)
        data["notes"].append(note)
        data["notes"] = data["notes"][-max_notes:]
        self._save(user_id, data)

    def push_exchange(self, user_id: str, q: str, a: str, max_rounds: int = 8) -> None:
        data = self.load(user_id)
        data["history"].append({"q": q, "a": a})
        data["history"] = data["history"][-max_rounds:]
        self._save(user_id, data)

    def clear(self, user_id: str) -> None:
        p = self._path(user_id)
        if os.path.exists(p):
            os.remove(p)

    def list_users(self) -> List[str]:
        if not os.path.isdir(self.memory_dir):
            return []
        return [f for f in os.listdir(self.memory_dir) if f.endswith(".json")]

    # ---- 供提示词注入的格式 ----
    def snapshot(self, user_id: str) -> str:
        """把记忆整理成一段可注入提示词的上下文文本。"""
        data = self.load(user_id)
        lines: List[str] = []
        if data["profile"]:
            lines.append("用户画像：" + "；".join(f"{k}={v}" for k, v in data["profile"].items()))
        if data["notes"]:
            lines.append("备注：" + "；".join(data["notes"][-6:]))
        if data["history"]:
            recent = data["history"][-4:]
            turns = []
            for h in recent:
                turns.append(f"（之前你说过：{h['q']} → 我当时回答：{h['a']}）")
            lines.append("历史互动：" + " ".join(turns))
        return "\n".join(lines) if lines else "（暂无记忆）"


_default_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    global _default_store
    if _default_store is None:
        _default_store = MemoryStore()
    return _default_store