"""教师预检与 OPSD 短跑共用的固定抽样规则。"""

from __future__ import annotations

import hashlib


def select_ids(items: list[tuple[str, str]], limit: int) -> set[str]:
    """按 seed 42 固定抽取相同数量的 fake/real；-1 表示全量。"""
    ids = [item_id for item_id, _ in items]
    if len(ids) != len(set(ids)):
        raise ValueError("proposal ID 重复")
    if limit == -1:
        return set(ids)
    if limit < 2 or limit > len(items):
        raise ValueError(f"抽样数量须为 2 到 {len(items)}，或 -1 表示全量")
    by_kind = {kind: [] for kind in ("fake", "real")}
    for item_id, kind in items:
        if kind not in by_kind:
            raise ValueError(f"未知片段类别：{kind}")
        by_kind[kind].append(item_id)
    fake_count = min(len(by_kind["fake"]), max(1, limit // 2))
    real_count = limit - fake_count
    if real_count > len(by_kind["real"]):
        real_count = len(by_kind["real"])
        fake_count = limit - real_count
    if fake_count < 1 or real_count < 1 or fake_count > len(by_kind["fake"]):
        raise ValueError("抽样必须同时包含 fake 和 real proposal")

    def rank(item_id: str) -> bytes:
        return hashlib.sha256(f"42:{item_id}".encode("utf-8")).digest()

    return set(sorted(by_kind["fake"], key=rank)[:fake_count]
               + sorted(by_kind["real"], key=rank)[:real_count])


def ids_sha256(ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()
