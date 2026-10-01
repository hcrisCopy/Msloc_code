"""按视频并行准备，按输入顺序返回；只有主线程提交进度记录。"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice


def ordered_video_results(process_video, rows, workers: int):
    """参考 Python concurrent.futures；限制在途任务，避免积压整个数据集。"""
    if workers < 1:
        raise ValueError("--workers 必须大于等于 1")
    if workers == 1:
        for row in rows:
            yield process_video(row)
        return

    source = iter(rows)
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prepare-video")
    pending = deque()
    try:
        for row in islice(source, workers):
            pending.append(executor.submit(process_video, row))
        while pending:
            # result() 原样抛出异常；不能跳过失败的视频。按顺序提交可沿用续建记录。
            yield pending.popleft().result()
            for row in islice(source, 1):
                pending.append(executor.submit(process_video, row))
    finally:
        # 中断或异常后不再启动排队任务；已运行任务结束后释放解码器和编码器。
        executor.shutdown(wait=True, cancel_futures=True)
