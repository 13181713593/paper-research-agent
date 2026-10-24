"""同一批本地 PDF 的串行/多进程对照实验，不重复下载或调用收费 API。"""

from __future__ import annotations

import asyncio
import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from paper_agent.adapters.markdown_parser import CONVERTER_VERSION, PdfMarkdownParser
from paper_agent.domain.models import Paper


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


async def _measure(paths: list[Path], output: Path, workers: int) -> dict:
    started = perf_counter()
    semaphore = asyncio.Semaphore(workers)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        parser = PdfMarkdownParser(backend="pypdf", executor=pool)

        async def one(index, path):
            async with semaphore:
                try:
                    # 每份文件使用全新输出目录，不读取正常业务的解析缓存。
                    markdown, _ = await parser.parse(path, output / str(index))
                    return await asyncio.to_thread(_digest, markdown)
                except Exception:  # 失败计入成功率，不删掉失败样本美化结果。 # noqa: BLE001
                    return None

        hashes = await asyncio.gather(*(one(i, p) for i, p in enumerate(paths)))
    elapsed = perf_counter() - started
    successes = sum(value is not None for value in hashes)
    return {
        "workers": workers,
        "seconds": elapsed,
        "successes": successes,
        "total": len(paths),
        "success_rate": successes / len(paths),
        "papers_per_second": successes / elapsed if elapsed else 0,
        "output_sha256": hashes,
    }


async def benchmark_pdf_parsing(
    papers: list[Paper],
    data_dir: str | Path,
    *,
    workers: int = 4,
    rounds: int = 2,
) -> dict:
    """交替运行 A/B 和 B/A，使用中位数；不保证小样本一定加速。"""
    if not 2 <= workers <= 8 or not 2 <= rounds <= 5:
        raise ValueError("并行进程数需为 2 至 8，对照轮数需为 2 至 5。")
    root = Path(data_dir).resolve()
    paths, dataset, seen = [], [], set()
    for paper in papers:
        if not paper.fulltext.local_pdf_path:
            continue
        path = Path(paper.fulltext.local_pdf_path).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() != ".pdf" or not path.is_file():
            raise ValueError("仅能测试当前数据目录内的本地 PDF。")
        digest = await asyncio.to_thread(_digest, path)
        if digest in seen:
            continue
        seen.add(digest)
        paths.append(path)
        dataset.append({"paper_id": str(paper.paper_id), "title": paper.title, "sha256": digest})
    if not 2 <= len(paths) <= 20:
        raise ValueError("需要至少两份不同内容的本地 PDF，最多 20 份；请先完成论文下载。")
    benchmark_dir = root / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    measurements = []
    with TemporaryDirectory(prefix="parse-", dir=benchmark_dir) as temporary:
        for index in range(rounds):
            order = [1, workers] if index % 2 == 0 else [workers, 1]
            for count in order:
                measurement = await _measure(
                    paths, Path(temporary) / f"round-{index}-workers-{count}", count
                )
                measurements.append({"round": index + 1, **measurement})
    after = [await asyncio.to_thread(_digest, path) for path in paths]
    input_unchanged = after == [entry["sha256"] for entry in dataset]
    baseline = measurements[0]["output_sha256"]
    comparable = input_unchanged and all(
        row["successes"] == len(paths) and row["output_sha256"] == baseline for row in measurements
    )
    serial = median(row["seconds"] for row in measurements if row["workers"] == 1)
    parallel = median(row["seconds"] for row in measurements if row["workers"] == workers)
    return {
        "scope": "本地 PDF → Markdown 解析；不含检索、下载和 LLM 总结",
        "backend": "pypdf",
        "converter_version": CONVERTER_VERSION,
        "cache_policy": "每轮独立空输出目录，禁用解析产物复用；操作系统文件缓存不可控",
        "timing_policy": "包含进程池启动和关闭；交替执行顺序，取各模式耗时中位数",
        "rounds": rounds,
        "workers": workers,
        "dataset": dataset,
        "measurements": measurements,
        "input_unchanged": input_unchanged,
        "comparable": comparable,
        "serial_seconds": serial,
        "parallel_seconds": parallel,
        "speedup": serial / parallel if comparable and parallel > 0 else None,
    }


def benchmark_to_markdown(result: dict) -> str:
    speedup = result["speedup"]
    lines = [
        "# 第三组并发优化对照实验",
        "",
        f"- 测试范围：{result['scope']}",
        f"- 后端：{result['backend']} / {result['converter_version']}",
        f"- 样本数：{len(result['dataset'])}，重复对照：{result['rounds']} 轮",
        f"- 缓存策略：{result['cache_policy']}",
        f"- 计时规则：{result['timing_policy']}",
        "",
        "| 模式 | 中位耗时（秒） |",
        "| --- | ---: |",
        f"| 串行（1 进程） | {result['serial_seconds']:.3f} |",
        f"| 并行（{result['workers']} 进程） | {result['parallel_seconds']:.3f} |",
        "",
    ]
    if speedup is None:
        lines.append("存在失败、输入变化或输出不一致，本轮不计算加速比。")
    else:
        lines.append(f"加速比（串行耗时 / 并行耗时）：{speedup:.2f}×。")
        lines.append("大于 1 才表示加速；小文件/少量论文可能因进程启动开销反而变慢。")
    lines.extend(
        [
            "",
            "## 各轮实测",
            "",
            "| 轮次 | 进程数 | 秒 | 成功数/总数 |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for row in result["measurements"]:
        lines.append(
            f"| {row['round']} | {row['workers']} | {row['seconds']:.3f} | "
            f"{row['successes']}/{row['total']} |"
        )
    lines.extend(["", "## 样本指纹（SHA-256）", ""])
    lines.extend(f"- {entry['paper_id']}：`{entry['sha256']}`" for entry in result["dataset"])
    return "\n".join(lines) + "\n"
