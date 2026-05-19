"""HybridPdM - Compute Efficiency 측정 스크립트.

각 RUL 모델(BiLSTM / DLinear / iTransformer)에 대해:
  - Parameter count (total / trainable)
  - Inference latency (mean / std / p50 / p95)
  - Memory:
      * CPU: psutil RSS delta (PyTorch C/C++ allocation 포함)
      * GPU: torch.cuda.max_memory_allocated (VRAM 정확)
  - Per-sample latency, throughput

설계 결정 (두 차례 코드리뷰 반영):
  1. CPU: psutil RSS delta — tracemalloc 은 Python heap 한정, PyTorch 미반영
  2. GPU: torch.cuda.max_memory_allocated — psutil RSS 는 VRAM 못 봄
  3. Memory baseline 측정 직전 gc.collect() — allocator 안정화
  4. 측정 중 gc.disable() — latency spike 방지
  5. torch.set_num_threads() 는 --single-thread 옵션 (기본은 OS 기본값, 운영 환경 반영)
  6. CUDA 환경에서 torch.cuda.synchronize() (async execution wall-clock 정확화)
  7. Random init weight 측정 (FLOPs 가 weight 값과 무관 — 학술 표준)
  8. MLflow run_name 은 dataset_key 가 모델별로 다르므로 자동 차별화 (run_id 충돌 없음)
  9. peak_delta = max(rss_now - rss_before) — "baseline 대비 추가 메모리" 의도

사용:
    python -m scripts.training.compute_efficiency
    python -m scripts.training.compute_efficiency --batch-size 64 --n-iters 50
    python -m scripts.training.compute_efficiency --single-thread  # 재현성 우선
    python -m scripts.training.compute_efficiency --device cuda     # GPU 측정
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from models_core import config
from models_core import models
from scripts.training import mlflow_logger as mlf


# ---------------------------------------------------------------------------
# 모델 인스턴스 헬퍼 (학습된 가중치 불필요 — random init 도 FLOPs 동일)
# ---------------------------------------------------------------------------

def build_models_for_benchmark(input_dim: int, seq_len: int) -> Dict[str, torch.nn.Module]:
    """3가지 RUL 모델을 동일 입력 차원으로 빌드."""
    out: Dict[str, torch.nn.Module] = {}

    out["bilstm"] = models.build_model(
        "lstm",
        input_dim=input_dim,
        hidden=config.LSTM_CFG["hidden"],
        num_layers=config.LSTM_CFG["num_layers"],
        dropout=config.LSTM_CFG["dropout"],
        input_format="BFL",
    )
    out["dlinear"] = models.build_model(
        "dlinear",
        input_dim=input_dim,
        seq_len=seq_len,
        kernel_size=config.DLINEAR_CFG["kernel_size"],
        individual=config.DLINEAR_CFG["individual"],
        input_format="BFL",
    )
    out["itransformer"] = models.build_model(
        "itransformer",
        input_dim=input_dim,
        seq_len=seq_len,
        d_model=config.ITRANSFORMER_CFG["d_model"],
        n_heads=config.ITRANSFORMER_CFG["n_heads"],
        n_layers=config.ITRANSFORMER_CFG["n_layers"],
        d_ffn=config.ITRANSFORMER_CFG["d_ffn"],
        dropout=config.ITRANSFORMER_CFG["dropout"],
        input_format="BFL",
    )
    return out


# ---------------------------------------------------------------------------
# 측정 헬퍼
# ---------------------------------------------------------------------------

def count_params(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def _maybe_sync(device: str) -> None:
    """CUDA async execution 동기화. CPU 에선 no-op."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    input_dim: int,
    seq_len: int,
    batch_size: int = 32,
    n_iters: int = 30,
    warmup: int = 5,
    device: str = "cpu",
) -> Dict[str, float]:
    """배치 추론 latency 측정 (mean / std / p50 / p95)."""
    model.eval()
    model.to(device)

    x = torch.randn(batch_size, input_dim, seq_len, device=device)

    # warm-up (JIT, cache, allocation 안정화)
    for _ in range(warmup):
        _ = model(x)
        _maybe_sync(device)

    # 측정 중 GC 중단 (spike 방지)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        times: List[float] = []
        for _ in range(n_iters):
            _maybe_sync(device)
            t0 = time.perf_counter()
            _ = model(x)
            _maybe_sync(device)
            times.append(time.perf_counter() - t0)
    finally:
        if gc_was_enabled:
            gc.enable()

    arr = np.array(times, dtype=np.float64)
    return {
        "batch_size":           batch_size,
        "n_iters":              n_iters,
        "latency_mean_ms":      float(arr.mean() * 1000),
        "latency_std_ms":       float(arr.std() * 1000),
        "latency_p50_ms":       float(np.percentile(arr, 50) * 1000),
        "latency_p95_ms":       float(np.percentile(arr, 95) * 1000),
        "per_sample_ms":        float(arr.mean() * 1000 / batch_size),
        "throughput_per_sec":   float(batch_size / arr.mean()),
    }


def measure_memory_mb(
    model: torch.nn.Module,
    input_dim: int,
    seq_len: int,
    batch_size: int = 32,
    device: str = "cpu",
    n_forward: int = 5,
) -> Dict[str, float]:
    """디바이스별 메모리 측정.

    CPU: psutil RSS delta (PyTorch C/C++ allocation 포함)
        - tracemalloc 은 Python heap 한정 → PyTorch 미반영
        - 측정 직전 gc.collect() 로 allocator 정리

    GPU: torch.cuda.max_memory_allocated (VRAM 정확)
        - PyTorch 공식 API, peak VRAM 정확 측정
    """
    # 모델 파라미터 byte (모든 device 공통)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mb = param_bytes / (1024 ** 2)

    model.eval()
    model.to(device)
    x = torch.randn(batch_size, input_dim, seq_len, device=device)

    # ── GPU 분기 ────────────────────────────────────────────────────────
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(n_forward):
                _ = model(x)
                torch.cuda.synchronize()
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return {
            "param_mb": float(param_mb),
            "peak_memory_mb": float(peak_vram),
            "activation_delta_mb": float(peak_vram - param_mb),
            "measurement": "cuda.max_memory_allocated",
        }

    # ── CPU 분기 (psutil) ───────────────────────────────────────────────
    if not _HAS_PSUTIL:
        return {
            "param_mb": float(param_mb),
            "peak_memory_mb": -1.0,
            "activation_delta_mb": -1.0,
            "measurement": "unavailable_psutil_missing",
        }

    proc = psutil.Process()
    # baseline 측정 전 allocator 정리
    gc.collect()
    rss_before = proc.memory_info().rss

    peak_delta = 0
    with torch.no_grad():
        for _ in range(n_forward):
            _ = model(x)
            rss_now = proc.memory_info().rss
            peak_delta = max(peak_delta, rss_now - rss_before)

    rss_after = proc.memory_info().rss
    return {
        "param_mb": float(param_mb),
        "peak_memory_mb": float(rss_after / (1024 ** 2)),
        "activation_delta_mb": float(peak_delta / (1024 ** 2)),
        "measurement": "psutil_rss",
    }


# ---------------------------------------------------------------------------
# 벤치마크 본체
# ---------------------------------------------------------------------------

def run_benchmark(
    input_dim: int,
    seq_len: int,
    batch_size: int,
    n_iters: int,
    label: str,
    run_id: str,
    device: str = "cpu",
) -> List[Dict]:
    """3 모델을 모두 측정해 결과 list 반환."""
    print(
        f"\n=== Benchmark: {label} "
        f"(F={input_dim}, L={seq_len}, batch={batch_size}, iters={n_iters}, device={device}) ==="
    )
    header = (
        f"{'model':<14}{'params':>12}{'lat_mean(ms)':>14}{'p95(ms)':>10}"
        f"{'per_smp(ms)':>13}{'thr/sec':>10}{'param_MB':>10}{'act_dMB':>10}"
    )
    print(header)
    print("-" * len(header))

    bench_models = build_models_for_benchmark(input_dim, seq_len)
    results: List[Dict] = []

    for name, m in bench_models.items():
        pc = count_params(m)
        lat = measure_latency(
            m, input_dim, seq_len,
            batch_size=batch_size, n_iters=n_iters, device=device,
        )
        mem = measure_memory_mb(
            m, input_dim, seq_len,
            batch_size=batch_size, device=device,
        )

        row = {
            "label":               label,
            "model":               name,
            "device":              device,
            "input_dim":           input_dim,
            "seq_len":             seq_len,
            "params_total":        pc["total"],
            "params_trainable":    pc["trainable"],
            "latency_mean_ms":     lat["latency_mean_ms"],
            "latency_std_ms":      lat["latency_std_ms"],
            "latency_p50_ms":      lat["latency_p50_ms"],
            "latency_p95_ms":      lat["latency_p95_ms"],
            "per_sample_ms":       lat["per_sample_ms"],
            "throughput_per_sec":  lat["throughput_per_sec"],
            "param_mb":            mem["param_mb"],
            "peak_memory_mb":      mem.get("peak_memory_mb", -1.0),
            "activation_delta_mb": mem.get("activation_delta_mb", -1.0),
            "memory_measurement":  mem.get("measurement", "unknown"),
        }
        results.append(row)

        print(
            f"{name:<14}{pc['total']:>12,}"
            f"{lat['latency_mean_ms']:>10.2f}±{lat['latency_std_ms']:<3.1f}"
            f"{lat['latency_p95_ms']:>10.2f}"
            f"{lat['per_sample_ms']:>13.3f}"
            f"{lat['throughput_per_sec']:>10.1f}"
            f"{mem['param_mb']:>10.2f}"
            f"{mem.get('activation_delta_mb', -1.0):>10.2f}"
        )

        # MLflow 기록 (run_name 은 dataset_key + run_id 로 자동 차별화)
        with mlf.start_run(
            dataset_key=f"compute_eff_{label}_{name}",
            run_id=run_id,
            experiment_name="hybridpdm_efficiency",
            tags={
                "benchmark_label": label,
                "model_family":    name,
                "device":          device,
            },
        ):
            mlf.log_params({
                "input_dim":  input_dim,
                "seq_len":    seq_len,
                "batch_size": batch_size,
                "n_iters":    n_iters,
                "model":      name,
                "device":     device,
            })
            mlf.log_metrics({k: v for k, v in row.items() if isinstance(v, (int, float))})

        del m
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HybridPdM Compute Efficiency Benchmark")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-iters", type=int, default=30)
    parser.add_argument(
        "--single-thread", action="store_true",
        help="torch.set_num_threads(1) — 재현성 우선 (기본은 OS 기본값, 운영 환경 반영)",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="추론 디바이스 (기본 cpu)",
    )
    args = parser.parse_args()

    config.set_seed(42)
    if args.single_thread:
        torch.set_num_threads(1)
        print("[compute_efficiency] torch.set_num_threads(1) — 단일 스레드 측정")

    if not _HAS_PSUTIL and args.device == "cpu":
        print("[compute_efficiency] WARNING: psutil 미설치 — activation_delta_mb 측정 불가")
        print("                       pip install psutil 권장")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[compute_efficiency] WARNING: --device cuda 지정했으나 CUDA 사용 불가, CPU 로 fallback")
        args.device = "cpu"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[compute_efficiency] run_id = {run_id}")
    print(f"[compute_efficiency] device = {args.device}")
    print(f"[compute_efficiency] batch  = {args.batch_size}, iters = {args.n_iters}")

    all_results: List[Dict] = []

    # C-MAPSS 차원 (F=14, L=30)
    all_results.extend(run_benchmark(
        input_dim=14, seq_len=30,
        batch_size=args.batch_size, n_iters=args.n_iters,
        label="C-MAPSS",
        run_id=run_id,
        device=args.device,
    ))

    # N-CMAPSS 차원 (F=43, L=30)
    all_results.extend(run_benchmark(
        input_dim=43, seq_len=30,
        batch_size=args.batch_size, n_iters=args.n_iters,
        label="N-CMAPSS",
        run_id=run_id,
        device=args.device,
    ))

    out_path = config.REPORT_DIR / f"compute_efficiency_{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[compute_efficiency] report saved -> {out_path}")


if __name__ == "__main__":
    main()
