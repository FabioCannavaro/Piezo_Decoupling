"""
Benchmark a trained model for:
- Parameters
- FLOPs
- Inference latency

Recommended paper setting:
- batch size = 1
- same GPU for every channel configuration
- same precision for every configuration
- warm-up before timing
- report mean ± SD

Example:
python benchmark_model.py --root "day15_2ch" --run_dir "outputs/..." --fold 1
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataset import SpectrogramDataset
from src.train import build_model


def count_parameters(model):
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    return total


def get_sample_shape(root):
    ds = SpectrogramDataset(
        root,
        augment=False,
        stats=None,
        index_json=None,
    )

    if len(ds.image_paths) == 0:
        raise RuntimeError(
            f"No samples found under: {root}"
        )

    arr = np.load(
        ds.image_paths[0]
    )

    if arr.ndim == 2:
        arr = arr[
            None,
            ...,
        ]

    if arr.ndim != 3:
        raise ValueError(
            f"Expected (C,H,W), got {arr.shape}"
        )

    return tuple(
        int(v)
        for v in arr.shape
    ), ds


def calculate_flops(
    model,
    sample_input,
):
    try:
        from fvcore.nn import (
            FlopCountAnalysis,
        )
    except ImportError:
        return None, (
            "fvcore is not installed. "
            "Run: pip install fvcore"
        )

    try:
        analyzer = FlopCountAnalysis(
            model,
            sample_input,
        )

        analyzer.unsupported_ops_warnings(
            False
        )
        analyzer.uncalled_modules_warnings(
            False
        )

        flops = float(
            analyzer.total()
        )

        return flops, None

    except Exception as e:
        return None, str(e)


@torch.no_grad()
def measure_latency(
    model,
    sample_input,
    device,
    warmup: int = 100,
    repeat: int = 500,
    use_amp: bool = True,
):
    model.eval()

    amp_enabled = bool(
        use_amp
        and device.type == "cuda"
    )

    for _ in range(warmup):
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            _ = model(
                sample_input
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []

    if device.type == "cuda":
        starter = torch.cuda.Event(
            enable_timing=True
        )
        ender = torch.cuda.Event(
            enable_timing=True
        )

        for _ in range(repeat):
            starter.record()

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                _ = model(
                    sample_input
                )

            ender.record()
            torch.cuda.synchronize()

            times_ms.append(
                float(
                    starter.elapsed_time(
                        ender
                    )
                )
            )

    else:
        for _ in range(repeat):
            t0 = time.perf_counter()

            _ = model(
                sample_input
            )

            t1 = time.perf_counter()

            times_ms.append(
                (t1 - t0)
                * 1000.0
            )

    times_ms = np.asarray(
        times_ms,
        dtype=np.float64,
    )

    return {
        "latency_mean_ms": float(
            times_ms.mean()
        ),
        "latency_sd_ms": float(
            times_ms.std(ddof=1)
            if len(times_ms) > 1
            else 0.0
        ),
        "latency_median_ms": float(
            np.median(times_ms)
        ),
        "latency_p95_ms": float(
            np.percentile(
                times_ms,
                95,
            )
        ),
        "latency_min_ms": float(
            times_ms.min()
        ),
        "latency_max_ms": float(
            times_ms.max()
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Params, FLOPs, "
            "and inference latency."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
    )

    parser.add_argument(
        "--run_dir",
        required=True,
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    parser.add_argument(
        "--no_amp",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    if (
        not args.cpu
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is unavailable. "
            "Use --cpu only if CPU latency "
            "is intentionally being reported."
        )

    device = torch.device(
        "cpu"
        if args.cpu
        else "cuda:0"
    )

    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(
            "high"
        )

    (
        sample_shape,
        ds,
    ) = get_sample_shape(
        args.root
    )

    in_chans, height, width = (
        sample_shape
    )

    num_classes = int(
        ds.labels.max()
    ) + 1

    checkpoint_path = os.path.join(
        args.run_dir,
        f"fold{args.fold}",
        "best_model.pth",
    )

    if not os.path.isfile(
        checkpoint_path
    ):
        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    checkpoint_stats = checkpoint.get(
        "stats"
    )

    if checkpoint_stats is None:
        raise KeyError(
            "Checkpoint has no 'stats'."
        )

    checkpoint_in_chans = len(
        checkpoint_stats["mean"]
    )

    if checkpoint_in_chans != in_chans:
        raise ValueError(
            "Dataset/checkpoint channel mismatch: "
            f"dataset={in_chans}, "
            f"checkpoint={checkpoint_in_chans}"
        )

    head_dropout = float(
        checkpoint.get(
            "head_dropout",
            0.3,
        )
    )

    model = build_model(
        num_classes=num_classes,
        in_chans=in_chans,
        head_dropout=head_dropout,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    x = torch.zeros(
        (
            1,
            in_chans,
            height,
            width,
        ),
        dtype=torch.float32,
        device=device,
    )

    total_params = count_parameters(
        model
    )

    flops, flop_error = (
        calculate_flops(
            model=model,
            sample_input=x,
        )
    )

    latency = measure_latency(
        model=model,
        sample_input=x,
        device=device,
        warmup=args.warmup,
        repeat=args.repeat,
        use_amp=(
            not args.no_amp
        ),
    )

    result = {
        "dataset": os.path.basename(
            os.path.normpath(
                args.root
            )
        ),
        "run_dir": os.path.abspath(
            args.run_dir
        ),
        "fold": args.fold,
        "device": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else "CPU"
        ),
        "precision": (
            "AMP_FP16"
            if (
                device.type == "cuda"
                and not args.no_amp
            )
            else "FP32"
        ),
        "batch_size": 1,
        "input_channels": in_chans,
        "input_height": height,
        "input_width": width,
        "num_classes": num_classes,
        "params": int(
            total_params
        ),
        "params_m": float(
            total_params / 1e6
        ),
        "flops": (
            float(flops)
            if flops is not None
            else np.nan
        ),
        "gflops": (
            float(flops / 1e9)
            if flops is not None
            else np.nan
        ),
        "warmup": args.warmup,
        "repeat": args.repeat,
        **latency,
    }

    output_path = (
        args.output
        or os.path.join(
            args.run_dir,
            f"benchmark_fold{args.fold}.csv",
        )
    )

    pd.DataFrame(
        [result]
    ).to_csv(
        output_path,
        index=False,
    )

    json_path = str(
        Path(output_path).with_suffix(
            ".json"
        )
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )

    print(
        "\n========== MODEL BENCHMARK =========="
    )
    print(
        f"Dataset          : {result['dataset']}"
    )
    print(
        f"Device           : {result['device']}"
    )
    print(
        f"Precision        : {result['precision']}"
    )
    print(
        f"Input            : "
        f"(1, {in_chans}, {height}, {width})"
    )
    print(
        f"Params           : "
        f"{result['params_m']:.4f} M"
    )

    if flops is not None:
        print(
            f"FLOPs            : "
            f"{result['gflops']:.4f} GFLOPs"
        )
    else:
        print(
            "FLOPs            : unavailable"
        )
        print(
            f"Reason           : {flop_error}"
        )

    print(
        f"Latency          : "
        f"{result['latency_mean_ms']:.4f} "
        f"± {result['latency_sd_ms']:.4f} ms/sample"
    )
    print(
        f"Median / P95     : "
        f"{result['latency_median_ms']:.4f} / "
        f"{result['latency_p95_ms']:.4f} ms"
    )
    print(
        f"Saved CSV        : {output_path}"
    )


if __name__ == "__main__":
    main()
