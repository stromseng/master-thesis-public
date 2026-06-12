from __future__ import annotations

from idun.vllm import build_serve_command


def test_build_serve_command_adds_tensor_parallel_by_default_for_vllm() -> None:
    cmd = build_serve_command("Qwen/Qwen2.5-7B-Instruct", tensor_parallel_size=2)

    assert "--tensor-parallel-size 2" in cmd


def test_build_serve_command_vllm_override_args_keep_auto_parallelism() -> None:
    cmd = build_serve_command(
        "meta-llama/Llama-3.1-70B-Instruct",
        tensor_parallel_size=8,
        override_args="--max-model-len 4096 --gpu-memory-utilization 0.7",
    )

    assert "--max-model-len 4096" in cmd
    assert "--gpu-memory-utilization 0.7" in cmd
    assert "--tensor-parallel-size 8" in cmd
    assert "--pipeline-parallel-size" not in cmd
    assert "--enable-chunked-prefill" not in cmd
    assert "--enforce-eager" not in cmd
    assert "--kv-cache-dtype fp8" not in cmd
    assert "--max-num-batched-tokens" not in cmd


def test_build_serve_command_sglang_still_adds_tp_without_explicit_override() -> None:
    cmd = build_serve_command(
        "Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=2,
        override_args="--context-length 8192",
        backend="sglang",
    )

    assert "--tp-size 2" in cmd
    assert "--context-length 8192" in cmd


def test_build_serve_command_vllm_override_args_respect_explicit_parallelism() -> None:
    cmd = build_serve_command(
        "meta-llama/Llama-3.1-70B-Instruct",
        tensor_parallel_size=8,
        override_args="--tensor-parallel-size 4 --max-model-len 4096",
    )

    assert "--tensor-parallel-size 4" in cmd
    assert "--pipeline-parallel-size" not in cmd


def test_build_serve_command_vllm_respects_equals_form_parallelism_override() -> None:
    cmd = build_serve_command(
        "meta-llama/Llama-3.1-70B-Instruct",
        tensor_parallel_size=8,
        override_args="--tensor-parallel-size=4 --max-model-len 4096",
    )

    assert "--tensor-parallel-size=4" in cmd
    assert "--pipeline-parallel-size" not in cmd


def test_build_serve_command_sglang_override_args_respect_explicit_parallelism() -> (
    None
):
    cmd = build_serve_command(
        "Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=2,
        override_args="--tp-size 4 --context-length 8192",
        backend="sglang",
    )

    assert "--tp-size 4" in cmd
    assert "--tp-size 2" not in cmd


def test_build_serve_command_sglang_respects_tp_alias_override() -> None:
    cmd = build_serve_command(
        "Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=2,
        override_args="--tp 4 --context-length 8192",
        backend="sglang",
    )

    assert "--tp 4" in cmd
    assert "--tp-size 2" not in cmd
