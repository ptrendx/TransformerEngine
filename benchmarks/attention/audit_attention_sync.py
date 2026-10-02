# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Audit CUDA host waits inside THD attention, excluding setup/harness waits."""

import argparse
import contextlib
import json
import os
from pathlib import Path


def extract_waits(path):
    """Count runtime/driver blocking APIs within each named CPU call range."""
    events = json.loads(Path(path).read_text())["traceEvents"]
    ranges = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and event.get("name", "").startswith("attention_sync/")
    ]
    result = []
    for call in ranges:
        waits = [
            event
            for event in events
            if event.get("ph") == "X"
            and event.get("cat") in ("cuda_runtime", "cuda_driver")
            and call["ts"] <= event["ts"] < call["ts"] + call["dur"]
            and (
                "Synchronize" in event["name"]
                or event["name"]
                in ("cudaMemcpy", "cudaMemcpy2D", "cudaMemcpy3D", "cuMemcpyDtoH", "cuMemcpyDtoH_v2")
            )
        ]
        result.append(dict(call=call["name"], waits=[event["name"] for event in waits]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["fused", "flash"], default="fused")
    parser.add_argument("--flash-version", choices=[2, 3, 4], type=int, default=2)
    parser.add_argument("--mode", choices=["inference", "train"], default="inference")
    parser.add_argument("--padded", action="store_true")
    parser.add_argument("--same-cu", action="store_true")
    parser.add_argument("--policies", choices=["none", "prepared", "fresh"], default="none")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for key, value in dict(
        NVTE_FUSED_ATTN=args.backend == "fused",
        NVTE_FLASH_ATTN=args.backend == "flash",
        NVTE_UNFUSED_ATTN=0,
        NVTE_FLASH_ATTN_V2=args.flash_version == 2,
        NVTE_FLASH_ATTN_V3=args.flash_version == 3,
        NVTE_FLASH_ATTN_V4=args.flash_version == 4,
        NVTE_ALLOW_NONDETERMINISTIC_ALGO=1,
        NVTE_NVTX_ENABLED=0,
    ).items():
        os.environ[key] = str(int(value))
    import torch
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex

    if args.backend == "flash" and args.flash_version == 2 and args.padded:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                dict(
                    status="skipped",
                    reason="FlashAttention 2 does not support padding between THD sequences",
                    args={**vars(args), "out": str(args.out)},
                ),
                indent=2,
            )
            + "\n"
        )
        print("Skipped: FlashAttention 2 does not support padded THD")
        return
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    qlengths = [63, 117, 31]
    klengths = qlengths if args.same_cu else [95, 129, 47]

    def cumulative(lengths):
        result = [0]
        for length in lengths:
            result.append(result[-1] + length)
        return result

    qp = [((length + 63) // 64) * 64 for length in qlengths] if args.padded else qlengths
    kp = [((length + 63) // 64) * 64 for length in klengths] if args.padded else klengths
    qo, ko = cumulative(qp), cumulative(kp)

    def tensor(values):
        return torch.tensor(values, device="cuda", dtype=torch.int32)

    cuq = tensor(cumulative(qlengths))
    cuk = cuq if args.same_cu else tensor(cumulative(klengths))
    kwargs = dict(cu_seqlens_q=cuq, cu_seqlens_kv=cuk, pad_between_seqs=args.padded)
    if args.padded:
        kwargs.update(cu_seqlens_q_padded=tensor(qo), cu_seqlens_kv_padded=tensor(ko))
    training = args.mode == "train"
    q, k, v = [
        torch.randn(tokens, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=training)
        for tokens in (qo[-1], ko[-1], ko[-1])
    ]
    module = (
        te.DotProductAttention(
            4, 64, qkv_format="thd", attn_mask_type="padding", attention_dropout=0
        )
        .cuda()
        .train(training)
    )

    def policies():
        return [
            dict(
                sequence_ids=torch.tensor(ids, device="cuda", dtype=torch.int64),
                mask_type="padding",
                window_size=(-1, -1),
            )
            for ids in ([0, 2], [1])
        ]

    maxq, maxk = ((max(qp) + 63) // 64 * 64), ((max(kp) + 63) // 64 * 64)
    supplies = dict(both=(maxq, maxk), q=(maxq, None), kv=(None, maxk), neither=(None, None))
    grad = torch.randn(qo[-1], 4 * 64, device="cuda", dtype=torch.bfloat16)

    def step(supply, policy=None):
        mq, mk = supplies[supply]
        result = module(
            q, k, v, max_seqlen_q=mq, max_seqlen_kv=mk, thd_attention_policies=policy, **kwargs
        )
        return (result, *torch.autograd.grad(result, (q, k, v), grad)) if training else (result,)

    context = contextlib.nullcontext if training else torch.no_grad
    selected = []
    handle = getattr(
        module, "fused_attention" if args.backend == "fused" else "flash_attention"
    ).register_forward_pre_hook(lambda *_unused: selected.append(True))
    with context():
        baseline = step("both")
    handle.remove()
    assert selected, "Requested backend was not invoked"
    # Independent reference: ignore physical padding while comparing logical outputs
    # and gradients, so padded offsets are checked rather than assumed contiguous.
    refs = [t.detach().float().requires_grad_(training) for t in (q, k, v)]
    from torch.nn.attention import SDPBackend, sdpa_kernel

    outputs = []
    with sdpa_kernel(SDPBackend.MATH):
        for index in range(3):
            qs = refs[0][qo[index] : qo[index] + qlengths[index]].transpose(0, 1).unsqueeze(0)
            ks, vs = [
                t[ko[index] : ko[index] + klengths[index]].transpose(0, 1).unsqueeze(0)
                for t in refs[1:]
            ]
            outputs.append(
                torch.nn.functional.scaled_dot_product_attention(qs, ks, vs)
                .squeeze(0)
                .transpose(0, 1)
                .flatten(1)
            )
    ref_output = torch.cat(outputs)
    qindices = torch.cat(
        [torch.arange(qo[index], qo[index] + qlengths[index], device="cuda") for index in range(3)]
    )
    expected = (ref_output,)
    if training:
        expected += torch.autograd.grad(ref_output, refs, grad.index_select(0, qindices).float())

    def check(result):
        torch.testing.assert_close(
            result[0].index_select(0, qindices).float(), expected[0], atol=0.025, rtol=0.025
        )
        for actual, reference in zip(result[1:], expected[1:]):
            torch.testing.assert_close(actual.float(), reference, atol=0.025, rtol=0.025)

    check(baseline)
    prepared = (
        te.DotProductAttention.prepare_thd_attention_policies(
            policies(),
            3,
            **{key: value for key, value in kwargs.items() if key != "pad_between_seqs"},
        )
        if args.policies == "prepared"
        else None
    )
    with context():
        for supply in supplies:
            policy = (
                prepared
                if args.policies == "prepared"
                else (policies() if args.policies == "fresh" else None)
            )
            check(step(supply, policy))
            for _ in range(15):
                step(supply, policy)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            for supply in supplies:
                for repeat in range(3):
                    # Fresh IDs are prepared outside the attention call budget.
                    policy = (
                        te.DotProductAttention.prepare_thd_attention_policies(
                            policies(),
                            3,
                            **{
                                key: value
                                for key, value in kwargs.items()
                                if key != "pad_between_seqs"
                            },
                        )
                        if args.policies == "prepared"
                        else (policies() if args.policies == "fresh" else None)
                    )
                    with torch.profiler.record_function(f"attention_sync/{supply}/{repeat}"):
                        step(supply, policy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace = args.out.with_suffix(".trace.json")
    profiler.export_chrome_trace(str(trace))
    calls = extract_waits(trace)
    assert len(calls) == 12, calls
    for call in calls:
        supply = call["call"].split("/")[1]
        expected_waits = int(supply != "both") + int(args.policies == "fresh")
        if args.backend == "flash" and args.flash_version == 2 and args.policies == "fresh":
            expected_waits += 4
        assert len(call["waits"]) == expected_waits, call
    result = dict(
        args={**vars(args), "out": str(args.out)},
        calls=calls,
        trace=str(trace),
        qlengths=qlengths,
        klengths=klengths,
        qoffsets=qo,
        koffsets=ko,
        dtype="bfloat16",
        shape=[list(t.shape) for t in (q, k, v)],
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        extension=tex.__file__,
        env={key: value for key, value in os.environ.items() if key.startswith("NVTE_")},
        correctness="FP32 outputs and gradients passed",
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(calls, indent=2))


if __name__ == "__main__":
    main()
