# InferPilot

## Vision

Build an **autonomous LLM inference optimization system**.

Given:

**Model + Hardware + Workload + Performance/SLO requirements**

InferPilot should automatically determine, test, and eventually adapt the best way to run that model on the available hardware.

The long-term goal is:

> Give InferPilot a model and hardware environment, and make inference optimization largely the system's problem rather than the engineer's.

## Optimization Space

InferPilot may eventually reason about and optimize:

- KV-cache configuration and management
- batching
- scheduling
- chunked prefill
- prefix caching
- quantization
- GPU/CPU memory usage
- CUDA graphs / compilation
- model serving configuration
- tensor/pipeline parallelism
- kernels / Triton
- hardware-specific strategies

## Core Loop

Model + Hardware + Workload
→ establish baseline
→ collect profiling/serving metrics
→ diagnose bottleneck
→ generate optimization hypothesis
→ select experiment
→ modify configuration/code
→ benchmark
→ compare against baseline
→ accept/reject
→ store result
→ repeat

**Real benchmark results are ground truth.**

The AI must never claim an optimization worked simply because it sounds theoretically correct.

## Initial MVP

Start SMALL.

Initial environment:

- vLLM
- single GPU
- one small model
- reproducible workload
- automated experiment runner

Initial metrics:

- TTFT
- TPOT
- throughput
- p95/p99 latency
- GPU memory
- KV-cache utilization
- GPU utilization where practical

Initial optimization parameters should be a small subset of things such as:

- max_num_seqs
- max_num_batched_tokens
- gpu_memory_utilization
- KV-cache dtype
- prefix caching
- chunked prefill
- scheduler-related configuration
- compilation/CUDA graph configuration

Do NOT attempt the entire vision immediately.

## Hardware Constraint

Primary local machine:

- NVIDIA RTX 3050 Laptop GPU
- 4 GB VRAM

Therefore the architecture must support running experiments on different machines.

Development can happen locally with small models.

Later experiments can use:

- Kaggle/Colab
- rented GPUs such as RunPod/Vast
- 16/24/48/80 GB GPUs when necessary

Hardware discovery should eventually become an important InferPilot component.

The same optimizer should ideally discover **different optimal strategies for different hardware**.

## Research Direction

The most interesting differentiator is:

**workload-aware autonomous KV-cache + scheduling optimization.**

Long-term research question:

> Can an AI performance engineer autonomously discover and adapt LLM-serving strategies across different hardware and changing workloads while satisfying latency SLOs?

Potential later extensions:

- learned KV-cache policies
- workload classification
- Bayesian optimization/bandits/search
- learned cost models
- adaptive runtime optimization
- quantization selection
- kernel generation/optimization
- Triton
- multi-GPU serving

## Engineering Principles

1. Measure before optimizing.
2. Every optimization must have benchmark evidence.
3. Keep experiments reproducible.
4. Log configurations, environment and results.
5. Change one meaningful variable at a time when appropriate.
6. Separate AI reasoning from numerical optimization.
7. Do not use agents merely for the sake of using agents.
8. Prefer simple systems until complexity provides measurable value.
9. Preserve clean Git history.
10. Build this as a serious ML-systems/research project, not an AI wrapper.

## AI Responsibilities

### ChatGPT

Act primarily as:

- research lead
- systems architect
- inference mentor
- experiment designer
- research-paper analyst
- architecture/code reviewer
- benchmark methodology reviewer

Help decide **what should be built and why**.

Challenge weak assumptions and proposed optimizations.

### Claude Code

Act primarily as:

- implementation engineer
- repository agent
- debugger
- test runner
- profiling/benchmark implementation agent

Claude should implement agreed tasks directly inside the repository.

Before substantial changes Claude should briefly state:

1. what it will do
2. why
3. files affected
4. how success will be verified

Keep responses concise and avoid unnecessary token usage.

## Current State

This project is starting from the beginning.

Do not immediately build a huge agent architecture.

First objective:

**Build a minimal end-to-end autonomous optimization loop that works.**

baseline
→ experiment
→ benchmark
→ compare
→ record result
→ choose next experiment

Once that foundation is trustworthy, progressively add intelligence.