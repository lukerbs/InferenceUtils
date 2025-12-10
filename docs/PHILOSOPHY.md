# Philosophy

## The Problem We're Solving

**You shouldn't need a PhD in Computer Architecture to `pip install` an LLM.**

Yet today, running local AI feels like navigating a maze:

- "Does my GPU support CUDA 12.1 or 12.4?"
- "Should I use Metal, ROCm, or just give up and use CPU?"
- "Will this 30GB model crash my system 10 minutes into a conversation?"
- "Why is my inference running at 1 token/sec when others get 40?"

These aren't AI questions. These are **infrastructure questions** that distract from building useful applications.

---

## The Missing Middleware Layer

The local AI ecosystem is fragmented:

```
┌─────────────────────────────────────────┐
│   End-User Applications                 │
│   (Ollama, LM Studio, Jan, etc.)        │
│   ✓ Great UX, but closed black boxes    │
└─────────────────────────────────────────┘
                   ▲
                   │
                   │  ❌ GAP ❌
                   │  (No developer tooling)
                   │
                   ▼
┌─────────────────────────────────────────┐
│   Raw Inference Libraries                │
│   (llama.cpp, vLLM, MLX, TensorRT-LLM)  │
│   ✓ Powerful, but assume sysadmin skills│
└─────────────────────────────────────────┘
```

**EdgeKit fills the gap.** It's the **middleware** that brings the intelligence from closed applications into the hands of Python developers.

---

## Design Principles

### 1. Engineering Over Guesswork
"Can I run this model?" should be a calculation, not a coin flip. We simulate the full runtime environment (weights + KV cache + overhead) before loading to give you a **guaranteed answer**, not a hope.

### 2. Native APIs Over Shell Hacks
We query hardware through stable binary interfaces (Mach kernel, NVML, AMDSMI) instead of parsing `nvidia-smi` output. When NVIDIA releases a new driver, your app doesn't break.

### 3. Cross-Platform Consistency
Whether you're on a MacBook Pro, a Linux workstation with 4x A100s, or a Windows laptop with an Intel iGPU, you import one library and get one coherent API. No `if sys.platform == "darwin"` spaghetti code.

### 4. Type Safety as Documentation
Every return value is a Pydantic model. Your IDE autocompletes `hw.gpu.nvidia[0].compute_capability`. You get runtime validation. You can serialize to JSON for logging. Types aren't an afterthought; they're the interface.

---

## What This Enables

By solving the "boring" infrastructure problems, we unlock entirely new application patterns:

### Smart Installation Scripts
```python
from edgekit import install_command
print(f"Run: {install_command()}")
# Output: CMAKE_ARGS="-DGGML_METAL=on ..." pip install llama-cpp-python
```

Instead of 3-page README instructions that 40% of users misunderstand, you generate the exact command for their hardware.

### Adaptive Context Windows
```python
result = model_preflight("mlx-community/Llama-3-70B-8bit", engine="mlx")
# Recommended: 8,192 tokens (safe)
# Max Safe: 16,384 tokens (tight)
```

Your chatbot automatically adjusts context length based on available RAM, preventing OOM crashes during long conversations.

### Multi-Backend Applications
```python
engine = recommended_engine()
if engine.name == "MLX":
    from mlx_lm import generate
elif engine.name == "vLLM":
    from vllm import LLM
```

Write one codebase that auto-selects the optimal backend (MLX on Apple Silicon, vLLM on NVIDIA, llama.cpp everywhere else).

### CI/CD Preflight Checks
```python
# In your GitHub Actions workflow
assert model_preflight("Qwen/Qwen2.5-72B", engine="vllm").can_load
```

Fail deployments **before** they reach production if the model won't fit on your instance type.

---

## Who This Is For

### App Developers
You're building a CLI tool, a Discord bot, or a web API. You just want to load an LLM without writing 500 lines of hardware detection code.

### Library Authors
You maintain an LLM wrapper library. Instead of telling users "make sure you have enough VRAM," you call `model_preflight()` and give them an actual answer.

### DevOps Engineers
You're deploying LLM inference services. You need reproducible builds and memory guarantees, not artisanal `CMAKE_ARGS` incantations.

### Power Users
You want to squeeze every bit of performance from your hardware. You need to know if that 70B parameter model will *actually* fit with a 16K context, not a guess.

---

## The Long-Term Vision

Right now, every serious LLM application re-implements this logic:

- Ollama does it in Go
- LM Studio does it in C++
- Every Hugging Face demo notebook has hand-coded VRAM checks
- Every llama.cpp tutorial has platform-specific build instructions

**This should be a solved problem.**

By centralizing this logic in a well-maintained, cross-platform Python library, we eliminate thousands of hours of duplicated effort across the ecosystem.

---

## Not Just a Library—An Infrastructure Layer

EdgeKit isn't trying to be an inference engine. It's the **standard library for local AI**—the layer that makes everything else work reliably.

When you import `requests`, you don't think about TCP sockets.  
When you import `pandas`, you don't think about CSV parsing edge cases.  
When you import `edgekit`, you shouldn't have to think about CUDA compute capabilities.

**We handle the infrastructure. You build the application.**

---

> *"The best tool is the one you forget you're using because it just works."*

That's EdgeKit.
