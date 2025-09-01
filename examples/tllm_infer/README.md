# Example: TrackieLLM Rust/C++ Inference

This directory contains a minimal example for running streaming inference with a TrackieLLM model (e.g., `Mistral-7B`).

## Building and Running

This example uses Cargo to build the Rust application, which may link to a C++ core via FFI.

```bash
# Build the example
cargo build --release

# Run inference
./target/release/infer_tllm --model /path/to/mistral-7b.gguf --prompt "Hello, world"
```

See `main.rs` for the implementation, which calls the core TrackieLLM runtime.
