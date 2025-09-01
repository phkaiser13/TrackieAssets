# Trackie Assets

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Repositório central para modelos, pesos, pipelines e ferramentas de integração do ecossistema Trackie.

Este repositório documenta e armazena os ativos para os sistemas **TrackieLINK** (visão computacional e sensores) e **TrackieLLM** (modelos de linguagem). Ele serve como a fonte da verdade para:
- Tipos de modelos e suas arquiteturas.
- Passos de treinamento, fine-tuning e avaliação.
- Pipelines de conversão e quantização (ONNX, GGUF, TensorRT).
- Estratégias de deployment para múltiplas plataformas (CUDA, ROCm, Metal, CPU).
- Bindings e stubs de integração em C++ e Rust.

## Core Features

- **Modelos Multimodais:** Uma coleção curada de modelos para visão, ASR, TTS, OCR e LLMs.
- **Otimização para Performance:** Foco em runtimes de alta performance em C++/Rust, com otimizações para hardware específico.
- **Reprodutibilidade:** Inclui templates, scripts e documentação para garantir que os processos de treinamento e exportação sejam reprodutíveis.
- **CI/CD Integrado:** Estruturado para suportar automação de builds, testes e publicação de artefatos.

## Estrutura do Repositório

- **`.docs/`**: Documentação de design, políticas e model cards detalhados.
- **`Link/` e `TLLM/`**: Código fonte para os wrappers, bindings e runtimes dos sistemas TrackieLINK e TrackieLLM.
- **`common/`**: Scripts, templates e configurações de CI compartilhados.
- **`examples/`**: Exemplos de inferência mínimos e executáveis para cada modelo.
- **`tests-data/`**: Pequenos artefatos para execução de testes unitários e de integração.

## Getting Started

Para começar, explore os exemplos de inferência em `examples/`. Cada exemplo contém um `README.md` com instruções de build e execução.

- **`examples/link_infer`**: Demonstra como executar um modelo de visão (ONNX) em C++.
- **`examples/tllm_infer`**: Demonstra como executar um modelo de linguagem (GGUF) em Rust com uma API de streaming.

## Licença

Este projeto é licenciado sob a licença Apache-2.0. Veja o arquivo `LICENSE` para mais detalhes.
