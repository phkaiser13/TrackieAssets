#!/bin/bash
#
# Script para clonar e compilar os motores de inferência Llama.cpp e Whisper.cpp
# com suporte otimizado para GPU (CUDA ou Metal).

set -e # Encerra o script imediatamente se um comando falhar.

echo "--- Iniciando setup dos motores de inferência (Llama.cpp e Whisper.cpp) ---"

# --- Llama.cpp ---
echo -e "\n[1/2] Configurando Llama.cpp..."
if [ -d "llama.cpp" ]; then
    echo "Diretório llama.cpp já existe. Fazendo git pull para atualizar..."
    cd llama.cpp
    git pull
    cd ..
else
    echo "Clonando repositório Llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp
# Limpa builds anteriores para garantir uma compilação nova
make clean

# Detecta o sistema e compila com o backend de GPU apropriado
echo "Detectando sistema para compilação otimizada do Llama.cpp..."
if [[ "$(uname)" == "Darwin" ]]; then
    echo "-> Sistema macOS detectado. Compilando com suporte a Metal (Apple GPU)."
    make LLAMA_METAL=1 -j
elif [[ "$(uname)" == "Linux" ]] && command -v nvcc &> /dev/null; then
    echo "-> Sistema Linux com NVCC detectado. Compilando com suporte a CUDA (NVIDIA GPU)."
    # Adiciona flags para compilar para arquiteturas de GPU comuns
    make LLAMA_CUDA=1 -j
else
    echo "-> Nenhum backend de GPU específico detectado. Compilando versão padrão para CPU."
    make -j
fi
echo "Llama.cpp compilado com sucesso."
cd ..

# --- Whisper.cpp ---
echo -e "\n[2/2] Configurando Whisper.cpp..."
if [ -d "whisper.cpp" ]; then
    echo "Diretório whisper.cpp já existe. Fazendo git pull para atualizar..."
    cd whisper.cpp
    git pull
    cd ..
else
    echo "Clonando repositório Whisper.cpp..."
    git clone https://github.com/ggerganov/whisper.cpp.git
fi

cd whisper.cpp
make clean

# Reutiliza a mesma lógica de detecção
echo "Detectando sistema para compilação otimizada do Whisper.cpp..."
if [[ "$(uname)" == "Darwin" ]]; then
    echo "-> Sistema macOS detectado. Compilando com suporte a Metal (Apple GPU)."
    make WHISPER_METAL=1 -j
elif [[ "$(uname)" == "Linux" ]] && command -v nvcc &> /dev/null; then
    echo "-> Sistema Linux com NVCC detectado. Compilando com suporte a CUDA (NVIDIA GPU)."
    make WHISPER_CUDA=1 -j
else
    echo "-> Nenhum backend de GPU específico detectado. Compilando versão padrão para CPU."
    make -j
fi
echo "Whisper.cpp compilado com sucesso."
cd ..

echo -e "\n\n--- Setup dos motores de inferência concluído! ---"
echo "Para usar, os principais executáveis são:"
echo "  - Llama: ./llama.cpp/main"
echo "  - Whisper: ./whisper.cpp/main"
echo "----------------------------------------------------"
