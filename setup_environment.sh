#!/bin/bash
#
# Script para configurar o ambiente de desenvolvimento para o projeto de IA.
#
# Este script irá:
# 1. Criar um ambiente virtual Python chamado 'venv'.
# 2. Ativar o ambiente virtual.
# 3. Instalar a versão do PyTorch compatível com CUDA 12.1.
#    (Edite a URL para corresponder à sua versão do CUDA ou para CPU/Metal).
# 4. Instalar todas as outras dependências do requirements.txt.

set -e # Encerra o script imediatamente se um comando falhar.

# --- Verificação de Comandos ---
if ! command -v python3 &> /dev/null
then
    echo "ERRO: python3 não encontrado. Por favor, instale Python 3.8 ou superior."
    exit 1
fi

if ! command -v pip3 &> /dev/null
then
    echo "ERRO: pip3 não encontrado. Por favor, instale o pip para Python 3."
    exit 1
fi

# --- Criação do Ambiente Virtual ---
echo "[1/4] Criando ambiente virtual Python em ./venv..."
python3 -m venv venv
source venv/bin/activate
echo "Ambiente virtual criado e ativado."

# --- Instalação do PyTorch ---
# A linha abaixo é para sistemas com GPU NVIDIA e CUDA 12.1.
# Para outras versões do CUDA, ou para CPU/Metal, acesse: https://pytorch.org/get-started/locally/
#
# Exemplo para Apple Metal (macOS):
# pip3 install torch torchvision torchaudio
#
# Exemplo para CPU-only:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "[2/4] Instalando PyTorch com suporte para CUDA 12.1..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "PyTorch instalado com sucesso."

# --- Instalação das Dependências ---
echo "[3/4] Instalando dependências do requirements.txt..."
pip3 install -r requirements.txt
echo "Dependências instaladas com sucesso."

# --- Conclusão ---
echo -e "\n[4/4] Configuração do ambiente concluída!"
echo "--------------------------------------------------"
echo "Para ativar o ambiente virtual, execute:"
echo "source venv/bin/activate"
echo "--------------------------------------------------"
echo "Para desativar, simplesmente execute: deactivate"
echo "--------------------------------------------------"
