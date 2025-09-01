import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """
    Classe base abstrata para todos os modelos do ecossistema.
    Força a implementação de métodos essenciais para um pipeline de treinamento padronizado.
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x):
        """
        Passagem forward do modelo.
        """
        pass

    @abstractmethod
    def load_from_pretrained(self, weights_path: str):
        """
        Carrega pesos pré-treinados de um arquivo.
        """
        pass

    @abstractmethod
    def get_loss(self, predictions, targets):
        """
        Calcula a perda (loss) para um determinado conjunto de predições e alvos.
        """
        pass

    @abstractmethod
    def get_optimizer(self, config):
        """
        Retorna o otimizador a ser usado no treinamento.
        """
        pass
