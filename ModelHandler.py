from abc import ABC, abstractmethod
                   
class ModelHandler(ABC):
    def __init__(self):
        super(ModelHandler, self).__init__()
        pass

    @abstractmethod
    def read_config(self):
        return NotImplemented

    @abstractmethod
    def load_model(self):
        return NotImplemented

    @abstractmethod
    def preprocessing(self):
        return NotImplemented