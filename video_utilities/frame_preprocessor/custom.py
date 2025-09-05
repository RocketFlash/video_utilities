import numpy as np
from typing import Dict, Any, Callable
from .base import FramePreprocessor

class FunctionPreprocessor(FramePreprocessor):
    """Wrapper for custom preprocessing functions."""
    
    def __init__(self, func: Callable[[np.ndarray], np.ndarray], name: str = "custom", **kwargs):
        """
        Args:
            func: Function that takes frame and returns processed frame
            name: Name for this preprocessor
            **kwargs: Additional parameters to store in config
        """
        self.func = func
        self.name = name
        self.kwargs = kwargs
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.func(frame)
    
    def get_name(self) -> str:
        return self.name
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'FunctionPreprocessor',
            'name': self.name,
            'function': str(self.func),
            **self.kwargs
        }

class LambdaPreprocessor(FramePreprocessor):
    """Quick preprocessor using lambda functions."""
    
    def __init__(self, lambda_func: Callable[[np.ndarray], np.ndarray], name: str = "lambda"):
        self.lambda_func = lambda_func
        self.name = name
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.lambda_func(frame)
    
    def get_name(self) -> str:
        return self.name
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'LambdaPreprocessor',
            'name': self.name,
            'function': str(self.lambda_func)
        }

class NoOpPreprocessor(FramePreprocessor):
    """No-operation preprocessor that returns frame unchanged."""
    
    def __init__(self, name: str = "noop"):
        self.name = name
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return frame
    
    def get_name(self) -> str:
        return self.name
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'NoOpPreprocessor',
            'name': self.name
        }