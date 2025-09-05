import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any

class FramePreprocessor(ABC):
    """Abstract base class for deterministic frame preprocessing."""
    
    @abstractmethod
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame deterministically."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get preprocessor name for logging."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for metadata."""
        pass
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({self.get_config()})"