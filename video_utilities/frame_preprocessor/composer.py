import numpy as np
import logging
from typing import List, Dict, Any, Union
from .base import FramePreprocessor

logger = logging.getLogger(__name__)

class PreprocessorComposer(FramePreprocessor):
    """Compose multiple preprocessors into a single pipeline."""
    
    def __init__(self, preprocessors: List[FramePreprocessor], name: str = "composed"):
        """
        Initialize composer with list of preprocessors.
        
        Args:
            preprocessors: List of preprocessors to apply in order
            name: Name for this composition
        """
        if not preprocessors:
            raise ValueError("At least one preprocessor must be provided")
        
        self.preprocessors = preprocessors
        self.name = name
        
        # Validate all items are preprocessors
        for i, preprocessor in enumerate(preprocessors):
            if not isinstance(preprocessor, FramePreprocessor):
                raise TypeError(f"Item {i} is not a FramePreprocessor: {type(preprocessor)}")
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Apply all preprocessors in sequence."""
        processed_frame = frame
        
        for preprocessor in self.preprocessors:
            try:
                processed_frame = preprocessor(processed_frame)
            except Exception as e:
                logger.error(f"Preprocessor {preprocessor.get_name()} failed: {e}")
                raise RuntimeError(f"Preprocessing failed at {preprocessor.get_name()}: {e}")
        
        return processed_frame
    
    def get_name(self) -> str:
        """Get composer name."""
        return self.name
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration including all sub-preprocessors."""
        return {
            'type': 'PreprocessorComposer',
            'name': self.name,
            'preprocessors': [p.get_config() for p in self.preprocessors],
            'count': len(self.preprocessors)
        }
    
    def add_preprocessor(self, preprocessor: FramePreprocessor):
        """Add a preprocessor to the end of the pipeline."""
        if not isinstance(preprocessor, FramePreprocessor):
            raise TypeError(f"Must be a FramePreprocessor: {type(preprocessor)}")
        self.preprocessors.append(preprocessor)
    
    def insert_preprocessor(self, index: int, preprocessor: FramePreprocessor):
        """Insert a preprocessor at specific index."""
        if not isinstance(preprocessor, FramePreprocessor):
            raise TypeError(f"Must be a FramePreprocessor: {type(preprocessor)}")
        self.preprocessors.insert(index, preprocessor)
    
    def remove_preprocessor(self, index: int):
        """Remove preprocessor at index."""
        if 0 <= index < len(self.preprocessors):
            self.preprocessors.pop(index)
        else:
            raise IndexError(f"Index {index} out of range")
    
    def get_preprocessor_names(self) -> List[str]:
        """Get names of all preprocessors in order."""
        return [p.get_name() for p in self.preprocessors]
    
    def __len__(self) -> int:
        """Number of preprocessors in composition."""
        return len(self.preprocessors)
    
    def __getitem__(self, index: int) -> FramePreprocessor:
        """Get preprocessor by index."""
        return self.preprocessors[index]

class ConditionalPreprocessor(FramePreprocessor):
    """Apply preprocessor conditionally based on frame properties."""
    
    def __init__(self, 
                 preprocessor: FramePreprocessor,
                 condition_func: callable,
                 name: str = "conditional"):
        """
        Args:
            preprocessor: Preprocessor to apply conditionally
            condition_func: Function that takes frame and returns bool
            name: Name for this conditional preprocessor
        """
        self.preprocessor = preprocessor
        self.condition_func = condition_func
        self.name = name
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessor only if condition is met."""
        if self.condition_func(frame):
            return self.preprocessor(frame)
        return frame
    
    def get_name(self) -> str:
        return f"{self.name}_conditional"
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'ConditionalPreprocessor',
            'name': self.name,
            'preprocessor': self.preprocessor.get_config(),
            'condition': str(self.condition_func)
        }