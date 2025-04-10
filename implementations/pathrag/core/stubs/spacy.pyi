"""
Type stubs for spaCy.
"""
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Iterator, Type, Sequence
import pathlib

T = TypeVar('T')

def load(model_name: str) -> 'Language':
    """Load a spaCy model."""
    ...

class Doc:
    """A processed document with tokens, entities, and linguistic annotations."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...
    
    @property
    def ents(self) -> Sequence['Span']:
        """Named entities found in the document."""
        ...
    
    @property
    def text(self) -> str:
        """The document text."""
        ...
    
    def __iter__(self) -> Iterator['Token']:
        """Iterate over tokens."""
        ...

class Span:
    """A slice from a Doc object."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...
    
    @property
    def text(self) -> str:
        """The span text."""
        ...
    
    @property
    def label_(self) -> str:
        """The span's label."""
        ...

class Token:
    """An individual token in a document."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...
    
    @property
    def text(self) -> str:
        """The token text."""
        ...

class Language:
    """Base language class."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...
    
    def __call__(self, text: str) -> 'Doc':
        """Process text with the pipeline."""
        ...
