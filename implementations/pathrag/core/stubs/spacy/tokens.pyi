"""
Type stubs for spacy.tokens module.
"""
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Iterator

T = TypeVar('T')

class Doc:
    """spaCy Doc class."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...
    
    def __iter__(self) -> Iterator['Token']:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __getitem__(self, key: Union[int, slice]) -> Union['Token', 'Span']:
        ...

class Token:
    """spaCy Token class."""
    i: int
    idx: int
    text: str
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

class Span:
    """spaCy Span class."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...
    
    def __iter__(self) -> Iterator[Token]:
        ...
    
    def __len__(self) -> int:
        ...
