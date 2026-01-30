from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib

class KnowledgeType(str, Enum):
    NARRATIVE = "narrative"
    FACTUAL = "factual"
    QUOTE = "quote"
    MIXED = "mixed"

@dataclass
class SourceInfo:
    file: str
    offset_start: int
    offset_end: int
    checksum: str = ""

@dataclass
class ChunkContext:
    chunk_id: str
    index: int
    context: Dict[str, Any] = field(default_factory=dict)
    text: str = ""

@dataclass
class ChunkFlags:
    verbatim: bool = False
    requires_llm: bool = True

@dataclass
class ProcessedChunk:
    type: KnowledgeType
    source: SourceInfo
    chunk: ChunkContext
    data: Dict[str, Any] = field(default_factory=dict) # triples, events, verbatim_entities
    flags: ChunkFlags = field(default_factory=ChunkFlags)

    def to_dict(self) -> Dict:
        return {
            "type": self.type.value,
            "source": {
                "file": self.source.file,
                "offset_start": self.source.offset_start,
                "offset_end": self.source.offset_end,
                "checksum": self.source.checksum
            },
            "chunk": {
                "text": self.chunk.text,
                "chunk_id": self.chunk.chunk_id,
                "index": self.chunk.index,
                "context": self.chunk.context
            },
            "data": self.data,
            "flags": {
                "verbatim": self.flags.verbatim,
                "requires_llm": self.flags.requires_llm
            }
        }

class BaseStrategy(ABC):
    def __init__(self, filename: str):
        self.filename = filename

    @abstractmethod
    def split(self, text: str) -> List[ProcessedChunk]:
        """Split text into chunks based on strategy."""
        pass

    @abstractmethod
    async def extract(self, chunk: ProcessedChunk, llm_func=None) -> ProcessedChunk:
        """Extract information from the chunk."""
        pass

    def calculate_checksum(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
