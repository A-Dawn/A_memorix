import re
from typing import List, Dict, Any
from .base import BaseStrategy, ProcessedChunk, KnowledgeType, SourceInfo, ChunkContext

class FactualStrategy(BaseStrategy):
    def split(self, text: str) -> List[ProcessedChunk]:
        # Structure-aware splitting
        lines = text.split('\n')
        chunks = []
        current_chunk_lines = []
        current_len = 0
        target_size = 600
        
        for i, line in enumerate(lines):
            # Check if we should split
            # Don't split if current block is list item or definition or table row
            is_structure = self._is_structural_line(line)
            
            current_len += len(line) + 1
            current_chunk_lines.append(line)
            
            # If we exceeded size and are NOT in a tight structural block (or just force split if too huge)
            if current_len >= target_size and not is_structure:
                 chunks.append(self._create_chunk(current_chunk_lines, len(chunks)))
                 current_chunk_lines = []
                 current_len = 0
            elif current_len >= target_size * 2: # Force split if way too big
                 chunks.append(self._create_chunk(current_chunk_lines, len(chunks)))
                 current_chunk_lines = []
                 current_len = 0

        if current_chunk_lines:
            chunks.append(self._create_chunk(current_chunk_lines, len(chunks)))
            
        return chunks

    def _is_structural_line(self, line: str) -> bool:
        line = line.strip()
        if not line: return False
        # List items
        if re.match(r'^[\-\*]\s+', line) or re.match(r'^\d+\.\s+', line):
            return True
        # Definitions (Term: Definition)
        if re.match(r'^[^：:]+[：:].+', line):
            return True
        # Table rows (assumed markdown)
        if line.startswith('|') and line.endswith('|'):
            return True
        return False

    def _create_chunk(self, lines: List[str], index: int) -> ProcessedChunk:
        text = "\n".join(lines)
        return ProcessedChunk(
            type=KnowledgeType.FACTUAL,
            source=SourceInfo(
                file=self.filename,
                offset_start=0, # Simplified, tracking real offset requires more state
                offset_end=0,
                checksum=self.calculate_checksum(text)
            ),
            chunk=ChunkContext(
                chunk_id=f"{self.filename}_{index}",
                index=index,
                text=text
            )
        )

    async def extract(self, chunk: ProcessedChunk, llm_func=None) -> ProcessedChunk:
        if not llm_func:
            raise ValueError("LLM function required for Factual extraction")

        prompt = f"""Extract factual knowledge as triples from the text.
Identify entities and their relationships.
Preserve lists and definitions accurately.

Text:
{chunk.chunk.text}

Output JSON format:
{{
  "triples": [
    {{"subject": "Entity", "predicate": "Relationship", "object": "Entity"}}
  ],
  "entities": ["Entity1", "Entity2"]
}}
"""
        result = await llm_func(prompt)
        
        # Normalize result to match unified schema as much as possible, or just store in data
        # vector_store expects relations with s,p,o
        chunk.data = result
        return chunk
