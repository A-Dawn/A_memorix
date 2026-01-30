import re
from typing import List, Dict, Any
from .base import BaseStrategy, ProcessedChunk, KnowledgeType, SourceInfo, ChunkContext

class NarrativeStrategy(BaseStrategy):
    def split(self, text: str) -> List[ProcessedChunk]:
        scenes = self._split_into_scenes(text)
        chunks = []
        
        for scene_idx, (scene_text, scene_title) in enumerate(scenes):
             scene_chunks = self._sliding_window(scene_text, scene_title, scene_idx)
             chunks.extend(scene_chunks)
             
        return chunks

    def _split_into_scenes(self, text: str) -> List[tuple[str, str]]:
        """Split text into scenes based on headers or separators."""
        # Simple heuristic: Split by markdown headers or specific separators
        # This regex looks for lines starting with #, Chapter, or specific separators like ***
        # This regex looks for lines starting with #, Chapter, or specific separators like ***
        scene_pattern_str = r'^(?:#{1,6}\s+.*|Chapter\s+\d+|^\*{3,}$|^={3,}$)'
        
        # We need to keep delimiters to know where scenes start
        parts = re.split(f"({scene_pattern_str})", text, flags=re.MULTILINE)
        
        scenes = []
        current_scene_title = "Start"
        current_scene_content = []
        
        if parts and parts[0].strip() == "":
            parts = parts[1:]
            
        for part in parts:
            if re.match(scene_pattern_str, part, re.MULTILINE):
                # Save previous scene
                if current_scene_content:
                    scenes.append(("".join(current_scene_content), current_scene_title))
                    current_scene_content = []
                current_scene_title = part.strip()
            else:
                current_scene_content.append(part)
                
        if current_scene_content:
             scenes.append(("".join(current_scene_content), current_scene_title))
             
        # Fallback if no scenes detected: treat whole text as one scene
        if not scenes:
            scenes = [(text, "Whole Text")]

        return scenes

    def _sliding_window(self, text: str, scene_id: str, scene_idx: int, window_size=800, overlap=200) -> List[ProcessedChunk]:
        chunks = []
        if len(text) <= window_size:
            chunks.append(self._create_chunk(text, scene_id, scene_idx, 0, 0))
            return chunks

        stride = window_size - overlap
        start = 0
        local_idx = 0
        while start < len(text):
            end = min(start + window_size, len(text))
            chunk_text = text[start:end]
            
            # Adjust to nearest newline to avoid cutting words/sentences too abruptly if possible
            # Check if we are not at end
            if end < len(text):
                last_newline = chunk_text.rfind('\n')
                if last_newline > window_size // 2: # Only cut back if it's not too far
                    end = start + last_newline + 1
                    chunk_text = text[start:end]
            
            chunks.append(self._create_chunk(chunk_text, scene_id, scene_idx, local_idx, start))
            
            start += len(chunk_text) - overlap if end < len(text) else len(chunk_text)
            local_idx += 1
            
        return chunks

    def _create_chunk(self, text: str, scene_id: str, scene_idx: int, local_idx: int, offset: int) -> ProcessedChunk:
        return ProcessedChunk(
            type=KnowledgeType.NARRATIVE,
            source=SourceInfo(
                file=self.filename,
                offset_start=offset,
                offset_end=offset + len(text),
                checksum=self.calculate_checksum(text)
            ),
            chunk=ChunkContext(
                chunk_id=f"{self.filename}_{scene_idx}_{local_idx}",
                index=local_idx,
                text=text,
                context={"scene_id": scene_id}
            )
        )

    async def extract(self, chunk: ProcessedChunk, llm_func=None) -> ProcessedChunk:
        if not llm_func:
            raise ValueError("LLM function required for Narrative extraction")
            
        prompt = f"""Analyze the following narrative text from scene '{chunk.chunk.context.get('scene_id')}'.
Identify key events and character relations.

Text:
{chunk.chunk.text}

Output JSON format:
{{
  "events": ["event description 1", "event description 2"],
  "relations": [
    {{"subject": "CharacterA", "predicate": "relation", "object": "CharacterB"}}
  ]
}}
"""
        result = await llm_func(prompt)
        chunk.data = result
        return chunk
