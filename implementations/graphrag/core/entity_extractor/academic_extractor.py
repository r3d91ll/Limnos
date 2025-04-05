"""
Academic Entity Extractor for GraphRAG

This module implements specialized entity extraction for academic and research documents.
It focuses on identifying academic concepts, citations, references, and domain-specific entities.
"""

import re
from typing import Dict, List, Optional, Any, Set, Tuple
import logging
from collections import defaultdict

from ..models.entity import Entity
from .entity_extractor import EntityExtractor
from .spacy_extractor import SpacyEntityExtractor


class AcademicEntityExtractor(EntityExtractor):
    """
    Specialized entity extractor for academic and research documents.
    
    This extractor combines NLP-based entity extraction with specialized patterns
    for academic content such as citations, references, equations, and technical terms.
    """
    
    # Regular expressions for academic entities
    CITATION_PATTERN = r'\[(\d+(?:,\s*\d+)*)\]|\(\s*([A-Za-z]+\s+et\s+al\.\s*,\s*\d{4}[a-z]?)\s*\)'
    REFERENCE_PATTERN = r'(?:^|\n)(\[\d+\])\s+(.+?)(?=\n\[\d+\]|\n\n|$)'
    EQUATION_PATTERN = r'(?:\$\$(.*?)\$\$|\$(.*?)\$)'
    SECTION_HEADER_PATTERN = r'(?:^|\n)(#+\s+(.+?)(?:\n|$))'
    
    # Academic entity types
    ACADEMIC_ENTITY_TYPES = {
        "citation": "citation",
        "reference": "reference",
        "equation": "equation",
        "section": "section",
        "term": "term",
        "algorithm": "algorithm",
        "dataset": "dataset",
        "method": "method",
        "metric": "metric",
        "model": "model",
        "task": "task",
        "technology": "technology",
        "tool": "tool"
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the academic entity extractor.
        
        Args:
            config: Configuration dictionary with the following options:
                - use_spacy: Whether to use spaCy for general NER (default: True)
                - spacy_config: Configuration for the SpacyEntityExtractor
                - extract_citations: Whether to extract citations (default: True)
                - extract_references: Whether to extract references (default: True)
                - extract_equations: Whether to extract equations (default: True)
                - extract_sections: Whether to extract section headers (default: True)
                - extract_terms: Whether to extract domain-specific terms (default: True)
                - domain_terms: Dictionary of domain-specific terms and their types
                - min_confidence: Minimum confidence threshold (default: 0.5)
        """
        super().__init__(config)
        
        # Get configuration options
        self.use_spacy = self.config.get("use_spacy", True)
        self.extract_citations = self.config.get("extract_citations", True)
        self.extract_references = self.config.get("extract_references", True)
        self.extract_equations = self.config.get("extract_equations", True)
        self.extract_sections = self.config.get("extract_sections", True)
        self.extract_terms = self.config.get("extract_terms", True)
        self.min_confidence = self.config.get("min_confidence", 0.5)
        
        # Domain-specific terms
        self.domain_terms = self.config.get("domain_terms", {})
        
        # Initialize spaCy extractor if needed
        if self.use_spacy:
            spacy_config = self.config.get("spacy_config", {})
            self.spacy_extractor = SpacyEntityExtractor(spacy_config)
            self.logger.info("Initialized SpacyEntityExtractor for academic document processing")
            
        # Compile regular expressions
        self.citation_regex = re.compile(self.CITATION_PATTERN)
        self.reference_regex = re.compile(self.REFERENCE_PATTERN, re.DOTALL)
        self.equation_regex = re.compile(self.EQUATION_PATTERN)
        self.section_regex = re.compile(self.SECTION_HEADER_PATTERN)
    
    def extract_entities(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """
        Extract academic entities from the given text.
        
        Args:
            text: The text to extract entities from
            metadata: Optional metadata about the text
            
        Returns:
            List of extracted entities
        """
        all_entities = []
        
        # Use spaCy for general NER if enabled
        if self.use_spacy:
            spacy_entities = self.spacy_extractor.extract_entities(text, metadata)
            all_entities.extend(spacy_entities)
            
        # Extract academic-specific entities
        academic_entities = self._extract_academic_entities(text, metadata)
        all_entities.extend(academic_entities)
        
        # Deduplicate entities
        return self.deduplicate_entities(all_entities)
    
    def _extract_academic_entities(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """
        Extract academic-specific entities from text.
        
        Args:
            text: The text to extract entities from
            metadata: Optional metadata about the text
            
        Returns:
            List of extracted academic entities
        """
        entities = []
        
        # Extract citations
        if self.extract_citations:
            citation_entities = self._extract_citations(text, metadata)
            entities.extend(citation_entities)
            
        # Extract references
        if self.extract_references:
            reference_entities = self._extract_references(text, metadata)
            entities.extend(reference_entities)
            
        # Extract equations
        if self.extract_equations:
            equation_entities = self._extract_equations(text, metadata)
            entities.extend(equation_entities)
            
        # Extract section headers
        if self.extract_sections:
            section_entities = self._extract_sections(text, metadata)
            entities.extend(section_entities)
            
        # Extract domain-specific terms
        if self.extract_terms:
            term_entities = self._extract_domain_terms(text, metadata)
            entities.extend(term_entities)
            
        return entities
    
    def _extract_citations(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """
        Extract citation entities from text.
        
        Args:
            text: The text to extract citations from
            metadata: Optional metadata about the text
            
        Returns:
            List of citation entities
        """
        entities = []
        
        # Find all citations
        for match in self.citation_regex.finditer(text):
            # Get the citation text
            citation_text = match.group(0)
            
            # Determine the citation format (numeric or author-year)
            if match.group(1):  # Numeric citation [1] or [1,2,3]
                citation_type = "numeric_citation"
                citation_ids = [id.strip() for id in match.group(1).split(',')]
                canonical_name = f"Citation {', '.join(citation_ids)}"
            else:  # Author-year citation (Smith et al., 2020)
                citation_type = "author_citation"
                author_year = match.group(2)
                canonical_name = f"Citation {author_year}"
                citation_ids = [author_year]
            
            # Create the entity
            entity = Entity(
                text=citation_text,
                entity_type="citation",
                canonical_name=canonical_name,
                confidence=0.95,
                metadata={
                    "citation_type": citation_type,
                    "citation_ids": citation_ids
                }
            )
            
            if metadata:
                entity.metadata.update(metadata)
                
            # Add position information
            entity.add_position(match.start(), match.end())
            
            entities.append(entity)
            
        return entities
    
    def _extract_references(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """
        Extract reference entities from text.
        
        Args:
            text: The text to extract references from
            metadata: Optional metadata about the text
            
        Returns:
            List of reference entities
        """
        entities = []
        
        # Find all references in the reference section
        for match in self.reference_regex.finditer(text):
            # Get the reference components
            ref_id = match.group(1)
            ref_text = match.group(2).strip()
            
            # Create the entity
            entity = Entity(
                text=ref_text,
                entity_type="reference",
                canonical_name=f"Reference {ref_id}",
                confidence=0.9,
                metadata={
                    "reference_id": ref_id,
                    "reference_text": ref_text
                }
            )
            
            if metadata:
                entity.metadata.update(metadata)
                
            # Add position information
            entity.add_position(match.start(), match.end())
            
            entities.append(entity)
            
        return entities
    
    def _extract_equations(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """
        Extract equation entities from text.
        
        Args:
            text: The text to extract equations from
            metadata: Optional metadata about the text
            
        Returns:
            List of equation entities
        """
        entities = []
        
        # Find all equations
        equation_count = 0
        for match in self.equation_regex.finditer(text):
            equation_count += 1
            
            # Get the equation content (either from group 1 or 2 depending on delimiter)
            equation_content = match.group(1) if match.group(1) else match.group(2)
            equation_text = match.group(0)  # Full match including delimiters
            
            # Create the entity
            entity = Entity(
                text=equation_text,
                entity_type="equation",
                canonical_name=f"Equation {equation_count}",
                confidence=0.9,
                metadata={
                    "equation_content": equation_content,
                    "equation_number": equation_count
                }
            )
            
            if metadata:
                entity.metadata.update(metadata)
                
            # Add position information
            entity.add_position(match.start(), match.end())
            
            entities.append(entity)
            
        return entities
    
    def _extract_sections(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """
        Extract section header entities from text.
        
        Args:
            text: The text to extract section headers from
            metadata: Optional metadata about the text
            
        Returns:
            List of section entities
        """
        entities = []
        
        # Find all section headers
        for match in self.section_regex.finditer(text):
            # Get the section header text
            header_with_markup = match.group(1)
            header_text = match.group(2)
            
            # Determine section level based on number of # characters
            level = header_with_markup.count('#')
            
            # Create the entity
            entity = Entity(
                text=header_text,
                entity_type="section",
                canonical_name=f"Section: {header_text}",
                confidence=1.0,
                metadata={
                    "section_level": level,
                    "section_text": header_text
                }
            )
            
            if metadata:
                entity.metadata.update(metadata)
                
            # Add position information
            entity.add_position(match.start(), match.end())
            
            entities.append(entity)
            
        return entities
    
    def _extract_domain_terms(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """
        Extract domain-specific terms from text.
        
        Args:
            text: The text to extract terms from
            metadata: Optional metadata about the text
            
        Returns:
            List of term entities
        """
        entities = []
        
        # Look for each domain term in the text
        for term, term_type in self.domain_terms.items():
            # Create a pattern that matches word boundaries
            pattern = r'\b' + re.escape(term) + r'\b'
            
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Get the actual matched text (preserving case)
                matched_text = match.group(0)
                
                # Create the entity
                entity = Entity(
                    text=matched_text,
                    entity_type=term_type,
                    canonical_name=term,  # Use the dictionary key as canonical name
                    confidence=0.85,
                    metadata={
                        "domain_term": True,
                        "term_type": term_type
                    }
                )
                
                if metadata:
                    entity.metadata.update(metadata)
                    
                # Add position information
                entity.add_position(match.start(), match.end())
                
                entities.append(entity)
                
        return entities
    
    def extract_with_sections(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, List[Entity]]:
        """
        Extract entities and organize them by document section.
        
        Args:
            text: The text to extract entities from
            metadata: Optional metadata about the text
            
        Returns:
            Dictionary mapping section names to lists of entities
        """
        # First extract all section headers
        section_entities = self._extract_sections(text, metadata)
        
        if not section_entities:
            # If no sections found, process the whole document
            entities = self.extract_entities(text, metadata)
            return {"document": entities}
            
        # Sort sections by position
        section_entities.sort(key=lambda e: e.positions[0]["start"])
        
        # Split the document into sections
        sections = {}
        for i, section in enumerate(section_entities):
            section_name = section.text
            start_pos = section.positions[0]["end"]
            
            # Determine end position (start of next section or end of text)
            if i < len(section_entities) - 1:
                end_pos = section_entities[i+1].positions[0]["start"]
            else:
                end_pos = len(text)
                
            # Extract section text
            section_text = text[start_pos:end_pos]
            
            # Create section metadata
            section_metadata = metadata.copy() if metadata else {}
            section_metadata["section"] = section_name
            section_metadata["section_level"] = section.metadata["section_level"]
            
            # Extract entities from this section
            section_entities = self.extract_entities(section_text, section_metadata)
            
            # Adjust positions to be relative to the original document
            for entity in section_entities:
                for pos in entity.positions:
                    pos["start"] += start_pos
                    pos["end"] += start_pos
                    
            sections[section_name] = section_entities
            
        return sections
