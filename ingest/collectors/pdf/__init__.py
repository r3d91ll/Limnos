"""
PDF Document Collector for Limnos

This package provides functionality for collecting and processing PDF documents
for use in the Limnos RAG system.
"""

from limnos.ingest.collectors.pdf.document_collector import UniversalDocumentCollector
from limnos.ingest.collectors.pdf.pdf_processor import PDFProcessor

__all__ = ['UniversalDocumentCollector', 'PDFProcessor']
