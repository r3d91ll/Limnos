#!/usr/bin/env python
"""
PDF to Markdown Converter

This script converts PDF documents to Markdown format, preserving as much
structure and formatting as possible. It's useful for making PDF content
more accessible and editable.

Usage:
    python pdf_to_markdown.py --input /path/to/input.pdf --output /path/to/output.md
    python pdf_to_markdown.py --input /path/to/input.pdf  # Output to input_file_name.md
    python pdf_to_markdown.py --dir /path/to/pdf/directory  # Convert all PDFs in directory
"""

import os
import re
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pypdf
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFToMarkdownConverter:
    """
    Converts PDF documents to Markdown format, attempting to preserve
    structure and formatting.
    """
    
    def __init__(self, 
                 extract_images: bool = False, 
                 image_dir: Optional[str] = None,
                 min_heading_font_size: int = 12,
                 preserve_tables: bool = True):
        """
        Initialize the PDF to Markdown converter.
        
        Args:
            extract_images: Whether to extract and save images
            image_dir: Directory to save extracted images
            min_heading_font_size: Minimum font size to consider text as a heading
            preserve_tables: Whether to attempt to preserve table structure
        """
        self.extract_images = extract_images
        self.image_dir = image_dir
        self.min_heading_font_size = min_heading_font_size
        self.preserve_tables = preserve_tables
        self.logger = logging.getLogger(__name__)
    
    def convert_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert a PDF file to Markdown.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Path to save the Markdown output (if None, return as string)
            
        Returns:
            Markdown content as a string if output_path is None, else empty string
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Set default output path if not provided
        if output_path is None:
            output_path = pdf_path.with_suffix('.md')
        else:
            output_path = Path(output_path)
        
        self.logger.info(f"Converting {pdf_path} to {output_path}")
        
        # Create image directory if extracting images
        if self.extract_images:
            if self.image_dir:
                image_dir = Path(self.image_dir)
            else:
                image_dir = output_path.parent / f"{output_path.stem}_images"
            image_dir.mkdir(exist_ok=True, parents=True)
            self.logger.info(f"Images will be saved to {image_dir}")
        
        # Extract text with formatting using PyMuPDF if available
        if PYMUPDF_AVAILABLE and self.extract_images:
            try:
                markdown_content = self._convert_with_pymupdf(pdf_path, image_dir if self.extract_images else None)
                self.logger.info("Successfully converted using PyMuPDF")
            except Exception as e:
                self.logger.warning(f"Error using PyMuPDF: {e}. Falling back to simpler extraction.")
                # Fall back to simpler extraction using PyPDF
                markdown_content = self._convert_with_pypdf(pdf_path)
        else:
            # Use simpler extraction with PyPDF
            markdown_content = self._convert_with_pypdf(pdf_path)
            self.logger.info("Using PyPDF for extraction")
        
        # Post-process content
        markdown_content = self._post_process_markdown(markdown_content)
        
        # Write to file if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            self.logger.info(f"Markdown saved to {output_path}")
        
        return markdown_content
    
    def _convert_with_pymupdf(self, pdf_path: Path, image_dir: Optional[Path] = None) -> str:
        """
        Convert PDF to Markdown using PyMuPDF (more advanced).
        
        Args:
            pdf_path: Path to the PDF file
            image_dir: Directory to save extracted images
            
        Returns:
            Markdown content as a string
        """
        markdown_content = []
        pdf_document = fitz.open(str(pdf_path))
        
        # Get document metadata
        metadata = pdf_document.metadata
        if metadata:
            # Add title as H1 heading
            if metadata.get('title'):
                markdown_content.append(f"# {metadata.get('title')}\n")
            
            # Add author information
            if metadata.get('author'):
                markdown_content.append(f"*Author: {metadata.get('author')}*\n")
            
            # Add creation date if available
            if metadata.get('creationDate'):
                date = metadata.get('creationDate')
                if date.startswith('D:'):
                    date = date[2:10]  # Extract YYYYMMDD
                    date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                markdown_content.append(f"*Date: {date}*\n")
            
            markdown_content.append("\n---\n\n")  # Separator
        
        # Process each page
        image_count = 0
        for page_num, page in enumerate(pdf_document):
            self.logger.info(f"Processing page {page_num + 1}/{len(pdf_document)}")
            
            # Add page number header
            markdown_content.append(f"\n## Page {page_num + 1}\n")
            
            # Extract text blocks with formatting
            blocks = page.get_text("dict")["blocks"]
            
            # Process text blocks
            for block in blocks:
                if block["type"] == 0:  # Text
                    # Process text lines
                    for line in block["lines"]:
                        line_text = ""
                        # Process spans in line
                        for span in line["spans"]:
                            text = span["text"]
                            font_size = span["size"]
                            flags = span.get("flags", 0)
                            
                            # Apply formatting based on font properties
                            if flags & 16:  # Bold
                                text = f"**{text}**"
                            if flags & 1:  # Italic
                                text = f"*{text}*"
                            
                            # Check for potential heading
                            if font_size > self.min_heading_font_size:
                                text_trimmed = text.strip()
                                if len(text_trimmed) < 100 and not text_trimmed.endswith("."):
                                    heading_level = min(6, max(3, int(24 / font_size)))
                                    text = f"\n{'#' * heading_level} {text_trimmed}\n"
                            
                            line_text += text + " "
                        
                        markdown_content.append(line_text.strip())
                        markdown_content.append("\n")
                
                # Extract images if requested
                elif block["type"] == 1 and image_dir:  # Image
                    try:
                        image_count += 1
                        image_filename = f"{pdf_path.stem}_image_{image_count:03d}.png"
                        image_path = image_dir / image_filename
                        
                        # Extract image
                        xref = block.get("xref", 0)
                        if xref > 0:
                            image = pdf_document.extract_image(xref)
                            with open(image_path, "wb") as img_file:
                                img_file.write(image["image"])
                            
                            # Add image reference to markdown
                            markdown_content.append(f"\n![Image {image_count}]({image_filename})\n")
                    except Exception as e:
                        self.logger.warning(f"Failed to extract image: {e}")
                        markdown_content.append("\n[Image extraction failed]\n")
            
            markdown_content.append("\n")  # Page separator
        
        pdf_document.close()
        
        return "\n".join(markdown_content)
    
    def _convert_with_pypdf(self, pdf_path: Path) -> str:
        """
        Convert PDF to Markdown using PyPDF (simpler fallback).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Markdown content as a string
        """
        markdown_content = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            # Extract metadata
            if pdf_reader.metadata:
                # Add title as H1 heading
                if hasattr(pdf_reader.metadata, 'title') and pdf_reader.metadata.title:
                    markdown_content.append(f"# {pdf_reader.metadata.title}\n")
                
                # Add author information
                if hasattr(pdf_reader.metadata, 'author') and pdf_reader.metadata.author:
                    markdown_content.append(f"*Author: {pdf_reader.metadata.author}*\n")
                
                markdown_content.append("\n---\n\n")  # Separator
            
            # Process each page
            for page_num, page in enumerate(pdf_reader.pages):
                markdown_content.append(f"\n## Page {page_num + 1}\n")
                
                # Extract text
                text = page.extract_text()
                
                # Very basic processing for potential headings and paragraphs
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        markdown_content.append("\n")  # Paragraph break
                    elif len(line) < 80 and not line.endswith('.') and line.isupper():
                        # Potential heading
                        markdown_content.append(f"\n### {line}\n")
                    else:
                        markdown_content.append(line + "\n")
                
                markdown_content.append("\n")  # Page separator
        
        return "\n".join(markdown_content)
    
    def _post_process_markdown(self, content: str) -> str:
        """
        Post-process the markdown content for better formatting.
        
        Args:
            content: Raw markdown content
            
        Returns:
            Processed markdown content
        """
        # Fix multiple blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Fix heading spacing
        content = re.sub(r'(#+[^#\n]+)\n+([^#\n])', r'\1\n\n\2', content)
        
        # Try to detect and format bullet points
        content = re.sub(r'\n(?:\s*[•∙○●◦-]\s+)([^\n]+)', r'\n* \1', content)
        
        # Try to detect numbered lists
        content = re.sub(r'\n(?:\s*\d+\.\s+)([^\n]+)', r'\n1. \1', content)
        
        # Try to identify code blocks (simplified approach)
        content = re.sub(r'\n((?:(?:\s{4}|\t)[^\n]+\n)+)', r'\n```\n\1```\n', content)
        
        return content
    
    def convert_directory(self, dir_path: str, output_dir: Optional[str] = None, recursive: bool = False) -> List[str]:
        """
        Convert all PDF files in a directory to Markdown.
        
        Args:
            dir_path: Path to directory containing PDF files
            output_dir: Directory to save Markdown files
            recursive: Whether to search subdirectories
            
        Returns:
            List of paths to the generated markdown files
        """
        dir_path = Path(dir_path)
        
        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")
        
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = dir_path
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        # Find all PDF files
        if recursive:
            pdf_files = list(dir_path.glob('**/*.pdf'))
        else:
            pdf_files = list(dir_path.glob('*.pdf'))
        
        self.logger.info(f"Found {len(pdf_files)} PDF files in {dir_path}")
        
        # Convert each PDF
        output_files = []
        for pdf_file in pdf_files:
            # Determine output path
            rel_path = pdf_file.relative_to(dir_path)
            output_path = output_dir / rel_path.with_suffix('.md')
            
            # Make sure output directory exists
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Convert PDF
            try:
                self.convert_pdf(pdf_file, output_path)
                output_files.append(str(output_path))
            except Exception as e:
                self.logger.error(f"Failed to convert {pdf_file}: {e}")
        
        return output_files


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description='Convert PDF documents to Markdown format')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', help='Input PDF file')
    input_group.add_argument('--dir', '-d', help='Directory containing PDF files')
    
    # Output options
    parser.add_argument('--output', '-o', help='Output Markdown file or directory')
    parser.add_argument('--recursive', '-r', action='store_true', help='Process directories recursively')
    
    # Conversion options
    parser.add_argument('--extract-images', action='store_true', help='Extract and include images')
    parser.add_argument('--image-dir', help='Directory to save extracted images')
    parser.add_argument('--min-heading-size', type=int, default=12, help='Minimum font size for headings')
    parser.add_argument('--no-tables', action='store_true', help='Do not attempt to preserve tables')
    
    args = parser.parse_args()
    
    # Create converter
    converter = PDFToMarkdownConverter(
        extract_images=args.extract_images,
        image_dir=args.image_dir,
        min_heading_font_size=args.min_heading_size,
        preserve_tables=not args.no_tables
    )
    
    # Convert single file
    if args.input:
        converter.convert_pdf(args.input, args.output)
    
    # Convert directory
    elif args.dir:
        converter.convert_directory(args.dir, args.output, args.recursive)


if __name__ == '__main__':
    main()