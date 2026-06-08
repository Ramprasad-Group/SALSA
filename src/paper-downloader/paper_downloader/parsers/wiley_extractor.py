#!/usr/bin/env python3
"""
Wiley publisher figure and table extraction
Handles HTML and PDF files from Wiley journals
"""

import os
import json
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import logging
import re

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("BeautifulSoup not found. Install with: pip install beautifulsoup4")
    BeautifulSoup = None

def extract_wiley_figures(file_path, output_dir, doi):
    """Extract figures from Wiley HTML files"""
    logger = logging.getLogger('extraction')
    
    if BeautifulSoup is None:
        logger.error("BeautifulSoup not available. Cannot extract from HTML files.")
        return 0
    
    try:
        # Read HTML file first to check if figures exist
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        figures = []
        
        # Find figure elements - Wiley uses specific patterns
        figure_selectors = [
            'figure.figure',
            '.article-section__inline-figure figure',
            'figure[id^="fig"]',
            '.figure'
        ]
        
        # Check if we have any figures before creating directory
        has_figures = False
        for selector in figure_selectors:
            if soup.select(selector):
                has_figures = True
                break
        
        if not has_figures:
            logger.info(f"No figures found in {file_path}")
            return 0
        
        # Create output directories only when we have figures
        fig_dir = os.path.join(output_dir, 'figures', doi.replace('/', '@'))
        os.makedirs(fig_dir, exist_ok=True)
        logger.debug(f"Created figure directory: {fig_dir}")
        
        for selector in figure_selectors:
            fig_elements = soup.select(selector)
            for i, fig_elem in enumerate(fig_elements):
                # Extract figure label from figure title
                label_elem = fig_elem.find('.figure-title')
                if label_elem:
                    label = label_elem.get_text(strip=True)
                else:
                    # Try to get from figcaption
                    caption_elem = fig_elem.find('figcaption')
                    if caption_elem:
                        label = caption_elem.get_text(strip=True).split('\n')[0]  # First line
                    else:
                        label = f"Figure_{i+1}"
                
                # Extract caption
                caption_elem = fig_elem.find('figcaption')
                caption = caption_elem.get_text(strip=True) if caption_elem else ""
                
                # Find image URL - try multiple sources
                url = None
                img_elem = fig_elem.find('img')
                if img_elem:
                    # Try different URL attributes
                    url = (img_elem.get('src') or 
                           img_elem.get('data-src') or 
                           img_elem.get('data-lg-src'))
                    
                    # Also check for picture/source elements
                    if not url:
                        source_elem = fig_elem.find('source')
                        if source_elem:
                            url = source_elem.get('srcset')
                            if url:
                                url = url.split()[0]  # Get first URL from srcset
                
                if url and not url.startswith('data:'):
                    # Download figure
                    if url:
                        try:
                            clean_label = _clean_figure_label(label)
                            img_ext = url.split("?")[0].split(".")[-1]
                            if not img_ext or len(img_ext) > 4:
                                img_ext = "jpg"
                            img_name = f"{clean_label}.{img_ext}"
                            img_path = os.path.join(fig_dir, img_name)
                            
                            headers = {
                                'Accept': 'image/*',
                                'User-agent': 'Mozilla/5.6 (Windows NT 11.0; rv:68.0 Gecko/20100101 Firefox/70.0'
                            }
                            
                            # Add API key if available (similar to Elsevier approach)
                            api_key = os.getenv('WILEY_API_KEY') or os.getenv('WILEY_TDM_TOKEN') or os.getenv('API_KEY')
                            if api_key and '?' not in url:
                                url += f"?APIKey={api_key}"
                            elif api_key and 'APIKey=' not in url:
                                url += f"&APIKey={api_key}"
                            
                            r = requests.get(url, headers=headers, timeout=15)
                            if r.status_code == 200:
                                content_type = r.headers.get('content-type', '').lower()
                                if 'image' in content_type or r.content.startswith(b'\xff\xd8') or r.content.startswith(b'\x89PNG'):
                                    with open(img_path, "wb") as f:
                                        f.write(r.content)
                                    logger.info(f"Downloaded {label} -> {img_path}")
                                    
                                    # Save figure metadata
                                    _save_figure_metadata(fig_dir, clean_label, label, caption, url, doi)
                                else:
                                    logger.warning(f"Got non-image content for {label}")
                            else:
                                logger.warning(f"Failed to download {label} - Status: {r.status_code}")
                        except Exception as e:
                            logger.error(f"Error downloading {label}: {e}")
                
                figures.append({"label": label, "caption": caption, "url": url if 'url' in locals() else None})
        
        logger.info(f"Found {len(figures)} figures in {file_path}")
        return len(figures)
        
    except Exception as e:
        logger.error(f"Error processing Wiley file {file_path}: {e}")
        return 0

def extract_wiley_tables(file_path, output_dir, doi, llm_extractor):
    """Extract tables from Wiley HTML files using LLM"""
    logger = logging.getLogger('extraction')
    try:
        logger.info(f"Extracting tables with LLM from {file_path}")
        tables = llm_extractor.extract_tables_from_file(file_path)
        logger.info(f"LLM found {len(tables)} tables")
        
        if tables:
            saved_files = llm_extractor.save_tables_to_csv(tables, output_dir, doi)
            logger.info(f"Saved {len(saved_files)} table CSV files")
            return len(tables), saved_files
        else:
            logger.warning(f"No tables found by LLM")
            return 0, []
    except Exception as e:
        logger.error(f"Error extracting tables with LLM from {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return 0, []

def _clean_figure_label(label):
    """Clean figure label for use as filename"""
    if not label:
        return "Figure"
    
    # Remove common prefixes and clean up
    clean_label = label.strip()
    clean_label = clean_label.replace("Fig.", "Figure")
    clean_label = clean_label.replace("Figure", "Figure")
    clean_label = clean_label.replace(" ", "_")
    clean_label = clean_label.replace(".", "_")
    clean_label = clean_label.replace(":", "_")
    clean_label = clean_label.replace("/", "_")
    clean_label = clean_label.replace("\\", "_")
    clean_label = clean_label.replace("(", "_")
    clean_label = clean_label.replace(")", "_")
    clean_label = clean_label.replace("[", "_")
    clean_label = clean_label.replace("]", "_")
    
    # Remove multiple underscores
    while "__" in clean_label:
        clean_label = clean_label.replace("__", "_")
    
    # Remove leading/trailing underscores
    clean_label = clean_label.strip("_")
    
    return clean_label or "Figure"

def _save_figure_metadata(fig_dir, clean_label, original_label, caption, url, doi):
    """Save figure metadata as JSON file"""
    import json
    logger = logging.getLogger('extraction')
    
    clean_caption = caption.strip() if caption else ""
    
    # Determine the figure file path
    img_ext = url.split("?")[0].split(".")[-1] if url else "jpg"
    if not img_ext or len(img_ext) > 4:
        img_ext = "jpg"
    figure_file_path = os.path.join(fig_dir, f"{clean_label}.{img_ext}")
    
    metadata = {
        "original_label": original_label,
        "clean_label": clean_label,
        "caption": clean_caption,
        "url": url,
        "doi": doi,
        "figure_file_path": figure_file_path,
        "figure_directory": fig_dir,
        "extraction_timestamp": pd.Timestamp.now().isoformat(),
        "file_type": "figure",
        "publisher": "wiley"
    }
    
    metadata_file = os.path.join(fig_dir, f"{clean_label}_metadata.json")
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved metadata: {metadata_file}")
    except Exception as e:
        logger.warning(f"Failed to save metadata: {e}")


# ---------------------------------------------------------------------------
# PDF extraction (used when Wiley papers are downloaded as PDFs via TDM API)
# ---------------------------------------------------------------------------

# Minimum image size in bytes to skip icons/logos
_MIN_IMAGE_BYTES = 5_000


def extract_wiley_figures_from_pdf(file_path, output_dir, doi):
    """Extract figures from a Wiley PDF using PyMuPDF (fitz)."""
    logger = logging.getLogger('extraction')
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF not installed. Run: pip install pymupdf")
        return 0

    try:
        doc = fitz.open(file_path)
        safe_doi = doi.replace('/', '@')
        fig_dir = os.path.join(output_dir, 'figures', safe_doi)

        # Collect images across all pages, deduplicate by xref
        seen_xrefs = set()
        figures = []
        for page in doc:
            for img in page.get_images(full=True):
                xref = img[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                if len(image_bytes) < _MIN_IMAGE_BYTES:
                    continue  # skip tiny decorative images
                figures.append((xref, image_bytes, base_image["ext"], page.number + 1))

        if not figures:
            logger.info(f"No figures found in PDF {file_path}")
            return 0

        os.makedirs(fig_dir, exist_ok=True)
        for idx, (xref, image_bytes, ext, page_num) in enumerate(figures):
            label = f"Figure_{idx + 1}"
            img_path = os.path.join(fig_dir, f"{label}.{ext}")
            with open(img_path, 'wb') as f:
                f.write(image_bytes)
            metadata = {
                "original_label": label,
                "clean_label": label,
                "caption": "",
                "page": page_num,
                "doi": doi,
                "figure_file_path": img_path,
                "figure_directory": fig_dir,
                "extraction_timestamp": pd.Timestamp.now().isoformat(),
                "file_type": "figure",
                "publisher": "wiley",
            }
            with open(os.path.join(fig_dir, f"{label}_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved {label} (page {page_num}) -> {img_path}")

        logger.info(f"Extracted {len(figures)} figures from PDF {file_path}")
        return len(figures)

    except Exception as e:
        logger.error(f"Error extracting figures from PDF {file_path}: {e}")
        return 0


def extract_wiley_tables_from_pdf(file_path, output_dir, doi, llm_extractor=None):
    """Extract tables from a Wiley PDF using pdfplumber."""
    logger = logging.getLogger('extraction')
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        return 0, []

    try:
        tables_found = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                for t_idx, raw_table in enumerate(page.extract_tables() or []):
                    if raw_table and len(raw_table) > 1:
                        tables_found.append((page_num + 1, raw_table))

        if not tables_found:
            logger.info(f"No tables found in PDF {file_path}")
            return 0, []

        table_dir = os.path.join(output_dir, 'tables', doi.replace('/', '@'))
        os.makedirs(table_dir, exist_ok=True)

        saved_files = []
        for i, (page_num, raw_table) in enumerate(tables_found):
            label = f"Table_{i + 1}"
            filepath = os.path.join(table_dir, f"{label}.csv")
            headers = [str(h) if h is not None else f"Column_{j+1}"
                       for j, h in enumerate(raw_table[0])]
            data = [[str(cell) if cell is not None else "" for cell in row]
                    for row in raw_table[1:]]
            pd.DataFrame(data, columns=headers).to_csv(filepath, index=False)
            saved_files.append(filepath)
            logger.info(f"Saved {label} (page {page_num}) -> {filepath}")

        logger.info(f"Extracted {len(tables_found)} tables from PDF {file_path}")
        return len(tables_found), saved_files

    except Exception as e:
        logger.error(f"Error extracting tables from PDF {file_path}: {e}")
        return 0, []
