#!/usr/bin/env python3
"""
Springer publisher figure and table extraction
Handles HTML files from Springer journals
"""

import os
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

def extract_springer_figures(file_path, output_dir, doi):
    """Extract figures from Springer HTML files"""
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
        
        # Find figure elements - Springer uses specific patterns
        # Use only the most specific selector to avoid duplicates
        figure_selectors = [
            '.c-article-section__figure'
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
        
        seen_figures = set()  # Track seen figures to avoid duplicates
        
        for selector in figure_selectors:
            fig_elements = soup.select(selector)
            for i, fig_elem in enumerate(fig_elements):
                # Create a unique identifier for this figure to avoid duplicates
                fig_id = fig_elem.get('id', f'fig_{i}')
                if fig_id in seen_figures:
                    continue
                seen_figures.add(fig_id)
                # Extract figure label from figcaption
                label_elem = fig_elem.find('figcaption')
                if label_elem:
                    # Look for figure number in the caption
                    caption_text = label_elem.get_text(strip=True)
                    if 'Figure' in caption_text:
                        label = caption_text.split('.')[0] if '.' in caption_text else caption_text
                    else:
                        label = f"Figure_{i+1}"
                else:
                    label = f"Figure_{i+1}"
                
                # Extract caption - look for description element first
                # Use find_all to search recursively for description elements
                caption_elem = fig_elem.find(class_=re.compile(r'description'))
                
                if caption_elem:
                    caption = caption_elem.get_text(strip=True)
                else:
                    # Fallback to figcaption but exclude the figure number
                    caption_elem = fig_elem.find('figcaption')
                    if caption_elem:
                        caption_text = caption_elem.get_text(strip=True)
                        # Remove "Figure X" from the beginning if present
                        caption = re.sub(r'^Figure\s+\d+\s*', '', caption_text).strip()
                    else:
                        caption = ""
                
                # Find image URL - try multiple sources
                url = None
                img_elem = fig_elem.find('img')
                if img_elem:
                    # Try different URL attributes
                    url = (img_elem.get('src') or 
                           img_elem.get('data-src') or 
                           img_elem.get('data-lazy-src'))
                    
                    if url and not url.startswith('data:'):
                        # Make URL absolute if relative
                        if url.startswith('//'):
                            url = f"https:{url}"
                        elif url.startswith('/'):
                            url = f"https://link.springer.com{url}"
                        elif url.startswith('./'):
                            url = f"https://link.springer.com/{url[2:]}"
                        
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
                                    'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                                }
                                
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
        logger.error(f"Error processing Springer file {file_path}: {e}")
        return 0

def extract_springer_tables(file_path, output_dir, doi, llm_extractor):
    """Extract tables from Springer HTML files using LLM"""
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
        "publisher": "springer"
    }
    
    metadata_file = os.path.join(fig_dir, f"{clean_label}_metadata.json")
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved metadata: {metadata_file}")
    except Exception as e:
        logger.warning(f"Failed to save metadata: {e}")