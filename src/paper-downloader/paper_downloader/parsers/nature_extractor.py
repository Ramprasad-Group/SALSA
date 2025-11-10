#!/usr/bin/env python3
"""
Nature publisher figure and table extraction
Handles HTML files from Nature journals
"""

import os
import requests
import pandas as pd
import logging
import re
import json

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("BeautifulSoup not found. Install with: pip install beautifulsoup4")
    BeautifulSoup = None

def _clean_figure_label(label):
    """Clean figure label for filename"""
    if not label:
        return "figure"
    # Remove special characters and replace spaces with underscores
    clean_label = "".join(c for c in label if c.isalnum() or c in (' ', '-', '_')).strip()
    clean_label = clean_label.replace(' ', '_').replace('-', '_')
    return clean_label or "figure"

def _save_figure_metadata(fig_dir, clean_label, original_label, caption, url, doi):
    """Save figure metadata as JSON file"""
    import json
    import pandas as pd
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
        "publisher": "nature"
    }
    
    metadata_file = os.path.join(fig_dir, f"{clean_label}_metadata.json")
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving metadata for {clean_label}: {e}")

def extract_nature_figures(file_path, output_dir, doi):
    """Extract figures from Nature HTML files"""
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
        
        # Find figure elements - Nature uses c-article-section__figure class
        figure_elements = soup.find_all('div', class_='c-article-section__figure')
        logger.info(f"Found {len(figure_elements)} figure elements")
        
        # Only create directory if we have figures to process
        if not figure_elements:
            logger.info(f"No figures found in {file_path}")
            return 0
        
        # Create output directories only when we have figures
        fig_dir = os.path.join(output_dir, 'figures', doi.replace('/', '@'))
        os.makedirs(fig_dir, exist_ok=True)
        logger.debug(f"Created figure directory: {fig_dir}")
        
        for i, fig_elem in enumerate(figure_elements):
            # Extract figure label from figcaption
            figcaption = fig_elem.find('figcaption')
            if figcaption:
                label_elem = figcaption.find('b', class_='c-article-section__figure-caption')
                if label_elem:
                    figure_label = label_elem.get_text(strip=True)
                else:
                    figure_label = f"Figure_{i+1}"
            else:
                figure_label = f"Figure_{i+1}"
            
            # Extract caption/description
            caption_elem = fig_elem.find('div', class_='c-article-section__figure-description')
            caption = ""
            if caption_elem:
                caption = caption_elem.get_text(strip=True)
            
            # Find image element
            img_elem = fig_elem.find('img')
            if not img_elem:
                logger.warning(f"No image found in {figure_label}")
                continue
            
            src = img_elem.get('src', '')
            if not src:
                logger.warning(f"No src found in {figure_label}")
                continue
            
            # Make URL absolute if relative
            if src.startswith('//'):
                url = f"https:{src}"
            elif src.startswith('/'):
                url = f"https://media.springernature.com{src}"
            elif src.startswith('http'):
                url = src
            else:
                url = f"https://media.springernature.com/{src}"
            
            # Download figure
            try:
                clean_label = _clean_figure_label(figure_label)
                img_ext = url.split("?")[0].split(".")[-1]
                if not img_ext or len(img_ext) > 4:
                    img_ext = "png"  # Nature typically uses PNG
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
                        logger.info(f"Downloaded {figure_label} -> {img_path}")
                        
                        # Save figure metadata
                        _save_figure_metadata(fig_dir, clean_label, figure_label, caption, url, doi)
                        figures.append({"label": figure_label, "caption": caption, "url": url})
                    else:
                        logger.warning(f"Got non-image content for {figure_label}")
                else:
                    logger.warning(f"Failed to download {figure_label} - Status: {r.status_code}")
            except Exception as e:
                logger.error(f"Error downloading {figure_label}: {e}")
        
        logger.info(f"Found {len(figures)} figures in {file_path}")
        return len(figures)
        
    except Exception as e:
        logger.error(f"Error processing Nature file {file_path}: {e}")
        return 0

def extract_nature_tables(file_path, output_dir, doi, llm_extractor):
    """Extract tables from Nature HTML files using LLM"""
    logger = logging.getLogger('extraction')
    try:
        # Create output directory
        table_dir = os.path.join(output_dir, 'tables', doi.replace('/', '@'))
        os.makedirs(table_dir, exist_ok=True)
        
        # Read HTML file
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = []
        
        # Find table elements - Nature uses c-article-table class
        table_elements = soup.find_all('div', class_='c-article-table')
        
        for i, table_elem in enumerate(table_elements):
            try:
                # Extract table caption
                caption_elem = table_elem.find('figcaption', class_='c-article-table__figcaption')
                caption = ""
                table_label = f"Table_{i+1}"
                
                if caption_elem:
                    caption_text = caption_elem.get_text(strip=True)
                    caption = caption_text
                    
                    # Extract table number from caption
                    tab_match = re.search(r'Table\s*(\d+)', caption_text, re.IGNORECASE)
                    if tab_match:
                        table_label = f"Table_{tab_match.group(1)}"
                
                # Look for actual table content within the div
                table_content = table_elem.find('table')
                if not table_content:
                    # If no table element, look for other table-like structures
                    table_content = table_elem.find(['div', 'section'], class_=re.compile(r'table|data', re.I))
                
                if table_content:
                    # Save table metadata
                    table_metadata = {
                        "label": table_label,
                        "caption": caption,
                        "doi": doi,
                        "content_type": "table_reference"
                    }
                    
                    metadata_file = os.path.join(table_dir, f"{table_label}_metadata.json")
                    with open(metadata_file, 'w') as f:
                        json.dump(table_metadata, f, indent=2)
                    
                    # Save table HTML content for LLM processing
                    table_html_file = os.path.join(table_dir, f"{table_label}_content.html")
                    with open(table_html_file, 'w', encoding='utf-8') as f:
                        f.write(str(table_content))
                    
                    tables.append({
                        "label": table_label,
                        "caption": caption,
                        "html_content": str(table_content)
                    })
                    
                    logger.info(f"Found {table_label}: {caption}")
                else:
                    logger.warning(f"Table {table_label} found but no content extracted")
                    
            except Exception as e:
                logger.error(f"Error processing table {i+1}: {e}")
        
        # Also try LLM extraction for additional table detection
        try:
            logger.info(f"Running LLM table extraction on {file_path}")
            llm_tables = llm_extractor.extract_tables_from_file(file_path)
            
            if llm_tables:
                for i, table in enumerate(llm_tables):
                    try:
                        # Save LLM-extracted table as CSV
                        table_name = f"llm_table_{i+1}.csv"
                        table_path = os.path.join(table_dir, table_name)
                        
                        # Convert table data to DataFrame and save
                        if 'data' in table and table['data']:
                            df = pd.DataFrame(table['data'])
                            df.to_csv(table_path, index=False)
                            logger.info(f"Saved LLM table {i+1} to {table_path}")
                        
                    except Exception as e:
                        logger.error(f"Error saving LLM table {i+1}: {e}")
        except Exception as e:
            logger.warning(f"LLM table extraction failed: {e}")
        
        logger.info(f"Found {len(tables)} tables in {file_path}")
        return len(tables), []
        
    except Exception as e:
        logger.error(f"Error extracting tables from {file_path}: {e}")
        return 0, []
