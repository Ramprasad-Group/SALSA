#!/usr/bin/env python3
"""
RSC publisher figure and table extraction
Handles HTML files from RSC journals
"""

import os
import pandas as pd
import logging
import re
import json

from paper_downloader.http_session import get_session

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
        "publisher": "rsc"
    }
    
    metadata_file = os.path.join(fig_dir, f"{clean_label}_metadata.json")
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving metadata for {clean_label}: {e}")

def extract_rsc_figures(file_path, output_dir, doi):
    """Extract figures from RSC HTML files"""
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
        
        # Find figure elements - RSC uses image_table divs for actual figures
        # Look specifically for image_table divs which contain the real figures
        image_tables = soup.find_all('div', class_='image_table')
        seen_figures = set()
        
        # Only create directory if we have figures to process
        if not image_tables:
            logger.info(f"No figures found in {file_path}")
            return 0
        
        # Create output directories only when we have figures
        fig_dir = os.path.join(output_dir, 'figures', doi.replace('/', '@'))
        os.makedirs(fig_dir, exist_ok=True)
        logger.debug(f"Created figure directory: {fig_dir}")
        
        for i, image_table in enumerate(image_tables):
            # Look for image inside the table, even if deeply nested
            img_elem = image_table.find('img')
            if not img_elem:
                # Look for embedded image inside table cells
                table_elem = image_table.find('table')
                if table_elem:
                    img_elem = table_elem.find('img')

            # If there's still no image, skip it
            if not img_elem:
                logger.debug(f"No <img> found in image_table {i}, skipping")
                continue
                
            # Try to get high-resolution version first
            src = None
            # Look for high-resolution link in parent anchor
            parent_link = img_elem.find_parent('a')
            if parent_link and parent_link.get('href'):
                href = parent_link.get('href')
                # Check if it's a high-resolution version
                if '_hi-res' in href or 'hi-res' in href:
                    src = href
                else:
                    # Fallback to regular img src
                    src = img_elem.get('src', '')
            else:
                src = img_elem.get('src', '')
            
            if not src:
                continue
            
            # Skip decorative images - only process actual figure images
            if any(skip_word in src.lower() for skip_word in ['icon', 'logo', 'orcid', 'char_', 'entities']):
                continue
            
            # Look for figure label and caption within the image_table
            figure_label = None
            caption = ""
            
            # Try to find figure number in the image_title
            image_title = image_table.find('td', class_='image_title')
            if image_title:
                title_text = image_title.get_text(strip=True)
                # Look for "Fig. X" or "Figure X" pattern
                fig_match = re.search(r'(?:Fig\.?|Figure)\s*(\d+)', title_text, re.IGNORECASE)
                if fig_match:
                    figure_label = f"Figure_{fig_match.group(1)}"
                
                # Extract caption from graphic_title
                graphic_title = image_title.find('span', class_='graphic_title')
                if graphic_title:
                    caption = graphic_title.get_text(strip=True)
                else:
                    # Fallback to the full title text
                    caption = title_text
            # Skip this image if there's no caption
            if not caption or len(caption.strip()) < 10:
                logger.debug(f"Skipping image {i} due to missing or short caption")
                continue
            
            # If no label found, use generic label
            if not figure_label:
                figure_label = f"Figure_{i+1}"
            
            # Skip if we've already processed this figure
            if figure_label in seen_figures:
                continue
            seen_figures.add(figure_label)
            
            # Make URL absolute if relative
            if src.startswith('//'):
                url = f"https:{src}"
            elif src.startswith('/'):
                url = f"https://pubs.rsc.org{src}"
            elif src.startswith('http'):
                url = src
            else:
                url = f"https://pubs.rsc.org/{src}"
            
            # Download figure
            try:
                clean_label = _clean_figure_label(figure_label)
                # Extract file extension from URL, preserving original format
                img_ext = url.split("?")[0].split(".")[-1].lower()
                if not img_ext or len(img_ext) > 4:
                    img_ext = "jpg"
                # Prefer higher quality formats
                if img_ext == 'gif' and '_hi-res' in url:
                    # For high-res GIFs, try to get better format if available
                    # Check if there's a TIFF version mentioned in alt text
                    alt_text = img_elem.get('alt', '')
                    if '.tif' in alt_text.lower():
                        # Try TIFF version first
                        tiff_url = url.replace('_hi-res.gif', '.tif')
                        # We'll try this URL first in the download section
                        pass
                img_name = f"{clean_label}.{img_ext}"
                img_path = os.path.join(fig_dir, img_name)
                
                headers = {
                    'Accept': 'image/*',
                    'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                # Try to download higher quality version first
                download_successful = False
                final_url = url
                
                # If we have a high-res GIF, try TIFF version first
                if img_ext == 'gif' and '_hi-res' in url and '.tif' in alt_text.lower():
                    tiff_url = url.replace('_hi-res.gif', '.tif')
                    try:
                        r = get_session().get(tiff_url, headers=headers, timeout=15)
                        if r.status_code == 200:
                            content_type = r.headers.get('content-type', '').lower()
                            if 'image' in content_type or r.content.startswith(b'II*\x00') or r.content.startswith(b'MM\x00*'):
                                # Update file extension and path for TIFF
                                img_name = f"{clean_label}.tif"
                                img_path = os.path.join(fig_dir, img_name)
                                with open(img_path, "wb") as f:
                                    f.write(r.content)
                                logger.info(f"Downloaded high-quality TIFF {figure_label} -> {img_path}")
                                final_url = tiff_url
                                download_successful = True
                    except Exception as e:
                        logger.debug(f"TIFF download failed for {figure_label}: {e}")
                
                # If TIFF failed, try the original high-res URL
                if not download_successful:
                    try:
                        r = get_session().get(url, headers=headers, timeout=15)
                        if r.status_code == 200:
                            content_type = r.headers.get('content-type', '').lower()
                            if 'image' in content_type or r.content.startswith(b'\xff\xd8') or r.content.startswith(b'\x89PNG') or r.content.startswith(b'GIF87a') or r.content.startswith(b'GIF89a'):
                                with open(img_path, "wb") as f:
                                    f.write(r.content)
                                logger.info(f"Downloaded {figure_label} -> {img_path}")
                                download_successful = True
                            else:
                                logger.warning(f"Got non-image content for {figure_label}")
                        else:
                            logger.warning(f"Failed to download {figure_label} - Status: {r.status_code}")
                    except Exception as e:
                        logger.error(f"Error downloading {figure_label}: {e}")
                
                if download_successful:
                    # Save figure metadata
                    _save_figure_metadata(fig_dir, clean_label, figure_label, caption, final_url, doi)
                    figures.append({"label": figure_label, "caption": caption, "url": final_url})
            except Exception as e:
                logger.error(f"Error downloading {figure_label}: {e}")
        
        logger.info(f"Found {len(figures)} figures in {file_path}")
        return len(figures)
        
    except Exception as e:
        logger.error(f"Error processing RSC file {file_path}: {e}")
        return 0

def extract_rsc_tables(file_path, output_dir, doi, llm_extractor):
    logger = logging.getLogger('extraction')
    try:
        logger.info(f"Extracting tables with LLM from {file_path}")
        tables = llm_extractor.extract_tables_from_file(file_path)
        logger.info(f"LLM found {len(tables)} tables")

        if tables:
            saved_files = llm_extractor.save_tables_to_csv(tables, output_dir, doi)
            logger.info(f"Saved {len(saved_files)} table CSV and metadata files")
            return len(tables), saved_files
        else:
            logger.info(f"No tables found in {file_path}")
            return 0, []

    except Exception as e:
        logger.error(f"Error extracting tables from {file_path}: {e}")
        return 0, []
