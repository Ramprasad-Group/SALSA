#!/usr/bin/env python3
"""
Elsevier publisher figure and table extraction
Handles XML files from Elsevier journals
"""

import os
import sys
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import logging
import json
import time

sys.path.append('/data/sonakshi/paper-downloader')
from paper_downloader import config

def extract_elsevier_figures(file_path, output_dir, doi):
    """Extract figures using rule-based approach (proven to work)"""
    logger = logging.getLogger('extraction')
    try:
        # Register namespaces
        ns = {
            "ce": "http://www.elsevier.com/xml/common/dtd",
            "cals": "http://www.elsevier.com/xml/common/cals/dtd",
            "svapi": "http://www.elsevier.com/xml/svapi/article/dtd"
        }
        
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        figures = []
        
        # Check if we have any figures before creating directory
        figure_elements = root.findall('.//ce:figure', ns)
        if not figure_elements:
            logger.info(f"No figures found in {file_path}")
            return 0
        
        # Create output directories only when we have figures
        fig_dir = os.path.join(output_dir, 'figures', doi.replace('/', '@'))
        os.makedirs(fig_dir, exist_ok=True)
        logger.debug(f"Created figure directory: {fig_dir}")
        
        # Extract Figures
        for fig in root.findall(".//ce:figure", ns):
            label = fig.findtext("ce:label", default="", namespaces=ns)
            caption = " ".join(fig.find("ce:caption", ns).itertext()) if fig.find("ce:caption", ns) is not None else ""
            locator = fig.find("ce:link", ns).attrib.get("locator") if fig.find("ce:link", ns) is not None else None

            # Match locator -> object URL
            url = None
            if locator:
                for obj in root.findall(".//svapi:object", ns):
                    if obj.attrib.get("ref") == locator and obj.attrib.get("category") == "standard":
                        url = (obj.text or "").strip()
                        break

            if url:
                max_retries = 5
                for i in range(max_retries):
                    try:
                        # Clean up the label for filename
                        clean_label = _clean_figure_label(label)
                        img_ext = url.split("?")[0].split(".")[-1]
                        img_name = f"{clean_label}.{img_ext}"
                        img_path = os.path.join(fig_dir, img_name)

                        headers = {
                            'Accept': 'image/*',
                            'User-agent': 'Mozilla/5.6 (Windows NT 11.0; rv:68.0 Gecko/20100101 Firefox/70.0)'
                        }
                        
                        # Add API key to the URL if it's an Elsevier API URL
                        if 'api.elsevier.com' in url:
                            separator = '&' if '?' in url else '?'
                            url_with_key = f"{url}{separator}APIKey={config.ELS_API_KEY}"
                        else:
                            url_with_key = url

                        r = requests.get(url_with_key, headers=headers, timeout=(5,60))
                        if r.status_code == 200:
                            content_type = r.headers.get('content-type', '').lower()
                            if 'image' in content_type or r.content.startswith(b'\xff\xd8') or r.content.startswith(b'\x89PNG'):
                                with open(img_path, "wb") as f:
                                    f.write(r.content)
                                logger.info(f"Downloaded {label} -> {img_path}")
                                _save_figure_metadata(fig_dir, clean_label, label, caption, url, doi)
                                break
                            else:
                                logger.warning(f"Got non-image content for {label}")
                        else:
                            logger.warning(f"Failed to download {label} - Status: {r.status_code}")
                    except Exception as e:
                        logger.error(f"Error downloading {label}: {e}")
                        time.sleep(5)

            figures.append({"label": label, "caption": caption, "url": url})
        
        logger.info(f"Found {len(figures)} figures in {file_path}")
        return len(figures)
        
    except Exception as e:
        logger.error(f"Error processing Elsevier file {file_path}: {e}")
        return 0

def extract_elsevier_tables(file_path, output_dir, doi, llm_extractor):
    """Extract tables using LLM approach"""
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
        "publisher": "elsevier"
    }
    
    metadata_file = os.path.join(fig_dir, f"{clean_label}_metadata.json")
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved metadata: {metadata_file}")
    except Exception as e:
        logger.warning(f"Failed to save metadata: {e}")