#!/usr/bin/env python3
"""
LLM-based table extraction using OpenAI GPT-4
Robust extraction of tables from XML/HTML content
"""

import os
import json
import xml.etree.ElementTree as ET
import requests
from openai import OpenAI
from typing import List, Dict, Any, Optional
import sys
import logging
from dotenv import load_dotenv
import traceback
import pandas as pd
from bs4 import Tag
from pathlib import Path
sys.path.append('/home/wschertzer3/AEM_OCR/src/ui')
from ui.openai_cost import append_usage, stage_from_env, run_id_from_env

load_dotenv()

sys.path.append('/home/wschertzer3/paper-downloader/paper_downloader')
from paper_downloader import config

def make_client(llm_url, api_key_string):
    base_url = os.getenv("LLM_BASE_URL", llm_url)
    api_key = os.getenv("LLM_API_KEY", api_key_string)
    return OpenAI(base_url=base_url, api_key=api_key)

class LLMTableExtractor:
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the LLM table extractor
        
        Args:
            model: model to use
        """
        self.model = model
        self.logger = logging.getLogger('extraction')
        
        api_key = os.getenv("OPENAI_API_KEY")
        org_id = os.getenv("OPENAI_ORGANIZATION")
        self.client = OpenAI(api_key=api_key, organization=org_id)
        # self.client = make_client(llm_url='http://127.0.0.1:11434/v1', api_key_string='ollama')
        
    
    def extract_tables_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables from a file using hybrid approach:
        1. Find table elements using rule-based parsing
        2. Send individual table XML to LLM for processing
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of extracted tables
        """
        try:
            self.logger.debug(f"Reading file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.logger.debug(f"File size: {len(content)} characters")
            
            if file_path.endswith('.xml'):
                self.logger.debug(f"Processing as XML with hybrid approach...")
                result = self.extract_tables_hybrid_xml(content)
                self.logger.debug(f"Hybrid processing result: {len(result)} tables")
                return result
            else:
                # For HTML files, use HTML-specific approach
                self.logger.debug(f"Processing as HTML with hybrid approach...")
                result = self.extract_tables_hybrid_html(content)
                self.logger.debug(f"Hybrid processing result: {len(result)} tables")
                return result
                
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def extract_tables_hybrid_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """
        Hybrid approach: Find table elements with rule-based parsing, then use LLM for processing
        
        Args:
            xml_content: Raw XML content as string
            
        Returns:
            List of extracted tables with metadata
        """
        try:
            # Parse XML to find table elements
            tree = ET.fromstring(xml_content)
            
            # Register namespaces
            ns = {
                "ce": "http://www.elsevier.com/xml/common/dtd",
                "cals": "http://www.elsevier.com/xml/common/cals/dtd",
                "svapi": "http://www.elsevier.com/xml/svapi/article/dtd"
            }
            
            # Find all table elements
            table_elements = tree.findall(".//ce:table", ns)
            self.logger.debug(f"Found {len(table_elements)} table elements using rule-based parsing")
            
            if not table_elements:
                self.logger.warning(f"No table elements found")
                return []
            
            # Process each table with LLM
            all_tables = []
            for i, table_elem in enumerate(table_elements):
                self.logger.debug(f"Processing table {i+1}/{len(table_elements)} with LLM...")
                
                # Convert table element to string
                table_xml = ET.tostring(table_elem, encoding='unicode')
                
                # Send individual table to LLM
                table_result = self.extract_single_table_with_llm_xml(table_xml)

                if table_result:
                    all_tables.append(table_result)
                    self.logger.debug(f"Successfully processed table {i+1}")
                else:
                    self.logger.warning(f"Failed to process table {i+1} from xml")
                    print('===table_result xml====', table_result)
                    quit()
            
            return all_tables
            
        except Exception as e:
            self.logger.error(f"Error in hybrid extraction: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def extract_tables_hybrid_html(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Hybrid extraction for HTML files (RSC, Springer, Wiley, etc.):
        Uses BeautifulSoup to find <table> elements and surrounding captions,
        including cases where tables are wrapped in <div class="table"> or <figure class="table">.
        Then uses LLM for semantic parsing of each table.
        """

        from bs4 import BeautifulSoup

        try:
            # --- 1. Parse HTML robustly ---
            soup = BeautifulSoup(html_content, "lxml")
            # Remove scripts/styles to simplify tree
            for s in soup(["script", "style"]):
                s.decompose()

            # --- 2. Find table containers (div.table, figure.table, or bare table) ---
            table_containers = []

            # Find explicit <table> tags
            table_containers.extend(soup.find_all("table"))

            # Find wrappers that include “table” in their class name (e.g. rtable__inner)
            for div in soup.find_all(["div", "figure"], recursive=True):
                cls = " ".join(div.get("class", []))
                if "table" in cls.lower():  # catches table, rtable, rtable__inner, image_table, etc.
                    # ensure we actually have a table inside
                    if div.find("table"):
                        table_containers.append(div)

            # Deduplicate
            table_containers = list(dict.fromkeys(table_containers))
            self.logger.debug(f"Found {len(table_containers)} potential table containers (expanded search)")

            # If nothing found, fall back to all <table>
            if not table_containers:
                table_containers = soup.find_all("table")

            self.logger.debug(f"Found {len(table_containers)} potential table containers")
            all_tables = []

            # --- 3. Iterate through containers ---
            for i, container in enumerate(table_containers):
                # Determine if this is an actual <table> or a wrapper
                table_elem = container if container.name == "table" else container.find("table")
                if not table_elem:
                    continue

                # Skip purely image-based “tables”
                if table_elem.find("img") and not table_elem.find("th"):
                    self.logger.debug(f"Skipping image-like table {i+1}")
                    continue

                # --- 4. Locate caption ---
                caption_text = None
                # Look for captions inside the container
                caption_tag = container.find(["caption", "p", "div"], class_=["caption", "table-caption"])
                if caption_tag:
                    caption_text = caption_tag.get_text(strip=True)
                # Look for RSC table_caption blocks nearby
                if not caption_text:
                    table_caption = container.find_previous("div", class_="table_caption")
                    if table_caption:
                        caption_text = table_caption.get_text(strip=True)
                if not caption_text:
                    # Try sibling before or after
                    prev = container.find_previous_sibling(["p", "div"])
                    if prev and "table" in prev.get_text().lower():
                        caption_text = prev.get_text(strip=True)
                    elif container.parent:
                        # Sometimes caption is parent text
                        parent_caption = container.parent.find(["p", "div"], class_=["caption", "table-caption"])
                        if parent_caption:
                            caption_text = parent_caption.get_text(strip=True)

                # --- 5. Extract table HTML and send to LLM ---
                table_html = str(table_elem)
                self.logger.debug(f"Processing table {i+1}/{len(table_containers)} with caption: {caption_text}")

                table_result = self.extract_single_table_with_llm_html(table_html, caption_text)
                if table_result:
                    all_tables.append(table_result)
                else:
                    self.logger.warning(f"Failed to process table {i+1} from html")
                    print('===table_result html====', table_result)
                    quit()

            return all_tables

        except Exception as e:
            self.logger.error(f"Error in HTML hybrid extraction: {e}", exc_info=True)
            return []

    def _find_table_caption(self, table_elem, soup):
        """Robustly detect the caption for a given table using nearby HTML context."""
        try:
            # 1. Check for <caption> tag inside the table
            caption_tag = table_elem.find('caption')
            if caption_tag:
                return caption_tag.get_text(strip=True)

            # 2. Check if the table itself has a class indicating a caption (not typical but possible)
            table_class = table_elem.get('class', [])
            if any("caption" in cls.lower() for cls in table_class):
                self.logger.debug("Found table with class indicating caption")
                return table_elem.get_text(strip=True)[:300]  # prevent overrun

            # 3. Check the previous sibling elements for a <div> or <p> with a table caption
            previous = table_elem
            for _ in range(5):  # Look up to 5 steps back
                previous = previous.find_previous_sibling()
                if not previous or not isinstance(previous, Tag):
                    continue

                class_list = previous.get('class', [])
                text = previous.get_text(strip=True)

                if any("caption" in cls.lower() for cls in class_list) and "table" in text.lower():
                    return text
                if text.lower().startswith("table"):
                    return text

            # 4. Look higher up in parent containers
            current = table_elem.parent
            while current and current.name != 'body':
                for tag in current.find_all(['div', 'p', 'header', 'span'], recursive=False):
                    class_list = tag.get('class', [])
                    text = tag.get_text(strip=True)
                    if any("caption" in cls.lower() for cls in class_list) and "table" in text.lower():
                        return text
                    if text.lower().startswith("table"):
                        return text
                current = current.parent

            # 5. Final fallback: global search for <div class="table_caption"> just before table
            all_divs = soup.find_all("div", class_="table_caption")
            for div in all_divs:
                if div.find_next("table") == table_elem:
                    return div.get_text(strip=True)

            return ""

        except Exception as e:
            self.logger.debug(f"[Caption Detection Error] {e}")
            return ""
        
    def extract_single_table_with_llm_xml(self, table_xml: str) -> Dict[str, Any]:
        """
        Extract a single table using LLM
        
        Args:
            table_xml: XML string of a single table element
            
        Returns:
            Dictionary with table data or None if failed
        """
        try:
            prompt = self._create_single_table_prompt_xml(table_xml)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting structured table data from XML files. Extract the table data and return it as JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=3000,
                temperature=0.0
            )
            self._log_cost(response,extra={"kind": "table_xml", "max_tokens": 3000, "temperature": 0.0})
            
            result = response.choices[0].message.content
            print('====PROMPT====', prompt)
            print('===RESPONSE====', response)
            print('====RESULT====', result)
            result = self._parse_single_table_response(result)
            if result:
                return result
            else:
                # retry if result is None
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at extracting structured table data from XML files. Extract the table data and return it as JSON. Return ONLY valid JSON. No explanations."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=3000,
                    temperature=0.0
                )
                self._log_cost(response,extra={"kind": "table_xml", "max_tokens": 3000, "temperature": 0.0})

                result = response.choices[0].message.content
                print('====RETRY RESULT====', result)
                print('===RETRY RESPONSE====', response)
                return self._parse_single_table_response(result)
            
        except Exception as e:
            self.logger.error(f"Error processing single XML table with LLM: {e}")
            return None
    
    def _create_single_table_prompt_xml(self, table_xml: str) -> str:
        """Create prompt for single table extraction"""
        return f"""
        Extract the table data from this XML table element. 

        XML Table:
        {table_xml}

        Return ONLY valid JSON in this exact format:
        {{
            "label": "Table 1",
            "caption": "Full table caption text",
            "headers": ["Column 1", "Column 2", "Column 3"],
            "data": [
                ["Row 1 Col 1", "Row 1 Col 2", "Row 1 Col 3"],
                ["Row 2 Col 1", "Row 2 Col 2", "Row 2 Col 3"]
            ],
            "notes": ["Any footnotes or notes"],
            "metadata": {{
                "row_count": 2,
                "column_count": 3,
                "has_merged_cells": false
            }}
        }}

        Extract:
        1. Label
        2. Caption
        3. Headers
        4. Data
        5. Clean up text content (remove XML tags, preserve data values)
        6. Handle merged cells appropriately (If there are merged cells, append merged content to connected cells)

        CRITICAL RULE FOR MERGED AND MULTI-ROW HEADERS:
        - If a parent header spans multiple columns (using namest/nameend),
        and there is a second header row that contains labels such as
        "Before" / "After", "30°C" / "80°C", "Cal."/"Exp", "in-plane"/"through-plane", etc.,
        you MUST combine them into final column headers.

        IMPORTANT: Use the actual caption text provided above, not placeholder text.
        """
    
    def _parse_single_table_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for single table"""
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                print('====start_idx == -1 or end_idx == 0====', start_idx, end_idx)
                return None
            
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing single table response: {e}")
            return None
    
    def extract_single_table_with_llm_html(self, table_html: str, caption_text: str = "") -> Dict[str, Any]:
        """
        Extract a single HTML table using LLM
        
        Args:
            table_html: HTML string of a single table element
            caption_text: Caption text found in surrounding context
            
        Returns:
            Dictionary with table data or None if failed
        """
        try:
            prompt = self._create_single_table_html_prompt(table_html, caption_text)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting structured table data from HTML files. Extract the table data and return it as JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=3000,
                temperature=0.0
            )
            self._log_cost(response,extra={"kind": "table_html", "max_tokens": 3000, "temperature": 0.0})
            
            result = response.choices[0].message.content
            print('====PROMPT====', prompt)
            print('===RESPONSE====', response)
            print('====RESULT====', result)
            result = self._parse_single_table_response(result)
            if result:
                return result
            else:
                # retry if result is None
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at extracting structured table data from html files. Extract the table data and return it as JSON. Return ONLY valid JSON. No explanations."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=3000,
                    temperature=0.0
                )
                self._log_cost(response,extra={"kind": "table_html", "max_tokens": 3000, "temperature": 0.0})
                result = response.choices[0].message.content
                print('====RETRY RESULT====', result)
                print('===RETRY RESPONSE====', response)
                return self._parse_single_table_response(result)
            
        except Exception as e:
            self.logger.error(f"Error processing single HTML table with LLM: {e}")
            return None
    
    def _create_single_table_html_prompt(self, table_html: str, caption_text: str = "") -> str:
        """Create prompt for single HTML table extraction"""
        caption_context = f"\n\nTable Caption: {caption_text}" if caption_text else ""
        
        return f"""
        Extract the table data from this HTML table element.{caption_context}

        HTML Table:
        {table_html}

        Return ONLY valid JSON in this exact format:
        {{
            "label": "Table 1",
            "caption": "Full table caption text",
            "headers": ["Column 1", "Column 2", "Column 3"],
            "data": [
                ["Row 1 Col 1", "Row 1 Col 2", "Row 1 Col 3"],
                ["Row 2 Col 1", "Row 2 Col 2", "Row 2 Col 3"]
            ],
            "notes": ["Any footnotes or notes"],
            "metadata": {{
                "row_count": 2,
                "column_count": 3,
                "has_merged_cells": false
            }}
        }}

        Extract:
        1. Label
        2. Caption
        3. Headers
        4. Data
        5. Clean up text content (remove HTML tags, preserve data values)
        6. Handle merged cells appropriately (If there are merged cells, append merged content to connected cells)

        CRITICAL RULE FOR MERGED AND MULTI-ROW HEADERS:
        - If a parent header spans multiple columns (using namest/nameend),
        and there is a second header row that contains labels such as
        "Before" / "After", "30°C" / "80°C", "Cal."/"Exp", "in-plane"/"through-plane", etc.,
        you MUST combine them into final column headers.

        IMPORTANT: Use the actual caption text provided above, not placeholder text.
        """
    
    def _log_cost(self, response, extra: dict):
        """
        Writes a JSONL record to <run_dir>/cost/usage.jsonl
        Requires env var AEM_OCR_RUN_DIR (or falls back to CWD).
        """
        append_usage(
            stage=stage_from_env("extract_tables"),
            run_id=run_id_from_env("unknown"),
            resp=response,
            extra=extra or {},
        )

    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse the LLM response and extract tables"""
        try:
            # Try to extract JSON from the response
            # Sometimes the LLM includes extra text
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                self.logger.warning(f"No JSON found in LLM response")
                return []
            
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            
            if 'tables' in result:
                return result['tables']
            else:
                self.logger.warning(f"No 'tables' key in LLM response")
                return []
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing LLM response as JSON: {e}")
            self.logger.debug(f"Response: {response[:500]}...")
            return []
        except Exception as e:
            self.logger.error(f"Error processing LLM response: {e}")
            return []
    
    def save_tables_to_csv(self, tables: List[Dict[str, Any]], output_dir: str, doi: str) -> List[str]:
        """
        Save extracted tables to CSV files
        
        Args:
            tables: List of extracted tables
            output_dir: Output directory
            doi: DOI for organizing files
            
        Returns:
            List of saved CSV file paths
        """
        import pandas as pd
        
        saved_files = []
        table_dir = os.path.join(output_dir, 'tables', doi.replace('/', '@'))
        os.makedirs(table_dir, exist_ok=True)
        
        for i, table in enumerate(tables):
            try:
                print("====TABLE====", table)
                # Create filename
                label = table.get('label', f'Table_{i+1}')
                filename = f"{label.replace(' ', '_').replace('.', '_')}.csv"
                filepath = os.path.join(table_dir, filename)
                
                # Create DataFrame
                headers = table.get('headers', [])
                data = table.get('data', [])

                print("====HEADERS====", headers)
                print("====DATA====", data)
                
                if headers and data:
                    df = pd.DataFrame(data, columns=headers)
                elif data:
                    # No headers, use generic column names
                    df = pd.DataFrame(data, columns=[f"Column_{j+1}" for j in range(len(data[0]) if data else 0)])
                else:
                    self.logger.warning(f"No data found for {label}")
                    continue
                
                # Save to CSV
                df.to_csv(filepath, index=False)
                saved_files.append(filepath)
                self.logger.info(f"Saved {label} -> {filepath}")
                
                # Save metadata
                metadata_file = filepath.replace('.csv', '_metadata.json')
                
                # Clean caption - remove extra whitespace and newlines
                caption = table.get('caption', '')
                clean_caption = caption.strip() if caption else ""
                
                with open(metadata_file, 'w') as f:
                    json.dump({
                        'label': table.get('label', f'Table_{i+1}'),
                        'caption': clean_caption,
                        'notes': table.get('notes', []),
                        'metadata': table.get('metadata', {}),
                        'doi': doi,
                        'table_file_path': filepath,
                        'table_directory': table_dir,
                        'extraction_timestamp': pd.Timestamp.now().isoformat(),
                        'file_type': 'table'
                    }, f, indent=2)
                
            except Exception as e:
                self.logger.error(f"Error saving table {i+1}: {e}")
        
        return saved_files

