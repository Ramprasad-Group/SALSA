#!/usr/bin/env python3
"""
LLM-based table extraction using OpenAI GPT-4
Robust extraction of tables from XML/HTML content
"""

import os
import json
import xml.etree.ElementTree as ET
import requests
from typing import List, Dict, Any, Optional
import sys
import logging
from dotenv import load_dotenv
import traceback
import pandas as pd
from bs4 import Tag
from pathlib import Path
_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _SRC_ROOT)

from llm_client import get_llm_client
from ui.openai_cost import append_usage, run_id_from_env

load_dotenv()

_PAPER_DL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PAPER_DL)
from paper_downloader import config


class LLMTableExtractor:

    def __init__(self, custom_table_extraction_rules: str = ""):
        """
        Initialize the LLM table extractor

        Args:
            custom_table_extraction_rules: Extra rules appended to the table
                extraction system prompt (e.g. merged-cell handling).
        """
        self.logger = logging.getLogger('extraction')
        self.client = get_llm_client()
        self.custom_table_extraction_rules = custom_table_extraction_rules or ""
    
    def _xml_text(self, elem: Optional[ET.Element]) -> str:
        """Flatten an XML element to clean text."""
        if elem is None:
            return ""
        # join itertext, normalize whitespace
        txt = " ".join(" ".join(elem.itertext()).split())
        return txt.strip()

    def _find_xml_caption(self,
                        table_elem: ET.Element,
                        parent_map: Dict[ET.Element, ET.Element],
                        ns: Dict[str, str]) -> str:
        """
        Robust caption finder for Elsevier-ish XML (ce/cals).
        Tries:
        - caption/label inside the table (rare)
        - caption/label in nearest ancestor container (common: ce:table-wrap)
        - nearby siblings of that container
        Returns a single string (can be empty).
        """

        def first(xpath: str, ctx: ET.Element) -> Optional[ET.Element]:
            return ctx.find(xpath, ns)

        def findall(xpath: str, ctx: ET.Element) -> List[ET.Element]:
            return ctx.findall(xpath, ns)

        # -------- 1) inside the table itself (sometimes present) --------
        # Elsevier captions are usually not inside ce:table, but safe to try.
        caption = first(".//ce:caption", table_elem)
        label = first(".//ce:label", table_elem)
        cap_txt = self._xml_text(caption)
        label_txt = self._xml_text(label)

        if cap_txt:
            return f"{label_txt} {cap_txt}".strip()

        # -------- 2) climb to nearest likely container (e.g., ce:table-wrap) --------
        # Walk upward and pick the first ancestor that contains ce:caption or ce:label.
        cur = table_elem
        container = None
        for _ in range(12):  # avoid infinite loops on weird trees
            parent = parent_map.get(cur)
            if parent is None:
                break
            # common wrappers in publisher XML:
            # ce:table-wrap, ce:float, ce:float-group, etc.
            if first(".//ce:caption", parent) is not None or first(".//ce:label", parent) is not None:
                container = parent
                break
            cur = parent

        if container is not None:
            label = first(".//ce:label", container)
            caption = first(".//ce:caption", container)

            # Elsevier often uses: <ce:caption><ce:simple-para>...</ce:simple-para></ce:caption>
            cap_txt = self._xml_text(caption)
            label_txt = self._xml_text(label)
            if cap_txt or label_txt:
                return f"{label_txt} {cap_txt}".strip()

            # fallback: sometimes caption text is split into multiple paras
            paras = findall(".//ce:caption//ce:simple-para", container)
            if paras:
                cap_txt = " ".join(self._xml_text(p) for p in paras if self._xml_text(p))
                return f"{label_txt} {cap_txt}".strip()

        # -------- 3) fallback: look at siblings of the container (if any) --------
        # Some XML puts captions in a preceding sibling node.
        if container is not None:
            parent = parent_map.get(container)
            if parent is not None:
                siblings = list(parent)
                try:
                    idx = siblings.index(container)
                except ValueError:
                    idx = -1

                # check a few siblings before/after for caption-ish nodes
                for j in range(max(0, idx - 3), min(len(siblings), idx + 4)):
                    sib = siblings[j]
                    cap = first(".//ce:caption", sib)
                    lab = first(".//ce:label", sib)
                    cap_txt = self._xml_text(cap)
                    lab_txt = self._xml_text(lab)
                    if cap_txt or lab_txt:
                        return f"{lab_txt} {cap_txt}".strip()

        return ""

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

            # --- Locate caption ---
            parent_map = {child: parent for parent in tree.iter() for child in parent}
            
            all_tables = []
            for i, table_elem in enumerate(table_elements):

                # --- Locate caption (XML) ---
                caption_text = self._find_xml_caption(table_elem, parent_map, ns)
                if caption_text is None:
                    caption_text = ""

                self.logger.debug(f"Processing table {i+1}/{len(table_elements)} with LLM...")
                
                # Convert table element to string
                table_xml = ET.tostring(table_elem, encoding='unicode')
                
                # Send individual table to LLM
                table_result = self.extract_single_table_with_llm_xml(table_xml, caption_text)

                if table_result:
                    all_tables.append(table_result)
                    self.logger.debug(f"Successfully processed table {i+1}")
                else:
                    self.logger.warning(f"Failed to process table {i+1} from xml")
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
            print(f"Found {len(table_containers)} potential table containers (expanded search)")
            self.logger.debug(f"Found {len(table_containers)} potential table containers (expanded search)")

            # If nothing found, fall back to all <table>
            if not table_containers:
                table_containers = soup.find_all("table")

            print(f"Found {len(table_containers)} potential table containers")
            self.logger.debug(f"Found {len(table_containers)} potential table containers")
            all_tables = []

            # --- 3. Iterate through containers ---
            for i, container in enumerate(table_containers):
                # Determine if this is an actual <table> or a wrapper
                table_elem = container if container.name == "table" else container.find("table")
                # print('===table_elem===', table_elem)
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
        
    def extract_single_table_with_llm_xml(self, table_xml: str, caption_text: str = "") -> Dict[str, Any]:
        """
        Extract a single table using LLM
        
        Args:
            table_xml: XML string of a single table element
            
        Returns:
            Dictionary with table data or None if failed
        """
        try:
            
            custom_block = ""
            if self.custom_table_extraction_rules.strip():
                custom_block = (
                    "\n\nAdditional user-specified rules (MUST follow):\n"
                    + self.custom_table_extraction_rules.strip()
                )

            instructions = f"""
                You are an expert at extracting structured table data from XML files.
                Extract the table data and return it as JSON.

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
                - Label
                - Caption
                - Headers
                - Data

                Rules:
                - Use the actual caption text provided above, not placeholder text.
                - Clean up text content (remove XML tags, preserve data values).
                - Do not return anything aside from the final json content. No explanations or anything that is not json
                - If a parent header spans multiple columns,and there is a second header row that contains labels such as
                    "Before" / "After", "30°C" / "80°C", "Cal."/"Exp", "in-plane"/"through-plane", etc., you MUST factor them into final column headers.{custom_block}
                """

                

            prompt = self._create_single_table_prompt_xml(table_xml, caption_text)
            
            text, meta = self.client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": instructions
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=3000,
                temperature=0.0,
                return_meta=True,
            )
            try:
                self._log_cost(meta, extra={"kind": "table_xml"})
            except Exception as e:
                self.logger.warning(f"[cost] cost could not be calculated: {e}")
            
            result = self._parse_single_table_response(text)
            if result:
                return result
            else:
                self.logger.error(f"Error processing single XML table with LLM: No result")
                return None
            
        except Exception as e:
            self.logger.error(f"Error processing single XML table with LLM: {e}")
            return None
    
    def _create_single_table_prompt_xml(self, table_xml: str, caption_text: str = "") -> str:
        """Create prompt for single table extraction"""

        caption_context = f"{caption_text}" if caption_text else ""

        return f"""
        Extract the table data from this XML table element. 

        Caption: {caption_context}, 
    

        XML Table: {table_xml}

        """
    
    def _parse_single_table_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for single table"""

        print("===RESPONSE===", response)
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

            custom_block = ""
            if self.custom_table_extraction_rules.strip():
                custom_block = (
                    "\n\nAdditional user-specified rules (MUST follow):\n"
                    + self.custom_table_extraction_rules.strip()
                )

            instructions = f"""
                You are an expert at extracting structured table data from HTML files.
                Extract the table data and return it as JSON.

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
                - Label
                - Caption
                - Headers
                - Data

                Rules:
                - Use the actual caption text provided above, not placeholder text.
                - Clean up text content (remove HTML tags, preserve data values).
                - Do not return anything aside from the final json content. No explanations or anything that is not json
                - If a parent header spans multiple columns,and there is a second header row that contains labels such as
                    "Before" / "After", "30°C" / "80°C", "Cal."/"Exp", "in-plane"/"through-plane", etc., you MUST factor them into final column headers.{custom_block}
                """


            prompt = self._create_single_table_html_prompt(table_html, caption_text)

            
            text, meta = self.client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": instructions
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=3000,
                temperature=0.0,
                return_meta=True,
            )
            try:
                self._log_cost(meta, extra={"kind": "table_html"})
            except Exception as e:
                self.logger.warning(f"[cost] cost could not be calculated: {e}")
            
            result = self._parse_single_table_response(text)
            if result:
                return result
            else:
                self.logger.error(f"Error processing single HTML table with LLM: No Results")
                return None
            
        except Exception as e:
            self.logger.error(f"Error processing single HTML table with LLM: {e}")
            return None
    
    def _create_single_table_html_prompt(self, table_html: str, caption_text: str = "") -> str:
        """Create prompt for single HTML table extraction"""

        caption_context = f"{caption_text}" if caption_text else ""
        
        return f"""
        Extract the table data from this HTML table element.
        
        Caption: 
        {caption_context}

        HTML Table:
        {table_html}
        """
        

        # Return ONLY valid JSON in this exact format:
        # {{
        #     "label": "Table 1",
        #     "caption": "Full table caption text",
        #     "headers": ["Column 1", "Column 2", "Column 3"],
        #     "data": [
        #         ["Row 1 Col 1", "Row 1 Col 2", "Row 1 Col 3"],
        #         ["Row 2 Col 1", "Row 2 Col 2", "Row 2 Col 3"]
        #     ],
        #     "notes": ["Any footnotes or notes"],
        #     "metadata": {{
        #         "row_count": 2,
        #         "column_count": 3,
        #         "has_merged_cells": false
        #     }}
        # }}

        # Extract:
        # 1. Label
        # 2. Caption
        # 3. Headers
        # 4. Data
        # 5. Clean up text content (remove HTML tags, preserve data values)
        # 6. Handle merged cells appropriately (If there are merged cells, append merged content to connected cells)

        # CRITICAL RULE FOR MERGED AND MULTI-ROW HEADERS:
        # - If a parent header spans multiple columns,
        # and there is a second header row that contains labels such as
        # "Before" / "After", "30°C" / "80°C", "Cal."/"Exp", "in-plane"/"through-plane", etc.,
        # you MUST factor them into final column headers.

        # IMPORTANT: Use the actual caption text provided above, not placeholder text.
        # """
    
    def _log_cost(self, text, extra: dict):
        """
        Writes a JSONL record to <run_dir>/cost/usage.jsonl
        Requires env var SALSA_RUN_DIR (or falls back to CWD).
        """
        append_usage(
            run_id=run_id_from_env("unknown"),
            resp=text,
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
        
        saved_files = []
        table_dir = os.path.join(output_dir, 'tables', doi.replace('/', '@'))
        os.makedirs(table_dir, exist_ok=True)
        
        for i, table in enumerate(tables):
            try:
                # Create filename
                label = table.get('label', f'Table_{i+1}')
                filename = f"{label.replace(' ', '_').replace('.', '_')}.csv"
                filepath = os.path.join(table_dir, filename)
                
                # Create DataFrame
                headers = table.get('headers', [])
                data = table.get('data', [])
                
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
