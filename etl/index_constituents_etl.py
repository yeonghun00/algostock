#!/usr/bin/env python3
"""
KRX Index Constituents Direct Processor

This script extracts stock codes from KRX index data and inserts them directly
into the database using multiprocessing for optimal performance.

Features:
1. Backfill mode: Process historical data from 2010-01-01 to present
2. Update mode: Update from latest date with overwrite or skip options
3. Multiprocessing: Parallel processing of multiple indices
4. Direct database insertion (minimal file I/O)
5. Robust encoding handling with multiple fallbacks
6. Temporary file cleanup

Usage:
    python krx_index_constituents_direct.py --mode backfill
    python krx_index_constituents_direct.py --mode update --strategy overwrite
    python krx_index_constituents_direct.py --mode update --strategy skip
"""

import argparse
import sqlite3
import pandas as pd
import re
import sys
import os
import tempfile
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.actions.wheel_input import ScrollOrigin
from selenium.webdriver import ActionChains

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config


class KRXIndexConstituentsDirect:
    """Direct processor for KRX index constituents with multiprocessing."""
    
    BASE_URL = 'https://index.krx.co.kr'
    OTP_URL = f'{BASE_URL}/contents/COM/GenerateOTP.jspx'
    DOWNLOAD_URL = 'https://file.krx.co.kr/download.jspx'
    
    MARKET_CONFIG = {
        'kospi': {
            'listing_path': '/contents/MKD/03/0304/03040200/MKD03040200.jsp',
            'upmidCd_default': '0102',
            'ind_tp_cd': '1',
        },
        'kosdaq': {
            'listing_path': '/contents/MKD/03/0304/03040300/MKD03040300.jsp',
            'upmidCd_default': '0103',
            'ind_tp_cd': '2',
        },
    }

    def __init__(self, config_path: str = "config.json"):
        """Initialize the processor with database connection."""
        # Get the parent directory (project root) for config file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_full_path = os.path.join(project_root, config_path)
        self.config = load_config(config_full_path)
        self.db_path = self.config.get('database', {}).get('path', 'krx_stock_data.db')
        
        # Create database table if it doesn't exist
        self._create_table()
        
        # Initialize session for HTTP requests
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        }
        
        # Cache for scraped indices to avoid repeated scraping
        self._cached_indices = {}  # {market: indices_list}
    
    def _create_table(self):
        """Create the index_constituents and index_category_mapping tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS index_constituents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    index_code TEXT NOT NULL,
                    stock_code TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, index_code, stock_code)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS index_category_mapping (
                    index_code TEXT PRIMARY KEY,
                    category TEXT,
                    market TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def format_index_name(self, index_name: str, market: str) -> str:
        """
        Format index name with underscores and market prefix.
        
        Args:
            index_name (str): Original index name
            market (str): Market type ('kospi' or 'kosdaq')
            
        Returns:
            str: Formatted index name (e.g., 'KOSPI_코스피_200')
        """
        # Convert market to uppercase prefix
        market_prefix = market.upper()
        
        # Replace spaces with underscores
        formatted_name = index_name.replace(' ', '_')
        
        # Replace special characters that might cause issues
        formatted_name = re.sub(r'[^\w\s\(\)]', '_', formatted_name)
        
        # Remove multiple consecutive underscores
        formatted_name = re.sub(r'_+', '_', formatted_name)
        
        # Remove leading/trailing underscores
        formatted_name = formatted_name.strip('_')
        
        # Add market prefix
        return f"{market_prefix}_{formatted_name}"
    
    def _create_driver(self) -> webdriver.Chrome:
        """Create and configure a headless Chrome WebDriver."""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument(f'--user-agent={self.headers["User-Agent"]}')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        # Selenium 4.6+ auto-manages chromedriver (no webdriver-manager needed)
        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_script('Object.defineProperty(navigator, "webdriver", {get: () => undefined})')
        return driver

    def scrape_listing_page(self, market: str) -> List[Dict]:
        """Fetch the KRX index listing page using Selenium and extract index links."""
        config = self.MARKET_CONFIG[market]
        listing_url = f'{self.BASE_URL}{config["listing_path"]}'
        indices = []
        driver = None

        print(f"Fetching {market.upper()} listing page with Selenium: {listing_url}")

        try:
            driver = self._create_driver()
            driver.get(listing_url)

            # Wait for the page to load
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="MKD03040101.jsp"]')))

            print(f"Page loaded successfully")

            # Try to find the scrollable table container for virtual scrolling
            try:
                table_container = driver.find_element(By.CSS_SELECTOR, "tbody.CI-GRID-BODY-TABLE-TBODY")
                print("Found scrollable table container, scrolling to load all indices...")

                # Track unique links using a set of (name, href) tuples
                seen_links = set()
                scroll_attempts = 0
                max_scroll_attempts = 200  # Safety limit
                scroll_increment = 300  # Pixels to scroll each time
                no_progress_count = 0
                current_category = ''  # Track 구분 category across rows (rowspan pattern)

                while scroll_attempts < max_scroll_attempts:
                    # Iterate rows to track category (구분 uses rowspan, so some rows only have the link td)
                    rows = driver.find_elements(By.CSS_SELECTOR, 'tbody.CI-GRID-BODY-TABLE-TBODY tr')

                    new_links_found = 0
                    for row in rows:
                        tds = row.find_elements(By.TAG_NAME, 'td')
                        if not tds:
                            continue

                        # Determine if first td is a category cell or a link cell
                        first_td_links = tds[0].find_elements(By.CSS_SELECTOR, 'a[href*="MKD03040101.jsp"]')

                        if len(tds) >= 2 and not first_td_links:
                            # First td is the 구분 category cell; second td has the index link
                            cat_text = tds[0].text.strip()
                            if cat_text:
                                current_category = cat_text
                            link_elements = tds[1].find_elements(By.CSS_SELECTOR, 'a[href*="MKD03040101.jsp"]')
                        else:
                            # Continuation row: no category td (rowspan from above), first td has link
                            link_elements = first_td_links

                        for link in link_elements:
                            href = link.get_attribute('href') or ''
                            text = link.text.strip()

                            if not text:
                                continue

                            link_key = (text, href)
                            if link_key not in seen_links:
                                seen_links.add(link_key)
                                new_links_found += 1

                                parsed = urlparse(href)
                                params = parse_qs(parsed.query)

                                upmid_cd = params.get('upmidCd', [''])[0]
                                idx_cd = params.get('idxCd', [''])[0]
                                idx_id = params.get('idxId', [''])[0]

                                if idx_cd and idx_id:
                                    indices.append({
                                        'name': text,
                                        'href': href,
                                        'upmidCd': upmid_cd,
                                        'idxCd': idx_cd,
                                        'idxId': idx_id,
                                        'category': current_category,
                                    })
                    
                    # Use Selenium 4 Wheel Actions to scroll the table container
                    scroll_origin = ScrollOrigin.from_element(table_container)
                    ActionChains(driver).scroll_from_origin(scroll_origin, 0, scroll_increment).perform()
                    
                    # Wait a bit for new content to load
                    time.sleep(0.5)
                    
                    # Check if we've made progress
                    if new_links_found == 0:
                        no_progress_count += 1
                        if no_progress_count > 5:  # Give it a few attempts
                            print(f"  No new links found after {no_progress_count} scroll attempts, stopping")
                            break
                    else:
                        no_progress_count = 0
                    
                    scroll_attempts += 1
                    
                    if scroll_attempts % 10 == 0:
                        print(f"  Scrolled {scroll_attempts} times, found {len(indices)} indices so far...")
                
                print(f"Finished scrolling. Total indices found: {len(indices)}")
                
            except Exception as scroll_error:
                # Fallback: if we can't find the scrollable container, use the original method
                print(f"Could not find scrollable table container ({scroll_error}), using fallback method")

                # Iterate rows to track 구분 category (non-scrolling fallback)
                current_category = ''
                rows = driver.find_elements(By.CSS_SELECTOR, 'tr')

                for row in rows:
                    tds = row.find_elements(By.TAG_NAME, 'td')
                    if not tds:
                        continue

                    first_td_links = tds[0].find_elements(By.CSS_SELECTOR, 'a[href*="MKD03040101.jsp"]')

                    if len(tds) >= 2 and not first_td_links:
                        cat_text = tds[0].text.strip()
                        if cat_text:
                            current_category = cat_text
                        link_elements = tds[1].find_elements(By.CSS_SELECTOR, 'a[href*="MKD03040101.jsp"]')
                    else:
                        link_elements = first_td_links

                    for link in link_elements:
                        href = link.get_attribute('href') or ''
                        text = link.text.strip()

                        if not text:
                            continue

                        parsed = urlparse(href)
                        params = parse_qs(parsed.query)

                        upmid_cd = params.get('upmidCd', [''])[0]
                        idx_cd = params.get('idxCd', [''])[0]
                        idx_id = params.get('idxId', [''])[0]

                        if idx_cd and idx_id:
                            indices.append({
                                'name': text,
                                'href': href,
                                'upmidCd': upmid_cd,
                                'idxCd': idx_cd,
                                'idxId': idx_id,
                                'category': current_category,
                            })

        except Exception as e:
            print(f"Error during Selenium scraping: {e}")
            raise
        finally:
            if driver:
                driver.quit()

        print(f"Found {len(indices)} index links on listing page")
        return indices

    def resolve_real_codes(self, market: str, index_info: Dict) -> Optional[Dict]:
        """
        Visit the index detail page and extract the REAL idxCd/idxId
        from the tab links (e.g. MKD03040101T1.jsp?idxCd=1167&idxId=K2Z01P).
        """
        href = index_info.get('href', '')
        if not href:
            return None

        detail_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href

        try:
            response = self.session.get(detail_url, headers=self.headers, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"   Failed to fetch detail page: {e}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find tab links like MKD03040101T1.jsp, MKD03040101T3.jsp, etc.
        tab_links = soup.find_all('a', href=re.compile(r'MKD03040101T\d\.jsp'))

        for tab_link in tab_links:
            tab_href = tab_link.get('href', '')
            parsed = urlparse(tab_href)
            params = parse_qs(parsed.query)

            real_idx_cd = params.get('idxCd', [''])[0]
            real_idx_id = params.get('idxId', [''])[0]
            real_upmid_cd = params.get('upmidCd', [''])[0]

            if real_idx_cd and real_idx_id:
                return {
                    'idxCd': real_idx_cd,
                    'idxId': real_idx_id,
                    'upmidCd': real_upmid_cd or index_info.get('upmidCd', self.MARKET_CONFIG[market]['upmidCd_default']),
                }
            
        return None

    def generate_otp(self, market: str, idx_cd: str, idx_id: str, upmid_cd: str) -> Optional[str]:
        """Generate OTP token using the real index codes."""
        from_date = '20260203'
        to_date = '20260210'

        config = self.MARKET_CONFIG[market]
        ind_tp_cd = config['ind_tp_cd']
        idx_ind_cd = idx_cd[-3:] if len(idx_cd) >= 3 else idx_cd

        otp_params = {
            'name': 'fileDown',
            'filetype': 'csv',
            'url': '/IDX/03/0304/03040101/mkd03040101T3_01',
            'ind_tp_cd': ind_tp_cd,
            'idx_ind_cd': idx_ind_cd,
            'idx_id': idx_id,
            'lang': 'ko',
            'compst_isu_tp': '1',
            'schdate': from_date,
            'fromdate': from_date,
            'todate': to_date,
            'pagePath': '/contents/MKD/03/0304/03040101/MKD03040101T3.jsp',
        }

        headers = {
            **self.headers,
            'Referer': f'{self.BASE_URL}/contents/MKD/03/0304/03040101/MKD03040101.jsp?upmidCd={upmid_cd}&idxCd={idx_cd}&idxId={idx_id}',
            'X-Requested-With': 'XMLHttpRequest',
        }

        try:
            response = self.session.get(self.OTP_URL, params=otp_params,
                                        headers=headers, timeout=30)
            if response.status_code == 200:
                otp_code = response.text.strip()
                if len(otp_code) > 20:
                    return otp_code
            print(f"   OTP generation failed: {response.status_code}")
            return None
        except Exception as e:
            print(f"   OTP request error: {e}")
            return None

    def process_index_data(self, market: str, index_info: Dict) -> Optional[List[Tuple[str, str]]]:
        """
        Process a single index and return constituents.
        This function is designed to be run in a separate process.
        """
        name = index_info['name']
        listing_cd = index_info['idxCd']
        listing_id = index_info['idxId']
        upmid = index_info.get('upmidCd', self.MARKET_CONFIG[market]['upmidCd_default'])
        
        print(f"   Processing {name}")
        print(f"   Listing codes: idxCd={listing_cd}, idxId={listing_id}")

        # Build detail page URL and resolve real constituent codes
        detail_info = {
            'href': f"/contents/MKD/03/0304/03040101/MKD03040101.jsp?upmidCd={upmid}&idxCd={listing_cd}&idxId={listing_id}",
            'upmidCd': upmid,
        }
        real_codes = self.resolve_real_codes(market, detail_info)
        if not real_codes:
            print(f"   Could not resolve real codes, using listing codes")
            real_codes = {
                'idxCd': listing_cd,
                'idxId': listing_id,
                'upmidCd': upmid,
            }

        idx_cd = real_codes['idxCd']
        idx_id = real_codes['idxId']
        upmid_cd = real_codes['upmidCd']

        print(f"   Real codes: idxCd={idx_cd}, idxId={idx_id}")

        # Generate OTP with real codes
        otp_code = self.generate_otp(market, idx_cd, idx_id, upmid_cd)
        if not otp_code:
            return None

        # Download CSV content
        constituents = self.download_and_process_csv(market, otp_code, name)
        return constituents

    def download_and_process_csv(self, market: str, otp_code: str, index_name: str) -> Optional[List[Tuple[str, str]]]:
        """Download CSV content and extract constituents directly."""
        headers = {
            'Referer': f'{self.BASE_URL}/',
            'User-Agent': self.headers['User-Agent'],
            'Accept': '*/*',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Origin': self.BASE_URL,
        }

        try:
            response = self.session.post(self.DOWNLOAD_URL, data={'code': otp_code},
                                         headers=headers, timeout=30)

            if response.status_code != 200:
                print(f"   Download failed: HTTP {response.status_code}")
                return None

            # Try to decode with multiple encodings
            csv_content = self._decode_csv_content(response.content)
            if csv_content is None:
                print(f"   Failed to decode CSV content")
                return None

            # Extract constituents from CSV content
            constituents = self.extract_constituents_from_content(csv_content, index_name, market)
            return constituents

        except Exception as e:
            print(f"   Download error: {e}")
            return None

    def _decode_csv_content(self, content: bytes) -> Optional[str]:
        """Try to decode CSV content with multiple fallback encodings."""
        encodings = ['cp949', 'utf-8', 'euc-kr', 'latin1']
        
        for encoding in encodings:
            try:
                decoded_content = content.decode(encoding)
                # Basic validation - check if it looks like CSV
                if ',' in decoded_content and '종목코드' in decoded_content:
                    return decoded_content
            except UnicodeDecodeError:
                continue
        
        return None

    def extract_constituents_from_content(self, csv_content: str, index_name: str, market: str) -> List[Tuple[str, str]]:
        """
        Extract stock codes directly from CSV content.
        Downloads to temporary file only as fallback, then deletes immediately.
        """
        try:
            # Try to read directly from string first
            try:
                from io import StringIO
                df = pd.read_csv(StringIO(csv_content))
                print(f"   Successfully read CSV from memory")
            except Exception as e:
                print(f"   Memory read failed: {e}, trying temporary file fallback")
                
                # Fallback: write to temporary file, read, then delete
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as temp_file:
                    temp_file.write(csv_content.encode('utf-8'))
                    temp_path = temp_file.name
                
                try:
                    df = pd.read_csv(temp_path)
                    print(f"   Successfully read CSV from temporary file")
                finally:
                    # Always delete the temporary file
                    try:
                        os.unlink(temp_path)
                        print(f"   Deleted temporary file")
                    except Exception as cleanup_error:
                        print(f"   Warning: Could not delete temporary file: {cleanup_error}")
            
            # Find the column with stock codes (should be '종목코드')
            stock_code_column = None
            for col in df.columns:
                if '종목코드' in col:
                    stock_code_column = col
                    break
            
            if not stock_code_column:
                print(f"   Warning: Could not find '종목코드' column")
                return []
            
            # Extract stock codes
            stock_codes = df[stock_code_column].dropna().astype(str).tolist()
            
            # Format index name
            formatted_index_name = self.format_index_name(index_name, market)
            
            # Create tuples for database insertion
            constituents = [(formatted_index_name, stock_code) for stock_code in stock_codes]
            
            print(f"   Extracted {len(constituents)} constituents for {formatted_index_name}")
            return constituents
            
        except Exception as e:
            print(f"   Error processing CSV content: {e}")
            return []

    def insert_constituents_batch(self, date: str, constituents: List[Tuple[str, str]], strategy: str = 'skip'):
        """Insert constituents into the database in batches."""
        if not constituents:
            return

        with sqlite3.connect(self.db_path) as conn:
            if strategy == 'overwrite':
                # Delete existing data for this date and indices
                index_codes = list(set([c[0] for c in constituents]))
                for index_code in index_codes:
                    conn.execute(
                        "DELETE FROM index_constituents WHERE date = ? AND index_code = ?",
                        (date, index_code)
                    )
            
            # Insert new data in batches
            batch_size = 100
            for i in range(0, len(constituents), batch_size):
                batch = constituents[i:i+batch_size]
                try:
                    conn.executemany(
                        "INSERT OR IGNORE INTO index_constituents (date, index_code, stock_code) VALUES (?, ?, ?)",
                        [(date, index_code, stock_code) for index_code, stock_code in batch]
                    )
                except sqlite3.IntegrityError:
                    if strategy == 'skip':
                        continue
                    else:
                        # For overwrite, we already deleted, so this shouldn't happen
                        pass
            
            conn.commit()
            print(f"   Inserted {len(constituents)} constituents for {date}")

    def save_category_mapping(self, indices: List[Dict], market: str):
        """Save index_code → category mappings to the index_category_mapping table."""
        if not indices:
            return

        with sqlite3.connect(self.db_path) as conn:
            for index_info in indices:
                category = index_info.get('category', '')
                if not category:
                    continue
                formatted_name = self.format_index_name(index_info['name'], market)
                conn.execute(
                    """INSERT OR REPLACE INTO index_category_mapping (index_code, category, market)
                       VALUES (?, ?, ?)""",
                    (formatted_name, category, market.upper())
                )
            conn.commit()
        print(f"   Saved category mappings for {market.upper()}: {len([i for i in indices if i.get('category')])} entries")

    def process_date_parallel(self, date: str, market: str = 'kospi', max_workers: int = 4, indices: Optional[List[Dict]] = None) -> bool:
        """
        Process a single date using multiprocessing.
        
        Args:
            date (str): Date to process (YYYY-MM-DD format)
            market (str): Market to process ('kospi' or 'kosdaq')
            max_workers (int): Number of parallel workers
            indices (Optional[List[Dict]]): Pre-scraped indices list. If None, will scrape.
            
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\nProcessing {date} for {market.upper()} with {max_workers} workers...")
        
        try:
            # Use cached indices if provided, otherwise scrape
            if indices is None:
                indices = self.scrape_listing_page(market)
                self.save_category_mapping(indices, market)
            else:
                print(f"   Using cached indices for {market.upper()}")
            
            if not indices:
                print(f"   No indices found for {market}")
                return False
            
            print(f"   Found {len(indices)} indices to process")
            
            # Process indices in parallel
            all_constituents = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all index processing tasks
                future_to_index = {
                    executor.submit(self.process_index_data, market, index_info): index_info 
                    for index_info in indices
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_index):
                    index_info = future_to_index[future]
                    try:
                        constituents = future.result()
                        if constituents:
                            all_constituents.extend(constituents)
                    except Exception as e:
                        print(f"   Error processing index {index_info.get('name', 'Unknown')}: {e}")
            
            # Insert all constituents at once
            if all_constituents:
                self.insert_constituents_batch(date, all_constituents, strategy='overwrite')
                return True
            else:
                print(f"   No constituents found for {date}")
                return False
                
        except Exception as e:
            print(f"   Error processing {date}: {e}")
            return False

    def get_latest_date(self) -> Optional[str]:
        """Get the latest date in the index_constituents table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT MAX(date) FROM index_constituents")
            result = cursor.fetchone()
            return result[0] if result[0] else None

    def backfill(self, start_date: str = "2010-01-01", max_workers: int = 4):
        """
        Backfill historical data from start_date to present using multiprocessing.
        Optimized to scrape index listings only once per market and reuse for all dates.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            max_workers (int): Number of parallel workers per date
        """
        print(f"Starting backfill from {start_date} to present with multiprocessing...")
        
        # Optimization: Scrape index listings once per market and cache them
        print("Optimization: Scraping index listings once per market...")
        kospi_indices = self.scrape_listing_page('kospi')
        kosdaq_indices = self.scrape_listing_page('kosdaq')

        # Save category mappings
        self.save_category_mapping(kospi_indices, 'kospi')
        self.save_category_mapping(kosdaq_indices, 'kosdaq')

        # Cache the indices for reuse
        self._cached_indices['kospi'] = kospi_indices
        self._cached_indices['kosdaq'] = kosdaq_indices

        print(f"Cached {len(kospi_indices)} KOSPI indices and {len(kosdaq_indices)} KOSDAQ indices")

        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.now()
        
        processed_count = 0
        failed_count = 0
        
        while current_date <= end_date:
            # Process monthly dates
            if current_date.day == 1:  # First day of month
                date_str = current_date.strftime("%Y-%m-%d")
                
                # Use cached indices for both markets
                kospi_success = self.process_date_parallel(date_str, 'kospi', max_workers, kospi_indices)
                kosdaq_success = self.process_date_parallel(date_str, 'kosdaq', max_workers, kosdaq_indices)
                
                if kospi_success and kosdaq_success:
                    processed_count += 1
                    print(f"✓ Successfully processed {date_str}")
                else:
                    failed_count += 1
                    print(f"✗ Failed to process {date_str}")
                
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            else:
                # Move to first day of current month
                current_date = current_date.replace(day=1)
        
        print(f"\nBackfill completed!")
        print(f"Processed: {processed_count} dates")
        print(f"Failed: {failed_count} dates")

    def update(self, strategy: str = 'skip', max_workers: int = 4):
        """
        Update from the latest date in the database using multiprocessing.
        Optimized to scrape index listings only once per market and reuse for all dates.
        
        Args:
            strategy (str): 'overwrite' or 'skip'
            max_workers (int): Number of parallel workers per date
        """
        latest_date = self.get_latest_date()
        
        if not latest_date:
            print("No existing data found. Starting backfill from 2010-01-01...")
            self.backfill(max_workers=max_workers)
            return
        
        print(f"Latest date in database: {latest_date}")
        
        # Optimization: Scrape index listings once per market and cache them
        print("Optimization: Scraping index listings once per market...")
        kospi_indices = self.scrape_listing_page('kospi')
        kosdaq_indices = self.scrape_listing_page('kosdaq')

        # Save category mappings
        self.save_category_mapping(kospi_indices, 'kospi')
        self.save_category_mapping(kosdaq_indices, 'kosdaq')

        # Cache the indices for reuse
        self._cached_indices['kospi'] = kospi_indices
        self._cached_indices['kosdaq'] = kosdaq_indices

        print(f"Cached {len(kospi_indices)} KOSPI indices and {len(kosdaq_indices)} KOSDAQ indices")

        # Start from next month
        latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
        if latest_dt.month == 12:
            start_date = latest_dt.replace(year=latest_dt.year + 1, month=1)
        else:
            start_date = latest_dt.replace(month=latest_dt.month + 1)
        
        print(f"Starting update from {start_date.strftime('%Y-%m-%d')}...")
        
        processed_count = 0
        failed_count = 0
        
        current_date = start_date
        end_date = datetime.now()
        
        while current_date <= end_date:
            if current_date.day == 1:  # First day of month
                date_str = current_date.strftime("%Y-%m-%d")
                
                # Check if date already exists (for skip strategy)
                if strategy == 'skip':
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.execute(
                            "SELECT COUNT(*) FROM index_constituents WHERE date = ?",
                            (date_str,)
                        )
                        if cursor.fetchone()[0] > 0:
                            print(f"   Skipping existing date: {date_str}")
                            current_date = current_date.replace(month=current_date.month + 1)
                            continue
                
                # Use cached indices for both markets
                kospi_success = self.process_date_parallel(date_str, 'kospi', max_workers, kospi_indices)
                kosdaq_success = self.process_date_parallel(date_str, 'kosdaq', max_workers, kosdaq_indices)
                
                if kospi_success and kosdaq_success:
                    processed_count += 1
                    print(f"✓ Successfully processed {date_str}")
                else:
                    failed_count += 1
                    print(f"✗ Failed to process {date_str}")
                
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            else:
                current_date = current_date.replace(day=1)
        
        print(f"\nUpdate completed!")
        print(f"Processed: {processed_count} dates")
        print(f"Failed: {failed_count} dates")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='KRX Index Constituents Direct Processor')
    parser.add_argument('--mode', choices=['backfill', 'update', 'map'], default='backfill',
                        help='Processing mode: backfill (historical), update (from latest), or map (category mapping only)')
    parser.add_argument('--strategy', choices=['overwrite', 'skip'], default='skip',
                        help='Update strategy: overwrite existing data or skip')
    parser.add_argument('--start-date', default='2010-01-01',
                        help='Start date for backfill mode (YYYY-MM-DD format)')
    parser.add_argument('--config', default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = KRXIndexConstituentsDirect(args.config)
    
    # Run based on mode
    if args.mode == 'backfill':
        processor.backfill(args.start_date, args.workers)
    elif args.mode == 'update':
        processor.update(args.strategy, args.workers)
    else:  # map
        print("Scraping category mappings only...")
        for market in ('kospi', 'kosdaq'):
            indices = processor.scrape_listing_page(market)
            processor.save_category_mapping(indices, market)
        print("Done.")


if __name__ == '__main__':
    main()