# src/wayback_client.py
"""
Client for interacting with the Wayback Machine CDX API and fetching archived content.
"""
import asyncio
import logging
from typing import Dict
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import config

logger = logging.getLogger(__name__)

class WaybackClient:
    """Manages finding snapshots and extracting content from the Wayback Machine."""

    def __init__(self, session: requests.Session):
        self.session = session

    async def find_snapshots(self, domain: str, window_name: str) -> Dict:
        """
        Finds the optimal snapshots for a domain within a given time window.
        Implements retry logic with exponential backoff for resilience.
        """
        window = config.ANALYSIS_WINDOWS[window_name]
        params = {
            'url': f'{domain}/*',
            'output': 'json',
            'from': window['start'],
            'to': window['end'],
            'limit': '10',  # Fetch more to find relevant pages
            'sort': 'closest',
            'filter': 'statuscode:200'
        }

        for attempt in range(config.MAX_RETRIES):
            try:
                await asyncio.sleep(config.CDX_RATE_LIMIT)
                logger.debug(f"Querying CDX for {domain} (Attempt {attempt + 1})")

                response = await asyncio.to_thread(
                    self.session.get,
                    'http://web.archive.org/cdx/search/cdx',
                    params=params,
                    timeout=config.REQUEST_TIMEOUT
                )
                response.raise_for_status()
                data = response.json()

                if len(data) <= 1:
                    continue # Try again if empty response

                # Prioritize strategically important pages
                for row in data[1:]:
                    timestamp, original_url = row[1], row[2]
                    url_path = urlparse(original_url).path.lower()
                    if any(page == url_path.rstrip('/') for page in config.TARGET_PAGES):
                        snapshot = {
                            'timestamp': timestamp,
                            'url': original_url,
                            'archive_url': f'https://web.archive.org/web/{timestamp}/{original_url}'
                        }
                        return {'success': True, 'snapshot': snapshot}

                return {'success': False, 'error': 'No relevant page snapshots found'}

            except requests.exceptions.RequestException as e:
                logger.warning(f"CDX query failed for {domain} (Attempt {attempt + 1}): {e}")
                if attempt == config.MAX_RETRIES - 1:
                    return {'success': False, 'error': str(e)}
                await asyncio.sleep(2 ** attempt)

        return {'success': False, 'error': 'Max retries exceeded'}

    async def extract_content(self, archive_url: str) -> Dict:
        """
        Extracts clean, readable text content from an archived webpage.
        """
        try:
            response = await asyncio.to_thread(
                self.session.get, archive_url, timeout=config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')

            # Remove standard navigation, scripts, and Wayback Machine's own UI
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            for element in soup.find_all(attrs={'id': lambda x: x and 'wm-' in x}):
                element.decompose()

            text = ' '.join(soup.get_text(separator=' ', strip=True).split())
            truncated_text = text[:config.CONTENT_TRUNCATION_LIMIT]
            return {
                'success': True,
                'content': truncated_text,
                'word_count': len(truncated_text.split())
            }
        except Exception as e:
            logger.warning(f"Content extraction failed for {archive_url}: {e}")
            return {'success': False, 'error': str(e)}
