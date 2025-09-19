# src/analyzer.py
"""
Core analysis class that orchestrates the entire workflow from data loading to final analysis.
"""
import asyncio
import logging
from typing import Dict
import pandas as pd
from tqdm.asyncio import tqdm
import requests
import openai

from src.wayback_client import WaybackClient
from src.llm_handler import LLMHandler

logger = logging.getLogger(__name__)

def _flatten_analysis_output(analysis_data: Dict) -> Dict:
    """Flattens the nested JSON from the LLM into a single-level dictionary for CSV export."""
    flat_dict = {}
    # Flatten change_analysis
    ca = analysis_data.get("change_analysis", {})
    for key, val in ca.get("lexical_change", {}).items():
        flat_dict[f"lexical_{key}"] = val
    for key, val in ca.get("strategic_framing", {}).items():
        flat_dict[f"framing_{key}"] = val
    for key, val in ca.get("target_audience", {}).items():
        flat_dict[f"audience_{key}"] = val
    # Flatten ira_alignment
    for key, val in analysis_data.get("ira_alignment", {}).items():
        flat_dict[f"ira_{key}"] = val
    # Flatten overall_assessment
    for key, val in analysis_data.get("overall_assessment", {}).items():
        flat_dict[f"overall_{key}"] = val
    return flat_dict

class IRACorporateAnalyzer:
    """Orchestrates the corporate messaging analysis process."""

    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Academic Research - IRA Corporate Analysis (Stanford GSB)'
        })
        self.openai_client = openai.AsyncOpenAI(api_key=api_key)
        self.wayback_client = WaybackClient(self.session)
        self.llm_handler = LLMHandler(self.openai_client)
        logger.info("IRA Corporate Analyzer initialized successfully.")

    async def analyze_company(self, company_row: pd.Series) -> Dict:
        """Runs the complete analysis pipeline for a single company."""
        company_name, domain = company_row['Companies'], company_row['Domain']
        logger.info(f"Analyzing: {company_name} ({domain})")
        result = {'company_name': company_name, 'domain': domain, 'website': company_row['Website']}

        pre_snapshot_res = await self.wayback_client.find_snapshots(domain, 'pre_ira')
        post_snapshot_res = await self.wayback_client.find_snapshots(domain, 'post_ira')

        if not pre_snapshot_res['success'] or not post_snapshot_res['success']:
            result['status'] = 'insufficient_snapshots'
            result['pre_snapshot_error'] = pre_snapshot_res.get('error', 'N/A')
            result['post_snapshot_error'] = post_snapshot_res.get('error', 'N/A')
            return result
        
        pre_url = pre_snapshot_res['snapshot']['archive_url']
        post_url = post_snapshot_res['snapshot']['archive_url']
        result.update({'pre_ira_snapshot_url': pre_url, 'post_ira_snapshot_url': post_url})

        pre_content_res = await self.wayback_client.extract_content(pre_url)
        post_content_res = await self.wayback_client.extract_content(post_url)

        if not pre_content_res['success'] or not post_content_res['success']:
            result['status'] = 'content_extraction_failed'
            result['pre_content_error'] = pre_content_res.get('error', 'N/A')
            result['post_content_error'] = post_content_res.get('error', 'N/A')
            return result

        result.update({
            'pre_word_count': pre_content_res.get('word_count'),
            'post_word_count': post_content_res.get('word_count')
        })

        analysis_res = await self.llm_handler.analyze_content_change(
            pre_content_res['content'], post_content_res['content']
        )

        if not analysis_res['success']:
            result['status'] = 'analysis_error'
            result['error_message'] = analysis_res.get('error')
            return result
        
        result['status'] = 'completed'
        result.update(_flatten_analysis_output(analysis_res['data']))
        return result

    async def run(self, companies_df: pd.DataFrame, max_concurrent: int = 2) -> pd.DataFrame:
        """Analyzes an entire dataset of companies with controlled concurrency."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(row):
            async with semaphore:
                return await self.analyze_company(row)

        tasks = [analyze_with_semaphore(row) for _, row in companies_df.iterrows()]
        results = await tqdm.gather(*tasks, desc="Analyzing companies")
        
        return pd.DataFrame([r for r in results if r])
