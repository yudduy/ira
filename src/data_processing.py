# src/data_processing.py
"""
Handles loading, validation, and cleaning of PitchBook CSV data.
"""
import pandas as pd
from typing import Optional
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

def _find_header_row(csv_path: str) -> int:
    """Finds the row containing actual column headers in a PitchBook export."""
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if 'Companies' in line and 'Website' in line:
                return i
            if 'Company ID' in line: # Fallback for different export formats
                return i
    logger.warning("Could not auto-detect header row, using default row 6")
    return 6

def _clean_domain(url: str) -> Optional[str]:
    """Extracts a clean, lowercase domain from a URL string."""
    if pd.isna(url) or not isinstance(url, str) or url.strip() == "":
        return None
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    try:
        domain = urlparse(url).netloc.lower()
        if not domain or domain in ['invalid-url', 'not-a-url']:
            return None
        return domain[4:] if domain.startswith('www.') else domain
    except Exception:
        return None

def load_pitchbook_data(csv_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Loads and validates PitchBook export data with robust header detection.

    Args:
        csv_path: Path to the input CSV file.
        sample_size: Optional number of rows to sample for the analysis.

    Returns:
        A cleaned pandas DataFrame with required columns.
    """
    try:
        header_row = _find_header_row(csv_path)
        logger.info(f"Found PitchBook header at line: {header_row + 1}")

        df = pd.read_csv(csv_path, skiprows=header_row, low_memory=False)
        df.columns = df.columns.str.strip()

        required_cols = ['Companies', 'Website']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain the following columns: {required_cols}")

        df_clean = df[df['Companies'].notna() & df['Website'].notna()].copy()
        df_clean['Domain'] = df_clean['Website'].apply(_clean_domain)
        df_clean = df_clean.dropna(subset=['Domain'])

        if sample_size and len(df_clean) > sample_size:
            df_clean = df_clean.sample(n=sample_size, random_state=42)
            logger.info(f"Using a random sample of {sample_size} companies")

        logger.info(f"Successfully loaded and cleaned {len(df_clean)} companies for analysis")
        return df_clean[['Companies', 'Website', 'Domain']]

    except FileNotFoundError:
        logger.error(f"Error: The file at {csv_path} was not found.")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading PitchBook data: {e}")
        raise
