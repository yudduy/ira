# main.py
"""
Main entry point for the IRA Corporate Messaging Analyzer.
Handles command-line argument parsing, setup, and execution.
"""
import os
import sys
import argparse
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

from src.data_processing import load_pitchbook_data
from src.analyzer import IRACorporateAnalyzer

def setup_logging():
    """Configures logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s',
        handlers=[
            logging.FileHandler('ira_analysis.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def get_api_key() -> str:
    """
    Retrieves the OpenAI API key from environment variables or Google Colab secrets.
    """
    load_dotenv() # Load .env file if it exists
    try:
        from google.colab import userdata
        key = userdata.get('OPENAI_API_KEY')
        if key: return key
    except (ImportError, KeyError):
        pass # Not in Colab or key not found, proceed to environment variable

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY in a .env file "
            "or as an environment variable."
        )
    return api_key

async def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(
        description='IRA Corporate Messaging Analyzer',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--csv', required=True, help='Path to the input PitchBook CSV file.')
    parser.add_argument('--sample', type=int, help='(Optional) Sample size to run for testing.')
    parser.add_argument('--output', help='(Optional) Output CSV file path. Defaults to a timestamped name.')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        api_key = get_api_key()
        
        logger.info("Loading and processing company data...")
        companies_df = load_pitchbook_data(args.csv, args.sample)
        
        analyzer = IRACorporateAnalyzer(api_key)
        results_df = await analyzer.run(companies_df)
        
        output_file = args.output or f"ira_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        
        logger.info(f"Analysis complete. Results saved to: {output_file}")

    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
