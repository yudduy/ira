# config.py
"""
Configuration file for the IRA Corporate Messaging Analyzer.
Contains constants for analysis windows, API settings, and request parameters.
"""

# Research-grade comparison windows (Annual periods for better results)
ANALYSIS_WINDOWS = {
    "pre_ira": {
        "name": "Pre-IRA Baseline (Annual)",
        "start": "20220101",
        "end": "20221231",
        "description": "Full Year 2022 - Pre-IRA corporate messaging baseline"
    },
    "post_ira": {
        "name": "Post-IRA Implementation (Annual)",
        "start": "20230101",
        "end": "20231231",
        "description": "Full Year 2023 - Post-IRA implementation period"
    }
}

# Academic standard rate limits and timeouts for Wayback Machine CDX API
CDX_RATE_LIMIT = 2.0  # Seconds to wait between requests
REQUEST_TIMEOUT = 45  # Seconds before a request times out
MAX_RETRIES = 3       # Max number of retries for failed requests

# List of target pages for snapshot searching to ensure construct validity
TARGET_PAGES = [
    '/', '/about', '/company', '/our-company', '/mission',
    '/products', '/services', '/solutions', '/technology',
    '/sustainability', '/environmental', '/esg'
]

# OpenAI Model Configuration
OPENAI_MODEL = "gpt-4.1-nano"
OPENAI_TEMPERATURE = 0.1
OPENAI_MAX_TOKENS = 1000

# Truncation limits for text sent to the LLM
CONTENT_TRUNCATION_LIMIT = 8000
PROMPT_CONTENT_LIMIT = 3500
