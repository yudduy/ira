"""
Pytest configuration and shared fixtures for the IRA analysis test suite.
"""
import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, AsyncMock
from pathlib import Path

@pytest.fixture
def sample_csv_content():
    """Sample PitchBook CSV content for testing."""
    return """Company Information Export
Generated: 2024-01-01

Companies,Website,Industry,Revenue
Acme Corp,https://acme.com,Technology,1000000
Green Energy Inc,www.greenenergy.com,Energy,5000000
Sustainable Solutions,https://sustainablesolutions.org,Consulting,2000000
Test Company,invalid-url,Technology,100000
"""

@pytest.fixture
def sample_csv_file(sample_csv_content):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(sample_csv_content)
        f.flush()
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def sample_companies_df():
    """Sample pandas DataFrame for testing."""
    return pd.DataFrame({
        'Companies': ['Acme Corp', 'Green Energy Inc', 'Sustainable Solutions'],
        'Website': ['https://acme.com', 'www.greenenergy.com', 'https://sustainablesolutions.org'],
        'Domain': ['acme.com', 'greenenergy.com', 'sustainablesolutions.org']
    })

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = AsyncMock()
    client.chat.completions.create = AsyncMock()
    return client

@pytest.fixture
def mock_requests_session():
    """Mock requests session for testing."""
    session = Mock()
    session.get = Mock()
    session.headers = {}
    return session

@pytest.fixture
def sample_wayback_cdx_response():
    """Sample Wayback Machine CDX API response."""
    return [
        ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"],
        ["com,acme)/", "20220315000000", "https://acme.com/", "text/html", "200", "ABC123", "50000"],
        ["com,acme)/about", "20220315000001", "https://acme.com/about", "text/html", "200", "DEF456", "30000"],
        ["com,acme)/products", "20220315000002", "https://acme.com/products", "text/html", "200", "GHI789", "40000"]
    ]

@pytest.fixture
def sample_html_content():
    """Sample HTML content for testing content extraction."""
    return """
    <html>
    <head><title>Test Company</title></head>
    <body>
        <nav>Navigation</nav>
        <header>Header Content</header>
        <main>
            <h1>Welcome to Our Company</h1>
            <p>We are a leading provider of sustainable solutions.</p>
            <p>Our mission is to create a better future through innovation.</p>
        </main>
        <footer>Footer Content</footer>
        <script>console.log('test');</script>
        <style>body { color: black; }</style>
    </body>
    </html>
    """

@pytest.fixture
def sample_llm_response():
    """Sample LLM analysis response."""
    return {
        "change_analysis": {
            "lexical_change": {
                "has_changed": True,
                "summary": "Added sustainability-related keywords"
            },
            "strategic_framing": {
                "has_changed": False,
                "from_narrative": "Technology company focused on innovation",
                "to_narrative": "Technology company focused on innovation",
                "summary": "No significant change"
            },
            "target_audience": {
                "has_changed": False,
                "primary_audience": "B2B Customers and Investors",
                "summary": "No significant change"
            }
        },
        "ira_alignment": {
            "alignment_detected": False,
            "evidence_type": "none",
            "specific_evidence": [],
            "reasoning": "No IRA-related content found"
        },
        "overall_assessment": {
            "change_level": "minor",
            "confidence": 0.8,
            "synthesis_reasoning": "Minor lexical changes only"
        }
    }
