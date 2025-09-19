"""
Unit tests for src/data_processing.py module.
Tests CSV loading, validation, and domain cleaning functionality.
"""
import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import patch, mock_open
from src.data_processing import load_pitchbook_data, _find_header_row, _clean_domain


class TestFindHeaderRow:
    """Test header row detection functionality."""
    
    def test_find_header_row_success(self):
        """Test successful header row detection."""
        csv_content = """Some metadata
More metadata
Header row with Companies and Website
Companies,Website,Industry
Acme Corp,https://acme.com,Technology"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            header_row = _find_header_row(f.name)
            assert header_row == 2  # 0-indexed, so row 2 contains headers
            
        os.unlink(f.name)
    
    def test_find_header_row_fallback(self):
        """Test fallback to default row when Companies not found."""
        csv_content = """Some metadata
More metadata
Different headers
Name,URL,Type
Acme Corp,https://acme.com,Technology"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            header_row = _find_header_row(f.name)
            assert header_row == 6  # Should fallback to default row 6
            
        os.unlink(f.name)
    
    def test_find_header_row_company_id_fallback(self):
        """Test fallback using Company ID."""
        csv_content = """Some metadata
More metadata
Company ID,Website,Industry
123,https://acme.com,Technology"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            header_row = _find_header_row(f.name)
            assert header_row == 2  # Should find Company ID header
            
        os.unlink(f.name)
    
    def test_find_header_row_nonexistent_file(self):
        """Test handling of nonexistent file."""
        with pytest.raises(FileNotFoundError):
            _find_header_row("nonexistent_file.csv")


class TestCleanDomain:
    """Test domain cleaning functionality."""
    
    def test_clean_domain_https(self):
        """Test cleaning domain with https URL."""
        url = "https://www.example.com"
        result = _clean_domain(url)
        assert result == "example.com"
    
    def test_clean_domain_http(self):
        """Test cleaning domain with http URL."""
        url = "http://example.com"
        result = _clean_domain(url)
        assert result == "example.com"
    
    def test_clean_domain_no_protocol(self):
        """Test cleaning domain without protocol."""
        url = "www.example.com"
        result = _clean_domain(url)
        assert result == "example.com"
    
    def test_clean_domain_no_www(self):
        """Test cleaning domain without www."""
        url = "https://example.com"
        result = _clean_domain(url)
        assert result == "example.com"
    
    def test_clean_domain_with_path(self):
        """Test cleaning domain with path."""
        url = "https://www.example.com/about"
        result = _clean_domain(url)
        assert result == "example.com"
    
    def test_clean_domain_with_port(self):
        """Test cleaning domain with port."""
        url = "https://www.example.com:8080"
        result = _clean_domain(url)
        assert result == "example.com:8080"
    
    def test_clean_domain_nan_input(self):
        """Test cleaning domain with NaN input."""
        result = _clean_domain(pd.NA)
        assert result is None
    
    def test_clean_domain_none_input(self):
        """Test cleaning domain with None input."""
        result = _clean_domain(None)
        assert result is None
    
    def test_clean_domain_empty_string(self):
        """Test cleaning domain with empty string."""
        result = _clean_domain("")
        assert result is None
    
    def test_clean_domain_invalid_url(self):
        """Test cleaning domain with invalid URL."""
        result = _clean_domain("not-a-url")
        assert result is None
    
    def test_clean_domain_mixed_case(self):
        """Test cleaning domain with mixed case."""
        url = "https://WWW.EXAMPLE.COM"
        result = _clean_domain(url)
        assert result == "example.com"


class TestLoadPitchbookData:
    """Test main data loading functionality."""
    
    def test_load_valid_csv(self, sample_csv_file):
        """Test loading valid CSV file."""
        df = load_pitchbook_data(sample_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # Should have 3 valid companies
        assert list(df.columns) == ['Companies', 'Website', 'Domain']
        
        # Check that domains were cleaned
        assert df['Domain'].iloc[0] == 'acme.com'
        assert df['Domain'].iloc[1] == 'greenenergy.com'
        assert df['Domain'].iloc[2] == 'sustainablesolutions.org'
    
    def test_load_csv_with_sampling(self, sample_csv_file):
        """Test loading CSV with sampling."""
        df = load_pitchbook_data(sample_csv_file, sample_size=2)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # Should sample 2 companies
        assert list(df.columns) == ['Companies', 'Website', 'Domain']
    
    def test_load_csv_no_sampling_when_sample_larger_than_data(self, sample_csv_file):
        """Test that no sampling occurs when sample size > data size."""
        df = load_pitchbook_data(sample_csv_file, sample_size=10)
        
        assert len(df) == 3  # Should return all 3 companies
    
    def test_load_csv_filters_invalid_domains(self):
        """Test that invalid domains are filtered out."""
        csv_content = """Companies,Website
Valid Company,https://valid.com
Invalid Company,invalid-url
Another Valid,https://another.com
Empty Company,
NaN Company,None"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            df = load_pitchbook_data(f.name)
            
            assert len(df) == 2  # Only valid companies should remain
            assert df['Domain'].notna().all()  # All domains should be valid
            
        os.unlink(f.name)
    
    def test_load_csv_missing_required_columns(self):
        """Test error when required columns are missing."""
        csv_content = """Name,URL
Company,https://example.com"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            
        with pytest.raises((ValueError, Exception), match="(CSV must contain the following columns|No columns to parse from file)"):
            load_pitchbook_data(f.name)
                
        os.unlink(f.name)
    
    def test_load_csv_nonexistent_file(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_pitchbook_data("nonexistent_file.csv")
    
    def test_load_csv_empty_file(self):
        """Test handling of empty CSV file."""
        csv_content = ""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            with pytest.raises((ValueError, KeyError)):  # Either error is acceptable
                load_pitchbook_data(f.name)
                
        os.unlink(f.name)
    
    def test_load_csv_with_whitespace_columns(self):
        """Test handling of columns with whitespace."""
        csv_content = """Companies , Website , Industry
Acme Corp, https://acme.com , Technology
Green Energy, www.green.com , Energy"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            df = load_pitchbook_data(f.name)
            
            assert len(df) == 2
            assert list(df.columns) == ['Companies', 'Website', 'Domain']
            
        os.unlink(f.name)
    
    def test_load_csv_preserves_company_names(self, sample_csv_file):
        """Test that company names are preserved correctly."""
        df = load_pitchbook_data(sample_csv_file)
        
        expected_names = ['Acme Corp', 'Green Energy Inc', 'Sustainable Solutions']
        assert list(df['Companies']) == expected_names
    
    def test_load_csv_preserves_website_urls(self, sample_csv_file):
        """Test that website URLs are preserved correctly."""
        df = load_pitchbook_data(sample_csv_file)
        
        expected_urls = ['https://acme.com', 'www.greenenergy.com', 'https://sustainablesolutions.org']
        assert list(df['Website']) == expected_urls
    
    def test_load_csv_deterministic_sampling(self):
        """Test that sampling is deterministic with same random state."""
        csv_content = """Companies,Website
Company1,https://company1.com
Company2,https://company2.com
Company3,https://company3.com
Company4,https://company4.com
Company5,https://company5.com"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            # Load with same sample size twice
            df1 = load_pitchbook_data(f.name, sample_size=3)
            df2 = load_pitchbook_data(f.name, sample_size=3)
            
            # Should get same results due to fixed random state
            assert list(df1['Companies']) == list(df2['Companies'])
            
        os.unlink(f.name)
    
    def test_load_csv_handles_encoding_issues(self):
        """Test handling of encoding issues."""
        csv_content = """Companies,Website
Company with émojis,https://emoji.com
Company with special chars,https://special.com"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(csv_content)
            f.flush()
            
            # Should handle encoding gracefully
            df = load_pitchbook_data(f.name)
            assert len(df) == 2
            
        os.unlink(f.name)


class TestDataProcessingIntegration:
    """Integration tests for data processing pipeline."""
    
    def test_full_pipeline_with_realistic_data(self):
        """Test full pipeline with realistic PitchBook-like data."""
        csv_content = """PitchBook Export
Generated: 2024-01-01
Export Type: Company List

Companies,Website,Industry,Revenue,Employees,Founded
Acme Corporation,https://www.acme-corp.com,Technology,10000000,500,2010
Green Energy Solutions,www.greenenergy.com,Energy,25000000,1200,2005
Sustainable Tech Inc,https://sustech.io,Clean Technology,5000000,200,2018
Invalid Company,not-a-url,Technology,1000000,50,2020
Empty Website Company,,Consulting,2000000,100,2015"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            df = load_pitchbook_data(f.name)
            
            # Should process 3 valid companies (excluding invalid URL and empty website)
            assert len(df) == 3
            assert list(df.columns) == ['Companies', 'Website', 'Domain']
            
            # Check domain cleaning worked
            domains = list(df['Domain'])
            assert 'acme-corp.com' in domains
            assert 'greenenergy.com' in domains
            assert 'sustech.io' in domains
            
        os.unlink(f.name)
    
    def test_error_handling_chain(self):
        """Test that error handling works through the chain."""
        # Test with malformed CSV that would cause multiple issues
        csv_content = """Invalid CSV structure
Companies,Website
Company1,https://valid.com
Company2,invalid-url
Company3,
Company4,https://another-valid.com"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            df = load_pitchbook_data(f.name)
            
            # Should handle all issues and return only valid entries
            assert len(df) == 2  # Only 2 valid companies
            assert df['Domain'].notna().all()
            
        os.unlink(f.name)
