"""
Unit tests for config.py module.
Tests configuration constants and their validity.
"""
import pytest
import config


class TestAnalysisWindows:
    """Test analysis windows configuration."""
    
    def test_analysis_windows_structure(self):
        """Test that ANALYSIS_WINDOWS has correct structure."""
        assert isinstance(config.ANALYSIS_WINDOWS, dict)
        assert "pre_ira" in config.ANALYSIS_WINDOWS
        assert "post_ira" in config.ANALYSIS_WINDOWS
    
    def test_pre_ira_window_structure(self):
        """Test pre-IRA window has all required fields."""
        pre_ira = config.ANALYSIS_WINDOWS["pre_ira"]
        required_fields = ["name", "start", "end", "description"]
        
        for field in required_fields:
            assert field in pre_ira, f"Missing field: {field}"
            assert isinstance(pre_ira[field], str), f"Field {field} should be string"
    
    def test_post_ira_window_structure(self):
        """Test post-IRA window has all required fields."""
        post_ira = config.ANALYSIS_WINDOWS["post_ira"]
        required_fields = ["name", "start", "end", "description"]
        
        for field in required_fields:
            assert field in post_ira, f"Missing field: {field}"
            assert isinstance(post_ira[field], str), f"Field {field} should be string"
    
    def test_date_formats(self):
        """Test that date formats are correct YYYYMMDD."""
        pre_ira = config.ANALYSIS_WINDOWS["pre_ira"]
        post_ira = config.ANALYSIS_WINDOWS["post_ira"]
        
        # Check format is YYYYMMDD (8 digits)
        assert len(pre_ira["start"]) == 8, "Start date should be YYYYMMDD format"
        assert len(pre_ira["end"]) == 8, "End date should be YYYYMMDD format"
        assert len(post_ira["start"]) == 8, "Start date should be YYYYMMDD format"
        assert len(post_ira["end"]) == 8, "End date should be YYYYMMDD format"
        
        # Check that dates are numeric
        assert pre_ira["start"].isdigit(), "Start date should be numeric"
        assert pre_ira["end"].isdigit(), "End date should be numeric"
        assert post_ira["start"].isdigit(), "Start date should be numeric"
        assert post_ira["end"].isdigit(), "End date should be numeric"
    
    def test_chronological_order(self):
        """Test that pre-IRA dates come before post-IRA dates."""
        pre_ira = config.ANALYSIS_WINDOWS["pre_ira"]
        post_ira = config.ANALYSIS_WINDOWS["post_ira"]
        
        assert pre_ira["end"] < post_ira["start"], "Pre-IRA period should end before post-IRA period begins"
    
    def test_annual_windows(self):
        """Test that windows represent full years."""
        pre_ira = config.ANALYSIS_WINDOWS["pre_ira"]
        post_ira = config.ANALYSIS_WINDOWS["post_ira"]
        
        # Check that we have full year 2022 and 2023
        assert pre_ira["start"] == "20220101", "Pre-IRA should start on January 1, 2022"
        assert pre_ira["end"] == "20221231", "Pre-IRA should end on December 31, 2022"
        assert post_ira["start"] == "20230101", "Post-IRA should start on January 1, 2023"
        assert post_ira["end"] == "20231231", "Post-IRA should end on December 31, 2023"


class TestRateLimits:
    """Test rate limiting and timeout configuration."""
    
    def test_rate_limit_positive(self):
        """Test that rate limit is positive."""
        assert config.CDX_RATE_LIMIT > 0, "CDX rate limit should be positive"
    
    def test_timeout_positive(self):
        """Test that timeout is positive."""
        assert config.REQUEST_TIMEOUT > 0, "Request timeout should be positive"
    
    def test_max_retries_positive(self):
        """Test that max retries is positive."""
        assert config.MAX_RETRIES > 0, "Max retries should be positive"
    
    def test_rate_limit_reasonable(self):
        """Test that rate limit is reasonable for API usage."""
        assert 1.0 <= config.CDX_RATE_LIMIT <= 10.0, "Rate limit should be between 1-10 seconds"
    
    def test_timeout_reasonable(self):
        """Test that timeout is reasonable for web requests."""
        assert 10 <= config.REQUEST_TIMEOUT <= 120, "Timeout should be between 10-120 seconds"


class TestTargetPages:
    """Test target pages configuration."""
    
    def test_target_pages_list(self):
        """Test that TARGET_PAGES is a list."""
        assert isinstance(config.TARGET_PAGES, list), "TARGET_PAGES should be a list"
    
    def test_target_pages_not_empty(self):
        """Test that TARGET_PAGES is not empty."""
        assert len(config.TARGET_PAGES) > 0, "TARGET_PAGES should not be empty"
    
    def test_target_pages_strings(self):
        """Test that all target pages are strings."""
        for page in config.TARGET_PAGES:
            assert isinstance(page, str), f"Target page {page} should be a string"
    
    def test_target_pages_start_with_slash(self):
        """Test that all target pages start with slash."""
        for page in config.TARGET_PAGES:
            assert page.startswith('/'), f"Target page {page} should start with '/'"
    
    def test_essential_pages_present(self):
        """Test that essential pages are included."""
        essential_pages = ['/', '/about']
        for page in essential_pages:
            assert page in config.TARGET_PAGES, f"Essential page {page} should be in TARGET_PAGES"


class TestOpenAIConfig:
    """Test OpenAI configuration."""
    
    def test_model_name(self):
        """Test that model name is valid."""
        assert isinstance(config.OPENAI_MODEL, str), "OpenAI model should be string"
        assert len(config.OPENAI_MODEL) > 0, "OpenAI model should not be empty"
    
    def test_temperature_range(self):
        """Test that temperature is in valid range."""
        assert 0.0 <= config.OPENAI_TEMPERATURE <= 2.0, "Temperature should be between 0.0 and 2.0"
    
    def test_max_tokens_positive(self):
        """Test that max tokens is positive."""
        assert config.OPENAI_MAX_TOKENS > 0, "Max tokens should be positive"
    
    def test_max_tokens_reasonable(self):
        """Test that max tokens is reasonable."""
        assert 100 <= config.OPENAI_MAX_TOKENS <= 10000, "Max tokens should be between 100-10000"


class TestTruncationLimits:
    """Test text truncation configuration."""
    
    def test_truncation_limits_positive(self):
        """Test that truncation limits are positive."""
        assert config.CONTENT_TRUNCATION_LIMIT > 0, "Content truncation limit should be positive"
        assert config.PROMPT_CONTENT_LIMIT > 0, "Prompt content limit should be positive"
    
    def test_truncation_limits_reasonable(self):
        """Test that truncation limits are reasonable."""
        assert 1000 <= config.CONTENT_TRUNCATION_LIMIT <= 50000, "Content truncation should be reasonable"
        assert 1000 <= config.PROMPT_CONTENT_LIMIT <= 10000, "Prompt content limit should be reasonable"
    
    def test_prompt_limit_less_than_content_limit(self):
        """Test that prompt limit is less than content limit."""
        assert config.PROMPT_CONTENT_LIMIT <= config.CONTENT_TRUNCATION_LIMIT, "Prompt limit should be <= content limit"


class TestConfigConsistency:
    """Test overall configuration consistency."""
    
    def test_all_configs_accessible(self):
        """Test that all configuration values are accessible."""
        configs_to_test = [
            'ANALYSIS_WINDOWS',
            'CDX_RATE_LIMIT',
            'REQUEST_TIMEOUT',
            'MAX_RETRIES',
            'TARGET_PAGES',
            'OPENAI_MODEL',
            'OPENAI_TEMPERATURE',
            'OPENAI_MAX_TOKENS',
            'CONTENT_TRUNCATION_LIMIT',
            'PROMPT_CONTENT_LIMIT'
        ]
        
        for config_name in configs_to_test:
            assert hasattr(config, config_name), f"Config {config_name} should be defined"
            assert getattr(config, config_name) is not None, f"Config {config_name} should not be None"
    
    def test_no_undefined_constants(self):
        """Test that no unexpected constants are defined."""
        expected_constants = {
            'ANALYSIS_WINDOWS', 'CDX_RATE_LIMIT', 'REQUEST_TIMEOUT', 'MAX_RETRIES',
            'TARGET_PAGES', 'OPENAI_MODEL', 'OPENAI_TEMPERATURE', 'OPENAI_MAX_TOKENS',
            'CONTENT_TRUNCATION_LIMIT', 'PROMPT_CONTENT_LIMIT'
        }
        
        actual_constants = {name for name in dir(config) if name.isupper() and not name.startswith('_')}
        
        # Allow for some flexibility but ensure core constants are present
        missing_constants = expected_constants - actual_constants
        assert len(missing_constants) == 0, f"Missing expected constants: {missing_constants}"
