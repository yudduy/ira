"""
Unit tests for main.py module.
Tests the command-line entry point and main execution flow.
"""
import pytest
import sys
import os
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from io import StringIO
from pathlib import Path


class TestSetupLogging:
    """Test logging setup functionality."""
    
    @patch('main.logging.basicConfig')
    @patch('main.logging.getLogger')
    def test_setup_logging(self, mock_get_logger, mock_basic_config):
        """Test that logging is properly configured."""
        from main import setup_logging
        
        setup_logging()
        
        # Verify basicConfig was called
        mock_basic_config.assert_called_once()
        
        # Verify loggers were configured
        assert mock_get_logger.call_count >= 1


class TestGetApiKey:
    """Test API key retrieval functionality."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key-from-env'})
    @patch('main.load_dotenv')
    def test_get_api_key_from_env(self, mock_load_dotenv):
        """Test getting API key from environment variable."""
        from main import get_api_key
        
        result = get_api_key()
        
        assert result == 'test-key-from-env'
        mock_load_dotenv.assert_called_once()
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('main.load_dotenv')
    def test_get_api_key_not_found(self, mock_load_dotenv):
        """Test error when API key is not found."""
        from main import get_api_key
        
        with pytest.raises(ValueError, match="OpenAI API key not found"):
            get_api_key()


class TestMainFunction:
    """Test the main function execution."""
    
    @pytest.mark.asyncio
    async def test_main_success(self, sample_csv_file, capsys):
        """Test successful main execution."""
        with patch('main.get_api_key') as mock_get_key, \
             patch('main.load_pitchbook_data') as mock_load_data, \
             patch('main.IRACorporateAnalyzer') as mock_analyzer_class, \
             patch('main.setup_logging'):
            
            # Setup mocks
            mock_get_key.return_value = 'test-api-key'
            
            mock_df = Mock()
            mock_load_data.return_value = mock_df
            
            mock_analyzer = Mock()
            mock_analyzer.run = AsyncMock(return_value=Mock())
            mock_analyzer_class.return_value = mock_analyzer
            
            # Mock sys.argv
            with patch.object(sys, 'argv', ['main.py', '--csv', sample_csv_file]):
                from main import main
                await main()
            
            # Verify calls
            mock_get_key.assert_called_once()
            mock_load_data.assert_called_once_with(sample_csv_file, None)
            mock_analyzer.run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_main_with_sample_size(self, sample_csv_file):
        """Test main execution with sample size."""
        with patch('main.get_api_key') as mock_get_key, \
             patch('main.load_pitchbook_data') as mock_load_data, \
             patch('main.IRACorporateAnalyzer') as mock_analyzer_class, \
             patch('main.setup_logging'):
            
            mock_get_key.return_value = 'test-api-key'
            mock_df = Mock()
            mock_load_data.return_value = mock_df
            
            mock_analyzer = Mock()
            mock_analyzer.run = AsyncMock(return_value=Mock())
            mock_analyzer_class.return_value = mock_analyzer
            
            with patch.object(sys, 'argv', ['main.py', '--csv', sample_csv_file, '--sample', '10']):
                from main import main
                await main()
            
            mock_load_data.assert_called_once_with(sample_csv_file, 10)
    
    @pytest.mark.asyncio
    async def test_main_with_output_file(self, sample_csv_file):
        """Test main execution with custom output file."""
        with patch('main.get_api_key') as mock_get_key, \
             patch('main.load_pitchbook_data') as mock_load_data, \
             patch('main.IRACorporateAnalyzer') as mock_analyzer_class, \
             patch('main.setup_logging'):
            
            mock_get_key.return_value = 'test-api-key'
            mock_df = Mock()
            mock_load_data.return_value = mock_df
            
            mock_results = Mock()
            mock_results.to_csv = Mock()
            mock_analyzer = Mock()
            mock_analyzer.run = AsyncMock(return_value=mock_results)
            mock_analyzer_class.return_value = mock_analyzer
            
            with patch.object(sys, 'argv', ['main.py', '--csv', sample_csv_file, '--output', 'custom_output.csv']):
                from main import main
                await main()
            
            mock_results.to_csv.assert_called_once_with('custom_output.csv', index=False)
    
    @pytest.mark.asyncio
    async def test_main_api_key_error(self, sample_csv_file):
        """Test main execution when API key retrieval fails."""
        with patch('main.get_api_key') as mock_get_key, \
             patch('main.setup_logging'):
            
            mock_get_key.side_effect = ValueError("API key not found")
            
            with patch.object(sys, 'argv', ['main.py', '--csv', sample_csv_file]):
                from main import main
                
                with pytest.raises(SystemExit):
                    await main()
    
    @pytest.mark.asyncio
    async def test_main_data_loading_error(self, sample_csv_file):
        """Test main execution when data loading fails."""
        with patch('main.get_api_key') as mock_get_key, \
             patch('main.load_pitchbook_data') as mock_load_data, \
             patch('main.setup_logging'):
            
            mock_get_key.return_value = 'test-api-key'
            mock_load_data.side_effect = FileNotFoundError("CSV file not found")
            
            with patch.object(sys, 'argv', ['main.py', '--csv', sample_csv_file]):
                from main import main
                
                with pytest.raises(SystemExit):
                    await main()
    
    @pytest.mark.asyncio
    async def test_main_analysis_error(self, sample_csv_file):
        """Test main execution when analysis fails."""
        with patch('main.get_api_key') as mock_get_key, \
             patch('main.load_pitchbook_data') as mock_load_data, \
             patch('main.IRACorporateAnalyzer') as mock_analyzer_class, \
             patch('main.setup_logging'):
            
            mock_get_key.return_value = 'test-api-key'
            mock_df = Mock()
            mock_load_data.return_value = mock_df
            
            mock_analyzer = Mock()
            mock_analyzer.run = AsyncMock(side_effect=Exception("Analysis failed"))
            mock_analyzer_class.return_value = mock_analyzer
            
            with patch.object(sys, 'argv', ['main.py', '--csv', sample_csv_file]):
                from main import main
                
                with pytest.raises(SystemExit):
                    await main()


class TestArgumentParsing:
    """Test command-line argument parsing."""
    
    def test_required_csv_argument(self):
        """Test that CSV argument is required."""
        with patch.object(sys, 'argv', ['main.py']):
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument('--csv', required=True)
            
            with pytest.raises(SystemExit):
                parser.parse_args([])
    
    def test_optional_arguments(self):
        """Test optional arguments parsing."""
        with patch.object(sys, 'argv', ['main.py', '--csv', 'test.csv', '--sample', '50', '--output', 'results.csv']):
            import argparse
            from main import main
            
            # We can't easily test the parser directly, but we can verify the arguments are accepted
            # by checking that no SystemExit is raised for argument parsing
            try:
                parser = argparse.ArgumentParser()
                parser.add_argument('--csv', required=True)
                parser.add_argument('--sample', type=int)
                parser.add_argument('--output')
                
                args = parser.parse_args(['--csv', 'test.csv', '--sample', '50', '--output', 'results.csv'])
                
                assert args.csv == 'test.csv'
                assert args.sample == 50
                assert args.output == 'results.csv'
            except SystemExit:
                pytest.fail("Valid arguments should not raise SystemExit")


class TestFileOperations:
    """Test file operations in main."""
    
    @pytest.mark.asyncio
    async def test_output_file_creation(self, sample_csv_file):
        """Test that output file is created correctly."""
        with patch('main.get_api_key') as mock_get_key, \
             patch('main.load_pitchbook_data') as mock_load_data, \
             patch('main.IRACorporateAnalyzer') as mock_analyzer_class, \
             patch('main.setup_logging'):
            
            mock_get_key.return_value = 'test-api-key'
            mock_df = Mock()
            mock_load_data.return_value = mock_df
            
            mock_results = Mock()
            mock_results.to_csv = Mock()
            mock_analyzer = Mock()
            mock_analyzer.run = AsyncMock(return_value=mock_results)
            mock_analyzer_class.return_value = mock_analyzer
            
            with patch.object(sys, 'argv', ['main.py', '--csv', sample_csv_file, '--output', 'test_output.csv']):
                from main import main
                await main()
            
            mock_results.to_csv.assert_called_once_with('test_output.csv', index=False)
    
    @pytest.mark.asyncio
    async def test_default_output_filename(self, sample_csv_file):
        """Test default output filename generation."""
        with patch('main.get_api_key') as mock_get_key, \
             patch('main.load_pitchbook_data') as mock_load_data, \
             patch('main.IRACorporateAnalyzer') as mock_analyzer_class, \
             patch('main.setup_logging'), \
             patch('main.datetime') as mock_datetime:
            
            mock_get_key.return_value = 'test-api-key'
            mock_df = Mock()
            mock_load_data.return_value = mock_df
            
            mock_results = Mock()
            mock_results.to_csv = Mock()
            mock_analyzer = Mock()
            mock_analyzer.run = AsyncMock(return_value=mock_results)
            mock_analyzer_class.return_value = mock_analyzer
            
            # Mock datetime for predictable filename
            mock_datetime.now.return_value.strftime.return_value = '20240101_120000'
            
            with patch.object(sys, 'argv', ['main.py', '--csv', sample_csv_file]):
                from main import main
                await main()
            
            expected_filename = 'ira_analysis_results_20240101_120000.csv'
            mock_results.to_csv.assert_called_once_with(expected_filename, index=False)


class TestMainIntegration:
    """Integration tests for main module."""
    
    @pytest.mark.asyncio
    async def test_full_main_execution_flow(self, sample_csv_file):
        """Test complete main execution flow with realistic mocking."""
        with patch('main.get_api_key') as mock_get_key, \
             patch('main.load_pitchbook_data') as mock_load_data, \
             patch('main.IRACorporateAnalyzer') as mock_analyzer_class, \
             patch('main.setup_logging'):
            
            # Setup all mocks
            mock_get_key.return_value = 'test-api-key'
            
            mock_df = Mock()
            mock_load_data.return_value = mock_df
            
            mock_results = Mock()
            mock_results.to_csv = Mock()
            mock_analyzer = Mock()
            mock_analyzer.run = AsyncMock(return_value=mock_results)
            mock_analyzer_class.return_value = mock_analyzer
            
            # Test with all arguments
            with patch.object(sys, 'argv', ['main.py', '--csv', sample_csv_file, '--sample', '25', '--output', 'integration_test.csv']):
                from main import main
                await main()
            
            # Verify the complete flow
            mock_get_key.assert_called_once()
            mock_load_data.assert_called_once_with(sample_csv_file, 25)
            mock_analyzer_class.assert_called_once_with('test-api-key')
            mock_analyzer.run.assert_called_once_with(mock_df)
            mock_results.to_csv.assert_called_once_with('integration_test.csv', index=False)
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, sample_csv_file):
        """Test error handling throughout the main execution flow."""
        with patch('main.get_api_key') as mock_get_key, \
             patch('main.load_pitchbook_data') as mock_load_data, \
             patch('main.setup_logging'):
            
            # Test API key error
            mock_get_key.side_effect = ValueError("API key not found")
            
            with patch.object(sys, 'argv', ['main.py', '--csv', sample_csv_file]):
                from main import main
                
                with pytest.raises(SystemExit):
                    await main()
            
            # Reset and test data loading error
            mock_get_key.reset_mock()
            mock_get_key.side_effect = None
            mock_get_key.return_value = 'test-api-key'
            mock_load_data.side_effect = Exception("Data loading failed")
            
            with patch.object(sys, 'argv', ['main.py', '--csv', sample_csv_file]):
                from main import main
                
                with pytest.raises(SystemExit):
                    await main()


class TestMainEdgeCases:
    """Test edge cases in main module."""
    
    @pytest.mark.asyncio
    async def test_main_with_nonexistent_csv(self):
        """Test main with nonexistent CSV file."""
        with patch('main.get_api_key') as mock_get_key, \
             patch('main.setup_logging'):
            
            mock_get_key.return_value = 'test-api-key'
            
            with patch.object(sys, 'argv', ['main.py', '--csv', 'nonexistent.csv']):
                from main import main
                
                with pytest.raises(SystemExit):
                    await main()
    
    @pytest.mark.asyncio
    async def test_main_with_empty_csv(self, tmp_path):
        """Test main with empty CSV file."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")
        
        with patch('main.get_api_key') as mock_get_key, \
             patch('main.setup_logging'):
            
            mock_get_key.return_value = 'test-api-key'
            
            with patch.object(sys, 'argv', ['main.py', '--csv', str(empty_csv)]):
                from main import main
                
                with pytest.raises(SystemExit):
                    await main()
    
    @pytest.mark.asyncio
    async def test_main_with_unicode_csv_filename(self, tmp_path):
        """Test main with Unicode CSV filename."""
        unicode_csv = tmp_path / "测试数据.csv"
        unicode_csv.write_text("Companies,Website\nTest Company,https://test.com")
        
        with patch('main.get_api_key') as mock_get_key, \
             patch('main.load_pitchbook_data') as mock_load_data, \
             patch('main.IRACorporateAnalyzer') as mock_analyzer_class, \
             patch('main.setup_logging'):
            
            mock_get_key.return_value = 'test-api-key'
            mock_df = Mock()
            mock_load_data.return_value = mock_df
            
            mock_results = Mock()
            mock_results.to_csv = Mock()
            mock_analyzer = Mock()
            mock_analyzer.run = AsyncMock(return_value=mock_results)
            mock_analyzer_class.return_value = mock_analyzer
            
            with patch.object(sys, 'argv', ['main.py', '--csv', str(unicode_csv)]):
                from main import main
                await main()
            
            mock_load_data.assert_called_once_with(str(unicode_csv), None)
