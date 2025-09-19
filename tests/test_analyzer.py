"""
Unit tests for src/analyzer.py module.
Tests the core analysis orchestrator and workflow integration.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch
from src.analyzer import IRACorporateAnalyzer, _flatten_analysis_output


class TestFlattenAnalysisOutput:
    """Test the analysis output flattening function."""
    
    def test_flatten_analysis_output_complete(self, sample_llm_response):
        """Test flattening of complete analysis output."""
        flattened = _flatten_analysis_output(sample_llm_response)
        
        # Check lexical change fields
        assert 'lexical_has_changed' in flattened
        assert 'lexical_summary' in flattened
        assert flattened['lexical_has_changed'] is True
        
        # Check strategic framing fields
        assert 'framing_has_changed' in flattened
        assert 'framing_from_narrative' in flattened
        assert 'framing_to_narrative' in flattened
        assert 'framing_summary' in flattened
        
        # Check target audience fields
        assert 'audience_has_changed' in flattened
        assert 'audience_primary_audience' in flattened
        assert 'audience_summary' in flattened
        
        # Check IRA alignment fields
        assert 'ira_alignment_detected' in flattened
        assert 'ira_evidence_type' in flattened
        assert 'ira_specific_evidence' in flattened
        assert 'ira_reasoning' in flattened
        
        # Check overall assessment fields
        assert 'overall_change_level' in flattened
        assert 'overall_confidence' in flattened
        assert 'overall_synthesis_reasoning' in flattened
    
    def test_flatten_analysis_output_empty(self):
        """Test flattening of empty analysis output."""
        empty_response = {}
        flattened = _flatten_analysis_output(empty_response)
        
        # Should return empty dict
        assert flattened == {}
    
    def test_flatten_analysis_output_partial(self):
        """Test flattening of partial analysis output."""
        partial_response = {
            "change_analysis": {
                "lexical_change": {
                    "has_changed": True,
                    "summary": "Added sustainability keywords"
                }
            },
            "overall_assessment": {
                "change_level": "minor"
            }
        }
        
        flattened = _flatten_analysis_output(partial_response)
        
        assert 'lexical_has_changed' in flattened
        assert 'lexical_summary' in flattened
        assert 'overall_change_level' in flattened
        assert flattened['lexical_has_changed'] is True
        assert flattened['overall_change_level'] == "minor"
    
    def test_flatten_analysis_output_nested_none(self):
        """Test flattening when nested values are None."""
        response_with_none = {
            "change_analysis": {
                "lexical_change": {
                    "has_changed": None,
                    "summary": None
                }
            }
        }
        
        flattened = _flatten_analysis_output(response_with_none)
        
        assert 'lexical_has_changed' in flattened
        assert 'lexical_summary' in flattened
        assert flattened['lexical_has_changed'] is None
        assert flattened['lexical_summary'] is None


class TestIRACorporateAnalyzerInit:
    """Test IRACorporateAnalyzer initialization."""
    
    @patch('src.analyzer.openai.AsyncOpenAI')
    @patch('src.analyzer.requests.Session')
    def test_analyzer_init(self, mock_session, mock_openai):
        """Test analyzer initialization."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        
        mock_openai_instance = Mock()
        mock_openai.return_value = mock_openai_instance
        
        analyzer = IRACorporateAnalyzer("test-api-key")
        
        assert analyzer.session == mock_session_instance
        assert analyzer.openai_client == mock_openai_instance
        assert analyzer.wayback_client is not None
        assert analyzer.llm_handler is not None
        
        # Check session headers were set
        assert hasattr(analyzer.session, 'headers')
        # The headers dict is mocked, so we can't check its contents directly


class TestAnalyzeCompany:
    """Test individual company analysis."""
    
    @pytest.mark.asyncio
    async def test_analyze_company_success(self, sample_companies_df, mock_openai_client, mock_requests_session, sample_llm_response):
        """Test successful company analysis."""
        # Mock all dependencies
        with patch('src.analyzer.openai.AsyncOpenAI') as mock_openai, \
             patch('src.analyzer.requests.Session') as mock_session:
            
            mock_openai.return_value = mock_openai_client
            mock_session.return_value = mock_requests_session
            
            # Mock wayback client responses
            wayback_client = Mock()
            wayback_client.find_snapshots = AsyncMock(side_effect=[
                {'success': True, 'snapshot': {'archive_url': 'https://web.archive.org/web/20220315000000/https://acme.com/'}},
                {'success': True, 'snapshot': {'archive_url': 'https://web.archive.org/web/20230315000000/https://acme.com/'}}
            ])
            wayback_client.extract_content = AsyncMock(side_effect=[
                {'success': True, 'content': 'Pre-IRA content', 'word_count': 100},
                {'success': True, 'content': 'Post-IRA content', 'word_count': 120}
            ])
            
            # Mock LLM handler
            llm_handler = Mock()
            llm_handler.analyze_content_change = AsyncMock(return_value={'success': True, 'data': sample_llm_response})
            
            analyzer = IRACorporateAnalyzer("test-api-key")
            analyzer.wayback_client = wayback_client
            analyzer.llm_handler = llm_handler
            
            company_row = sample_companies_df.iloc[0]
            result = await analyzer.analyze_company(company_row)
            
            assert result['status'] == 'completed'
            assert result['company_name'] == 'Acme Corp'
            assert result['domain'] == 'acme.com'
            assert result['pre_word_count'] == 100
            assert result['post_word_count'] == 120
            assert 'lexical_has_changed' in result
    
    @pytest.mark.asyncio
    async def test_analyze_company_insufficient_snapshots(self, sample_companies_df, mock_openai_client, mock_requests_session):
        """Test handling when snapshots are not available."""
        with patch('src.analyzer.openai.AsyncOpenAI') as mock_openai, \
             patch('src.analyzer.requests.Session') as mock_session:
            
            mock_openai.return_value = mock_openai_client
            mock_session.return_value = mock_requests_session
            
            # Mock wayback client to return failure
            wayback_client = Mock()
            wayback_client.find_snapshots = AsyncMock(return_value={'success': False, 'error': 'No snapshots found'})
            
            analyzer = IRACorporateAnalyzer("test-api-key")
            analyzer.wayback_client = wayback_client
            
            company_row = sample_companies_df.iloc[0]
            result = await analyzer.analyze_company(company_row)
            
            assert result['status'] == 'insufficient_snapshots'
            assert 'pre_snapshot_error' in result
            assert 'post_snapshot_error' in result
    
    @pytest.mark.asyncio
    async def test_analyze_company_content_extraction_failed(self, sample_companies_df, mock_openai_client, mock_requests_session):
        """Test handling when content extraction fails."""
        with patch('src.analyzer.openai.AsyncOpenAI') as mock_openai, \
             patch('src.analyzer.requests.Session') as mock_session:
            
            mock_openai.return_value = mock_openai_client
            mock_session.return_value = mock_requests_session
            
            # Mock wayback client
            wayback_client = Mock()
            wayback_client.find_snapshots = AsyncMock(return_value={'success': True, 'snapshot': {'archive_url': 'test-url'}})
            wayback_client.extract_content = AsyncMock(return_value={'success': False, 'error': 'Content extraction failed'})
            
            analyzer = IRACorporateAnalyzer("test-api-key")
            analyzer.wayback_client = wayback_client
            
            company_row = sample_companies_df.iloc[0]
            result = await analyzer.analyze_company(company_row)
            
            assert result['status'] == 'content_extraction_failed'
            assert 'pre_content_error' in result
            assert 'post_content_error' in result
    
    @pytest.mark.asyncio
    async def test_analyze_company_analysis_error(self, sample_companies_df, mock_openai_client, mock_requests_session):
        """Test handling when LLM analysis fails."""
        with patch('src.analyzer.openai.AsyncOpenAI') as mock_openai, \
             patch('src.analyzer.requests.Session') as mock_session:
            
            mock_openai.return_value = mock_openai_client
            mock_session.return_value = mock_requests_session
            
            # Mock wayback client
            wayback_client = Mock()
            wayback_client.find_snapshots = AsyncMock(return_value={'success': True, 'snapshot': {'archive_url': 'test-url'}})
            wayback_client.extract_content = AsyncMock(return_value={'success': True, 'content': 'test content', 'word_count': 100})
            
            # Mock LLM handler to fail
            llm_handler = Mock()
            llm_handler.analyze_content_change = AsyncMock(return_value={'success': False, 'error': 'LLM analysis failed'})
            
            analyzer = IRACorporateAnalyzer("test-api-key")
            analyzer.wayback_client = wayback_client
            analyzer.llm_handler = llm_handler
            
            company_row = sample_companies_df.iloc[0]
            result = await analyzer.analyze_company(company_row)
            
            assert result['status'] == 'analysis_error'
            assert 'error_message' in result
            assert 'LLM analysis failed' in result['error_message']


class TestRun:
    """Test the main run method."""
    
    @pytest.mark.asyncio
    async def test_run_success(self, sample_companies_df, mock_openai_client, mock_requests_session, sample_llm_response):
        """Test successful run of analysis on multiple companies."""
        with patch('src.analyzer.openai.AsyncOpenAI') as mock_openai, \
             patch('src.analyzer.requests.Session') as mock_session:
            
            mock_openai.return_value = mock_openai_client
            mock_session.return_value = mock_requests_session
            
            # Mock wayback client
            wayback_client = Mock()
            wayback_client.find_snapshots = AsyncMock(return_value={'success': True, 'snapshot': {'archive_url': 'test-url'}})
            wayback_client.extract_content = AsyncMock(return_value={'success': True, 'content': 'test content', 'word_count': 100})
            
            # Mock LLM handler
            llm_handler = Mock()
            llm_handler.analyze_content_change = AsyncMock(return_value={'success': True, 'data': sample_llm_response})
            
            analyzer = IRACorporateAnalyzer("test-api-key")
            analyzer.wayback_client = wayback_client
            analyzer.llm_handler = llm_handler
            
            # Run with a small subset for testing
            small_df = sample_companies_df.head(2)
            results_df = await analyzer.run(small_df, max_concurrent=1)
            
            assert isinstance(results_df, pd.DataFrame)
            assert len(results_df) == 2
            assert 'company_name' in results_df.columns
            assert 'status' in results_df.columns
            assert all(results_df['status'] == 'completed')
    
    @pytest.mark.asyncio
    async def test_run_with_mixed_results(self, sample_companies_df, mock_openai_client, mock_requests_session):
        """Test run with mixed success/failure results."""
        with patch('src.analyzer.openai.AsyncOpenAI') as mock_openai, \
             patch('src.analyzer.requests.Session') as mock_session:
            
            mock_openai.return_value = mock_openai_client
            mock_session.return_value = mock_requests_session
            
            # Mock wayback client with mixed results
            wayback_client = Mock()
            async def mock_find_snapshots(domain, window):
                if domain == 'acme.com':
                    return {'success': True, 'snapshot': {'archive_url': 'test-url'}}
                elif domain == 'greenenergy.com':
                    return {'success': False, 'error': 'No snapshots'}
                elif domain == 'sustainablesolutions.org':
                    return {'success': True, 'snapshot': {'archive_url': 'test-url'}}
                else:
                    return {'success': False, 'error': 'Unknown domain'}
            
            wayback_client.find_snapshots = mock_find_snapshots
            wayback_client.extract_content = AsyncMock(return_value={'success': True, 'content': 'test content', 'word_count': 100})
            
            # Mock LLM handler
            llm_handler = Mock()
            llm_handler.analyze_content_change = AsyncMock(return_value={'success': True, 'data': {'overall_assessment': {'change_level': 'minor'}}})
            
            analyzer = IRACorporateAnalyzer("test-api-key")
            analyzer.wayback_client = wayback_client
            analyzer.llm_handler = llm_handler
            
            results_df = await analyzer.run(sample_companies_df, max_concurrent=1)
            
            assert isinstance(results_df, pd.DataFrame)
            assert len(results_df) == 3  # All companies should be in results
            
            # Check that we have mixed statuses
            statuses = results_df['status'].unique()
            assert 'completed' in statuses
            assert 'insufficient_snapshots' in statuses
    
    @pytest.mark.asyncio
    async def test_run_concurrency_control(self, sample_companies_df, mock_openai_client, mock_requests_session):
        """Test that concurrency is properly controlled."""
        with patch('src.analyzer.openai.AsyncOpenAI') as mock_openai, \
             patch('src.analyzer.requests.Session') as mock_session, \
             patch('src.analyzer.asyncio.Semaphore') as mock_semaphore:
            
            mock_openai.return_value = mock_openai_client
            mock_session.return_value = mock_requests_session
            
            # Mock semaphore
            mock_semaphore_instance = AsyncMock()
            mock_semaphore.return_value = mock_semaphore_instance
            
            # Mock wayback client
            wayback_client = Mock()
            wayback_client.find_snapshots = AsyncMock(return_value={'success': True, 'snapshot': {'archive_url': 'test-url'}})
            wayback_client.extract_content = AsyncMock(return_value={'success': True, 'content': 'test content', 'word_count': 100})
            
            # Mock LLM handler
            llm_handler = Mock()
            llm_handler.analyze_content_change = AsyncMock(return_value={'success': True, 'data': {'overall_assessment': {'change_level': 'minor'}}})
            
            analyzer = IRACorporateAnalyzer("test-api-key")
            analyzer.wayback_client = wayback_client
            analyzer.llm_handler = llm_handler
            
            results_df = await analyzer.run(sample_companies_df, max_concurrent=2)
            
            # Verify semaphore was created with correct limit
            mock_semaphore.assert_called_once_with(2)
    
    @pytest.mark.asyncio
    async def test_run_empty_dataframe(self, mock_openai_client, mock_requests_session):
        """Test run with empty DataFrame."""
        with patch('src.analyzer.openai.AsyncOpenAI') as mock_openai, \
             patch('src.analyzer.requests.Session') as mock_session:
            
            mock_openai.return_value = mock_openai_client
            mock_session.return_value = mock_requests_session
            
            analyzer = IRACorporateAnalyzer("test-api-key")
            
            empty_df = pd.DataFrame(columns=['Companies', 'Website', 'Domain'])
            results_df = await analyzer.run(empty_df)
            
            assert isinstance(results_df, pd.DataFrame)
            assert len(results_df) == 0


class TestAnalyzerIntegration:
    """Integration tests for the analyzer."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_analysis_simulation(self, sample_companies_df):
        """Test end-to-end analysis with realistic mocking."""
        with patch('src.analyzer.openai.AsyncOpenAI') as mock_openai, \
             patch('src.analyzer.requests.Session') as mock_session:
            
            # Set up mocks
            mock_openai_instance = Mock()
            mock_openai.return_value = mock_openai_instance
            
            mock_session_instance = Mock()
            mock_session_instance.headers = {}
            mock_session.return_value = mock_session_instance
            
            # Create analyzer
            analyzer = IRACorporateAnalyzer("test-api-key")
            
            # Mock the wayback client methods
            analyzer.wayback_client.find_snapshots = AsyncMock(return_value={
                'success': True, 
                'snapshot': {'archive_url': 'https://web.archive.org/web/20220315000000/https://acme.com/'}
            })
            analyzer.wayback_client.extract_content = AsyncMock(return_value={
                'success': True, 
                'content': 'Company content for analysis',
                'word_count': 150
            })
            
            # Mock the LLM handler
            analyzer.llm_handler.analyze_content_change = AsyncMock(return_value={
                'success': True,
                'data': {
                    'change_analysis': {
                        'lexical_change': {'has_changed': True, 'summary': 'Added sustainability terms'},
                        'strategic_framing': {'has_changed': False, 'from_narrative': 'Tech company', 'to_narrative': 'Tech company', 'summary': 'No change'},
                        'target_audience': {'has_changed': False, 'primary_audience': 'B2B', 'summary': 'No change'}
                    },
                    'ira_alignment': {'alignment_detected': True, 'evidence_type': 'conceptual_language', 'specific_evidence': ['clean energy'], 'reasoning': 'Mentions clean energy'},
                    'overall_assessment': {'change_level': 'minor', 'confidence': 0.8, 'synthesis_reasoning': 'Minor changes detected'}
                }
            })
            
            # Run analysis on first company
            company_row = sample_companies_df.iloc[0]
            result = await analyzer.analyze_company(company_row)
            
            # Verify the result structure
            assert result['status'] == 'completed'
            assert result['company_name'] == 'Acme Corp'
            assert result['domain'] == 'acme.com'
            assert result['pre_word_count'] == 150
            assert result['post_word_count'] == 150
            assert result['lexical_has_changed'] is True
            assert result['overall_change_level'] == 'minor'
            assert result['ira_alignment_detected'] is True
    
    @pytest.mark.asyncio
    async def test_error_handling_chain(self, sample_companies_df):
        """Test that errors are properly handled through the analysis chain."""
        with patch('src.analyzer.openai.AsyncOpenAI') as mock_openai, \
             patch('src.analyzer.requests.Session') as mock_session:
            
            mock_openai.return_value = Mock()
            mock_session.return_value = Mock()
            
            analyzer = IRACorporateAnalyzer("test-api-key")
            
            # Mock wayback client to fail
            analyzer.wayback_client.find_snapshots = AsyncMock(return_value={'success': False, 'error': 'Network timeout'})
            
            company_row = sample_companies_df.iloc[0]
            result = await analyzer.analyze_company(company_row)
            
            # Should handle the error gracefully
            assert result['status'] == 'insufficient_snapshots'
            assert 'Network timeout' in result['pre_snapshot_error']


class TestAnalyzerEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    async def test_analyzer_with_unicode_company_names(self, mock_openai_client, mock_requests_session):
        """Test analyzer with Unicode company names."""
        with patch('src.analyzer.openai.AsyncOpenAI') as mock_openai, \
             patch('src.analyzer.requests.Session') as mock_session:
            
            mock_openai.return_value = mock_openai_client
            mock_session.return_value = mock_requests_session
            
            analyzer = IRACorporateAnalyzer("test-api-key")
            
            # Create DataFrame with Unicode names
            unicode_df = pd.DataFrame({
                'Companies': ['测试公司', 'Компания', 'شركة'],
                'Website': ['https://test.com', 'https://company.ru', 'https://company.ae'],
                'Domain': ['test.com', 'company.ru', 'company.ae']
            })
            
            # Mock successful responses
            analyzer.wayback_client.find_snapshots = AsyncMock(return_value={'success': True, 'snapshot': {'archive_url': 'test-url'}})
            analyzer.wayback_client.extract_content = AsyncMock(return_value={'success': True, 'content': 'content', 'word_count': 100})
            analyzer.llm_handler.analyze_content_change = AsyncMock(return_value={'success': True, 'data': {'overall_assessment': {'change_level': 'minor'}}})
            
            result = await analyzer.analyze_company(unicode_df.iloc[0])
            
            assert result['company_name'] == '测试公司'
            assert result['status'] == 'completed'
    
    @pytest.mark.asyncio
    async def test_analyzer_with_very_long_content(self, sample_companies_df, mock_openai_client, mock_requests_session):
        """Test analyzer with very long content."""
        with patch('src.analyzer.openai.AsyncOpenAI') as mock_openai, \
             patch('src.analyzer.requests.Session') as mock_session:
            
            mock_openai.return_value = mock_openai_client
            mock_session.return_value = mock_requests_session
            
            analyzer = IRACorporateAnalyzer("test-api-key")
            
            # Mock with very long content
            long_content = "Very long content. " * 10000
            analyzer.wayback_client.find_snapshots = AsyncMock(return_value={'success': True, 'snapshot': {'archive_url': 'test-url'}})
            analyzer.wayback_client.extract_content = AsyncMock(return_value={'success': True, 'content': long_content, 'word_count': 20000})
            analyzer.llm_handler.analyze_content_change = AsyncMock(return_value={'success': True, 'data': {'overall_assessment': {'change_level': 'major'}}})
            
            company_row = sample_companies_df.iloc[0]
            result = await analyzer.analyze_company(company_row)
            
            assert result['status'] == 'completed'
            assert result['pre_word_count'] == 20000
            assert result['overall_change_level'] == 'major'
