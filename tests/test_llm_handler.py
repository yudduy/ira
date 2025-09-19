"""
Unit tests for src/llm_handler.py module.
Tests LLM-based content analysis functionality.
"""
import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from src.llm_handler import LLMHandler


class TestLLMHandlerInit:
    """Test LLMHandler initialization."""
    
    def test_llm_handler_init(self, mock_openai_client):
        """Test LLMHandler initialization."""
        handler = LLMHandler(mock_openai_client)
        assert handler.client == mock_openai_client


class TestAnalyzeContentChange:
    """Test content change analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_analyze_content_change_success(self, mock_openai_client, sample_llm_response):
        """Test successful content analysis."""
        # Mock OpenAI response
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(sample_llm_response)
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        handler = LLMHandler(mock_openai_client)
        
        pre_content = "We are a technology company focused on innovation."
        post_content = "We are a sustainable technology company focused on green innovation."
        
        result = await handler.analyze_content_change(pre_content, post_content)
        
        assert result['success'] is True
        assert 'data' in result
        assert result['data'] == sample_llm_response
    
    @pytest.mark.asyncio
    async def test_analyze_content_change_invalid_json(self, mock_openai_client):
        """Test handling of invalid JSON response from LLM."""
        # Mock OpenAI response with invalid JSON
        mock_choice = Mock()
        mock_choice.message.content = "This is not valid JSON"
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        handler = LLMHandler(mock_openai_client)
        
        pre_content = "Test content"
        post_content = "Test content"
        
        result = await handler.analyze_content_change(pre_content, post_content)
        
        assert result['success'] is False
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_analyze_content_change_openai_error(self, mock_openai_client):
        """Test handling of OpenAI API errors."""
        # Mock OpenAI API error
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        handler = LLMHandler(mock_openai_client)
        
        pre_content = "Test content"
        post_content = "Test content"
        
        result = await handler.analyze_content_change(pre_content, post_content)
        
        assert result['success'] is False
        assert 'error' in result
        assert 'API Error' in result['error']
    
    @pytest.mark.asyncio
    async def test_analyze_content_change_content_truncation(self, mock_openai_client, sample_llm_response):
        """Test that content is properly truncated before sending to LLM."""
        # Mock OpenAI response
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(sample_llm_response)
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        handler = LLMHandler(mock_openai_client)
        
        # Create very long content
        long_content = "This is a test sentence. " * 1000  # Much longer than PROMPT_CONTENT_LIMIT
        
        result = await handler.analyze_content_change(long_content, long_content)
        
        assert result['success'] is True
        
        # Verify that the content was truncated in the API call
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        user_message = messages[1]['content']
        
        # Content should be truncated to PROMPT_CONTENT_LIMIT (3500)
        assert len(user_message) < len(long_content) * 2  # Should be much shorter
    
    @pytest.mark.asyncio
    async def test_analyze_content_change_prompt_structure(self, mock_openai_client, sample_llm_response):
        """Test that the prompt has the correct structure."""
        # Mock OpenAI response
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(sample_llm_response)
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        handler = LLMHandler(mock_openai_client)
        
        pre_content = "Pre-IRA content about technology."
        post_content = "Post-IRA content about sustainable technology."
        
        result = await handler.analyze_content_change(pre_content, post_content)
        
        # Verify the API call structure
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        
        # Check system message
        assert messages[0]['role'] == 'system'
        assert 'research analyst' in messages[0]['content'].lower()
        assert 'corporate communications' in messages[0]['content'].lower()
        
        # Check user message
        assert messages[1]['role'] == 'user'
        user_content = messages[1]['content']
        assert 'PRE-IRA TEXT' in user_content
        assert 'POST-IRA TEXT' in user_content
        assert 'JSON object' in user_content
        assert 'change_analysis' in user_content
        assert 'ira_alignment' in user_content
        assert 'overall_assessment' in user_content
    
    @pytest.mark.asyncio
    async def test_analyze_content_change_api_parameters(self, mock_openai_client, sample_llm_response):
        """Test that API is called with correct parameters."""
        # Mock OpenAI response
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(sample_llm_response)
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        handler = LLMHandler(mock_openai_client)
        
        pre_content = "Test content"
        post_content = "Test content"
        
        result = await handler.analyze_content_change(pre_content, post_content)
        
        # Verify API call parameters
        call_args = mock_openai_client.chat.completions.create.call_args
        kwargs = call_args[1]
        
        assert kwargs['model'] == 'gpt-4.1-nano'
        assert kwargs['temperature'] == 0.1
        assert kwargs['max_tokens'] == 1000
        assert kwargs['response_format'] == {'type': 'json_object'}
    
    @pytest.mark.asyncio
    async def test_analyze_content_change_empty_content(self, mock_openai_client, sample_llm_response):
        """Test handling of empty content."""
        # Mock OpenAI response
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(sample_llm_response)
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        handler = LLMHandler(mock_openai_client)
        
        result = await handler.analyze_content_change("", "")
        
        assert result['success'] is True
        
        # Verify that empty content was handled in the prompt
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        user_message = messages[1]['content']
        
        assert 'PRE-IRA TEXT (Full Year 2022):' in user_message
        assert 'POST-IRA TEXT (Full Year 2023):' in user_message
    
    @pytest.mark.asyncio
    async def test_analyze_content_change_malformed_json_response(self, mock_openai_client):
        """Test handling of malformed JSON in LLM response."""
        # Mock OpenAI response with malformed JSON
        mock_choice = Mock()
        mock_choice.message.content = '{"change_analysis": {"lexical_change": {"has_changed": true'  # Missing closing braces
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        handler = LLMHandler(mock_openai_client)
        
        result = await handler.analyze_content_change("Test", "Test")
        
        assert result['success'] is False
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_analyze_content_change_unicode_content(self, mock_openai_client, sample_llm_response):
        """Test handling of Unicode content."""
        # Mock OpenAI response
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(sample_llm_response)
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        handler = LLMHandler(mock_openai_client)
        
        pre_content = "我们是一家科技公司，专注于创新。"
        post_content = "我们是一家可持续发展的科技公司，专注于绿色创新。"
        
        result = await handler.analyze_content_change(pre_content, post_content)
        
        assert result['success'] is True
        
        # Verify Unicode content was handled properly
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        user_message = messages[1]['content']
        
        assert '我们是一家科技公司' in user_message
        assert '可持续发展' in user_message


class TestLLMHandlerIntegration:
    """Integration tests for LLMHandler."""
    
    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, mock_openai_client):
        """Test complete analysis workflow with realistic content."""
        # Mock a realistic LLM response
        realistic_response = {
            "change_analysis": {
                "lexical_change": {
                    "has_changed": True,
                    "summary": "Added 'sustainable', 'green', and 'ESG' terminology"
                },
                "strategic_framing": {
                    "has_changed": True,
                    "from_narrative": "Technology company focused on innovation and growth",
                    "to_narrative": "Sustainable technology company focused on green innovation and ESG impact",
                    "summary": "Shifted from general innovation focus to sustainability-driven innovation"
                },
                "target_audience": {
                    "has_changed": False,
                    "primary_audience": "B2B Customers, Investors, and Enterprise Clients",
                    "summary": "No significant change in target audience"
                }
            },
            "ira_alignment": {
                "alignment_detected": True,
                "evidence_type": "conceptual_language",
                "specific_evidence": ["domestic content", "clean energy", "tax incentives"],
                "reasoning": "Content mentions clean energy and tax incentives, which align with IRA provisions"
            },
            "overall_assessment": {
                "change_level": "moderate",
                "confidence": 0.85,
                "synthesis_reasoning": "Moderate change due to clear shift toward sustainability messaging and potential IRA alignment"
            }
        }
        
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(realistic_response)
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        handler = LLMHandler(mock_openai_client)
        
        pre_content = "We are a technology company focused on innovation and growth. Our solutions help businesses scale efficiently."
        post_content = "We are a sustainable technology company focused on green innovation and ESG impact. Our clean energy solutions help businesses scale efficiently while meeting domestic content requirements for tax incentives."
        
        result = await handler.analyze_content_change(pre_content, post_content)
        
        assert result['success'] is True
        assert result['data']['overall_assessment']['change_level'] == 'moderate'
        assert result['data']['ira_alignment']['alignment_detected'] is True
    
    @pytest.mark.asyncio
    async def test_error_recovery_mechanisms(self, mock_openai_client):
        """Test various error conditions and recovery."""
        # Test timeout error
        mock_openai_client.chat.completions.create.side_effect = Exception("Request timeout")
        
        handler = LLMHandler(mock_openai_client)
        
        result = await handler.analyze_content_change("Test", "Test")
        
        assert result['success'] is False
        assert 'Request timeout' in result['error']


class TestLLMHandlerEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    async def test_extremely_long_content(self, mock_openai_client, sample_llm_response):
        """Test handling of extremely long content."""
        # Mock OpenAI response
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(sample_llm_response)
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        handler = LLMHandler(mock_openai_client)
        
        # Create content that exceeds even CONTENT_TRUNCATION_LIMIT
        extremely_long_content = "Word " * 100000  # Much longer than any limit
        
        result = await handler.analyze_content_change(extremely_long_content, extremely_long_content)
        
        assert result['success'] is True
        
        # Verify content was truncated appropriately
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        user_message = messages[1]['content']
        
        # Should be much shorter than the input
        assert len(user_message) < len(extremely_long_content) * 2
    
    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, mock_openai_client, sample_llm_response):
        """Test handling of special characters in content."""
        # Mock OpenAI response
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(sample_llm_response)
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        handler = LLMHandler(mock_openai_client)
        
        special_content = "Content with special chars: @#$%^&*()_+-=[]{}|;':\",./<>?`~"
        
        result = await handler.analyze_content_change(special_content, special_content)
        
        assert result['success'] is True
        
        # Verify special characters were handled
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        user_message = messages[1]['content']
        
        assert special_content in user_message
    
    @pytest.mark.asyncio
    async def test_identical_content_analysis(self, mock_openai_client):
        """Test analysis when pre and post content are identical."""
        identical_response = {
            "change_analysis": {
                "lexical_change": {
                    "has_changed": False,
                    "summary": "No changes in vocabulary or keywords detected"
                },
                "strategic_framing": {
                    "has_changed": False,
                    "from_narrative": "Same narrative in both periods",
                    "to_narrative": "Same narrative in both periods",
                    "summary": "No significant change"
                },
                "target_audience": {
                    "has_changed": False,
                    "primary_audience": "Same audience in both periods",
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
                "change_level": "none",
                "confidence": 1.0,
                "synthesis_reasoning": "No changes detected between pre and post-IRA content"
            }
        }
        
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(identical_response)
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        handler = LLMHandler(mock_openai_client)
        
        identical_content = "We are a technology company focused on innovation."
        
        result = await handler.analyze_content_change(identical_content, identical_content)
        
        assert result['success'] is True
        assert result['data']['overall_assessment']['change_level'] == 'none'
        assert result['data']['overall_assessment']['confidence'] == 1.0
