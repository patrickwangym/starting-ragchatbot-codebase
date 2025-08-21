"""Tests for AIGenerator and Claude API integration"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test AIGenerator functionality"""

    def test_init(self):
        """Test AIGenerator initialization"""
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        
        assert generator.model == "claude-sonnet-4-20250514"
        assert generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_simple(self, mock_anthropic):
        """Test simple response generation without tools"""
        # Setup mock client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a test response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = generator.generate_response("What is machine learning?")
        
        assert result == "This is a test response"
        
        # Verify API call
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["messages"][0]["content"] == "What is machine learning?"
        assert "tools" not in call_args

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test response generation with conversation history"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with history")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "Follow up question",
            conversation_history="Previous conversation content"
        )
        
        assert result == "Response with history"
        
        # Verify system prompt includes history
        call_args = mock_client.messages.create.call_args[1]
        assert "Previous conversation content" in call_args["system"]

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tools(self, mock_anthropic, mock_tool_manager):
        """Test response generation with tools available"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response using tools")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        tools = [{"name": "search_course_content", "description": "Search courses"}]
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "Search for ML content",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Response using tools"
        
        # Verify tools were included in API call
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_tool_use(self, mock_anthropic, mock_tool_manager):
        """Test response generation when Claude uses tools"""
        mock_client = Mock()
        
        # Mock initial tool use response
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "machine learning"}
        tool_block.id = "tool_123"
        initial_response.content = [tool_block]
        
        # Mock final response after tool execution
        final_response = Mock()
        final_response.content = [Mock(text="Final response with tool results")]
        
        # Setup client to return different responses for different calls
        mock_client.messages.create.side_effect = [initial_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool manager
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        tools = [{"name": "search_course_content"}]
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "What is machine learning?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Final response with tool results"
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning"
        )
        
        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_multiple_tool_calls(self, mock_anthropic, mock_tool_manager):
        """Test handling multiple tool calls in one response"""
        mock_client = Mock()
        
        # Mock initial response with multiple tool uses
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        
        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "search_course_content"
        tool_block1.input = {"query": "ML"}
        tool_block1.id = "tool_1"
        
        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.name = "get_course_outline"
        tool_block2.input = {"course_title": "ML Course"}
        tool_block2.id = "tool_2"
        
        initial_response.content = [tool_block1, tool_block2]
        
        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Response with multiple tools")]
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool manager returns
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        tools = [{"name": "search_course_content"}, {"name": "get_course_outline"}]
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "Tell me about ML courses",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Response with multiple tools"
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Check the tool result message structure
        final_call_args = mock_client.messages.create.call_args_list[1][1]
        tool_result_message = final_call_args["messages"][-1]
        assert tool_result_message["role"] == "user"
        assert len(tool_result_message["content"]) == 2  # Two tool results

    def test_handle_tool_execution_message_flow(self, mock_tool_manager):
        """Test message flow during tool execution"""
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        
        # Mock initial response
        initial_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "test"}
        tool_block.id = "tool_123"
        initial_response.content = [tool_block]
        
        # Mock base params (like what would be passed to the first API call)
        base_params = {
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "System prompt"
        }
        
        # Mock tool execution
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        # Mock final API call
        with patch.object(generator.client, 'messages') as mock_messages:
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="Final response")]
            mock_messages.create.return_value = mock_final_response
            
            result = generator._handle_tool_execution(
                initial_response, base_params, mock_tool_manager
            )
            
            assert result == "Final response"
            
            # Verify the final API call structure
            final_call_args = mock_messages.create.call_args[1]
            messages = final_call_args["messages"]
            
            # Should have: original user message + assistant tool use + user tool results
            assert len(messages) == 3
            assert messages[0]["role"] == "user"  # Original query
            assert messages[1]["role"] == "assistant"  # Tool use
            assert messages[2]["role"] == "user"  # Tool results
            
            # Check tool result structure
            tool_results = messages[2]["content"]
            assert len(tool_results) == 1
            assert tool_results[0]["type"] == "tool_result"
            assert tool_results[0]["tool_use_id"] == "tool_123"
            assert tool_results[0]["content"] == "Tool result"

    @patch('ai_generator.anthropic.Anthropic')
    def test_api_error_handling(self, mock_anthropic):
        """Test handling of API errors"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        
        with pytest.raises(Exception, match="API Error"):
            generator.generate_response("Test query")

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_error_handling(self, mock_anthropic, mock_tool_manager):
        """Test handling of tool execution errors"""
        mock_client = Mock()
        
        # Mock tool use response
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "test"}
        tool_block.id = "tool_123"
        initial_response.content = [tool_block]
        
        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Error handled response")]
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool execution error
        mock_tool_manager.execute_tool.return_value = "Tool execution failed: Database error"
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Should still return a response, with error passed to Claude
        assert result == "Error handled response"
        
        # Verify tool error was included in the message
        final_call_args = mock_client.messages.create.call_args_list[1][1]
        tool_result = final_call_args["messages"][-1]["content"][0]
        assert "Tool execution failed: Database error" in tool_result["content"]

    def test_system_prompt_content(self):
        """Test that system prompt contains expected instructions"""
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        
        system_prompt = generator.SYSTEM_PROMPT
        
        # Check for key instructions
        assert "search_course_content" in system_prompt
        assert "get_course_outline" in system_prompt
        assert "One tool use per query maximum" in system_prompt
        assert "Brief, Concise and focused" in system_prompt

    @patch('ai_generator.anthropic.Anthropic')
    def test_no_conversation_history(self, mock_anthropic):
        """Test response generation without conversation history"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response without history")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = generator.generate_response("Test query", conversation_history=None)
        
        assert result == "Response without history"
        
        # Verify system prompt doesn't include history section
        call_args = mock_client.messages.create.call_args[1]
        assert "Previous conversation:" not in call_args["system"]

    def test_base_params_configuration(self):
        """Test that base parameters are properly configured"""
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        
        expected_params = {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0,
            "max_tokens": 800
        }
        
        for key, value in expected_params.items():
            assert generator.base_params[key] == value