"""
Tests for sequential tool calling functionality in ai_generator.py
"""
from unittest.mock import Mock, patch
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


def create_mock_response(content, tool_calls=None):
    """Helper to create a mock Azure OpenAI response"""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = content
    mock_response.choices[0].message.tool_calls = tool_calls
    return mock_response


def create_mock_tool_call(tool_id, function_name, arguments):
    """Helper to create a mock tool call"""
    import json
    tool_call = Mock()
    tool_call.id = tool_id
    tool_call.function.name = function_name
    tool_call.function.arguments = json.dumps(arguments) if isinstance(arguments, dict) else arguments
    return tool_call


class TestSequentialToolCalling:
    """Tests for multi-round sequential tool calling"""

    @patch('ai_generator.AzureOpenAI')
    def test_two_round_sequential_calls(self, mock_azure_client_class, ai_generator_factory):
        """Test AI making 2 sequential tool calls based on previous results"""
        mock_client = Mock()

        # Round 1: AI calls get_course_outline
        tool_call_1 = create_mock_tool_call(
            "call_outline",
            "get_course_outline",
            {"course_title": "MCP"}
        )
        response_1 = create_mock_response(None, [tool_call_1])

        # Round 2: AI calls search_course_content using outline result
        tool_call_2 = create_mock_tool_call(
            "call_search",
            "search_course_content",
            {"query": "Building Custom Servers"}
        )
        response_2 = create_mock_response(None, [tool_call_2])

        # Final: AI synthesizes answer
        response_3 = create_mock_response(
            "The course 'Advanced RAG' lesson 3 covers the same topic as lesson 4 of MCP."
        )

        mock_client.chat.completions.create.side_effect = [
            response_1, response_2, response_3
        ]
        mock_azure_client_class.return_value = mock_client

        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.side_effect = [
            "Course: MCP\nLesson 4: Building Custom Servers",
            "[Advanced RAG - Lesson 3] Building custom servers..."
        ]

        generator = ai_generator_factory()

        # Execute query
        response = generator.generate_response(
            query="Find a course discussing the same topic as lesson 4 of MCP",
            tools=[{"type": "function", "function": {"name": "get_course_outline"}},
                   {"type": "function", "function": {"name": "search_course_content"}}],
            tool_manager=tool_manager
        )

        # Assertions
        assert mock_client.chat.completions.create.call_count == 3
        assert "Advanced RAG" in response

        # Verify both tools were executed
        assert tool_manager.execute_tool.call_count == 2

        # Verify tools were included in ALL API calls
        for call_args in mock_client.chat.completions.create.call_args_list:
            assert "tools" in call_args[1]

    @patch('ai_generator.AzureOpenAI')
    def test_stops_after_max_rounds(self, mock_azure_client_class, ai_generator_factory):
        """Test that tool calling stops after reaching max_tool_rounds"""
        mock_client = Mock()

        # Both rounds request tools
        response_1 = create_mock_response(None, [create_mock_tool_call("c1", "get_course_outline", {})])
        response_2 = create_mock_response(None, [create_mock_tool_call("c2", "search_course_content", {})])
        # Final synthesis after max rounds
        response_3 = create_mock_response("Based on available information...")

        mock_client.chat.completions.create.side_effect = [
            response_1, response_2, response_3
        ]
        mock_azure_client_class.return_value = mock_client

        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Tool result"

        generator = ai_generator_factory()

        response = generator.generate_response(
            query="Complex query",
            tools=[{"type": "function"}],
            tool_manager=tool_manager,
            max_tool_rounds=2
        )

        # Should make 3 API calls: 2 rounds + 1 final synthesis
        assert mock_client.chat.completions.create.call_count == 3
        assert tool_manager.execute_tool.call_count == 2
        assert "available information" in response

    @patch('ai_generator.AzureOpenAI')
    def test_stops_when_no_tools_requested(self, mock_azure_client_class, ai_generator_factory):
        """Test that loop stops when AI doesn't request tools"""
        mock_client = Mock()

        # Round 1: Tool call
        response_1 = create_mock_response(None, [create_mock_tool_call("c1", "search", {"query": "test"})])
        # Round 2: Direct answer (no tools)
        response_2 = create_mock_response("Here's the answer based on the search.")

        mock_client.chat.completions.create.side_effect = [response_1, response_2]
        mock_azure_client_class.return_value = mock_client

        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Search results"

        generator = ai_generator_factory()

        response = generator.generate_response(
            query="What topics are covered?",
            tools=[{"type": "function"}],
            tool_manager=tool_manager
        )

        # Should stop after round 2 (no third call needed)
        assert mock_client.chat.completions.create.call_count == 2
        assert "answer based on the search" in response

    @patch('ai_generator.AzureOpenAI')
    def test_handles_tool_execution_errors(self, mock_azure_client_class, ai_generator_factory):
        """Test graceful handling of tool execution errors"""
        mock_client = Mock()

        response_1 = create_mock_response(None, [create_mock_tool_call("c1", "search", {"query": "test"})])
        response_2 = create_mock_response("I encountered an error accessing the database.")

        mock_client.chat.completions.create.side_effect = [response_1, response_2]
        mock_azure_client_class.return_value = mock_client

        tool_manager = Mock()
        tool_manager.execute_tool.side_effect = Exception("Database connection failed")

        generator = ai_generator_factory()

        response = generator.generate_response(
            query="Search for X",
            tools=[{"type": "function"}],
            tool_manager=tool_manager
        )

        # Should handle error gracefully and get AI's response
        assert mock_client.chat.completions.create.call_count == 2
        assert "error" in response.lower() or "database" in response.lower()
