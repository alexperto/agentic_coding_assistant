"""
Integration tests for ai_generator.py - AIGenerator and tool calling
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


class TestAIGeneratorBasics:
    """Basic tests for AIGenerator"""

    def test_initialization(self, mock_token_manager):
        """Test AIGenerator can be initialized"""
        generator = AIGenerator(
            endpoint="https://test.openai.azure.com",
            token_manager=mock_token_manager,
            api_version="2024-02-01",
            deployment="gpt-4"
        )

        assert generator.deployment == "gpt-4"
        assert generator.base_params["model"] == "gpt-4"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    def test_system_prompt_exists(self):
        """Test that system prompt is defined"""
        assert AIGenerator.SYSTEM_PROMPT is not None
        assert "assistant" in AIGenerator.SYSTEM_PROMPT.lower()
        assert "tools" in AIGenerator.SYSTEM_PROMPT.lower()


class TestAIGeneratorWithoutTools:
    """Test AIGenerator when tools are not used"""

    @patch('ai_generator.AzureOpenAI')
    def test_generate_response_without_tools(self, mock_azure_client_class, ai_generator_factory):
        """Test generating a simple response without tools"""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = create_mock_response("This is a test response.")
        mock_azure_client_class.return_value = mock_client

        generator = ai_generator_factory()

        response = generator.generate_response(query="What is 2+2?")

        assert response == "This is a test response."
        mock_client.chat.completions.create.assert_called_once()

    @patch('ai_generator.AzureOpenAI')
    def test_generate_response_with_conversation_history(self, mock_azure_client_class, ai_generator_factory):
        """Test generating response with conversation history"""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = create_mock_response("Continuing the conversation.")
        mock_azure_client_class.return_value = mock_client

        generator = ai_generator_factory()

        history = "User: Hello\nAssistant: Hi there!"
        response = generator.generate_response(query="How are you?", conversation_history=history)

        assert response == "Continuing the conversation."
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) >= 3


class TestAIGeneratorWithTools:
    """Test AIGenerator with tool calling"""

    @patch('ai_generator.AzureOpenAI')
    def test_tools_parameter_passed_to_api(self, mock_azure_client_class, mock_vector_store, ai_generator_factory):
        """Test that tools are correctly passed to the API"""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = create_mock_response("Here's the answer.")
        mock_azure_client_class.return_value = mock_client

        tool_manager = ToolManager()
        tool_manager.register_tool(CourseSearchTool(mock_vector_store))

        generator = ai_generator_factory()

        generator.generate_response(
            query="What is Anthropic?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"
        assert len(call_kwargs["tools"]) > 0

    @patch('ai_generator.AzureOpenAI')
    def test_ai_calls_search_tool(self, mock_azure_client_class, mock_vector_store_with_results, ai_generator_factory):
        """Test that AI correctly calls the search tool"""
        mock_client = Mock()

        # First response: AI wants to use tool
        tool_call = create_mock_tool_call(
            "call_123",
            "search_course_content",
            {"query": "What is Anthropic?", "course_name": "Computer Use"}
        )
        initial_response = create_mock_response(None, [tool_call])

        # Second response: AI provides final answer
        final_response = create_mock_response("Based on the course material, Anthropic is an AI safety company.")

        mock_client.chat.completions.create.side_effect = [initial_response, final_response]
        mock_azure_client_class.return_value = mock_client

        tool_manager = ToolManager()
        tool_manager.register_tool(CourseSearchTool(mock_vector_store_with_results))

        generator = ai_generator_factory()

        response = generator.generate_response(
            query="What is Anthropic?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        assert "Anthropic is an AI safety company" in response
        mock_vector_store_with_results.search.assert_called_once_with(
            query="What is Anthropic?",
            course_name="Computer Use",
            lesson_number=None
        )
        assert mock_client.chat.completions.create.call_count == 2

    @patch('ai_generator.AzureOpenAI')
    def test_tool_result_formatting(self, mock_azure_client_class, mock_vector_store_with_results, ai_generator_factory):
        """Test that tool results are correctly formatted in messages"""
        mock_client = Mock()

        tool_call = create_mock_tool_call("call_456", "search_course_content", {"query": "test query"})
        initial_response = create_mock_response("Let me search for that.", [tool_call])
        final_response = create_mock_response("Here's what I found.")

        mock_client.chat.completions.create.side_effect = [initial_response, final_response]
        mock_azure_client_class.return_value = mock_client

        tool_manager = ToolManager()
        tool_manager.register_tool(CourseSearchTool(mock_vector_store_with_results))

        generator = ai_generator_factory()

        generator.generate_response(query="test", tools=tool_manager.get_tool_definitions(), tool_manager=tool_manager)

        second_call_kwargs = mock_client.chat.completions.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        tool_messages = [msg for msg in messages if msg.get("role") == "tool"]

        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "call_456"
        assert "content" in tool_messages[0]


class TestAIGeneratorToolCallingBug:
    """Tests specifically for the MAX_RESULTS=0 bug affecting tool calling"""

    @patch('ai_generator.AzureOpenAI')
    def test_ai_tool_call_with_zero_results(self, mock_azure_client_class, empty_search_results, ai_generator_factory):
        """Test what happens when tool returns empty results due to MAX_RESULTS=0"""
        mock_client = Mock()

        tool_call = create_mock_tool_call("call_789", "search_course_content", {"query": "What is Anthropic?"})
        initial_response = create_mock_response(None, [tool_call])
        final_response = create_mock_response("I couldn't retrieve any information about that topic.")

        mock_client.chat.completions.create.side_effect = [initial_response, final_response]
        mock_azure_client_class.return_value = mock_client

        mock_vector_store = Mock()
        mock_vector_store.search.return_value = empty_search_results

        tool_manager = ToolManager()
        tool_manager.register_tool(CourseSearchTool(mock_vector_store))

        generator = ai_generator_factory()

        response = generator.generate_response(
            query="What is Anthropic?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        assert "couldn't retrieve" in response.lower()
        mock_vector_store.search.assert_called_once()

    @patch('ai_generator.AzureOpenAI')
    def test_multiple_tool_calls(self, mock_azure_client_class, mock_vector_store_with_results, ai_generator_factory):
        """Test AI making multiple tool calls in sequence"""
        mock_client = Mock()

        tool_call = create_mock_tool_call("call_multi", "search_course_content", {"query": "test"})
        initial_response = create_mock_response(None, [tool_call])
        final_response = create_mock_response("Based on multiple searches, here's the answer.")

        mock_client.chat.completions.create.side_effect = [initial_response, final_response]
        mock_azure_client_class.return_value = mock_client

        tool_manager = ToolManager()
        tool_manager.register_tool(CourseSearchTool(mock_vector_store_with_results))

        generator = ai_generator_factory()

        response = generator.generate_response(
            query="test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        assert "answer" in response.lower()
