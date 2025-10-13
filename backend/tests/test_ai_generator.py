"""
Integration tests for ai_generator.py - AIGenerator and tool calling
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class TestAIGeneratorBasics:
    """Basic tests for AIGenerator"""

    def test_initialization(self):
        """Test AIGenerator can be initialized"""
        generator = AIGenerator(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
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
        assert "course materials" in AIGenerator.SYSTEM_PROMPT.lower()
        assert "tools" in AIGenerator.SYSTEM_PROMPT.lower()


class TestAIGeneratorWithoutTools:
    """Test AIGenerator when tools are not used"""

    @patch('ai_generator.AzureOpenAI')
    def test_generate_response_without_tools(self, mock_azure_client_class):
        """Test generating a simple response without tools"""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a test response."
        mock_response.choices[0].message.tool_calls = None

        mock_client.chat.completions.create.return_value = mock_response
        mock_azure_client_class.return_value = mock_client

        # Create generator
        generator = AIGenerator(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            api_version="2024-02-01",
            deployment="gpt-4"
        )

        # Generate response
        response = generator.generate_response(
            query="What is 2+2?",
            conversation_history=None,
            tools=None,
            tool_manager=None
        )

        assert response == "This is a test response."

        # Verify API was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4"
        assert "messages" in call_kwargs

    @patch('ai_generator.AzureOpenAI')
    def test_generate_response_with_conversation_history(self, mock_azure_client_class):
        """Test generating response with conversation history"""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Continuing the conversation."
        mock_response.choices[0].message.tool_calls = None

        mock_client.chat.completions.create.return_value = mock_response
        mock_azure_client_class.return_value = mock_client

        # Create generator
        generator = AIGenerator(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            api_version="2024-02-01",
            deployment="gpt-4"
        )

        # Generate response with history
        history = "User: Hello\nAssistant: Hi there!"
        response = generator.generate_response(
            query="How are you?",
            conversation_history=history,
            tools=None,
            tool_manager=None
        )

        assert response == "Continuing the conversation."

        # Verify history was included in messages
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        # Should have system prompt, history, and user query
        assert len(messages) >= 3


class TestAIGeneratorWithTools:
    """Test AIGenerator with tool calling"""

    @patch('ai_generator.AzureOpenAI')
    def test_tools_parameter_passed_to_api(self, mock_azure_client_class, mock_vector_store):
        """Test that tools are correctly passed to the API"""
        # Setup mock - no tool call, direct response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Here's the answer."
        mock_response.choices[0].message.tool_calls = None

        mock_client.chat.completions.create.return_value = mock_response
        mock_azure_client_class.return_value = mock_client

        # Setup tool manager
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Create generator
        generator = AIGenerator(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            api_version="2024-02-01",
            deployment="gpt-4"
        )

        # Generate response with tools
        response = generator.generate_response(
            query="What is Anthropic?",
            conversation_history=None,
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Verify tools were passed to API
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"
        assert len(call_kwargs["tools"]) > 0

    @patch('ai_generator.AzureOpenAI')
    def test_ai_calls_search_tool(self, mock_azure_client_class, mock_vector_store_with_results):
        """Test that AI correctly calls the search tool"""
        # Setup mock client
        mock_client = Mock()

        # First response: AI wants to use tool
        initial_response = Mock()
        initial_response.choices = [Mock()]
        initial_response.choices[0].message.content = None

        # Create mock tool call
        tool_call = Mock()
        tool_call.id = "call_123"
        tool_call.function.name = "search_course_content"
        tool_call.function.arguments = json.dumps({
            "query": "What is Anthropic?",
            "course_name": "Computer Use"
        })
        initial_response.choices[0].message.tool_calls = [tool_call]

        # Second response: AI provides final answer after tool execution
        final_response = Mock()
        final_response.choices = [Mock()]
        final_response.choices[0].message.content = "Based on the course material, Anthropic is an AI safety company."
        final_response.choices[0].message.tool_calls = None

        # Return initial response first, then final response
        mock_client.chat.completions.create.side_effect = [initial_response, final_response]
        mock_azure_client_class.return_value = mock_client

        # Setup tool manager
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store_with_results)
        tool_manager.register_tool(search_tool)

        # Create generator
        generator = AIGenerator(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            api_version="2024-02-01",
            deployment="gpt-4"
        )

        # Generate response
        response = generator.generate_response(
            query="What is Anthropic?",
            conversation_history=None,
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Should get final response after tool execution
        assert "Anthropic is an AI safety company" in response

        # Verify tool was executed
        mock_vector_store_with_results.search.assert_called_once_with(
            query="What is Anthropic?",
            course_name="Computer Use",
            lesson_number=None
        )

        # Verify API was called twice (initial + final)
        assert mock_client.chat.completions.create.call_count == 2

    @patch('ai_generator.AzureOpenAI')
    def test_tool_result_formatting(self, mock_azure_client_class, mock_vector_store_with_results):
        """Test that tool results are correctly formatted in messages"""
        # Setup mock client
        mock_client = Mock()

        # First response with tool call
        initial_response = Mock()
        initial_response.choices = [Mock()]
        initial_response.choices[0].message.content = "Let me search for that."

        tool_call = Mock()
        tool_call.id = "call_456"
        tool_call.function.name = "search_course_content"
        tool_call.function.arguments = json.dumps({"query": "test query"})
        initial_response.choices[0].message.tool_calls = [tool_call]

        # Final response
        final_response = Mock()
        final_response.choices = [Mock()]
        final_response.choices[0].message.content = "Here's what I found."
        final_response.choices[0].message.tool_calls = None

        mock_client.chat.completions.create.side_effect = [initial_response, final_response]
        mock_azure_client_class.return_value = mock_client

        # Setup tools
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store_with_results)
        tool_manager.register_tool(search_tool)

        # Create generator
        generator = AIGenerator(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            api_version="2024-02-01",
            deployment="gpt-4"
        )

        # Generate response
        generator.generate_response(
            query="test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Check the second API call (after tool execution)
        second_call_kwargs = mock_client.chat.completions.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        # Should include tool result message
        tool_messages = [msg for msg in messages if msg.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "call_456"
        assert "content" in tool_messages[0]


class TestAIGeneratorToolCallingBug:
    """Tests specifically for the MAX_RESULTS=0 bug affecting tool calling"""

    @patch('ai_generator.AzureOpenAI')
    def test_ai_tool_call_with_zero_results(self, mock_azure_client_class):
        """Test what happens when tool returns empty results due to MAX_RESULTS=0"""
        # Setup mock client
        mock_client = Mock()

        # First response: AI wants to search
        initial_response = Mock()
        initial_response.choices = [Mock()]
        initial_response.choices[0].message.content = None

        tool_call = Mock()
        tool_call.id = "call_789"
        tool_call.function.name = "search_course_content"
        tool_call.function.arguments = json.dumps({"query": "What is Anthropic?"})
        initial_response.choices[0].message.tool_calls = [tool_call]

        # Second response: AI responds to empty tool results
        final_response = Mock()
        final_response.choices = [Mock()]
        final_response.choices[0].message.content = "I couldn't retrieve any information about that topic."
        final_response.choices[0].message.tool_calls = None

        mock_client.chat.completions.create.side_effect = [initial_response, final_response]
        mock_azure_client_class.return_value = mock_client

        # Setup tool with mock that returns empty results (simulating MAX_RESULTS=0)
        mock_vector_store = Mock()
        from vector_store import SearchResults
        mock_vector_store.search.return_value = SearchResults(
            documents=[],  # Empty due to MAX_RESULTS=0
            metadata=[],
            distances=[],
            error=None
        )

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Create generator
        generator = AIGenerator(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            api_version="2024-02-01",
            deployment="gpt-4"
        )

        # Generate response
        response = generator.generate_response(
            query="What is Anthropic?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # This is the bug! Even though there might be content in the DB,
        # MAX_RESULTS=0 causes empty results, leading to this error message
        assert "couldn't retrieve" in response.lower()

        # Verify tool was called but returned nothing
        mock_vector_store.search.assert_called_once()

    @patch('ai_generator.AzureOpenAI')
    def test_multiple_tool_calls(self, mock_azure_client_class, mock_vector_store_with_results):
        """Test AI making multiple tool calls in sequence"""
        mock_client = Mock()

        # Response with tool call
        initial_response = Mock()
        initial_response.choices = [Mock()]
        initial_response.choices[0].message.content = None

        tool_call = Mock()
        tool_call.id = "call_multi"
        tool_call.function.name = "search_course_content"
        tool_call.function.arguments = json.dumps({"query": "test"})
        initial_response.choices[0].message.tool_calls = [tool_call]

        # Final response
        final_response = Mock()
        final_response.choices = [Mock()]
        final_response.choices[0].message.content = "Based on multiple searches, here's the answer."
        final_response.choices[0].message.tool_calls = None

        mock_client.chat.completions.create.side_effect = [initial_response, final_response]
        mock_azure_client_class.return_value = mock_client

        # Setup tools
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store_with_results)
        tool_manager.register_tool(search_tool)

        # Create generator
        generator = AIGenerator(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            api_version="2024-02-01",
            deployment="gpt-4"
        )

        response = generator.generate_response(
            query="test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        assert "answer" in response.lower()
