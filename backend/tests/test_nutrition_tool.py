"""
Unit tests for NutritionTool in search_tools.py

These tests verify that the NutritionTool correctly:
1. Handles OAuth authentication via TokenManager
2. Makes API requests to the UCSF nutrition API
3. Parses various response formats
4. Handles errors gracefully
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from search_tools import NutritionTool, ToolManager


class TestNutritionToolDefinition:
    """Test NutritionTool metadata and configuration"""

    def test_get_tool_definition(self, mock_token_manager):
        """Test that tool definition is correctly formatted for OpenAI"""
        tool = NutritionTool(mock_token_manager)
        definition = tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "ask_nutrition_expert"
        assert "nutrition" in definition["function"]["description"].lower()
        assert "parameters" in definition["function"]
        assert definition["function"]["parameters"]["required"] == ["question"]

    def test_tool_definition_has_clear_description(self, mock_token_manager):
        """Test that description clearly indicates this is ONLY for nutrition questions"""
        tool = NutritionTool(mock_token_manager)
        definition = tool.get_tool_definition()

        description = definition["function"]["description"]
        assert "ONLY" in description or "only" in description
        assert any(word in description.lower() for word in ["nutrition", "food", "diet"])

    def test_api_endpoint_is_set(self, mock_token_manager):
        """Test that NutritionTool has the correct API endpoint"""
        tool = NutritionTool(mock_token_manager)

        assert tool.api_endpoint is not None
        assert "ucsf.edu" in tool.api_endpoint
        assert "versaassistant" in tool.api_endpoint


class TestNutritionToolAuthentication:
    """Test OAuth token handling"""

    def test_execute_with_none_token_manager(self):
        """Test that tool returns error when token_manager is None"""
        tool = NutritionTool(None)
        result = tool.execute(question="What are good sources of protein?")

        assert "Error" in result
        assert "Authentication not configured" in result

    def test_execute_gets_token_from_manager(self, mock_token_manager):
        """Test that execute() calls get_token() on the token manager"""
        tool = NutritionTool(mock_token_manager)

        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"answer": "Test response"}

            tool.execute(question="Test question")

            # Verify token was retrieved
            mock_token_manager.get_token.assert_called_once()

    def test_execute_uses_token_in_authorization_header(self, mock_token_manager):
        """Test that the token is correctly used in the Authorization header"""
        tool = NutritionTool(mock_token_manager)

        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"answer": "Test response"}

            tool.execute(question="Test question")

            # Verify Authorization header was set correctly
            call_args = mock_post.call_args
            headers = call_args[1]['headers']
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer mock-oauth-token-12345"

    def test_execute_with_failing_token_manager(self, mock_token_manager_failing):
        """Test handling when token manager fails to get a token"""
        tool = NutritionTool(mock_token_manager_failing)

        result = tool.execute(question="What are good sources of protein?")

        # Should return an error message, not crash
        assert "Error" in result or "error" in result.lower()


class TestNutritionToolAPIRequest:
    """Test API request construction and execution"""

    @patch('requests.post')
    def test_api_request_structure(self, mock_post, mock_token_manager, successful_nutrition_response):
        """Test that API request has the correct structure"""
        mock_post.return_value = successful_nutrition_response

        tool = NutritionTool(mock_token_manager)
        tool.execute(question="What vitamins are in oranges?")

        # Verify request was made
        mock_post.assert_called_once()

        # Check endpoint
        call_args = mock_post.call_args
        assert "versaassistant" in call_args[0][0]

        # Check headers
        headers = call_args[1]['headers']
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" in headers

        # Check body structure
        body = call_args[1]['json']
        assert "userid" in body
        assert "datasource" in body
        assert "model" in body
        assert "messages" in body
        assert len(body["messages"]) == 2  # system + user message

    @patch('requests.post')
    def test_question_passed_to_api(self, mock_post, mock_token_manager, successful_nutrition_response):
        """Test that user's question is passed to the API"""
        mock_post.return_value = successful_nutrition_response

        tool = NutritionTool(mock_token_manager)
        question = "How much protein is in chicken breast?"
        tool.execute(question=question)

        body = mock_post.call_args[1]['json']
        user_message = next(msg for msg in body["messages"] if msg["role"] == "user")
        assert user_message["content"] == question

    @patch('requests.post')
    def test_api_timeout_is_set(self, mock_post, mock_token_manager, successful_nutrition_response):
        """Test that API request has a reasonable timeout"""
        mock_post.return_value = successful_nutrition_response

        tool = NutritionTool(mock_token_manager)
        tool.execute(question="Test question")

        call_args = mock_post.call_args
        assert "timeout" in call_args[1]
        assert call_args[1]["timeout"] > 0


class TestNutritionToolResponseParsing:
    """Test parsing of various API response formats"""

    @patch('requests.post')
    def test_successful_response_with_answer_field(self, mock_post, mock_token_manager, successful_nutrition_response):
        """Test parsing response with 'answer' field"""
        mock_post.return_value = successful_nutrition_response

        tool = NutritionTool(mock_token_manager)
        result = tool.execute(question="What nutrients are in bananas?")

        assert "potassium" in result.lower()
        assert "Error" not in result

    @patch('requests.post')
    def test_response_with_alternative_field_names(self, mock_post, mock_token_manager, nutrition_response_alternative_format):
        """Test parsing response with 'response' field instead of 'answer'"""
        mock_post.return_value = nutrition_response_alternative_format

        tool = NutritionTool(mock_token_manager)
        result = tool.execute(question="What nutrients are in apples?")

        assert "Apples" in result or "fiber" in result.lower()
        assert "Error" not in result

    @patch('requests.post')
    def test_response_with_unknown_format(self, mock_post, mock_token_manager, nutrition_response_unknown_format):
        """Test handling of response with unrecognized format"""
        mock_post.return_value = nutrition_response_unknown_format

        tool = NutritionTool(mock_token_manager)
        result = tool.execute(question="Test question")

        # Should return something (either parsed data or JSON dump)
        assert result is not None
        assert len(result) > 0


class TestNutritionToolErrorHandling:
    """Test error handling for various failure scenarios"""

    @patch('requests.post')
    def test_401_unauthorized_error(self, mock_post, mock_token_manager, nutrition_401_error):
        """Test handling of 401 Unauthorized error (bad OAuth token)"""
        mock_post.return_value = nutrition_401_error

        tool = NutritionTool(mock_token_manager)
        result = tool.execute(question="Test question")

        assert "Error" in result
        assert "401" in result or "Unauthorized" in result or "Failed to connect" in result

    @patch('requests.post')
    def test_timeout_error(self, mock_post, mock_token_manager, nutrition_timeout_error):
        """Test handling of API timeout"""
        mock_post.side_effect = nutrition_timeout_error

        tool = NutritionTool(mock_token_manager)
        result = tool.execute(question="Test question")

        assert "Error" in result
        assert "timed out" in result.lower() or "timeout" in result.lower()

    @patch('requests.post')
    def test_invalid_json_response(self, mock_post, mock_token_manager, nutrition_invalid_json):
        """Test handling of invalid JSON in response"""
        mock_post.return_value = nutrition_invalid_json

        tool = NutritionTool(mock_token_manager)
        result = tool.execute(question="Test question")

        assert "Error" in result
        assert "Invalid response format" in result or "response format" in result.lower()

    @patch('requests.post')
    def test_network_error(self, mock_post, mock_token_manager):
        """Test handling of network connection errors"""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Network error")

        tool = NutritionTool(mock_token_manager)
        result = tool.execute(question="Test question")

        assert "Error" in result
        assert "Failed to connect" in result or "network" in result.lower()

    @patch('requests.post')
    def test_unexpected_exception(self, mock_post, mock_token_manager):
        """Test handling of unexpected exceptions"""
        mock_post.side_effect = Exception("Unexpected error")

        tool = NutritionTool(mock_token_manager)
        result = tool.execute(question="Test question")

        assert "Error" in result
        # Should not crash, should return error message


class TestNutritionToolIntegrationWithToolManager:
    """Test NutritionTool integration with ToolManager"""

    def test_tool_can_be_registered(self, mock_token_manager):
        """Test that NutritionTool can be registered with ToolManager"""
        manager = ToolManager()
        tool = NutritionTool(mock_token_manager)

        manager.register_tool(tool)

        assert "ask_nutrition_expert" in manager.tools

    def test_tool_appears_in_definitions(self, mock_token_manager):
        """Test that NutritionTool appears in tool definitions"""
        manager = ToolManager()
        manager.register_tool(NutritionTool(mock_token_manager))

        definitions = manager.get_tool_definitions()
        tool_names = {d["function"]["name"] for d in definitions}

        assert "ask_nutrition_expert" in tool_names

    @patch('requests.post')
    def test_tool_can_be_executed_via_manager(self, mock_post, mock_token_manager, successful_nutrition_response):
        """Test that NutritionTool can be executed through ToolManager"""
        mock_post.return_value = successful_nutrition_response

        manager = ToolManager()
        manager.register_tool(NutritionTool(mock_token_manager))

        result = manager.execute_tool("ask_nutrition_expert", question="Test nutrition question")

        assert "Error" not in result
        assert len(result) > 0


class TestNutritionToolSourceExtraction:
    """Test source extraction and formatting for UI"""

    def test_extract_sources_from_answer(self, mock_token_manager):
        """Test that sources are extracted from HTML links in answer"""
        tool = NutritionTool(mock_token_manager)

        answer_with_sources = """Spinach is nutritious.

Cited Sources:
<a href="/get_document/example.pdf" target="_blank">Example Document.pdf</a>"""

        cleaned, sources = tool._extract_sources(answer_with_sources)

        # Verify source was extracted
        assert len(sources) == 1
        assert sources[0]["text"] == "Example Document.pdf"
        assert "https://dev-unified-api.ucsf.edu/get_document/example.pdf" in sources[0]["url"]

    def test_cleaned_answer_removes_sources_section(self, mock_token_manager):
        """Test that cleaned answer doesn't contain 'Cited Sources' section"""
        tool = NutritionTool(mock_token_manager)

        answer_with_sources = """Spinach is nutritious.

Cited Sources:
<a href="/example.pdf" target="_blank">Example.pdf</a>"""

        cleaned, sources = tool._extract_sources(answer_with_sources)

        # Verify "Cited Sources" section is removed
        assert "Cited Sources" not in cleaned
        assert "Example.pdf" not in cleaned
        assert "Spinach is nutritious." in cleaned

    def test_multiple_sources_extracted(self, mock_token_manager):
        """Test extracting multiple sources from answer"""
        tool = NutritionTool(mock_token_manager)

        answer = """Info here.

Cited Sources:
<a href="/doc1.pdf" target="_blank">Document 1</a>
<a href="/doc2.pdf" target="_blank">Document 2</a>"""

        cleaned, sources = tool._extract_sources(answer)

        assert len(sources) == 2
        assert sources[0]["text"] == "Document 1"
        assert sources[1]["text"] == "Document 2"

    @patch('requests.post')
    def test_last_sources_tracked(self, mock_post, mock_token_manager):
        """Test that sources are stored in last_sources after execute"""
        # Mock response with sources
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {
            "answer": """Protein sources include chicken.

Cited Sources:
<a href="/protein.pdf" target="_blank">Protein Guide.pdf</a>"""
        }
        mock_post.return_value = mock_response

        tool = NutritionTool(mock_token_manager)
        result = tool.execute(question="What are protein sources?")

        # Verify sources were tracked
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Protein Guide.pdf"
        assert "url" in tool.last_sources[0]

        # Verify result doesn't contain HTML or "Cited Sources"
        assert "Cited Sources" not in result
        assert "<a href" not in result

    @patch('requests.post')
    def test_sources_accessible_via_tool_manager(self, mock_post, mock_token_manager):
        """Test that sources can be retrieved through ToolManager"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {
            "answer": """Info.

Cited Sources:
<a href="/test.pdf" target="_blank">Test.pdf</a>"""
        }
        mock_post.return_value = mock_response

        manager = ToolManager()
        tool = NutritionTool(mock_token_manager)
        manager.register_tool(tool)

        # Execute via manager
        manager.execute_tool("ask_nutrition_expert", question="Test")

        # Retrieve sources via manager
        sources = manager.get_last_sources()

        assert len(sources) == 1
        assert sources[0]["text"] == "Test.pdf"

    def test_sources_cleared_on_error(self, mock_token_manager):
        """Test that sources are cleared when there's an error"""
        tool = NutritionTool(mock_token_manager)

        # Set some dummy sources
        tool.last_sources = [{"text": "old", "url": "old"}]

        # Execute with None token manager (should error)
        tool.token_manager = None
        result = tool.execute(question="Test")

        # Verify sources were cleared
        assert len(tool.last_sources) == 0
        assert "Error" in result
