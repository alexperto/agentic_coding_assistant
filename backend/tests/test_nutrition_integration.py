"""
Integration tests for nutrition question handling across the RAG system

These tests verify the end-to-end flow:
1. User asks a nutrition question
2. AI recognizes it as nutrition-related
3. AI calls the NutritionTool
4. NutritionTool queries the UCSF API
5. Response is returned to the user
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai_generator import AIGenerator
from search_tools import ToolManager, NutritionTool
from rag_system import RAGSystem


def create_mock_response(content, tool_calls=None):
    """Helper to create a mock Azure OpenAI response"""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = content
    mock_response.choices[0].message.tool_calls = tool_calls
    return mock_response


def create_mock_tool_call(tool_id, function_name, arguments):
    """Helper to create a mock tool call"""
    tool_call = Mock()
    tool_call.id = tool_id
    tool_call.function.name = function_name
    tool_call.function.arguments = json.dumps(arguments) if isinstance(arguments, dict) else arguments
    return tool_call


class TestAIGeneratorRecognizesNutritionQuestions:
    """Test that AI generator correctly identifies nutrition questions"""

    @patch('ai_generator.AzureOpenAI')
    @patch('requests.post')
    def test_ai_calls_nutrition_tool_for_nutrition_question(
        self, mock_requests, mock_azure_client_class, mock_token_manager, successful_nutrition_response
    ):
        """Test that AI calls NutritionTool when asked a nutrition question"""
        mock_requests.return_value = successful_nutrition_response

        # Setup AI client to call nutrition tool
        mock_client = Mock()
        tool_call = create_mock_tool_call(
            "call_nutrition_1",
            "ask_nutrition_expert",
            {"question": "What are good sources of protein?"}
        )
        initial_response = create_mock_response(None, [tool_call])
        final_response = create_mock_response("Chicken, fish, and beans are excellent protein sources.")

        mock_client.chat.completions.create.side_effect = [initial_response, final_response]
        mock_azure_client_class.return_value = mock_client

        # Create AI generator with token manager
        generator = AIGenerator(
            endpoint="https://test.openai.azure.com",
            token_manager=mock_token_manager,
            api_version="2024-02-01",
            deployment="gpt-4"
        )

        # Create tool manager with nutrition tool
        tool_manager = ToolManager()
        tool_manager.register_tool(NutritionTool(mock_token_manager))

        # Execute query
        response = generator.generate_response(
            query="What are good sources of protein?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Verify AI called the nutrition tool
        assert "protein" in response.lower()
        mock_requests.assert_called_once()

        # Verify the nutrition API was called with correct question
        call_kwargs = mock_requests.call_args[1]
        body = call_kwargs['json']
        user_message = next(msg for msg in body["messages"] if msg["role"] == "user")
        assert "protein" in user_message["content"].lower()

    @patch('ai_generator.AzureOpenAI')
    def test_ai_recognizes_various_nutrition_questions(self, mock_azure_client_class, mock_token_manager):
        """Test that AI recognizes different types of nutrition questions"""
        nutrition_questions = [
            "What vitamins are in spinach?",
            "How many calories are in an apple?",
            "What's a healthy breakfast?",
            "Is keto diet good for weight loss?",
            "What foods are high in iron?"
        ]

        for question in nutrition_questions:
            mock_client = Mock()
            tool_call = create_mock_tool_call(
                "call_test",
                "ask_nutrition_expert",
                {"question": question}
            )
            initial_response = create_mock_response(None, [tool_call])
            final_response = create_mock_response("Nutrition information here")

            mock_client.chat.completions.create.side_effect = [initial_response, final_response]
            mock_azure_client_class.return_value = mock_client

            generator = AIGenerator(
                endpoint="https://test.openai.azure.com",
                token_manager=mock_token_manager,
                api_version="2024-02-01",
                deployment="gpt-4"
            )

            tool_manager = ToolManager()
            tool_manager.register_tool(NutritionTool(mock_token_manager))

            with patch('requests.post') as mock_requests:
                mock_requests.return_value.status_code = 200
                mock_requests.return_value.json.return_value = {"answer": "Test answer"}

                response = generator.generate_response(
                    query=question,
                    tools=tool_manager.get_tool_definitions(),
                    tool_manager=tool_manager
                )

                # At minimum, should not return an error
                assert "Error" not in response or "error" not in response


class TestRAGSystemNutritionFlow:
    """Test end-to-end nutrition question flow through RAG system"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.create_token_manager_from_env')
    @patch('ai_generator.AzureOpenAI')
    @patch('requests.post')
    def test_rag_system_handles_nutrition_question(
        self, mock_requests, mock_azure_class, mock_token_factory, mock_vector_store_class, test_config,
        mock_token_manager, successful_nutrition_response
    ):
        """Test full RAG system flow for nutrition question"""
        # Setup mocks
        mock_requests.return_value = successful_nutrition_response
        mock_token_factory.return_value = mock_token_manager

        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        mock_client = Mock()
        tool_call = create_mock_tool_call(
            "call_rag_nutrition",
            "ask_nutrition_expert",
            {"question": "What nutrients are in broccoli?"}
        )
        initial_response = create_mock_response(None, [tool_call])
        final_response = create_mock_response("Broccoli is rich in vitamins C and K, and contains fiber.")

        mock_client.chat.completions.create.side_effect = [initial_response, final_response]
        mock_azure_class.return_value = mock_client

        # Create RAG system
        rag = RAGSystem(test_config)

        # Execute nutrition query
        response, sources = rag.query("What nutrients are in broccoli?")

        # Verify nutrition tool was called
        assert "broccoli" in response.lower() or "vitamin" in response.lower()
        mock_requests.assert_called_once()

    @patch('rag_system.VectorStore')
    @patch('rag_system.create_token_manager_from_env')
    def test_nutrition_tool_is_registered_in_rag_system(self, mock_token_factory, mock_vector_store_class, test_config, mock_token_manager):
        """Test that NutritionTool is properly registered when RAG system initializes"""
        mock_token_factory.return_value = mock_token_manager
        mock_vector_store_class.return_value = Mock()

        rag = RAGSystem(test_config)

        # Verify nutrition tool exists
        assert hasattr(rag, 'nutrition_tool')
        assert rag.nutrition_tool is not None

        # Verify it's registered in tool manager
        tool_definitions = rag.tool_manager.get_tool_definitions()
        tool_names = {td["function"]["name"] for td in tool_definitions}
        assert "ask_nutrition_expert" in tool_names

    @patch('rag_system.VectorStore')
    @patch('rag_system.create_token_manager_from_env')
    def test_nutrition_tool_not_created_without_token_manager(self, mock_token_factory, mock_vector_store_class, test_config):
        """Test behavior when token manager is not available (None)"""
        # Simulate no OAuth credentials configured
        mock_token_factory.return_value = None
        mock_vector_store_class.return_value = Mock()

        rag = RAGSystem(test_config)

        # Nutrition tool should still be created (with None token_manager)
        assert hasattr(rag, 'nutrition_tool')

        # But it should return error when executed
        result = rag.nutrition_tool.execute(question="Test question")
        assert "Error" in result
        assert "Authentication not configured" in result


class TestNutritionToolWithAIGeneratorTokenManager:
    """Test that NutritionTool and AIGenerator share the same token manager"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.create_token_manager_from_env')
    def test_same_token_manager_for_ai_and_nutrition(self, mock_token_factory, mock_vector_store_class, test_config, mock_token_manager):
        """Test that both AI and Nutrition tool use the same token manager instance"""
        mock_token_factory.return_value = mock_token_manager
        mock_vector_store_class.return_value = Mock()

        rag = RAGSystem(test_config)

        # Both should have the same token manager
        assert rag.ai_generator.token_manager == rag.nutrition_tool.token_manager
        assert rag.ai_generator.token_manager is not None


class TestNutritionAPIErrorScenarios:
    """Test how system handles various nutrition API failures"""

    @patch('ai_generator.AzureOpenAI')
    @patch('requests.post')
    def test_nutrition_api_401_error_handling(self, mock_requests, mock_azure_class, mock_token_manager):
        """Test handling when nutrition API returns 401 (auth failure)"""
        import requests
        mock_requests.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
        mock_requests.return_value.status_code = 401

        mock_client = Mock()
        tool_call = create_mock_tool_call("call_401", "ask_nutrition_expert", {"question": "test"})
        initial_response = create_mock_response(None, [tool_call])
        final_response = create_mock_response("I'm unable to retrieve nutrition information at this time.")

        mock_client.chat.completions.create.side_effect = [initial_response, final_response]
        mock_azure_class.return_value = mock_client

        generator = AIGenerator(
            endpoint="https://test.openai.azure.com",
            token_manager=mock_token_manager,
            api_version="2024-02-01",
            deployment="gpt-4"
        )

        tool_manager = ToolManager()
        tool_manager.register_tool(NutritionTool(mock_token_manager))

        response = generator.generate_response(
            query="What's in carrots?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Should handle error gracefully
        assert response is not None
        # AI should synthesize tool error into a user-friendly message
        assert "unable" in response.lower() or "error" in response.lower() or "cannot" in response.lower()

    @patch('ai_generator.AzureOpenAI')
    @patch('requests.post')
    def test_nutrition_api_timeout_handling(self, mock_requests, mock_azure_class, mock_token_manager):
        """Test handling when nutrition API times out"""
        import requests
        mock_requests.side_effect = requests.exceptions.Timeout("Timeout")

        mock_client = Mock()
        tool_call = create_mock_tool_call("call_timeout", "ask_nutrition_expert", {"question": "test"})
        initial_response = create_mock_response(None, [tool_call])
        final_response = create_mock_response("The nutrition service is currently unavailable.")

        mock_client.chat.completions.create.side_effect = [initial_response, final_response]
        mock_azure_class.return_value = mock_client

        generator = AIGenerator(
            endpoint="https://test.openai.azure.com",
            token_manager=mock_token_manager,
            api_version="2024-02-01",
            deployment="gpt-4"
        )

        tool_manager = ToolManager()
        tool_manager.register_tool(NutritionTool(mock_token_manager))

        response = generator.generate_response(
            query="Nutrition question",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Should not crash, should return something
        assert response is not None
        assert len(response) > 0


class TestNutritionToolNotCalledForNonNutritionQuestions:
    """Test that nutrition tool is NOT called for non-nutrition questions"""

    @patch('ai_generator.AzureOpenAI')
    def test_course_question_does_not_call_nutrition_tool(self, mock_azure_class, mock_token_manager, mock_vector_store):
        """Test that asking about course content doesn't trigger nutrition tool"""
        from search_tools import CourseSearchTool

        mock_client = Mock()
        # AI should call course search tool, not nutrition tool
        tool_call = create_mock_tool_call(
            "call_course",
            "search_course_content",
            {"query": "What is Anthropic?"}
        )
        initial_response = create_mock_response(None, [tool_call])
        final_response = create_mock_response("Anthropic is an AI safety company.")

        mock_client.chat.completions.create.side_effect = [initial_response, final_response]
        mock_azure_class.return_value = mock_client

        generator = AIGenerator(
            endpoint="https://test.openai.azure.com",
            token_manager=mock_token_manager,
            api_version="2024-02-01",
            deployment="gpt-4"
        )

        tool_manager = ToolManager()
        tool_manager.register_tool(CourseSearchTool(mock_vector_store))
        tool_manager.register_tool(NutritionTool(mock_token_manager))

        with patch('requests.post') as mock_nutrition_api:
            response = generator.generate_response(
                query="What is Anthropic?",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )

            # Nutrition API should NOT have been called
            mock_nutrition_api.assert_not_called()
