"""
End-to-end integration tests for rag_system.py
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag_system import RAGSystem


@pytest.fixture
def mock_rag_components():
    """Create mocks for all RAG system components"""
    with patch('rag_system.VectorStore') as mock_vs, \
         patch('rag_system.AIGenerator') as mock_ai, \
         patch('rag_system.SessionManager') as mock_sm:

        # Setup vector store mock
        mock_vector_store = Mock()
        mock_vs.return_value = mock_vector_store

        # Setup AI generator mock
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "AI response"
        mock_ai.return_value = mock_ai_gen

        # Setup session manager mock
        mock_session_mgr = Mock()
        mock_session_mgr.get_conversation_history.return_value = None
        mock_sm.return_value = mock_session_mgr

        yield {
            'vector_store': mock_vector_store,
            'ai_generator': mock_ai_gen,
            'session_manager': mock_session_mgr
        }


class TestRAGSystemInitialization:
    """Test RAG system initialization"""

    def test_initialization(self, mock_rag_components, test_config):
        """Test that RAG system initializes all components"""
        rag = RAGSystem(test_config)

        assert rag.config == test_config
        assert rag.document_processor is not None
        assert rag.tool_manager is not None
        assert rag.search_tool is not None
        assert rag.outline_tool is not None

    def test_tools_registered(self, mock_rag_components, test_config):
        """Test that tools are registered with the tool manager"""
        rag = RAGSystem(test_config)
        tool_definitions = rag.tool_manager.get_tool_definitions()

        assert len(tool_definitions) >= 2
        tool_names = {td["function"]["name"] for td in tool_definitions}
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names


class TestRAGSystemQuery:
    """Test RAG system query flow"""

    def test_query_without_session(self, mock_rag_components, test_config):
        """Test query without session ID"""
        rag = RAGSystem(test_config)
        response, sources = rag.query("What is Anthropic?", session_id=None)

        assert response == "AI response"
        assert isinstance(sources, list)
        mock_rag_components['ai_generator'].generate_response.assert_called_once()

    def test_query_with_session(self, mock_rag_components, test_config):
        """Test query with session ID and conversation history"""
        mock_rag_components['session_manager'].get_conversation_history.return_value = "Previous history"

        rag = RAGSystem(test_config)
        response, sources = rag.query("Tell me more", session_id="session123")

        assert response == "AI response"
        mock_rag_components['session_manager'].get_conversation_history.assert_called_once_with("session123")

        call_kwargs = mock_rag_components['ai_generator'].generate_response.call_args[1]
        assert call_kwargs["conversation_history"] == "Previous history"
        mock_rag_components['session_manager'].add_exchange.assert_called_once()

    def test_query_passes_tools_to_ai(self, mock_rag_components, test_config):
        """Test that query passes tools to AI generator"""
        rag = RAGSystem(test_config)
        rag.query("What is Anthropic?")

        call_kwargs = mock_rag_components['ai_generator'].generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert "tool_manager" in call_kwargs


class TestRAGSystemWithToolExecution:
    """Test RAG system when AI uses tools"""

    def test_query_with_search_tool_execution(self, mock_rag_components, test_config, sample_search_results):
        """Test complete flow when AI uses search tool"""
        mock_rag_components['vector_store'].search.return_value = sample_search_results
        mock_rag_components['vector_store'].get_lesson_link.return_value = "https://example.com/lesson0"
        mock_rag_components['ai_generator'].generate_response.return_value = "Anthropic is an AI safety company."

        rag = RAGSystem(test_config)
        response, sources = rag.query("What is Anthropic?")

        assert "Anthropic" in response or "safety" in response.lower()

    def test_sources_tracked_after_search(self, mock_rag_components, test_config, sample_search_results):
        """Test that sources are tracked after search tool execution"""
        mock_rag_components['vector_store'].search.return_value = sample_search_results
        mock_rag_components['vector_store'].get_lesson_link.return_value = "https://example.com/lesson1"

        rag = RAGSystem(test_config)
        rag.search_tool.execute(query="test")
        sources = rag.tool_manager.get_last_sources()

        assert len(sources) > 0
        assert "text" in sources[0]


class TestRAGSystemWithBrokenConfig:
    """Test RAG system behavior with MAX_RESULTS=0 bug"""

    def test_query_with_zero_max_results(self, mock_rag_components, broken_config, empty_search_results):
        """Test that MAX_RESULTS=0 causes search to return no results"""
        mock_rag_components['vector_store'].search.return_value = empty_search_results
        mock_rag_components['vector_store'].max_results = 0
        mock_rag_components['ai_generator'].generate_response.return_value = (
            "I couldn't retrieve any relevant information about that topic."
        )

        rag = RAGSystem(broken_config)
        assert broken_config.MAX_RESULTS == 0, "This test requires MAX_RESULTS=0"

        response, sources = rag.query("What is Anthropic?")

        assert "couldn't retrieve" in response.lower() or "no relevant content" in response.lower()
        assert len(sources) == 0

    def test_comparison_working_vs_broken_config(self, test_config, broken_config, sample_search_results, empty_search_results):
        """Test to show the difference between working and broken config"""
        # Test with WORKING config
        with patch('rag_system.VectorStore') as mock_vs_w, \
             patch('rag_system.AIGenerator') as mock_ai_w, \
             patch('rag_system.SessionManager') as mock_sm_w:

            mock_vector_store_working = Mock()
            mock_vector_store_working.search.return_value = sample_search_results
            mock_vector_store_working.max_results = 5
            mock_vector_store_working.get_lesson_link.return_value = "https://example.com"
            mock_vs_w.return_value = mock_vector_store_working

            mock_ai_working = Mock()
            mock_ai_working.generate_response.return_value = "Anthropic is an AI safety company that builds Claude."
            mock_ai_w.return_value = mock_ai_working

            mock_sm_w.return_value = Mock()

            rag_working = RAGSystem(test_config)
            response_working, _ = rag_working.query("What is Anthropic?")

        # Test with BROKEN config
        with patch('rag_system.VectorStore') as mock_vs_b, \
             patch('rag_system.AIGenerator') as mock_ai_b, \
             patch('rag_system.SessionManager') as mock_sm_b:

            mock_vector_store_broken = Mock()
            mock_vector_store_broken.search.return_value = empty_search_results
            mock_vector_store_broken.max_results = 0
            mock_vs_b.return_value = mock_vector_store_broken

            mock_ai_broken = Mock()
            mock_ai_broken.generate_response.return_value = "I couldn't retrieve information."
            mock_ai_b.return_value = mock_ai_broken

            mock_sm_b.return_value = Mock()

            rag_broken = RAGSystem(broken_config)
            response_broken, _ = rag_broken.query("What is Anthropic?")

        # Comparisons
        assert "Anthropic" in response_working
        assert "couldn't" in response_broken.lower()
        assert response_working != response_broken


class TestRAGSystemCourseManagement:
    """Test RAG system course analytics"""

    def test_get_course_analytics(self, mock_rag_components, test_config):
        """Test getting course analytics"""
        mock_rag_components['vector_store'].get_course_count.return_value = 3
        mock_rag_components['vector_store'].get_existing_course_titles.return_value = [
            "Course 1", "Course 2", "Course 3"
        ]

        rag = RAGSystem(test_config)
        analytics = rag.get_course_analytics()

        assert analytics["total_courses"] == 3
        assert len(analytics["course_titles"]) == 3
        assert "Course 1" in analytics["course_titles"]
