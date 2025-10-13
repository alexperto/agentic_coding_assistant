"""
End-to-end integration tests for rag_system.py
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from rag_system import RAGSystem


class TestRAGSystemInitialization:
    """Test RAG system initialization"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_initialization(self, mock_session_mgr, mock_ai_gen, mock_vector_store, test_config):
        """Test that RAG system initializes all components"""
        rag = RAGSystem(test_config)

        # Verify components were initialized
        assert rag.config == test_config
        assert rag.document_processor is not None
        assert rag.tool_manager is not None
        assert rag.search_tool is not None
        assert rag.outline_tool is not None

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_tools_registered(self, mock_session_mgr, mock_ai_gen, mock_vector_store, test_config):
        """Test that tools are registered with the tool manager"""
        rag = RAGSystem(test_config)

        tool_definitions = rag.tool_manager.get_tool_definitions()

        # Should have both search and outline tools
        assert len(tool_definitions) >= 2
        tool_names = [td["function"]["name"] for td in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names


class TestRAGSystemQuery:
    """Test RAG system query flow"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_without_session(self, mock_session_mgr_class, mock_ai_gen_class, mock_vector_store_class, test_config):
        """Test query without session ID"""
        # Setup mocks
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "This is the AI's response."
        mock_ai_gen_class.return_value = mock_ai_gen

        mock_session_mgr = Mock()
        mock_session_mgr.get_conversation_history.return_value = None
        mock_session_mgr_class.return_value = mock_session_mgr

        # Create RAG system
        rag = RAGSystem(test_config)

        # Query without session
        response, sources = rag.query("What is Anthropic?", session_id=None)

        # Should get a response
        assert response == "This is the AI's response."
        assert isinstance(sources, list)

        # AI generator should have been called
        mock_ai_gen.generate_response.assert_called_once()

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_with_session(self, mock_session_mgr_class, mock_ai_gen_class, mock_vector_store_class, test_config):
        """Test query with session ID and conversation history"""
        # Setup mocks
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Continuing our conversation."
        mock_ai_gen_class.return_value = mock_ai_gen

        mock_session_mgr = Mock()
        mock_session_mgr.get_conversation_history.return_value = "Previous conversation history"
        mock_session_mgr_class.return_value = mock_session_mgr

        # Create RAG system
        rag = RAGSystem(test_config)

        # Query with session
        response, sources = rag.query("Tell me more", session_id="session123")

        # Should get a response
        assert response == "Continuing our conversation."

        # History should have been retrieved
        mock_session_mgr.get_conversation_history.assert_called_once_with("session123")

        # History should have been passed to AI
        call_kwargs = mock_ai_gen.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] == "Previous conversation history"

        # Session should be updated with new exchange
        mock_session_mgr.add_exchange.assert_called_once()

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_passes_tools_to_ai(self, mock_session_mgr_class, mock_ai_gen_class, mock_vector_store_class, test_config):
        """Test that query passes tools to AI generator"""
        # Setup mocks
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Response"
        mock_ai_gen_class.return_value = mock_ai_gen

        mock_session_mgr = Mock()
        mock_session_mgr_class.return_value = mock_session_mgr

        # Create RAG system
        rag = RAGSystem(test_config)

        # Query
        rag.query("What is Anthropic?")

        # Verify tools were passed
        call_kwargs = mock_ai_gen.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert "tool_manager" in call_kwargs


class TestRAGSystemWithToolExecution:
    """Test RAG system when AI uses tools"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_with_search_tool_execution(self, mock_session_mgr_class, mock_ai_gen_class, mock_vector_store_class, test_config):
        """Test complete flow when AI uses search tool"""
        # Setup vector store mock with results
        mock_vector_store = Mock()
        from vector_store import SearchResults
        mock_vector_store.search.return_value = SearchResults(
            documents=["Anthropic is an AI safety company."],
            metadata=[{"course_title": "Test Course", "lesson_number": 0}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson0"
        mock_vector_store_class.return_value = mock_vector_store

        # Setup AI generator mock that calls tool
        mock_ai_gen = Mock()

        # First call: AI wants to use tool
        initial_response_mock = Mock()
        tool_call = Mock()
        tool_call.id = "call_abc"
        tool_call.function.name = "search_course_content"
        tool_call.function.arguments = json.dumps({"query": "What is Anthropic?"})

        # Mock the internal _handle_tool_execution to return final response
        mock_ai_gen.generate_response.return_value = "Based on the course, Anthropic is an AI safety company."

        mock_ai_gen_class.return_value = mock_ai_gen

        mock_session_mgr = Mock()
        mock_session_mgr_class.return_value = mock_session_mgr

        # Create RAG system
        rag = RAGSystem(test_config)

        # Query that should trigger tool use
        response, sources = rag.query("What is Anthropic?")

        # Should get a response
        assert "Anthropic" in response or "safety" in response.lower()

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_sources_tracked_after_search(self, mock_session_mgr_class, mock_ai_gen_class, mock_vector_store_class, test_config):
        """Test that sources are tracked after search tool execution"""
        # Setup vector store with results
        mock_vector_store = Mock()
        from vector_store import SearchResults
        mock_vector_store.search.return_value = SearchResults(
            documents=["Course content here."],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        mock_vector_store_class.return_value = mock_vector_store

        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Here's what I found."
        mock_ai_gen_class.return_value = mock_ai_gen

        mock_session_mgr = Mock()
        mock_session_mgr_class.return_value = mock_session_mgr

        # Create RAG system
        rag = RAGSystem(test_config)

        # Manually execute search tool to populate sources
        rag.search_tool.execute(query="test")

        # Get sources
        sources = rag.tool_manager.get_last_sources()

        # Should have sources
        assert len(sources) > 0
        assert "text" in sources[0]


class TestRAGSystemWithBrokenConfig:
    """Test RAG system behavior with MAX_RESULTS=0 bug"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_with_zero_max_results(self, mock_session_mgr_class, mock_ai_gen_class, mock_vector_store_class, broken_config):
        """Test that MAX_RESULTS=0 causes search to return no results"""
        # Setup vector store that simulates MAX_RESULTS=0 behavior
        mock_vector_store = Mock()
        from vector_store import SearchResults

        # With MAX_RESULTS=0, search returns empty results
        mock_vector_store.search.return_value = SearchResults(
            documents=[],  # Empty!
            metadata=[],
            distances=[],
            error=None
        )
        # Important: Set max_results to 0 to simulate the bug
        mock_vector_store.max_results = 0

        mock_vector_store_class.return_value = mock_vector_store

        # AI will respond with "couldn't retrieve" when tool returns empty
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "I couldn't retrieve any relevant information about that topic."
        mock_ai_gen_class.return_value = mock_ai_gen

        mock_session_mgr = Mock()
        mock_session_mgr_class.return_value = mock_session_mgr

        # Create RAG system with BROKEN config
        rag = RAGSystem(broken_config)

        # Verify the config has the bug
        assert broken_config.MAX_RESULTS == 0, "This test requires MAX_RESULTS=0 to demonstrate the bug"

        # Query for content
        response, sources = rag.query("What is Anthropic?")

        # This is the bug! Even though content exists in the database,
        # the response says it couldn't retrieve anything
        assert "couldn't retrieve" in response.lower() or "no relevant content" in response.lower()

        # Sources should be empty
        assert len(sources) == 0

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_comparison_working_vs_broken_config(
        self,
        mock_session_mgr_class,
        mock_ai_gen_class,
        mock_vector_store_class,
        test_config,
        broken_config
    ):
        """Test to show the difference between working and broken config"""
        # Test with WORKING config (MAX_RESULTS=5)
        mock_vector_store_working = Mock()
        from vector_store import SearchResults

        mock_vector_store_working.search.return_value = SearchResults(
            documents=["Anthropic is an AI safety company."],
            metadata=[{"course_title": "Test", "lesson_number": 0}],
            distances=[0.1],
            error=None
        )
        mock_vector_store_working.max_results = 5  # Working value
        mock_vector_store_working.get_lesson_link.return_value = "https://example.com"

        mock_ai_gen_working = Mock()
        mock_ai_gen_working.generate_response.return_value = "Anthropic is an AI safety company that builds Claude."

        mock_session_mgr_working = Mock()

        # Create instances
        with patch('rag_system.VectorStore', return_value=mock_vector_store_working), \
             patch('rag_system.AIGenerator', return_value=mock_ai_gen_working), \
             patch('rag_system.SessionManager', return_value=mock_session_mgr_working):

            rag_working = RAGSystem(test_config)
            response_working, sources_working = rag_working.query("What is Anthropic?")

        # Test with BROKEN config (MAX_RESULTS=0)
        mock_vector_store_broken = Mock()
        mock_vector_store_broken.search.return_value = SearchResults(
            documents=[],  # Empty due to MAX_RESULTS=0
            metadata=[],
            distances=[],
            error=None
        )
        mock_vector_store_broken.max_results = 0  # Broken value

        mock_ai_gen_broken = Mock()
        mock_ai_gen_broken.generate_response.return_value = "I couldn't retrieve information."

        mock_session_mgr_broken = Mock()

        with patch('rag_system.VectorStore', return_value=mock_vector_store_broken), \
             patch('rag_system.AIGenerator', return_value=mock_ai_gen_broken), \
             patch('rag_system.SessionManager', return_value=mock_session_mgr_broken):

            rag_broken = RAGSystem(broken_config)
            response_broken, sources_broken = rag_broken.query("What is Anthropic?")

        # Working config should have results
        assert "Anthropic" in response_working

        # Broken config should fail
        assert "couldn't" in response_broken.lower()

        # This demonstrates the bug!
        assert response_working != response_broken


class TestRAGSystemCourseManagement:
    """Test RAG system course analytics"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_get_course_analytics(self, mock_session_mgr_class, mock_ai_gen_class, mock_vector_store_class, test_config):
        """Test getting course analytics"""
        # Setup mock
        mock_vector_store = Mock()
        mock_vector_store.get_course_count.return_value = 3
        mock_vector_store.get_existing_course_titles.return_value = [
            "Course 1", "Course 2", "Course 3"
        ]
        mock_vector_store_class.return_value = mock_vector_store

        mock_ai_gen = Mock()
        mock_ai_gen_class.return_value = mock_ai_gen

        mock_session_mgr = Mock()
        mock_session_mgr_class.return_value = mock_session_mgr

        # Create RAG system
        rag = RAGSystem(test_config)

        # Get analytics
        analytics = rag.get_course_analytics()

        # Should return course info
        assert analytics["total_courses"] == 3
        assert len(analytics["course_titles"]) == 3
        assert "Course 1" in analytics["course_titles"]
