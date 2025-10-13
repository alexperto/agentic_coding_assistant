"""
Unit tests for search_tools.py - CourseSearchTool and CourseOutlineTool
"""
import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool"""

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly formatted"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "search_course_content"
        assert "description" in definition["function"]
        assert "parameters" in definition["function"]

        # Check required parameters
        params = definition["function"]["parameters"]
        assert "query" in params["properties"]
        assert "course_name" in params["properties"]
        assert "lesson_number" in params["properties"]
        assert params["required"] == ["query"]

    def test_execute_with_results(self, mock_vector_store_with_results):
        """Test execute with successful search results"""
        tool = CourseSearchTool(mock_vector_store_with_results)

        result = tool.execute(query="What is Anthropic?")

        # Should return formatted results
        assert "Building Towards Computer Use with Anthropic" in result
        assert "Lesson 0" in result
        assert "Welcome to the course" in result

        # Sources should be tracked
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Building Towards Computer Use with Anthropic - Lesson 0"
        assert "url" in tool.last_sources[0]

    def test_execute_with_empty_results(self, mock_vector_store):
        """Test execute when no results are found"""
        # Mock returns empty results
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_with_course_filter(self, mock_vector_store):
        """Test execute with course_name filter"""
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test", course_name="Computer Use")

        # Should indicate the course filter in the message
        assert "in course 'Computer Use'" in result

        # Verify the search was called with course_name
        mock_vector_store.search.assert_called_once_with(
            query="test",
            course_name="Computer Use",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Test execute with lesson_number filter"""
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test", lesson_number=1)

        # Should indicate the lesson filter
        assert "in lesson 1" in result

        # Verify the search was called with lesson_number
        mock_vector_store.search.assert_called_once_with(
            query="test",
            course_name=None,
            lesson_number=1
        )

    def test_execute_with_both_filters(self, mock_vector_store):
        """Test execute with both course_name and lesson_number filters"""
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test", course_name="Computer Use", lesson_number=2)

        # Should indicate both filters
        assert "in course 'Computer Use'" in result
        assert "in lesson 2" in result

        # Verify the search was called with both parameters
        mock_vector_store.search.assert_called_once_with(
            query="test",
            course_name="Computer Use",
            lesson_number=2
        )

    def test_execute_with_error(self, mock_vector_store):
        """Test execute when search returns an error"""
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="No course found matching 'NonExistent'"
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test", course_name="NonExistent")

        # Should return the error message
        assert "No course found matching 'NonExistent'" in result

    def test_format_results_with_lesson_links(self, mock_vector_store_with_results):
        """Test that results are formatted with lesson links"""
        tool = CourseSearchTool(mock_vector_store_with_results)

        result = tool.execute(query="Anthropic")

        # Check sources contain URLs
        assert len(tool.last_sources) > 0
        for source in tool.last_sources:
            assert "text" in source
            assert "url" in source

    def test_format_results_without_lesson_number(self, mock_vector_store):
        """Test formatting when metadata doesn't include lesson_number"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Some course content without lesson number"],
            metadata=[{
                "course_title": "Test Course",
                # No lesson_number
            }],
            distances=[0.1],
            error=None
        )
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test")

        # Should still format correctly
        assert "[Test Course]" in result
        assert "Some course content" in result

        # Source should not have lesson info
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course"

    def test_last_sources_reset_between_searches(self, mock_vector_store_with_results):
        """Test that last_sources is properly updated on each search"""
        tool = CourseSearchTool(mock_vector_store_with_results)

        # First search
        tool.execute(query="first query")
        first_sources = tool.last_sources.copy()

        # Second search with different results
        mock_vector_store_with_results.search.return_value = SearchResults(
            documents=["Different content"],
            metadata=[{
                "course_title": "Different Course",
                "lesson_number": 5
            }],
            distances=[0.1],
            error=None
        )
        tool.execute(query="second query")

        # Sources should be updated
        assert tool.last_sources != first_sources
        assert tool.last_sources[0]["text"] == "Different Course - Lesson 5"


class TestCourseOutlineTool:
    """Test suite for CourseOutlineTool"""

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly formatted"""
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "get_course_outline"
        assert "description" in definition["function"]
        assert definition["function"]["parameters"]["required"] == ["course_title"]

    def test_execute_with_valid_course(self, mock_vector_store):
        """Test execute with a valid course"""
        # Mock outline data
        mock_vector_store.get_course_outline.return_value = {
            "course_title": "Building Towards Computer Use with Anthropic",
            "course_link": "https://example.com/course",
            "instructor": "Colt Steele",
            "lessons": [
                {
                    "lesson_number": 0,
                    "lesson_title": "Introduction",
                    "lesson_link": "https://example.com/lesson0"
                },
                {
                    "lesson_number": 1,
                    "lesson_title": "Overview",
                    "lesson_link": "https://example.com/lesson1"
                }
            ]
        }

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_title="Computer Use")

        # Should contain course info
        assert "Building Towards Computer Use with Anthropic" in result
        assert "Colt Steele" in result
        assert "Lesson 0: Introduction" in result
        assert "Lesson 1: Overview" in result
        assert "2 total" in result

    def test_execute_with_nonexistent_course(self, mock_vector_store):
        """Test execute with a course that doesn't exist"""
        mock_vector_store.get_course_outline.return_value = None

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_title="NonExistentCourse")

        assert "No course found matching 'NonExistentCourse'" in result

    def test_format_outline_without_instructor(self, mock_vector_store):
        """Test formatting outline when instructor is not available"""
        mock_vector_store.get_course_outline.return_value = {
            "course_title": "Test Course",
            "course_link": "https://example.com",
            "lessons": []
        }

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_title="Test Course")

        # Should not have instructor line
        assert "Instructor:" not in result
        assert "Test Course" in result

    def test_format_outline_without_link(self, mock_vector_store):
        """Test formatting outline when course link is not available"""
        mock_vector_store.get_course_outline.return_value = {
            "course_title": "Test Course",
            "instructor": "Test Instructor",
            "lessons": []
        }

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_title="Test Course")

        # Should not have link line
        assert "Link:" not in result
        assert "Test Course" in result


class TestToolManager:
    """Test suite for ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """Test registering a tool"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        # Tool should be registered
        assert "search_course_content" in manager.tools

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        assert any(d["function"]["name"] == "search_course_content" for d in definitions)
        assert any(d["function"]["name"] == "get_course_outline" for d in definitions)

    def test_execute_tool(self, mock_vector_store_with_results):
        """Test executing a tool by name"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store_with_results)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test query")

        # Should return formatted results
        assert "Building Towards Computer Use with Anthropic" in result

    def test_execute_nonexistent_tool(self, mock_vector_store):
        """Test executing a tool that doesn't exist"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self, mock_vector_store_with_results):
        """Test getting sources from last search"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store_with_results)
        manager.register_tool(tool)

        # Execute a search
        manager.execute_tool("search_course_content", query="test")

        # Should return sources
        sources = manager.get_last_sources()
        assert len(sources) > 0
        assert "text" in sources[0]

    def test_reset_sources(self, mock_vector_store_with_results):
        """Test resetting sources"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store_with_results)
        manager.register_tool(tool)

        # Execute a search
        manager.execute_tool("search_course_content", query="test")
        assert len(manager.get_last_sources()) > 0

        # Reset sources
        manager.reset_sources()

        # Sources should be empty
        assert len(manager.get_last_sources()) == 0


class TestMaxResultsBug:
    """Tests specifically designed to expose the MAX_RESULTS=0 bug"""

    def test_vector_store_with_zero_max_results(self, broken_config):
        """Test that MAX_RESULTS=0 causes issues"""
        # This test simulates what happens in the real system with MAX_RESULTS=0
        from vector_store import VectorStore

        # Create a mock ChromaDB collection
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [[]],  # Empty because n_results=0
            "metadatas": [[]],
            "distances": [[]]
        }

        # Create vector store with broken config
        with patch('chromadb.PersistentClient') as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.get_or_create_collection.return_value = mock_collection
            mock_client.return_value = mock_client_instance

            store = VectorStore(
                chroma_path=broken_config.CHROMA_PATH,
                embedding_model=broken_config.EMBEDDING_MODEL,
                max_results=broken_config.MAX_RESULTS  # This is 0!
            )

            # Try to search
            results = store.search(query="test query")

            # Verify that n_results was set to 0
            mock_collection.query.assert_called_once()
            call_kwargs = mock_collection.query.call_args[1]
            assert call_kwargs["n_results"] == 0, "MAX_RESULTS=0 should cause n_results=0"

            # Results should be empty
            assert results.is_empty()

    def test_search_tool_with_zero_results(self):
        """Test CourseSearchTool when vector store returns 0 results due to MAX_RESULTS=0"""
        # Create a mock that simulates MAX_RESULTS=0 behavior
        mock_store = Mock()
        mock_store.search.return_value = SearchResults(
            documents=[],  # Empty due to MAX_RESULTS=0
            metadata=[],
            distances=[],
            error=None
        )

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="What is Anthropic?")

        # With MAX_RESULTS=0, the tool will find no results
        assert "No relevant content found" in result
        # This is the bug! User asks valid question but gets no results
