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
        assert definition["function"]["parameters"]["required"] == ["query"]

    def test_execute_with_results(self, mock_vector_store_with_results):
        """Test execute with successful search results"""
        tool = CourseSearchTool(mock_vector_store_with_results)
        result = tool.execute(query="What is Anthropic?")

        assert "Building Towards Computer Use with Anthropic" in result
        assert "Lesson 0" in result
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Building Towards Computer Use with Anthropic - Lesson 0"
        assert "url" in tool.last_sources[0]

    def test_execute_with_empty_results(self, mock_vector_store, empty_search_results):
        """Test execute when no results are found"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_with_course_filter(self, mock_vector_store, empty_search_results):
        """Test execute with course_name filter"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test", course_name="Computer Use")

        assert "in course 'Computer Use'" in result
        mock_vector_store.search.assert_called_once_with(
            query="test", course_name="Computer Use", lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store, empty_search_results):
        """Test execute with lesson_number filter"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test", lesson_number=1)

        assert "in lesson 1" in result
        mock_vector_store.search.assert_called_once_with(
            query="test", course_name=None, lesson_number=1
        )

    def test_execute_with_both_filters(self, mock_vector_store, empty_search_results):
        """Test execute with both course_name and lesson_number filters"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test", course_name="Computer Use", lesson_number=2)

        assert "in course 'Computer Use'" in result
        assert "in lesson 2" in result
        mock_vector_store.search.assert_called_once_with(
            query="test", course_name="Computer Use", lesson_number=2
        )

    def test_execute_with_error(self, mock_vector_store, error_search_results):
        """Test execute when search returns an error"""
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test", course_name="NonExistent")

        assert "No course found matching" in result

    def test_format_results_with_lesson_links(self, mock_vector_store_with_results):
        """Test that results are formatted with lesson links"""
        tool = CourseSearchTool(mock_vector_store_with_results)
        result = tool.execute(query="Anthropic")

        assert len(tool.last_sources) > 0
        for source in tool.last_sources:
            assert "text" in source
            assert "url" in source

    def test_format_results_without_lesson_number(self, mock_vector_store):
        """Test formatting when metadata doesn't include lesson_number"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Some course content without lesson number"],
            metadata=[{"course_title": "Test Course"}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test")

        assert "[Test Course]" in result
        assert "Some course content" in result
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course"

    def test_last_sources_reset_between_searches(self, mock_vector_store_with_results):
        """Test that last_sources is properly updated on each search"""
        tool = CourseSearchTool(mock_vector_store_with_results)

        tool.execute(query="first query")
        first_sources = tool.last_sources.copy()

        mock_vector_store_with_results.search.return_value = SearchResults(
            documents=["Different content"],
            metadata=[{"course_title": "Different Course", "lesson_number": 5}],
            distances=[0.1],
            error=None
        )
        tool.execute(query="second query")

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
        mock_vector_store.get_course_outline.return_value = {
            "course_title": "Building Towards Computer Use with Anthropic",
            "course_link": "https://example.com/course",
            "instructor": "Colt Steele",
            "lessons": [
                {"lesson_number": 0, "lesson_title": "Introduction", "lesson_link": "https://example.com/lesson0"},
                {"lesson_number": 1, "lesson_title": "Overview", "lesson_link": "https://example.com/lesson1"}
            ]
        }

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_title="Computer Use")

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

    @pytest.mark.parametrize("missing_field,expected_missing", [
        ("instructor", "Instructor:"),
        ("course_link", "Link:")
    ])
    def test_format_outline_with_missing_fields(self, mock_vector_store, missing_field, expected_missing):
        """Test formatting outline when optional fields are missing"""
        outline_data = {
            "course_title": "Test Course",
            "course_link": "https://example.com",
            "instructor": "Test Instructor",
            "lessons": []
        }
        del outline_data[missing_field]
        mock_vector_store.get_course_outline.return_value = outline_data

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_title="Test Course")

        assert expected_missing not in result
        assert "Test Course" in result


class TestToolManager:
    """Test suite for ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """Test registering a tool"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_vector_store))
        manager.register_tool(CourseOutlineTool(mock_vector_store))

        definitions = manager.get_tool_definitions()
        assert len(definitions) == 2
        tool_names = {d["function"]["name"] for d in definitions}
        assert tool_names == {"search_course_content", "get_course_outline"}

    def test_execute_tool(self, mock_vector_store_with_results):
        """Test executing a tool by name"""
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_vector_store_with_results))

        result = manager.execute_tool("search_course_content", query="test query")
        assert "Building Towards Computer Use with Anthropic" in result

    def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist"""
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self, mock_vector_store_with_results):
        """Test getting sources from last search"""
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_vector_store_with_results))

        manager.execute_tool("search_course_content", query="test")
        sources = manager.get_last_sources()

        assert len(sources) > 0
        assert "text" in sources[0]

    def test_reset_sources(self, mock_vector_store_with_results):
        """Test resetting sources"""
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_vector_store_with_results))

        manager.execute_tool("search_course_content", query="test")
        assert len(manager.get_last_sources()) > 0

        manager.reset_sources()
        assert len(manager.get_last_sources()) == 0


class TestMaxResultsBug:
    """Tests specifically designed to expose the MAX_RESULTS=0 bug"""

    def test_vector_store_with_zero_max_results(self, broken_config):
        """Test that MAX_RESULTS=0 causes issues"""
        from vector_store import VectorStore

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

        with patch('chromadb.PersistentClient') as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.get_or_create_collection.return_value = mock_collection
            mock_client.return_value = mock_client_instance

            store = VectorStore(
                chroma_path=broken_config.CHROMA_PATH,
                embedding_model=broken_config.EMBEDDING_MODEL,
                max_results=broken_config.MAX_RESULTS
            )

            results = store.search(query="test query")

            mock_collection.query.assert_called_once()
            call_kwargs = mock_collection.query.call_args[1]
            assert call_kwargs["n_results"] == 0, "MAX_RESULTS=0 should cause n_results=0"
            assert results.is_empty()

    def test_search_tool_with_zero_results(self, empty_search_results):
        """Test CourseSearchTool when vector store returns 0 results due to MAX_RESULTS=0"""
        mock_store = Mock()
        mock_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="What is Anthropic?")

        assert "No relevant content found" in result
