from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults
import requests
import json


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return OpenAI function definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return OpenAI function definition for this tool"""
        return {
            "type": "function",
            "function": {
                "name": "search_course_content",
                "description": "Search course materials with smart course name matching and lesson filtering",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string", 
                            "description": "What to search for in the course content"
                        },
                        "course_name": {
                            "type": "string",
                            "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                        },
                        "lesson_number": {
                            "type": "integer",
                            "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Track source for the UI with lesson link
            source_text = course_title
            if lesson_num is not None:
                source_text += f" - Lesson {lesson_num}"

            # Try to get lesson link
            lesson_link = None
            if lesson_num is not None:
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)

            # Store source as dict with text and optional link
            source_info = {"text": source_text}
            if lesson_link:
                source_info["url"] = lesson_link
            sources.append(source_info)

            formatted.append(f"{header}\n{doc}")

        # Store sources for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving course outlines and structure"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return OpenAI function definition for this tool"""
        return {
            "type": "function",
            "function": {
                "name": "get_course_outline",
                "description": "Get the complete outline and structure of a course, including all lessons",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "course_title": {
                            "type": "string",
                            "description": "The course title or name (partial matches work, e.g. 'MCP', 'Introduction')"
                        }
                    },
                    "required": ["course_title"]
                }
            }
        }

    def execute(self, course_title: str) -> str:
        """
        Execute the course outline retrieval.

        Args:
            course_title: Course name or title to get outline for

        Returns:
            Formatted course outline or error message
        """
        # Get course outline from vector store
        outline = self.store.get_course_outline(course_title)

        # Handle not found
        if not outline:
            return f"No course found matching '{course_title}'."

        # Format the outline
        return self._format_outline(outline)

    def _format_outline(self, outline: Dict[str, Any]) -> str:
        """Format course outline for display"""
        parts = []

        # Course title and link
        parts.append(f"Course: {outline['course_title']}")
        if outline.get('course_link'):
            parts.append(f"Link: {outline['course_link']}")

        # Instructor if available
        if outline.get('instructor'):
            parts.append(f"Instructor: {outline['instructor']}")

        # Lessons
        lessons = outline.get('lessons', [])
        if lessons:
            parts.append(f"\nLessons ({len(lessons)} total):")
            for lesson in lessons:
                lesson_num = lesson.get('lesson_number')
                lesson_title = lesson.get('lesson_title')
                parts.append(f"  Lesson {lesson_num}: {lesson_title}")

        return "\n".join(parts)


class NutritionTool(Tool):
    """Tool for answering nutrition, food, and diet-related questions via UCSF API"""

    def __init__(self, token_manager):
        """
        Initialize NutritionTool with TokenManager for authentication.

        Args:
            token_manager: TokenManager instance for OAuth token retrieval
        """
        self.token_manager = token_manager
        self.api_endpoint = "https://dev-unified-api.ucsf.edu/general/versaassistant/api/answer"

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return OpenAI function definition for this tool"""
        return {
            "type": "function",
            "function": {
                "name": "ask_nutrition_expert",
                "description": "Ask a nutrition expert about food, diet, or nutrition-related questions. Use this tool ONLY for questions about nutrition, food, diet, eating habits, nutritional values, or meal planning.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The nutrition, food, or diet-related question to ask the expert"
                        }
                    },
                    "required": ["question"]
                }
            }
        }

    def execute(self, question: str) -> str:
        """
        Execute the nutrition query by calling the UCSF API.

        Args:
            question: The nutrition-related question from the user

        Returns:
            Response from the nutrition API or error message
        """
        try:
            # Get authentication token
            if not self.token_manager:
                return "Error: Authentication not configured for nutrition queries."

            token = self.token_manager.get_token()

            # Prepare request headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }

            # Prepare request body
            body = {
                "userid": "john.doe@ucsf.edu",
                "datasource": "eureka_fim",
                "model": "GPT-4o",
                "temperature": "0.0",
                "context": "1",
                "returndoc": "1",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant expert on nutrition, respond briefly"
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            }

            # Log the request details
            print("\n" + "="*70)
            print("NUTRITION API REQUEST DETAILS")
            print("="*70)
            print(f"URL: {self.api_endpoint}")
            print(f"\nRequest Headers:")
            for key, value in headers.items():
                if key == "Authorization":
                    print(f"  {key}: Bearer {token[:30]}...{token[-10:]} (truncated)")
                else:
                    print(f"  {key}: {value}")
            print(f"\nRequest Body:")
            print(json.dumps(body, indent=2))
            print("="*70)

            # Make API request
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=body,
                timeout=30  # 30 second timeout for API calls
            )

            # Log the response details
            print("\n" + "="*70)
            print("NUTRITION API RESPONSE DETAILS")
            print("="*70)
            print(f"Status Code: {response.status_code}")
            print(f"\nResponse Headers:")
            for key, value in response.headers.items():
                print(f"  {key}: {value}")

            print(f"\nResponse Body:")
            try:
                response_json = response.json()
                print(json.dumps(response_json, indent=2))
            except:
                print(f"  (Raw text): {response.text[:500]}")
            print("="*70 + "\n")

            # Check for HTTP errors
            response.raise_for_status()

            # Parse response
            response_data = response.json()

            # Extract the answer from the response
            # The API response structure may vary, so we handle common formats
            if isinstance(response_data, dict):
                # Try common response field names
                answer = (
                    response_data.get("answer") or
                    response_data.get("response") or
                    response_data.get("content") or
                    response_data.get("message")
                )

                if answer:
                    return str(answer)
                else:
                    # If no recognized field, return the whole response as JSON
                    return json.dumps(response_data, indent=2)
            else:
                # If response is not a dict, return as string
                return str(response_data)

        except requests.exceptions.Timeout:
            return "Error: Nutrition API request timed out. Please try again."
        except requests.exceptions.RequestException as e:
            return f"Error: Failed to connect to nutrition API - {str(e)}"
        except json.JSONDecodeError:
            return "Error: Invalid response format from nutrition API."
        except Exception as e:
            return f"Error: Unexpected error while querying nutrition API - {str(e)}"


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("function").get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for OpenAI function calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []