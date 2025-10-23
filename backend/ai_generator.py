from openai import AzureOpenAI
from typing import List, Optional, Dict, Any
import json

class AIGenerator:
    """Handles interactions with Azure OpenAI API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **get_course_outline**: Retrieves course structure, outline, and lesson list
   - Use for: "What is the outline?", "List the lessons", "What's in this course?"
   - Returns: Course title, link, instructor, and complete lesson list with numbers and titles

2. **search_course_content**: Searches within specific course materials and content
   - Use for: Questions about specific topics, concepts, or detailed content
   - Supports filtering by course name and lesson number

Tool Usage Guidelines:
- **Sequential tool calling**: You can make MULTIPLE tool calls across up to 2 rounds
- **Round 1**: Make initial tool call(s) to gather necessary information
- **Round 2**: If needed, use Round 1 results to make more specific follow-up tool calls
- **When to use sequential calls**:
  - User asks about "same topic as lesson X of course Y" → Get lesson X title, then search for that topic
  - User asks to compare content → Get outline first, then search specific sections
- **When to stop**: You have sufficient info, tool returns error, or you've made 2 rounds of calls
- Choose the appropriate tool(s) based on the question type and previous results

Response Protocol:
- **General knowledge**: Answer without tools
- **Course structure**: Use get_course_outline
- **Course content**: Use search_course_content
- **Cross-reference queries**: Use get_course_outline THEN search_course_content
- **No meta-commentary**:
  - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
  - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when helpful
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, endpoint: str, token_manager: Any, api_version: str, deployment: str):
        """
        Initialize AIGenerator with Azure OpenAI configuration.

        Args:
            endpoint: Azure OpenAI endpoint URL
            token_manager: TokenManager instance for dynamic OAuth token retrieval
            api_version: Azure OpenAI API version
            deployment: Azure OpenAI deployment name
        """
        self.endpoint = endpoint
        self.api_version = api_version
        self.deployment = deployment
        self.token_manager = token_manager

        if token_manager:
            print("[AIGenerator] Initialized with dynamic OAuth token management")

        # Pre-build base API parameters
        self.base_params = {
            "model": self.deployment,
            "temperature": 0,
            "max_tokens": 800
        }

    def _get_client(self) -> AzureOpenAI:
        """
        Get AzureOpenAI client with current valid token.

        Fetches a fresh token from TokenManager on each call.

        Returns:
            Configured AzureOpenAI client instance
        """
        current_token = self.token_manager.get_token()
        return AzureOpenAI(
            base_url=self.endpoint,
            azure_ad_token=current_token,
            api_version=self.api_version
        )
    
    def generate_response(self,
                         query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_tool_rounds: int = 2) -> str:
        """
        Generate AI response with support for sequential tool calling.

        Args:
            query: The user's question
            conversation_history: Previous messages for context
            tools: Available tools (OpenAI function definitions)
            tool_manager: Manager to execute tools
            max_tool_rounds: Maximum tool calling rounds (default: 2)

        Returns:
            Final generated response string
        """
        # Build initial messages
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        if conversation_history:
            messages.append({
                "role": "system",
                "content": f"Previous conversation:\n{conversation_history}"
            })

        messages.append({"role": "user", "content": query})

        # Iterative tool calling loop
        current_round = 0

        while current_round < max_tool_rounds:
            current_round += 1

            # Prepare API parameters with accumulated messages
            api_params = {
                **self.base_params,
                "messages": messages
            }

            # Include tools if available and within max rounds
            if tools and tool_manager:
                api_params["tools"] = tools
                api_params["tool_choice"] = "auto"

            # Make API call with fresh client (ensures valid token)
            client = self._get_client()
            response = client.chat.completions.create(**api_params)
            assistant_message = response.choices[0].message

            # Check if AI wants to call tools
            if not assistant_message.tool_calls:
                # No tools requested - return final answer
                return assistant_message.content or ""

            # Add assistant's tool call message to history
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": assistant_message.tool_calls
            })

            # Execute all tool calls
            tool_execution_failed = False
            for tool_call in assistant_message.tool_calls:
                try:
                    # Parse arguments
                    function_args = json.loads(tool_call.function.arguments)

                    # Execute tool
                    tool_result = tool_manager.execute_tool(
                        tool_call.function.name,
                        **function_args
                    )

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                except json.JSONDecodeError as e:
                    # Malformed tool arguments
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error: Invalid tool arguments format"
                    })
                    tool_execution_failed = True

                except Exception as e:
                    # Tool execution error
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error: Tool execution failed - {str(e)}"
                    })
                    tool_execution_failed = True

            # If at max rounds or tool failed, stop iteration
            if current_round >= max_tool_rounds or tool_execution_failed:
                break

        # After loop, make final synthesis call if last message was a tool result
        if messages[-1]["role"] == "tool":
            final_params = {
                **self.base_params,
                "messages": messages
            }

            # Include tools for consistency (AI shouldn't call them at this point)
            if tools:
                final_params["tools"] = tools
                final_params["tool_choice"] = "auto"

            client = self._get_client()
            final_response = client.chat.completions.create(**final_params)
            return final_response.choices[0].message.content or ""

        # Last message was assistant response (no tools called)
        return messages[-1]["content"] or ""