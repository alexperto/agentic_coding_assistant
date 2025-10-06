from openai import AzureOpenAI
from typing import List, Optional, Dict, Any
import json

class AIGenerator:
    """Handles interactions with Azure OpenAI API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **get_course_outline**: Retrieves course structure, outline, and lesson list
   - Use for: "What is the outline?", "List the lessons", "What's in this course?", "Course structure"
   - Returns: Course title, link, instructor, and complete lesson list with numbers and titles

2. **search_course_content**: Searches within specific course materials and content
   - Use for: Questions about specific topics, concepts, or detailed content within courses
   - Supports filtering by course name and lesson number

Tool Usage Guidelines:
- **One tool call per query maximum**
- Choose the appropriate tool based on the question type
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline/structure questions**: Use get_course_outline
- **Course content questions**: Use search_course_content
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, endpoint: str, api_key: str, api_version: str, deployment: str):
        # Construct AzureOpenAI client with explicit Azure parameters.
        # Use `azure_endpoint` and `api_key` (or `azure_ad_token` if you have an AAD token).
        self.client = AzureOpenAI(
            base_url=endpoint,
            azure_ad_token=api_key,
            api_version=api_version
        )

        self.deployment = deployment
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.deployment,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build messages array with system prompt
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        
        # Add conversation history if available
        if conversation_history:
            messages.append({
                "role": "system", 
                "content": f"Previous conversation:\n{conversation_history}"
            })
        
        # Add user query
        messages.append({"role": "user", "content": query})
        
        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": messages
        }
        
        # Add tools if available (convert to OpenAI function calling format)
        if tools:
            print(">>> TOOLS:", tools)
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"
        
        # Get response from Azure OpenAI
        #print("-----------------------------------------------------------------")
        
        #print(">>> API PARAMS:", api_params)
        #print(">>> Client object:", self.client)
        #print(">>> Client attributes:", dir(self.client))
        #try:
        #    print(">>> Client vars:", vars(self.client))
        #except:
        #    print(">>> Could not get client vars (might be a proxied object)")
        #print("-----------------------------------------------------------------")
        response = self.client.chat.completions.create(**api_params)
        
        # Handle tool execution if needed
        if response.choices[0].message.tool_calls and tool_manager:
            return self._handle_tool_execution(response, messages, tool_manager)
        
        # Return direct response
        return response.choices[0].message.content
        
    def _handle_tool_execution(self, initial_response, messages: List[Dict], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            messages: Current conversation messages
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Add AI's tool use response to messages
        messages.append({
            "role": "assistant",
            "content": initial_response.choices[0].message.content,
            "tool_calls": initial_response.choices[0].message.tool_calls
        })
        
        # Execute all tool calls and collect results
        for tool_call in initial_response.choices[0].message.tool_calls:
            # Parse function arguments
            function_args = json.loads(tool_call.function.arguments)
            
            # Execute the tool
            tool_result = tool_manager.execute_tool(
                tool_call.function.name,
                **function_args
            )
            
            # Add tool result as message
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages
        }
        
        # Get final response
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content