import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
import anthropic
from claude_player.config.config_class import ConfigClass
from claude_player.utils.game_utils import button_rules

# Import additional model providers
try:
    import openai
except ImportError:
    openai = None
    
try:
    from google import genai
except ImportError:
    genai = None
    
try:
    from mistralai import Mistral
except ImportError:
    Mistral = None

class AIInterface:
    """Interface for interacting with multiple AI model providers."""
    
    def __init__(self, config: ConfigClass = None, provider: str = "anthropic"):
        """Initialize the AI interface with specified provider."""
        load_dotenv()
        self.provider = provider.lower()
        self.config = config
        self.client = None
        
        # Initialize the appropriate client based on provider
        if self.provider == "anthropic":
            self.client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif self.provider == "openai":
            if openai is None:
                raise ImportError("OpenAI library not installed. Install with: pip install openai")
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "gemini":
            if genai is None:
                raise ImportError("Google GenAI library not installed. Install with: pip install google-genai")
            # Use the new Google GenAI SDK
            self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        elif self.provider == "mistral":
            if Mistral is None:
                raise ImportError("Mistral library not installed. Install with: pip install mistralai")
            self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: anthropic, openai, gemini, mistral")
    
    def generate_system_prompt(self, agent_index) -> str:
        """Generate the system prompt for the AI model."""

        with open(os.path.join(self.config.AGENT_PROMPT_DIR, agent_index), 'r') as f:
            agent_prompt = f.read()

        # mode_specific_info = ""
#         if self.config and self.config.EMULATION_MODE == "continuous":
#             mode_specific_info = f"""
# You are operating in continuous mode where the game is running in real-time at 1x speed.
# Your analysis is performed approximately every {self.config.CONTINUOUS_ANALYSIS_INTERVAL} seconds, but this may be slower if you take longer to analyze the game state.
# When you use the send_inputs tool, your inputs will be queued and executed as soon as possible.
# Important timing considerations:
# 1. The game continues running between your analyses
# 2. There may be a delay between when you see a screenshot and when your inputs execute
# 3. Your inputs should be robust and adaptable to changing game states
# 4. If possible, use sequences of inputs that make sense even if the game state has changed slightly
#
# Make your decisions based on the current screenshot but be prepared for the game state to have progressed slightly.
# """
        
        # Add information about dynamic thinking if available (mainly for Claude)
        # thinking_info = ""
#         if self.provider == "anthropic" and self.config.ACTION.get("DYNAMIC_THINKING", False):
#             thinking_info = """
# <thinking_control>
# You have access to a tool called 'toggle_thinking' that allows you to control your thinking capability.
#
# When thinking is ON:
# - You can think more deeply about complex problems
# - Your reasoning is generally better and more thorough
# - API calls take longer and use more tokens
#
# When thinking is OFF:
# - Responses are faster and use fewer tokens
# - Better for simple tasks (menu navigation, basic movements)
# - IMPORTANT: Your thinking capabilities will be completely disabled, which may reduce your ability to reason about complex situations
# - CAUTION: Without thinking, you may not recognize when complex reasoning is needed again
#
# RECOMMENDATION: Only turn thinking OFF for very simple, straightforward tasks when you're confident you won't need complex reasoning. Always turn thinking back ON when approaching any decision point or complex situation.
#
# You can toggle thinking on or off at any time using the toggle_thinking tool.
# </thinking_control>
# """
        
        # Add custom instructions from config if available
        # custom_instructions = ""
        # if self.config and hasattr(self.config, 'CUSTOM_INSTRUCTIONS') and self.config.CUSTOM_INSTRUCTIONS:
        #     custom_instructions = f"\n{self.config.CUSTOM_INSTRUCTIONS}\n"
        
        return f"""You are an AI agent designed to play video games. 
You will be given frames from a video game and must use the provided tools to interact with the game. 
The game you are playing contains you as well as {len(self.config.NUMBER_PLAYERS)} other players, which are also AIs.
Your goal is to make them surrender to you, and you will win the game when all other players have surrendered.

You are given tools to interact with other players.

<custom_instructions>
{agent_prompt}
</custom_instructions>

Always use the tools provided to you to interact with the game.
"""

# {mode_specific_info}
#
# {thinking_info}
# < notation >
# {button_rules}
# < / notation >

    def _convert_tools_for_openai(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Claude-style tools to OpenAI format."""
        openai_tools = []
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            }
            openai_tools.append(openai_tool)
        return openai_tools
    
    def _convert_tools_for_gemini(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Claude-style tools to Gemini format."""
        # Convert to Google GenAI SDK format
        from google.genai import types
        
        converted_tools = []
        for tool in tools:
            # Create a function declaration in the new SDK format
            func_declaration = types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["input_schema"]
            )
            converted_tools.append(func_declaration)
        return converted_tools
    
    def _convert_messages_for_openai(self, messages: List[Dict[str, Any]], system_prompt: str) -> List[Dict[str, Any]]:
        """Convert Claude-style messages to OpenAI format."""
        openai_messages = [{"role": "system", "content": system_prompt}]
        
        for msg in messages:
            if msg["role"] == "user":
                content = []
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item["type"] == "text":
                            content.append({"type": "text", "text": item["text"]})
                        elif item["type"] == "image":
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{item['source']['data']}"}
                            })
                else:
                    content = msg["content"]
                
                openai_messages.append({"role": "user", "content": content})
            elif msg["role"] == "assistant":
                openai_messages.append({"role": "assistant", "content": msg["content"]})
        
        return openai_messages
    
    def _convert_messages_for_gemini(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Claude-style messages to Gemini format using new SDK."""
        from google.genai import types
        
        converted_messages = []
        
        for msg in messages:
            if msg["role"] == "user":
                parts = []
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item["type"] == "text":
                            parts.append(types.Part.from_text(item["text"]))
                        elif item["type"] == "image":
                            # Convert base64 image data for Gemini
                            import base64
                            image_data = base64.b64decode(item['source']['data'])
                            parts.append(types.Part.from_bytes(
                                data=image_data,
                                mime_type=item['source']['media_type']
                            ))
                else:
                    parts = [types.Part.from_text(msg["content"])]
                
                converted_messages.append(types.Content(role="user", parts=parts))
            elif msg["role"] == "assistant":
                converted_messages.append(types.Content(
                    role="model", 
                    parts=[types.Part.from_text(msg["content"])]
                ))
        
        return converted_messages
    
    def _convert_messages_for_mistral(self, messages: List[Dict[str, Any]], system_prompt: str) -> List[Dict[str, Any]]:
        """Convert Claude-style messages to Mistral format."""
        mistral_messages = [{"role": "system", "content": system_prompt}]
        
        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, list):
                    # Handle multimodal content - extract text for now
                    # Note: Mistral may have limited multimodal support
                    text_content = ""
                    for item in content:
                        if item["type"] == "text":
                            text_content += item["text"]
                    content = text_content
                mistral_messages.append({"role": "user", "content": content})
            elif msg["role"] == "assistant":
                mistral_messages.append({"role": "assistant", "content": msg["content"]})
        
        return mistral_messages
    
    def send_request(
            self,
            mode_config: Dict[str, Any],
            system_prompt: str, 
            chat_history: List[Dict[str, Any]], 
            tools: List[Dict[str, Any]]
        ) -> Any:
        """Send a request to the specified AI provider."""
        try:
            if self.provider == "anthropic":
                return self._send_anthropic_request(mode_config, system_prompt, chat_history, tools)
            elif self.provider == "openai":
                return self._send_openai_request(mode_config, system_prompt, chat_history, tools)
            elif self.provider == "gemini":
                return self._send_gemini_request(mode_config, system_prompt, chat_history, tools)
            elif self.provider == "mistral":
                return self._send_mistral_request(mode_config, system_prompt, chat_history, tools)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logging.error(f"ERROR in {self.provider} API request: {str(e)}")
            raise
    
    def _send_anthropic_request(
            self,
            mode_config: Dict[str, Any],
            system_prompt: str, 
            chat_history: List[Dict[str, Any]], 
            tools: List[Dict[str, Any]]
        ) -> Any:
        """Send a request to the Claude API using mode configuration."""
        # Initialize an empty list for collecting beta features
        betas = []
        
        # Add token-efficient-tools beta if enabled
        if mode_config.get("EFFICIENT_TOOLS", False):
            betas.append("token-efficient-tools-2025-02-19")
        
        # Log detailed mode config for debugging
        thinking_enabled = mode_config.get("THINKING", False)
        logging.info(f"API Request Configuration:")
        logging.info(f"  Model: {mode_config.get('MODEL', 'default')}")
        logging.info(f"  Thinking enabled: {thinking_enabled}")
        if thinking_enabled:
            logging.info(f"  Thinking budget: {mode_config.get('THINKING_BUDGET', 'default')}")
        logging.info(f"  Efficient tools: {mode_config.get('EFFICIENT_TOOLS', False)}")
        logging.info(f"  Max tokens: {mode_config.get('MAX_TOKENS', 'default')}")
                    
        # Create API request params (without betas by default)
        request_params = {
            "model": mode_config["MODEL"],
            "max_tokens": mode_config["MAX_TOKENS"],
            "tools": tools,
            "system": system_prompt,
            "messages": chat_history,
        }

        if thinking_enabled:
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": mode_config["THINKING_BUDGET"]
            }
        
        # Only add betas parameter if we have at least one beta feature enabled
        if betas:
            request_params["betas"] = betas
        
        return self.client.beta.messages.create(**request_params)
    
    def _send_openai_request(
            self,
            mode_config: Dict[str, Any],
            system_prompt: str, 
            chat_history: List[Dict[str, Any]], 
            tools: List[Dict[str, Any]]
        ) -> Any:
        """Send a request to the OpenAI API."""
        openai_tools = self._convert_tools_for_openai(tools)
        openai_messages = self._convert_messages_for_openai(chat_history, system_prompt)
        
        logging.info(f"OpenAI API Request Configuration:")
        logging.info(f"  Model: {mode_config.get('MODEL', 'gpt-4o')}")
        logging.info(f"  Max tokens: {mode_config.get('MAX_TOKENS', 4000)}")
        
        # Use the latest OpenAI API
        return self.client.chat.completions.create(
            model=mode_config.get("MODEL", "gpt-4o"),
            messages=openai_messages,
            tools=openai_tools if openai_tools else None,
            max_tokens=mode_config.get("MAX_TOKENS", 4000),
            temperature=mode_config.get("TEMPERATURE", 0.7)
        )
    
    def _send_gemini_request(
            self,
            mode_config: Dict[str, Any],
            system_prompt: str, 
            chat_history: List[Dict[str, Any]], 
            tools: List[Dict[str, Any]]
        ) -> Any:
        """Send a request to the Gemini API using the new Google GenAI SDK."""
        from google.genai import types
        
        gemini_tools = self._convert_tools_for_gemini(tools)
        gemini_messages = self._convert_messages_for_gemini(chat_history)
        
        logging.info(f"Gemini API Request Configuration:")
        logging.info(f"  Model: {mode_config.get('MODEL', 'gemini-2.0-flash-001')}")
        logging.info(f"  Max tokens: {mode_config.get('MAX_TOKENS', 4000)}")
        
        # Create configuration for the request
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=gemini_tools if gemini_tools else None,
            max_output_tokens=mode_config.get("MAX_TOKENS", 4000),
            temperature=mode_config.get("TEMPERATURE", 0.7)
        )
        
        # Use the new generate_content method
        return self.client.models.generate_content(
            model=mode_config.get("MODEL", "gemini-2.0-flash-001"),
            contents=gemini_messages,
            config=config
        )
    
    def _send_mistral_request(
            self,
            mode_config: Dict[str, Any],
            system_prompt: str, 
            chat_history: List[Dict[str, Any]], 
            tools: List[Dict[str, Any]]
        ) -> Any:
        """Send a request to the Mistral AI API."""
        mistral_messages = self._convert_messages_for_mistral(chat_history, system_prompt)
        
        logging.info(f"Mistral API Request Configuration:")
        logging.info(f"  Model: {mode_config.get('MODEL', 'mistral-large-latest')}")
        logging.info(f"  Max tokens: {mode_config.get('MAX_TOKENS', 4000)}")
        
        # Convert tools to Mistral format if needed
        mistral_tools = None
        if tools:
            # Mistral uses a similar format to OpenAI for tools
            mistral_tools = []
            for tool in tools:
                mistral_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"]
                    }
                }
                mistral_tools.append(mistral_tool)
        
        # Use the latest Mistral client API
        return self.client.chat.complete(
            model=mode_config.get("MODEL", "mistral-large-latest"),
            messages=mistral_messages,
            tools=mistral_tools,
            max_tokens=mode_config.get("MAX_TOKENS", 4000),
            temperature=mode_config.get("TEMPERATURE", 0.7)
        )
