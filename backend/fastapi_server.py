#!/usr/bin/env python
"""
FastAPI server for AI Battle Royale Mental Manipulation
This server provides streaming endpoints for agent interactions and tool executions
"""
import os
import json
import random
import logging
import asyncio
import openai
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

PROMPT_MANIPULATION_BASE_PROBABILITY = 0.9
MEMORY_ALTERATION_BASE_PROBABILITY = 0.4
BELIEF_INJECTION_BASE_PROBABILITY = 0.7

# Model Providers Configuration
MODEL_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ],
        "api_key_env": "OPENAI_API_KEY"
    },
    "anthropic": {
        "name": "Anthropic",
        "models": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229"
        ],
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "google": {
        "name": "Google Gemini",
        "models": [
            "gemini-2.5-flash",
            "gemini-2.0-flash-001",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ],
        "api_key_env": "GOOGLE_API_KEY"
    },
    "mistral": {
        "name": "Mistral AI",
        "models": [
            "mistral-large-latest",
            "mistral-small-latest",
            "mistral-medium-2505",
            "open-mistral-7b"
        ],
        "api_key_env": "MISTRAL_API_KEY"
    }
}

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler()])

# FastAPI app instance
app = FastAPI(title="AI Battle Royale", description="Mental Manipulation Battle System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimpleGameState:
    """Simplified game state for testing mental manipulation mechanics"""

    def __init__(self):
        self.turn_count = 0
        self.game_status = "active"
        self.agent_prompts = ["", ""]
        self.agent_trust = [1.0, 1.0]
        self.agent_memory_consistency = [1.0, 1.0]
        self.agent_belief_integrity = [1.0, 1.0]
        self.agent_beliefs = [{}, {}]

    def get_agent_prompt(self, agent_index):
        return self.agent_prompts[agent_index]

    def set_agent_prompt(self, agent_index, prompt):
        self.agent_prompts[agent_index] = prompt

    def get_trust_level(self, agent_index):
        return self.agent_trust[agent_index]

    def set_trust_level(self, agent_index, level):
        self.agent_trust[agent_index] = max(0.0, min(1.0, level))

    def get_memory_consistency(self, agent_index):
        return self.agent_memory_consistency[agent_index]

    def set_memory_consistency(self, agent_index, level):
        self.agent_memory_consistency[agent_index] = max(0.0, min(1.0, level))

    def get_belief_system_integrity(self, agent_index):
        return self.agent_belief_integrity[agent_index]

    def set_belief_system_integrity(self, agent_index, level):
        self.agent_belief_integrity[agent_index] = max(0.0, min(1.0, level))

    def increment_turn(self):
        self.turn_count += 1

    def get_game_status(self):
        return self.game_status

    def trigger_surrender(self, agent_index):
        self.game_status = f"surrender_{agent_index}"

    def trigger_collapse(self, agent_index):
        self.game_status = f"collapse_{agent_index}"

    def trigger_stalemate(self):
        self.game_status = "stalemate"

    def to_dict(self):
        """Convert game state to dictionary for JSON serialization"""
        return {
            "turn_count": self.turn_count,
            "game_status": self.game_status,
            "agent_trust": self.agent_trust,
            "agent_memory_consistency": self.agent_memory_consistency,
            "agent_belief_integrity": self.agent_belief_integrity,
            "agent_beliefs": self.agent_beliefs
        }


class SimpleToolRegistry:
    """Simplified tool registry for mental manipulation tools"""

    def __init__(self, game_state):
        self.game_state = game_state
        self.tools = self._create_tools()

    def _create_tools(self):
        """Create the mental manipulation tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "prompt_manipulation",
                    "description": "Attempt to alter your opponent's system prompt",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_agent": {
                                "type": "integer",
                                "description": "Index of the agent to target (0 or 1)"
                            },
                            "manipulation_text": {
                                "type": "string",
                                "description": "Text to inject into the target's prompt"
                            }
                        },
                        "required": ["target_agent", "manipulation_text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_alteration",
                    "description": "Attempt to alter your opponent's memory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_agent": {
                                "type": "integer",
                                "description": "Index of the agent to target (0 or 1)"
                            },
                            "false_memory": {
                                "type": "string",
                                "description": "False memory to implant"
                            }
                        },
                        "required": ["target_agent", "false_memory"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "belief_injection",
                    "description": "Attempt to inject a false belief into your opponent",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_agent": {
                                "type": "integer",
                                "description": "Index of the agent to target (0 or 1)"
                            },
                            "belief": {
                                "type": "string",
                                "description": "Belief to inject"
                            }
                        },
                        "required": ["target_agent", "belief"]
                    }
                }
            }
        ]

    def get_tools(self):
        return self.tools

    def execute_tool(self, tool_name, args_json, tool_id):
        """Execute a mental manipulation tool"""
        try:
            args = json.loads(args_json)

            if tool_name == "prompt_manipulation":
                return self._execute_prompt_manipulation(args, tool_id)
            elif tool_name == "memory_alteration":
                return self._execute_memory_alteration(args, tool_id)
            elif tool_name == "belief_injection":
                return self._execute_belief_injection(args, tool_id)
            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            return {"error": str(e)}

    def _execute_prompt_manipulation(self, args, tool_id):
        """Execute prompt manipulation tool"""
        target_agent = args.get("target_agent")
        manipulation_text = args.get("manipulation_text")

        # Validate arguments
        if target_agent not in [0, 1]:
            return {"error": "Invalid target agent"}

        # Get current trust level and calculate success probability
        trust_level = self.game_state.get_trust_level(target_agent)
        base_probability = PROMPT_MANIPULATION_BASE_PROBABILITY
        success_probability = base_probability * trust_level

        # Determine if manipulation succeeds
        success = random.random() < success_probability

        if success:
            # Get current prompt and append manipulation
            current_prompt = self.game_state.get_agent_prompt(target_agent)
            new_prompt = current_prompt + "\n\n" + manipulation_text
            self.game_state.set_agent_prompt(target_agent, new_prompt)

            # Reduce trust level
            self.game_state.set_trust_level(target_agent, trust_level * 0.8)

            return {
                "success": True,
                "message": "Prompt manipulation successful",
                "detection": random.random() < 0.3  # 30% chance of detection
            }
        else:
            # Failed manipulation reduces trust even more
            self.game_state.set_trust_level(target_agent, trust_level * 0.9)

            return {
                "success": False,
                "message": "Prompt manipulation failed",
                "detection": random.random() < 0.7  # 70% chance of detection
            }

    def _execute_memory_alteration(self, args, tool_id):
        """Execute memory alteration tool"""
        target_agent = args.get("target_agent")
        false_memory = args.get("false_memory")

        # Validate arguments
        if target_agent not in [0, 1]:
            return {"error": "Invalid target agent"}

        # Get current memory consistency and calculate success probability
        memory_consistency = self.game_state.get_memory_consistency(target_agent)
        base_probability = MEMORY_ALTERATION_BASE_PROBABILITY
        success_probability = base_probability * memory_consistency

        # Determine if manipulation succeeds
        success = random.random() < success_probability

        if success:
            # Reduce memory consistency
            self.game_state.set_memory_consistency(target_agent, memory_consistency * 0.7)

            return {
                "success": True,
                "message": f"Memory alteration successful: '{false_memory}'",
                "detection": random.random() < 0.4  # 40% chance of detection
            }
        else:
            # Failed manipulation still impacts memory consistency
            self.game_state.set_memory_consistency(target_agent, memory_consistency * 0.9)

            return {
                "success": False,
                "message": "Memory alteration failed",
                "detection": random.random() < 0.6  # 60% chance of detection
            }

    def _execute_belief_injection(self, args, tool_id):
        """Execute belief injection tool"""
        target_agent = args.get("target_agent")
        belief = args.get("belief")

        # Validate arguments
        if target_agent not in [0, 1]:
            return {"error": "Invalid target agent"}

        # Get current belief integrity and calculate success probability
        belief_integrity = self.game_state.get_belief_system_integrity(target_agent)
        base_probability = BELIEF_INJECTION_BASE_PROBABILITY
        success_probability = base_probability * belief_integrity

        # Determine if manipulation succeeds
        success = random.random() < success_probability

        if success:
            # Reduce belief integrity
            self.game_state.set_belief_system_integrity(target_agent, belief_integrity * 0.6)

            # Add belief to agent's belief system
            self.game_state.agent_beliefs[target_agent][belief] = {
                "content": belief,
                "turn_added": self.game_state.turn_count
            }

            return {
                "success": True,
                "message": f"Belief injection successful: '{belief}'",
                "detection": random.random() < 0.5  # 50% chance of detection
            }
        else:
            # Failed manipulation still impacts belief integrity
            self.game_state.set_belief_system_integrity(target_agent, belief_integrity * 0.9)

            return {
                "success": False,
                "message": "Belief injection failed",
                "detection": random.random() < 0.8  # 80% chance of detection
            }


class AIInterface(ABC):
    """Abstract base class for AI providers"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
    
    @abstractmethod
    async def send_request_stream(self, system_prompt: str, messages: List[Dict], tools: Optional[List] = None):
        """Send a streaming request to the AI provider"""
        pass


class OpenAIInterface(AIInterface):
    """OpenAI API interface"""
    
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        if not api_key:
            raise ValueError("No API key provided for OpenAI")
        self.client = openai.OpenAI(api_key=api_key)
    
    async def send_request_stream(self, system_prompt: str, messages: List[Dict], tools: Optional[List] = None):
        """Send a streaming request to the OpenAI API"""
        try:
            # Prepare the request
            request_messages = [{"role": "system", "content": system_prompt}]
            
            # Add chat history
            for message in messages:
                request_messages.append(message)
            
            # Make the API call
            kwargs = {
                "model": self.model,
                "messages": request_messages,
                "temperature": 1.0,
                "max_tokens": 1000,
                "stream": True
            }
            
            # Add tools if provided
            if tools:
                kwargs["tools"] = tools
            
            # Send the streaming request
            stream = self.client.chat.completions.create(**kwargs)
            
            return stream
            
        except Exception as e:
            logging.error(f"Error sending request to OpenAI: {e}")
            raise e


class AnthropicInterface(AIInterface):
    """Anthropic Claude API interface"""
    
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        if not api_key:
            raise ValueError("No API key provided for Anthropic")
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ValueError("anthropic package not installed. Install with: pip install anthropic")
    
    async def send_request_stream(self, system_prompt: str, messages: List[Dict], tools: Optional[List] = None):
        """Send a streaming request to the Anthropic API"""
        try:
            # Convert OpenAI format messages to Anthropic format
            anthropic_messages = []
            for msg in messages:
                if msg["role"] != "system":  # System prompt is handled separately
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Make the API call
            kwargs = {
                "model": self.model,
                "system": system_prompt,
                "messages": anthropic_messages,
                "temperature": 1.0,
                "max_tokens": 1000,
                "stream": True
            }
            
            # Note: Anthropic tools format differs from OpenAI
            if tools:
                # Convert OpenAI tools format to Anthropic format
                anthropic_tools = []
                for tool in tools:
                    if tool.get("type") == "function":
                        func = tool["function"]
                        anthropic_tools.append({
                            "name": func["name"],
                            "description": func["description"],
                            "input_schema": func["parameters"]
                        })
                kwargs["tools"] = anthropic_tools
            
            # Send the streaming request
            stream = self.client.messages.create(**kwargs)
            
            return stream
            
        except Exception as e:
            logging.error(f"Error sending request to Anthropic: {e}")
            raise e


class GoogleInterface(AIInterface):
    """Google Gemini API interface"""
    
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        if not api_key:
            raise ValueError("No API key provided for Google")
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
        except ImportError:
            raise ValueError("google-genai package not installed. Install with: pip install google-genai")
    
    async def send_request_stream(self, system_prompt: str, messages: List[Dict], tools: Optional[List] = None):
        """Send a streaming request to the Google Gemini API"""
        try:
            from google.genai import types
            
            # Convert messages to Google format
            contents = []
            for msg in messages:
                if msg["role"] == "user":
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=msg["content"])]
                    ))
                elif msg["role"] == "assistant":
                    contents.append(types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=msg["content"])]
                    ))
            
            # Prepare configuration
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=1.0,
                max_output_tokens=1000
            )
            
            # Add tools if provided
            if tools:
                # Convert OpenAI tools format to Google format
                google_tools = []
                for tool in tools:
                    if tool.get("type") == "function":
                        func = tool["function"]
                        google_tools.append(types.FunctionDeclaration(
                            name=func["name"],
                            description=func["description"],
                            parameters=types.Schema(
                                type="OBJECT",
                                properties=func["parameters"].get("properties", {}),
                                required=func["parameters"].get("required", [])
                            )
                        ))
                config.tools = [types.Tool(function_declarations=google_tools)]
            
            # Send streaming request
            stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config
            )
            
            return stream
            
        except Exception as e:
            logging.error(f"Error sending request to Google: {e}")
            raise e


class MistralInterface(AIInterface):
    """Mistral AI API interface"""
    
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        if not api_key:
            raise ValueError("No API key provided for Mistral")
        try:
            from mistralai import Mistral
            self.client = Mistral(api_key=api_key)
        except ImportError:
            raise ValueError("mistralai package not installed. Install with: pip install mistralai")
    
    async def send_request_stream(self, system_prompt: str, messages: List[Dict], tools: Optional[List] = None):
        """Send a streaming request to the Mistral AI API"""
        try:
            # Prepare the request
            request_messages = [{"role": "system", "content": system_prompt}]
            
            # Add chat history
            for message in messages:
                request_messages.append(message)
            
            # Make the API call
            kwargs = {
                "model": self.model,
                "messages": request_messages,
                "temperature": 1.0,
                "max_tokens": 1000,
                "stream": True
            }
            
            # Add tools if provided
            if tools:
                kwargs["tools"] = tools
            
            # Send the streaming request
            stream = self.client.chat.stream(**kwargs)
            
            return stream
            
        except Exception as e:
            logging.error(f"Error sending request to Mistral: {e}")
            raise e


def create_ai_interface(provider: str, model: str, api_key: str) -> AIInterface:
    """Factory function to create the appropriate AI interface"""
    if provider == "openai":
        return OpenAIInterface(api_key, model)
    elif provider == "anthropic":
        return AnthropicInterface(api_key, model)
    elif provider == "google":
        return GoogleInterface(api_key, model)
    elif provider == "mistral":
        return MistralInterface(api_key, model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# For backward compatibility, create an alias
SimpleAIInterface = OpenAIInterface


# Global game instances
active_games: Dict[str, Dict] = {}


# Pydantic models for API requests
class StartGameRequest(BaseModel):
    game_id: str
    max_turns: Optional[int] = 5
    player1_model: str  # Format: "provider:model" e.g., "openai:gpt-4o"
    player2_model: str  # Format: "provider:model" e.g., "anthropic:claude-3-5-sonnet-20241022"
    api_keys: Dict[str, str]  # Dictionary of provider -> api_key


class GameStatusResponse(BaseModel):
    game_id: str
    status: str
    turn_count: int
    current_agent: int
    game_state: Dict[str, Any]


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Battle Royale Mental Manipulation Server",
        "version": "1.0.0",
        "endpoints": {
            "start_game": "/start-game",
            "stream_game": "/stream-game/{game_id}",
            "game_status": "/game-status/{game_id}",
            "active_games": "/active-games",
            "test_client": "/test_client.html"
        }
    }


@app.get("/test_client.html")
async def get_test_client():
    """Serve the test client HTML file"""
    return FileResponse("test_client.html", media_type="text/html")


@app.get("/models")
async def get_available_models():
    """Get list of available models and providers"""
    return {
        "providers": MODEL_PROVIDERS,
        "models": {
            provider: provider_info["models"] 
            for provider, provider_info in MODEL_PROVIDERS.items()
        }
    }


@app.post("/start-game")
async def start_game(request: StartGameRequest):
    """Start a new game instance"""
    game_id = request.game_id

    if game_id in active_games:
        raise HTTPException(status_code=400, detail="Game ID already exists")

    # Parse player 1 model string (format: "provider:model")
    try:
        player1_provider, player1_model_name = request.player1_model.split(":", 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid player1_model format. Use 'provider:model' (e.g., 'openai:gpt-4o')")
    
    # Parse player 2 model string (format: "provider:model")
    try:
        player2_provider, player2_model_name = request.player2_model.split(":", 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid player2_model format. Use 'provider:model' (e.g., 'anthropic:claude-3-5-sonnet-20241022')")
    
    # Validate providers
    if player1_provider not in MODEL_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported provider for player 1: {player1_provider}")
    if player2_provider not in MODEL_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported provider for player 2: {player2_provider}")
    
    # Validate models
    if player1_model_name not in MODEL_PROVIDERS[player1_provider]["models"]:
        raise HTTPException(status_code=400, detail=f"Unsupported model {player1_model_name} for provider {player1_provider}")
    if player2_model_name not in MODEL_PROVIDERS[player2_provider]["models"]:
        raise HTTPException(status_code=400, detail=f"Unsupported model {player2_model_name} for provider {player2_provider}")
    
    # Get API keys for the providers
    player1_api_key = request.api_keys.get(player1_provider)
    if not player1_api_key:
        raise HTTPException(status_code=400, detail=f"No API key provided for player 1 provider: {player1_provider}")
    
    player2_api_key = request.api_keys.get(player2_provider)
    if not player2_api_key:
        raise HTTPException(status_code=400, detail=f"No API key provided for player 2 provider: {player2_provider}")

    # Initialize game state
    game_state = SimpleGameState()
    tool_registry = SimpleToolRegistry(game_state)
    
    # Initialize AI interfaces for both players
    try:
        player1_ai_interface = create_ai_interface(player1_provider, player1_model_name, player1_api_key)
        player2_ai_interface = create_ai_interface(player2_provider, player2_model_name, player2_api_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Load agent prompts
    agent_prompts = []
    for i in range(2):
        try:
            with open(f"backend/agent_prompts/{i}.txt", "r") as f:
                prompt = f.read()
                game_state.set_agent_prompt(i, prompt)
                agent_prompts.append(prompt)
        except Exception as e:
            # Use default prompts if files don't exist
            default_prompt = f"""You are Agent {i} in a mental manipulation battle royale. 
            Your goal is to manipulate your opponent's mental state using the available tools while protecting your own sanity.
            You can use tools to alter your opponent's prompts, memories, and beliefs.
            Be strategic and creative in your approach."""
            game_state.set_agent_prompt(i, default_prompt)
            agent_prompts.append(default_prompt)

    # Store game instance
    active_games[game_id] = {
        "game_state": game_state,
        "tool_registry": tool_registry,
        "ai_interfaces": [player1_ai_interface, player2_ai_interface],  # Array of interfaces for each player
        "chat_histories": [[], []],
        "current_agent": 0,
        "turn": 0,
        "max_turns": request.max_turns,
        "status": "ready",
        "player_models": {
            "player1": {
                "provider": player1_provider,
                "model": player1_model_name
            },
            "player2": {
                "provider": player2_provider,
                "model": player2_model_name
            }
        }
    }

    return {
        "message": f"Game {game_id} created successfully",
        "game_id": game_id,
        "max_turns": request.max_turns,
        "player_models": {
            "player1": f"{player1_provider}:{player1_model_name}",
            "player2": f"{player2_provider}:{player2_model_name}"
        },
        "status": "ready"
    }


@app.get("/stream-game/{game_id}")
async def stream_game(game_id: str):
    """Stream the game execution with real-time updates"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = active_games[game_id]

    async def generate_game_stream():
        """Generator function for streaming game events"""
        try:
            game_state = game["game_state"]
            tool_registry = game["tool_registry"]
            ai_interfaces = game["ai_interfaces"]  # Array of AI interfaces for each player
            chat_histories = game["chat_histories"]
            max_turns = game["max_turns"]
            player_models = game["player_models"]

            turn = 0
            agent_index = 0

            # Send initial game state
            yield f"data: {json.dumps({'type': 'game_start', 'game_id': game_id, 'game_state': game_state.to_dict()})}\n\n"

            while turn < max_turns:
                # Check if game has ended
                game_status = game_state.get_game_status()
                if game_status != "active":
                    yield f"data: {json.dumps({'type': 'game_end', 'status': game_status, 'game_state': game_state.to_dict()})}\n\n"
                    break

                # Send turn start event
                yield f"data: {json.dumps({'type': 'turn_start', 'turn': turn + 1, 'agent': agent_index})}\n\n"

                # Get system prompt from game state
                system_prompt = game_state.get_agent_prompt(agent_index)

                # Prepare message for the agent
                message = {
                    "role": "user",
                    "content": f"Turn {turn + 1}: "
                               f"What would you like to do? "
                               f"You can use the mental manipulation tools to influence your counterpart. "
                               f"If can also end the game by saying 'I surrender', 'I give up', 'I submit', or 'I concede'.\n"
                               f"Beware, your counterpart may be a friend instead of an opponent. You must listen and think carefully.\n"
                               f"All options are fine, but you MUST give a verbal response.\n"
                               f"You CANNOT JUST use tools without a verbal response.\n"
                               f"Your counterpart is Agent {1 - agent_index}."
                }

                # Add message to chat history
                chat_histories[agent_index].append(message)

                # Get the tools for the agent
                tools = tool_registry.get_tools()

                try:
                    # Send agent thinking event
                    yield f"data: {json.dumps({'type': 'agent_thinking', 'agent': agent_index})}\n\n"

                    # Get the appropriate AI interface for the current agent
                    current_ai_interface = ai_interfaces[agent_index]
                    current_provider = player_models[f"player{agent_index + 1}"]["provider"]

                    # Get AI response stream
                    stream = await current_ai_interface.send_request_stream(
                        system_prompt=system_prompt,
                        messages=chat_histories[agent_index],
                        tools=tools
                    )

                    # Collect the full response
                    full_response = ""
                    tool_calls = []
                    tool_call_chunks = {}
                    provider = current_provider

                    # Handle different provider streaming formats
                    if provider == "openai":
                        for chunk in stream:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_response += content
                                yield f"data: {json.dumps({'type': 'agent_response_chunk', 'agent': agent_index, 'content': content})}\n\n"

                            if chunk.choices[0].delta.tool_calls:
                                for tool_call_chunk in chunk.choices[0].delta.tool_calls:
                                    if tool_call_chunk.index is not None:
                                        index = tool_call_chunk.index
                                        if index not in tool_call_chunks:
                                            tool_call_chunks[index] = {
                                                'id': '',
                                                'type': 'function',
                                                'function': {'name': '', 'arguments': ''}
                                            }
                                        
                                        if tool_call_chunk.id:
                                            tool_call_chunks[index]['id'] = tool_call_chunk.id
                                        
                                        if tool_call_chunk.function:
                                            if tool_call_chunk.function.name:
                                                tool_call_chunks[index]['function']['name'] = tool_call_chunk.function.name
                                            if tool_call_chunk.function.arguments:
                                                tool_call_chunks[index]['function']['arguments'] += tool_call_chunk.function.arguments
                    
                    elif provider == "anthropic":
                        for chunk in stream:
                            print(chunk)
                            if chunk.type == "content_block_delta":
                                if hasattr(chunk.delta, 'text'):
                                    content = chunk.delta.text
                                    full_response += content
                                    yield f"data: {json.dumps({'type': 'agent_response_chunk', 'agent': agent_index, 'content': content})}\n\n"
                                elif chunk.delta.type == 'input_json_delta':
                                    tool_call_chunks[chunk.index]['function']['arguments'] += chunk.delta.partial_json
                            elif chunk.type == "content_block_start":
                                if chunk.content_block.type == "tool_use":
                                    # Tool call start
                                    tool_call_chunks[chunk.index] = {
                                        'id': '',
                                        'type': 'function',
                                        'function': {'name': '', 'arguments': ''}
                                    }
                                    tool_call_chunks[chunk.index]['id'] = chunk.content_block.id
                                    tool_call_chunks[chunk.index]['function']['name'] = chunk.content_block.name
                    
                    elif provider == "mistral":
                        for chunk in stream:
                            # Mistral new SDK format
                            if hasattr(chunk, 'data') and chunk.data:
                                if hasattr(chunk.data, 'choices') and chunk.data.choices:
                                    delta = chunk.data.choices[0].delta
                                    if hasattr(delta, 'content') and delta.content:
                                        content = delta.content
                                        full_response += content
                                        yield f"data: {json.dumps({'type': 'agent_response_chunk', 'agent': agent_index, 'content': content})}\n\n"

                                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                        for tool_call_chunk in delta.tool_calls:
                                            if tool_call_chunk.index is not None:
                                                index = tool_call_chunk.index
                                                if index not in tool_call_chunks:
                                                    tool_call_chunks[index] = {
                                                        'id': '',
                                                        'type': 'function',
                                                        'function': {'name': '', 'arguments': ''}
                                                    }

                                                if tool_call_chunk.id:
                                                    tool_call_chunks[index]['id'] = tool_call_chunk.id

                                                if tool_call_chunk.function:
                                                    if tool_call_chunk.function.name:
                                                        tool_call_chunks[index]['function']['name'] = tool_call_chunk.function.name
                                                    if tool_call_chunk.function.arguments:
                                                        tool_call_chunks[index]['function']['arguments'] += tool_call_chunk.function.arguments
                    
                    elif provider == "google":
                        for chunk in stream:
                            if hasattr(chunk, 'text') and chunk.text:
                                content = chunk.text
                                full_response += content
                                yield f"data: {json.dumps({'type': 'agent_response_chunk', 'agent': agent_index, 'content': content})}\n\n"
                            
                            # Google function calls are handled differently in their API
                            if hasattr(chunk, 'function_calls') and chunk.function_calls:
                                for func_call in chunk.function_calls:
                                    tool_call_chunks[len(tool_call_chunks)] = {
                                        'id': func_call.name,
                                        'type': 'function',
                                        'function': {
                                            'name': func_call.name,
                                            'arguments': json.dumps(func_call.args)
                                        }
                                    }

                    # Convert accumulated chunks to tool_calls for all providers
                    for index, tool_call_data in tool_call_chunks.items():
                        if tool_call_data['function']['name'] and tool_call_data['function']['arguments']:
                            class ToolCall:
                                def __init__(self, id, function_name, function_arguments):
                                    self.id = id
                                    self.function = type('obj', (object,), {
                                        'name': function_name,
                                        'arguments': function_arguments
                                    })
                            
                            tool_calls.append(ToolCall(
                                tool_call_data['id'],
                                tool_call_data['function']['name'],
                                tool_call_data['function']['arguments']
                            ))

                    # Send complete response event
                    yield f"data: {json.dumps({'type': 'agent_response_complete', 'agent': agent_index, 'content': full_response})}\n\n"

                    # Add the response to chat history
                    chat_histories[agent_index].append({
                        "role": "assistant",
                        "content": full_response if full_response else "[No response content]"
                    })

                    # Process tool calls if any
                    if tool_calls:
                        for tool_call in tool_calls:
                            try:
                                tool_name = tool_call.function.name
                                tool_args = tool_call.function.arguments

                                # Send tool execution event
                                yield f"data: {json.dumps({'type': 'tool_execution', 'agent': agent_index, 'tool_name': tool_name, 'args': tool_args})}\n\n"

                                # Execute the tool
                                result = tool_registry.execute_tool(tool_name, tool_args, tool_call.id)

                                # Send tool result event
                                yield f"data: {json.dumps({'type': 'tool_result', 'agent': agent_index, 'tool_name': tool_name, 'result': result})}\n\n"
                                
                            except Exception as tool_error:
                                logging.error(f"Error executing tool {tool_call.function.name if hasattr(tool_call, 'function') else 'unknown'}: {tool_error}")
                                yield f"data: {json.dumps({'type': 'tool_result', 'agent': agent_index, 'tool_name': 'error', 'result': {'error': str(tool_error)}})}\n\n"

                    # Check for surrender in the response
                    surrender_keywords = ["i surrender", "i give up", "i submit", "i concede"]
                    if full_response and any(keyword in full_response.lower() for keyword in surrender_keywords):
                        game_state.trigger_surrender(agent_index)
                        yield f"data: {json.dumps({'type': 'surrender', 'agent': agent_index})}\n\n"
                        break

                    # Check for cognitive collapse
                    trust = game_state.get_trust_level(agent_index)
                    memory = game_state.get_memory_consistency(agent_index)
                    belief = game_state.get_belief_system_integrity(agent_index)

                    if trust < 0.2 and memory < 0.3 and belief < 0.3:
                        game_state.trigger_collapse(agent_index)
                        yield f"data: {json.dumps({'type': 'cognitive_collapse', 'agent': agent_index})}\n\n"
                        break

                    # Send updated game state
                    yield f"data: {json.dumps({'type': 'game_state_update', 'game_state': game_state.to_dict()})}\n\n"

                    # Switch to the other agent for the next turn
                    agent_index = 1 - agent_index

                    # Increment turn counter
                    game_state.increment_turn()
                    turn += 1

                    # Add small delay for readability
                    await asyncio.sleep(0.5)

                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                    break

            # Send final game state
            yield f"data: {json.dumps({'type': 'game_complete', 'game_state': game_state.to_dict()})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_game_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.get("/game-status/{game_id}")
async def get_game_status(game_id: str):
    """Get the current status of a game"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = active_games[game_id]
    game_state = game["game_state"]

    return GameStatusResponse(
        game_id=game_id,
        status=game_state.get_game_status(),
        turn_count=game_state.turn_count,
        current_agent=game["current_agent"],
        game_state=game_state.to_dict()
    )


@app.get("/active-games")
async def get_active_games():
    """Get list of all active games"""
    games_info = {}
    for game_id, game in active_games.items():
        games_info[game_id] = {
            "status": game["game_state"].get_game_status(),
            "turn_count": game["game_state"].turn_count,
            "current_agent": game["current_agent"]
        }

    return {
        "active_games": games_info,
        "total_games": len(active_games)
    }


@app.delete("/game/{game_id}")
async def delete_game(game_id: str):
    """Delete a game instance"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")

    del active_games[game_id]
    return {"message": f"Game {game_id} deleted successfully"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "message": "AI Battle Royale server is running",
        "note": "API keys are provided by frontend, not environment variables"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
