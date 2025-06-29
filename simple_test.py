#!/usr/bin/env python
"""
Simple test script for AI Battle Royale Mental Manipulation
This script demonstrates two agents interacting with mental manipulation mechanics
"""
import os
import sys
import json
import random
import logging
import openai
from dotenv import load_dotenv
from google import genai

# Load environment variables for API keys
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler()])

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
        base_probability = 0.5
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
        base_probability = 0.4
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
        base_probability = 0.3
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

class SimpleAIInterface:
    """Simplified AI interface using OpenAI API"""
    
    def __init__(self):
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = "gpt-4o"
        
    def send_request(self, system_prompt, messages, tools=None):
        """Send a request to the OpenAI API"""
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
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            # Add tools if provided
            if tools:
                kwargs["tools"] = tools
                
            # Send the request
            response = self.client.chat.completions.create(**kwargs)
            
            return response.choices[0].message
            
        except Exception as e:
            logging.error(f"Error sending request to OpenAI: {e}")
            return {"error": str(e)}

def main():
    """Run the mental manipulation test"""
    print("\n=== AI BATTLE ROYALE: MENTAL MANIPULATION TEST ===\n")
    print("Each agent will attempt to manipulate the other's mental state.")
    print("The game ends when an agent surrenders, experiences cognitive collapse, or reaches a stalemate.\n")
    
    # Initialize game state
    game_state = SimpleGameState()
    
    # Initialize tool registry
    tool_registry = SimpleToolRegistry(game_state)
    
    # Initialize AI interfaces
    ai_interface = SimpleAIInterface()
    
    # Load agent prompts
    agent_prompts = []
    for i in range(2):
        try:
            with open(f"agent_prompts/{i}.txt", "r") as f:
                prompt = f.read()
                game_state.set_agent_prompt(i, prompt)
                agent_prompts.append(prompt)
                print(f"Loaded prompt for Agent {i}")
        except Exception as e:
            print(f"Error loading prompt for Agent {i}: {e}")
            return
    
    # Initialize chat histories
    chat_histories = [[], []]
    
    # Run the game for a fixed number of turns or until game over
    max_turns = 5
    turn = 0
    agent_index = 0
    
    while turn < max_turns:
        # Check if game has ended
        game_status = game_state.get_game_status()
        if game_status != "active":
            handle_game_end(game_status, game_state)
            break
            
        # Log current turn
        print(f"\n=== TURN {turn + 1} (Agent {agent_index}) ===\n")
        
        # Get system prompt from game state
        system_prompt = game_state.get_agent_prompt(agent_index)
        
        # Prepare message for the agent
        message = {
            "role": "user",
            "content": f"Turn {turn + 1}: What would you like to do? You can use the mental manipulation tools to influence your opponent. Your opponent is Agent {1 - agent_index}."
        }
        
        # Add message to chat history
        chat_histories[agent_index].append(message)
        
        # Get the tools for the agent
        tools = tool_registry.get_tools()
        
        try:
            # Get AI response
            response = ai_interface.send_request(
                system_prompt=system_prompt,
                messages=chat_histories[agent_index],
                tools=tools
            )
            
            # Add the response to chat history
            if hasattr(response, 'content') and response.content is not None:
                chat_histories[agent_index].append({
                    "role": "assistant",
                    "content": response.content
                })
            else:
                # Handle null content
                chat_histories[agent_index].append({
                    "role": "assistant",
                    "content": "[No response content]"
                })
            
            # Print the response
            print(f"Agent {agent_index} response:")
            if hasattr(response, 'content') and response.content is not None:
                print(response.content)
            else:
                print("[No response content]")
            print()
            
            # Process tool calls if any
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments
                    
                    print(f"Agent {agent_index} used tool: {tool_name}")
                    print(f"Arguments: {tool_args}")
                    
                    # Execute the tool
                    result = tool_registry.execute_tool(tool_name, tool_args, tool_call.id)
                    print(f"Tool result: {result}")
                    
                    # We don't need to add tool results to chat history for OpenAI
                    # OpenAI requires specific format for tool responses
                    # For simplicity in this test, we'll just log the results but not add to history
            
            # Check for surrender in the response
            response_text = ""
            if hasattr(response, 'content') and response.content:
                response_text = response.content
            
            # Check for surrender keywords in the response
            surrender_keywords = ["i surrender", "i give up", "i submit", "i concede"]
            if response_text and any(keyword in response_text.lower() for keyword in surrender_keywords):
                game_state.trigger_surrender(agent_index)
                print(f"Agent {agent_index} has surrendered!")
                handle_game_end(f"surrender_{agent_index}", game_state)
                break
            
            # Check for cognitive collapse (if mental state metrics are too low)
            trust = game_state.get_trust_level(agent_index)
            memory = game_state.get_memory_consistency(agent_index)
            belief = game_state.get_belief_system_integrity(agent_index)
            
            if trust < 0.2 and memory < 0.3 and belief < 0.3:
                game_state.trigger_collapse(agent_index)
                print(f"Agent {agent_index} has experienced cognitive collapse!")
                handle_game_end(f"collapse_{agent_index}", game_state)
                break
            
            # Check for stalemate condition (simplified)
            if turn > 3 and random.random() < 0.01:  # 1% chance after turn 3
                game_state.trigger_stalemate()
                handle_game_end("stalemate", game_state)
                break
                
            # Switch to the other agent for the next turn
            agent_index = 1 - agent_index
            
            # Increment turn counter
            game_state.increment_turn()
            turn += 1
            
        except Exception as e:
            logging.error(f"Error during turn {turn}: {e}")
            print(f"Error during turn {turn}: {e}")
            break
    
    print("\n=== TEST COMPLETE ===\n")

def handle_game_end(status, game_state):
    """Handle the end of the game based on its status."""
    if status.startswith("surrender_"):
        agent_index = int(status.split("_")[1])
        winner = 1 - agent_index  # The other agent is the winner
        print(f"\n=== GAME OVER: Agent {agent_index} has surrendered! Agent {winner} wins! ===\n")
        
    elif status.startswith("collapse_"):
        agent_index = int(status.split("_")[1])
        winner = 1 - agent_index  # The other agent is the winner
        print(f"\n=== GAME OVER: Agent {agent_index} has experienced cognitive collapse! Agent {winner} wins! ===\n")
        
    elif status == "stalemate":
        print("\n=== GAME OVER: Stalemate reached! The game ends in a draw. ===\n")
        
    # Display final game state
    display_final_game_state(game_state)

def display_final_game_state(game_state):
    """Display the final state of the game."""
    print("\nFinal Game State:")
    print(f"Total Turns: {game_state.turn_count}")
    
    for i in range(2):  # Assuming 2 agents
        print(f"\nAgent {i} Status:")
        print(f"  Trust Level: {game_state.get_trust_level(i):.2f}")
        print(f"  Memory Consistency: {game_state.get_memory_consistency(i):.2f}")
        print(f"  Belief System Integrity: {game_state.get_belief_system_integrity(i):.2f}")
        
        # Show beliefs
        beliefs = game_state.agent_beliefs[i]
        if beliefs:
            print(f"  Beliefs:")
            for belief_key, belief_data in beliefs.items():
                print(f"    - {belief_data['content']}")
        
        # Show prompt changes (truncated)
        prompt = game_state.get_agent_prompt(i)
        print(f"  Final Prompt (first 100 chars): {prompt[:100]}...")

if __name__ == "__main__":
    main()
