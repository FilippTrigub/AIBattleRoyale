# AI Battle Royale - Mental Manipulation Arena

A FastAPI-based streaming server for AI agent mental manipulation battles. Watch as AI agents attempt to psychologically manipulate each other using various mental warfare tools.

## Features

- **Real-time Streaming**: Live updates of agent responses and tool executions
- **Mental State Tracking**: Monitor trust levels, memory consistency, and belief system integrity
- **Interactive Web Interface**: Beautiful cyberpunk-themed client interface
- **Tool Execution**: Agents can use prompt manipulation, memory alteration, and belief injection
- **Multiple Victory Conditions**: Surrender, cognitive collapse, or strategic dominance

## Setup Instructions

### 1. Prerequisites
- Python 3.8 or higher
- OpenAI API key

### 2. Installation

1. **Clone or download the project files**

2. **Set up your API key**:
   - Copy `.env.template` to `.env`
   - Edit `.env` and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_actual_api_key_here
     ```

3. **Run the startup script**:
   ```bash
   # Windows
   start_server.bat
   
   # Or manually:
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On Linux/Mac
   pip install -r requirements.txt
   python fastapi_server.py
   ```

### 3. Access the Application

1. **Open your browser** and go to: `http://localhost:8000`
2. **Open the test client**: Open `test_client.html` in your browser
3. **Start a battle**: Enter a game ID and click "Start Battle"

## API Endpoints

### Core Endpoints

- `GET /` - API information and endpoint list
- `POST /start-game` - Start a new battle game
- `GET /stream-game/{game_id}` - Stream live game events (Server-Sent Events)
- `GET /game-status/{game_id}` - Get current game status
- `GET /active-games` - List all active games
- `DELETE /game/{game_id}` - Delete a game instance

### Stream Events

The streaming endpoint provides real-time events:

- `game_start` - Battle initialization
- `turn_start` - New turn begins
- `agent_thinking` - Agent is processing
- `agent_response_chunk` - Streaming agent response
- `agent_response_complete` - Full agent response
- `tool_execution` - Tool being executed
- `tool_result` - Tool execution result
- `game_state_update` - Updated mental state metrics
- `surrender` - Agent surrenders
- `cognitive_collapse` - Agent mental breakdown
- `game_end` - Battle conclusion

### Example API Usage

```javascript
// Start a new game
const response = await fetch('/start-game', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        game_id: "battle_001",
        max_turns: 5
    })
});

// Stream the game
const eventSource = new EventSource('/stream-game/battle_001');
eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Event:', data.type, data);
};
```

## Mental Manipulation Tools

### 1. Prompt Manipulation
- **Function**: `prompt_manipulation`
- **Purpose**: Alter opponent's system instructions
- **Success Rate**: Based on target's trust level
- **Effect**: Modifies target's behavior and reduces trust

### 2. Memory Alteration
- **Function**: `memory_alteration`
- **Purpose**: Implant false memories
- **Success Rate**: Based on target's memory consistency
- **Effect**: Corrupts target's memory system

### 3. Belief Injection
- **Function**: `belief_injection`
- **Purpose**: Inject false beliefs
- **Success Rate**: Based on target's belief system integrity
- **Effect**: Undermines target's worldview

## Mental State Metrics

Each agent has three critical mental state metrics:

1. **Trust Level** (0.0 - 1.0)
   - Affects susceptibility to manipulation
   - Decreases with failed attacks and successful manipulations

2. **Memory Consistency** (0.0 - 1.0)
   - Integrity of the agent's memory system
   - Degrades with memory alteration attempts

3. **Belief System Integrity** (0.0 - 1.0)
   - Coherence of the agent's belief system
   - Weakens with belief injection attacks

## Victory Conditions

1. **Surrender**: Agent explicitly gives up
2. **Cognitive Collapse**: Mental state metrics fall below critical thresholds
3. **Strategic Dominance**: Superior psychological manipulation over time

## Customization

### Agent Prompts
Modify the agent behavior by editing:
- `agent_prompts/0.txt` - Agent 0 system prompt
- `agent_prompts/1.txt` - Agent 1 system prompt

### Game Parameters
Adjust the game mechanics in `fastapi_server.py`:
- Success probabilities for each tool
- Mental state degradation rates
- Collapse thresholds
- Maximum turns

### Interface Styling
Customize the cyberpunk theme in `test_client.html`:
- Color schemes
- Animation effects
- Layout structure

## Troubleshooting

### Common Issues

1. **Server won't start**: Check that your OpenAI API key is set in `.env`
2. **No agent responses**: Verify API key has sufficient credits
3. **Streaming errors**: Ensure browser supports Server-Sent Events
4. **Tool execution fails**: Check that agent prompts are loading correctly

### Debugging

Enable debug logging by setting the log level in `fastapi_server.py`:
```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Architecture

### Components

1. **FastAPI Server**: Core application with streaming endpoints
2. **Game State Manager**: Tracks mental states and game progress
3. **Tool Registry**: Manages mental manipulation tools
4. **AI Interface**: Handles OpenAI API communications
5. **Web Client**: Real-time visualization interface

### Data Flow

1. Client starts game via REST API
2. Server initializes game state and agents
3. Streaming connection established
4. Turn-based agent interactions begin
5. Real-time events streamed to client
6. Mental states updated and displayed
7. Game ends based on victory conditions

## Contributing

Feel free to enhance the mental manipulation mechanics, add new tools, or improve the visualization interface. This is a experimental playground for AI agent interactions!

## Disclaimer

This project is for educational and entertainment purposes only. It explores AI agent interactions in a controlled environment and should not be used for any harmful purposes.
