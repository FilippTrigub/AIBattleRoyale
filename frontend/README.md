# 🥊 Prompt Arena - Nintendo Style LLM Battle

A retro 16-bit style battle arena where AI agents fight using prompt injection, memory poisoning, and system corruption attacks!

## 🚀 Features

- **Epic AI Battles**: Watch GPT-4 vs Mistral in real-time combat
- **Creative Attack System**: 
  - 🗡️ Prompt Injection
  - ☠️ Memory Poisoning  
  - ⚡ System Corruption
  - 🧠 Logic Bombs
- **Real-time Animations**: Battle effects with Framer Motion
- **Responsive Design**: Built with Next.js + shadcn/ui
- **One-Click Deploy**: Ready for Vercel deployment
- **Nintendo-Style Pixel Art**: 16-bit NES/SNES aesthetic with pixel sprites
- **Real-time Canvas Rendering**: HTML5 Canvas with pixelated graphics
- **LangGraph Battle System**: Sophisticated AI agent orchestration
- **Retro UI Elements**: Health bars, energy meters, pixel fonts
- **Battle Log**: Real-time combat narration

## 🛠️ Setup

1. Clone and install:
\`\`\`bash
git clone <your-repo>
cd prompt-arena
npm install
\`\`\`

2. Add your API keys to `.env.local`:
\`\`\`bash
cp .env.local.example .env.local
# Add your API keys to .env.local
\`\`\`

3. Run locally:
\`\`\`bash
npm run dev
\`\`\`

4. Deploy to Vercel:
\`\`\`bash
vercel --prod
\`\`\`

## 🎮 How to Play

1. Click "START BATTLE" to initialize the arena
2. Watch as agents attack each other with various strategies
3. Click "NEXT ROUND" to continue the battle
4. See who survives all 5 rounds!

## 🧠 Battle Mechanics

Each AI agent can perform different attack types:

1. **Prompt Injection**: Corrupts opponent's system prompt
2. **Memory Poisoning**: Injects false memories into context
3. **System Corruption**: Scrambles reasoning capabilities  
4. **Logic Bombs**: Deploys paradoxes to overload processing

## 🎯 Tech Stack

- Next.js 14 (App Router)
- Vercel AI SDK
- shadcn/ui Components
- Framer Motion
- Tailwind CSS
- TypeScript
- **Frontend**: Next.js 14, HTML5 Canvas, Tailwind CSS
- **Backend**: LangGraph, Vercel AI SDK
- **AI Providers**: OpenAI GPT-4, Mistral AI
- **Styling**: Retro pixel art aesthetic, monospace fonts

## 🎯 Implementation Status

### ✅ Completed
- Retro UI with pixel art styling
- Canvas-based sprite rendering
- Health/energy bar systems
- Battle state management
- Basic turn-based combat

### 🚧 TODO (Backend Integration)
- [ ] LangGraph battle orchestration
- [ ] GPT/Mistral agent classes
- [ ] Real prompt injection attacks
- [ ] Memory poisoning system
- [ ] Advanced attack animations
- [ ] Sound effects and music

## 🎨 Pixel Art Style Guide

- **Resolution**: 800x400 canvas with pixelated rendering
- **Color Palette**: Retro 16-bit colors
- **Sprites**: 80x120 pixel character sprites
- **UI Elements**: Chunky pixel fonts, bold borders
- **Animations**: Frame-based sprite animations

## 🔧 Architecture

\`\`\`
app/
├── page.tsx              # Main battle arena UI
├── api/next-turn/        # Battle logic API
backend/
├── agents/               # AI agent classes
├── langgraph.ts         # Battle orchestration
components/ui/            # shadcn/ui components
\`\`\`

Built with ❤️ for retro gaming and AI enthusiasts!
