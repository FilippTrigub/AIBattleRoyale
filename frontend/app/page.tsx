"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Target, Zap, Settings, Volume2, VolumeX } from "lucide-react"

// --- State Interfaces ---
interface MentalState {
  trust: number
  memory: number
  belief: number
}

interface AgentState {
  mentalState: MentalState
  beliefs: string[]
}

interface SpeechBubble {
  message: string
  isVisible: boolean
  timestamp: number
  type: 'tool' | 'success' | 'failure' | 'detection'
  toolName?: string
}

interface AgentSpeechBubbles {
  tool: SpeechBubble
  result: SpeechBubble
  detection: SpeechBubble
}

interface LogoAnimation {
  x: number
  y: number
  rotation: number
  scale: number
  isAnimating: boolean
  animationType: string
  animationProgress: number
}

interface AudioState {
  isEnabled: boolean
  volume: number
  audioContext: AudioContext | null
  sounds: Record<string, AudioBuffer | null>
}

interface BattleState {
  isActive: boolean
  isStreaming: boolean
  winner: string | null
  reason: string
  battleLog: string[]
  agents: [AgentState, AgentState]
  turn: number
  currentAgent: number | null
  logoAnimations: [LogoAnimation, LogoAnimation]
  speechBubbles: [AgentSpeechBubbles, AgentSpeechBubbles]
}

const initialAgentState: AgentState = {
  mentalState: { trust: 1, memory: 1, belief: 1 },
  beliefs: [],
}

const initialSpeechBubble: SpeechBubble = {
  message: '',
  isVisible: false,
  timestamp: 0,
  type: 'tool',
  toolName: undefined,
}

const initialAgentSpeechBubbles: AgentSpeechBubbles = {
  tool: JSON.parse(JSON.stringify(initialSpeechBubble)),
  result: JSON.parse(JSON.stringify(initialSpeechBubble)),
  detection: JSON.parse(JSON.stringify(initialSpeechBubble)),
}

const initialLogoAnimation: LogoAnimation = {
  x: 0,
  y: 0,
  rotation: 0,
  scale: 1,
  isAnimating: false,
  animationType: 'idle',
  animationProgress: 0,
}

const initialAudioState: AudioState = {
  isEnabled: true,
  volume: 0.5,
  audioContext: null,
  sounds: {
    'tool-use': null,
    'success': null,
    'failure': null,
    'detection': null,
  },
}

const initialBattleState: BattleState = {
  isActive: false,
  isStreaming: false,
  winner: null,
  reason: "",
  battleLog: ["> WAITING FOR BATTLE..."],
  agents: [JSON.parse(JSON.stringify(initialAgentState)), JSON.parse(JSON.stringify(initialAgentState))],
  turn: 0,
  currentAgent: null,
  logoAnimations: [JSON.parse(JSON.stringify(initialLogoAnimation)), JSON.parse(JSON.stringify(initialLogoAnimation))],
  speechBubbles: [JSON.parse(JSON.stringify(initialAgentSpeechBubbles)), JSON.parse(JSON.stringify(initialAgentSpeechBubbles))],
}

// --- Mental State Bar Component ---
const MentalStateBar = ({ label, value, color }: { label: string; value: number; color: string }) => (
  <div className="mb-2">
    <div className="flex justify-between text-[10px] mb-1 text-white">
      <span>{label}</span>
      <span>{(value * 100).toFixed(0)}%</span>
    </div>
    <div className="w-full bg-gray-800 h-3 border-2 border-gray-600 relative overflow-hidden">
      <div
        className="h-full transition-all duration-500 ease-out"
        style={{
          width: `${value * 100}%`,
          backgroundColor: color,
          boxShadow: `inset 0 1px 0 rgba(255,255,255,0.3), inset 0 -1px 0 rgba(0,0,0,0.3)`
        }}
      />
      {/* Pixelated overlay effect */}
      <div className="absolute inset-0 opacity-20"
           style={{
             backgroundImage: `linear-gradient(90deg, transparent 50%, rgba(255,255,255,0.1) 50%)`,
             backgroundSize: '4px 4px'
           }}
      />
    </div>
  </div>
)

const detectServerURL = () => {
  // In development, check if we're running on port 3000 (Next.js dev server)
  // In production (Docker), use nginx proxy route
  if (typeof window !== 'undefined') {
    const isDevMode = window.location.port === '3000' && window.location.hostname === 'localhost'
    if (isDevMode) {
      // Development: connect directly to backend
      return 'http://localhost:8000'
    } else {
      // Production/Docker: use nginx proxy
      return '/api'
    }
  }
  // Fallback for SSR
  return '/api'
}

const serverURL = detectServerURL()

export default function AgentBattleArena() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const eventSourceRef = useRef<EventSource | null>(null)
  const [battleState, setBattleState] = useState<BattleState>(initialBattleState)
  const [audioState, setAudioState] = useState<AudioState>(initialAudioState)
  const logContainerRef = useRef<HTMLDivElement>(null)
  const logoImagesRef = useRef<Record<string, HTMLImageElement>>({})
  const animationFrameRef = useRef<number | null>(null)

  // Game configuration state
  const [maxTurns, setMaxTurns] = useState<number>(5)
  const [selectedPlayer1Model, setSelectedPlayer1Model] = useState<string>("openai:gpt-4o")
  const [selectedPlayer2Model, setSelectedPlayer2Model] = useState<string>("anthropic:claude-3-5-sonnet-20241022")
  const [apiKeys, setApiKeys] = useState<Record<string, string>>({
    openai: "",
    anthropic: "",
    google: "",
    mistral: ""
  })
  const [showConfig, setShowConfig] = useState<boolean>(true)
  const [availableModels, setAvailableModels] = useState<any>({
    providers: {
      openai: {
        name: "OpenAI",
        models: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
      },
      anthropic: {
        name: "Anthropic",
        models: ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
      },
      google: {
        name: "Google Gemini",
        models: ["gemini-2.5-flash", "gemini-2.0-flash-001", "gemini-1.5-pro", "gemini-1.5-flash"]
      },
      mistral: {
        name: "Mistral AI",
        models: ["mistral-large-latest", "mistral-small-latest", "mistral-medium-2505", "open-mistral-7b"]
      }
    }
  })

  // Audio System Functions
  const initializeAudioContext = useCallback(async () => {
    if (!audioState.audioContext) {
      try {
        const context = new (window.AudioContext || (window as any).webkitAudioContext)()
        setAudioState(prev => ({ ...prev, audioContext: context }))
        console.log('Audio context initialized')
        return context
      } catch (error) {
        console.warn('Failed to initialize audio context:', error)
        return null
      }
    }
    return audioState.audioContext
  }, [audioState.audioContext])

  const loadSound = useCallback(async (soundName: string, url: string, context: AudioContext): Promise<AudioBuffer | null> => {
    try {
      console.log(`Loading sound: ${soundName} from ${url}`)
      const response = await fetch(url)
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const arrayBuffer = await response.arrayBuffer()
      const audioBuffer = await context.decodeAudioData(arrayBuffer)
      console.log(`Sound loaded successfully: ${soundName}`)
      return audioBuffer
    } catch (error) {
      console.warn(`Failed to load sound ${soundName}:`, error)
      return null
    }
  }, [])

  const loadAllSounds = useCallback(async () => {
    const context = await initializeAudioContext()
    if (!context) return

    const soundFiles = {
      'tool-use': '/sounds/tool-use.wav',
      'success': '/sounds/success.wav', 
      'failure': '/sounds/failure.wav',
      'detection': '/sounds/detection.wav',
    }

    console.log('Loading all sound files...')
    const loadPromises = Object.entries(soundFiles).map(async ([name, url]) => {
      const buffer = await loadSound(name, url, context)
      return { name, buffer }
    })

    const results = await Promise.all(loadPromises)
    const newSounds: Record<string, AudioBuffer | null> = {}
    results.forEach(({ name, buffer }) => {
      newSounds[name] = buffer
    })

    setAudioState(prev => ({
      ...prev,
      sounds: newSounds
    }))
    
    console.log('Sound loading complete:', Object.keys(newSounds).filter(key => newSounds[key] !== null))
  }, [initializeAudioContext, loadSound])

  const playSound = useCallback((soundName: string) => {
    if (!audioState.isEnabled || !audioState.audioContext || !audioState.sounds[soundName]) {
      if (!audioState.isEnabled) {
        console.log(`Audio disabled, skipping sound: ${soundName}`)
      } else if (!audioState.audioContext) {
        console.warn(`No audio context available for sound: ${soundName}`)
      } else {
        console.warn(`Sound not loaded: ${soundName}`)
      }
      return
    }

    try {
      const source = audioState.audioContext.createBufferSource()
      const gainNode = audioState.audioContext.createGain()
      
      source.buffer = audioState.sounds[soundName]
      gainNode.gain.value = audioState.volume
      
      source.connect(gainNode)
      gainNode.connect(audioState.audioContext.destination)
      
      source.start(0)
      console.log(`Playing sound: ${soundName} at volume ${audioState.volume}`)
    } catch (error) {
      console.error(`Failed to play sound ${soundName}:`, error)
    }
  }, [audioState])

  const toggleAudio = useCallback(() => {
    setAudioState(prev => ({ ...prev, isEnabled: !prev.isEnabled }))
  }, [])

  const setVolume = useCallback((volume: number) => {
    const clampedVolume = Math.max(0, Math.min(1, volume))
    setAudioState(prev => ({ ...prev, volume: clampedVolume }))
  }, [])

  // Auto-scroll battle log
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [battleState.battleLog])

  // Initialize audio on component mount
  useEffect(() => {
    loadAllSounds()
  }, [loadAllSounds])

  // Load provider logos
  useEffect(() => {
    const loadLogos = async () => {
      console.log('Loading provider logos...')
      const providers = ['openai', 'anthropic', 'google', 'mistral']
      const logoPromises = providers.map(provider => {
        return new Promise<void>((resolve) => {
          const img = new Image()
          img.onload = () => {
            logoImagesRef.current[provider] = img
            console.log(`Logo loaded for ${provider}:`, img.src)
            resolve()
          }
          img.onerror = () => {
            console.warn(`Failed to load logo for ${provider}`)
            resolve()
          }
          img.src = `/images/${provider}.png`
          console.log(`Attempting to load logo: ${img.src}`)
        })
      })
      await Promise.all(logoPromises)
      console.log('All logos loaded:', Object.keys(logoImagesRef.current))
    }
    loadLogos()
  }, [])

  // Fetch available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        console.log('Fetching models from server...')
        const response = await fetch(`${serverURL}/models`)
        if (response.ok) {
          const data = await response.json()
          console.log('Models loaded from server:', data)
          setAvailableModels(data)
        } else {
          console.warn('Failed to fetch models from server, using fallback')
        }
      } catch (error) {
        console.warn("Failed to fetch available models from server, using fallback:", error)
      }
    }
    fetchModels()
  }, [])

  // Helper functions
  const getRequiredApiKeys = () => {
    const player1Provider = selectedPlayer1Model.split(':')[0]
    const player2Provider = selectedPlayer2Model.split(':')[0]
    const providers = [player1Provider]
    if (player2Provider !== player1Provider) {
      providers.push(player2Provider)
    }
    return providers
  }

  const getProviderDisplayName = (provider: string) => {
    return availableModels?.providers?.[provider]?.name || provider
  }

  const updateApiKey = (provider: string, value: string) => {
    setApiKeys(prev => ({ ...prev, [provider]: value }))
  }

  // Speech bubble functions with audio
  const showSpeechBubble = (agentIndex: number, message: string, type: SpeechBubble['type'], toolName?: string) => {
    console.log(`Showing speech bubble for agent ${agentIndex}: ${message} (${type})`)
    const timestamp = Date.now()
    
    // Play appropriate sound based on type
    switch (type) {
      case 'tool':
        playSound('tool-use')
        break
      case 'success':
        playSound('success')
        break
      case 'failure':
        playSound('failure')
        break
      case 'detection':
        playSound('detection')
        break
    }
    
    setBattleState(prev => {
      const newBubbles = [...prev.speechBubbles] as [AgentSpeechBubbles, AgentSpeechBubbles]
      
      // Determine which bubble slot to use based on type
      let bubbleSlot: keyof AgentSpeechBubbles
      if (type === 'tool') {
        bubbleSlot = 'tool'
      } else if (type === 'detection') {
        bubbleSlot = 'detection'
      } else {
        bubbleSlot = 'result' // success or failure
      }
      
      newBubbles[agentIndex][bubbleSlot] = {
        message,
        isVisible: true,
        timestamp,
        type,
        toolName,
      }
      
      return {
        ...prev,
        speechBubbles: newBubbles
      }
    })
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
      setBattleState(prev => {
        const newBubbles = [...prev.speechBubbles] as [AgentSpeechBubbles, AgentSpeechBubbles]
        
        // Find and hide the bubble with matching timestamp
        Object.keys(newBubbles[agentIndex]).forEach(key => {
          const bubbleKey = key as keyof AgentSpeechBubbles
          if (newBubbles[agentIndex][bubbleKey].timestamp === timestamp) {
            newBubbles[agentIndex][bubbleKey].isVisible = false
          }
        })
        
        return {
          ...prev,
          speechBubbles: newBubbles
        }
      })
    }, 3000)
  }

  // Animation logic - simplified without complex refs
  const startLogoAnimation = (agentIndex: number) => {
    console.log(`Starting animation for agent ${agentIndex}`)
    const animationTypes = ['bounce', 'float', 'wobble', 'spin', 'pulse']
    const randomType = animationTypes[Math.floor(Math.random() * animationTypes.length)]
    
    setBattleState(prev => {
      const newAnimations = [...prev.logoAnimations] as [LogoAnimation, LogoAnimation]
      newAnimations[agentIndex] = {
        ...newAnimations[agentIndex],
        isAnimating: true,
        animationType: randomType,
        animationProgress: 0,
      }
      console.log(`Animation started: ${randomType} for agent ${agentIndex}`)
      return {
        ...prev,
        logoAnimations: newAnimations
      }
    })
  }

  // Simplified animation system using useCallback
  const updateAnimations = useCallback(() => {
    setBattleState(prev => {
      const newAnimations = [...prev.logoAnimations] as [LogoAnimation, LogoAnimation]
      let hasChanges = false
      
      for (let i = 0; i < 2; i++) {
        if (newAnimations[i].isAnimating) {
          hasChanges = true
          newAnimations[i].animationProgress += 0.05
          
          const progress = newAnimations[i].animationProgress
          const type = newAnimations[i].animationType
          
          switch (type) {
            case 'bounce':
              newAnimations[i].y = Math.sin(progress * Math.PI * 6) * 30 * (1 - progress / 2)
              newAnimations[i].x = (Math.random() - 0.5) * 20 * (1 - progress / 2)
              break
            case 'float':
              newAnimations[i].y = Math.sin(progress * Math.PI * 4) * 20
              newAnimations[i].x = Math.cos(progress * Math.PI * 3) * 15
              break
            case 'wobble':
              newAnimations[i].rotation = Math.sin(progress * Math.PI * 8) * 15 * (1 - progress / 3)
              newAnimations[i].x = Math.sin(progress * Math.PI * 10) * 10
              break
            case 'spin':
              newAnimations[i].rotation = progress * 360 * 2
              newAnimations[i].scale = 1 + Math.sin(progress * Math.PI * 4) * 0.2
              break
            case 'pulse':
              newAnimations[i].scale = 1 + Math.sin(progress * Math.PI * 8) * 0.3
              break
          }
          
          if (progress >= 2) {
            // Return to resting position
            newAnimations[i] = {
              x: 0,
              y: 0,
              rotation: 0,
              scale: 1,
              isAnimating: false,
              animationType: 'idle',
              animationProgress: 0,
            }
            console.log(`Animation completed for agent ${i}`)
          }
        }
      }
      
      // Only update if there are changes
      if (!hasChanges) {
        return prev
      }
      
      return {
        ...prev,
        logoAnimations: newAnimations
      }
    })
  }, [])

  // Animation loop - simplified
  useEffect(() => {
    const animate = () => {
      // Check if any animation is active
      const hasActiveAnimation = battleState.logoAnimations.some(anim => anim.isAnimating)
      
      if (hasActiveAnimation) {
        updateAnimations()
      }
      
      animationFrameRef.current = requestAnimationFrame(animate)
    }
    
    if (!animationFrameRef.current) {
      animationFrameRef.current = requestAnimationFrame(animate)
    }
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }
    }
  }, [battleState.logoAnimations, updateAnimations])

  // Trigger animation when current agent changes
  useEffect(() => {
    if (battleState.currentAgent !== null && battleState.isActive) {
      startLogoAnimation(battleState.currentAgent!)
    }
  }, [battleState.currentAgent, battleState.isActive])

  // Initial canvas render
  useEffect(() => {
    renderCanvas()
  }, [])

  // Render canvas whenever battle state changes OR when model selection changes
  useEffect(() => {
    console.log('Rendering canvas due to state change')
    renderCanvas()
  }, [battleState.agents, battleState.currentAgent, battleState.isActive, selectedPlayer1Model, selectedPlayer2Model, battleState.logoAnimations, battleState.speechBubbles])

  // Cleanup EventSource on component unmount
  useEffect(() => {
    return () => {
      eventSourceRef.current?.close()
    }
  }, [])

  const renderCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    canvas.width = 800
    canvas.height = 200  // Set to 200
    renderMentalArena(ctx, battleState.agents)
  }

  const getProviderFromModel = (model: string): string => {
    return model.split(':')[0]
  }

  const drawSpeechBubble = (ctx: CanvasRenderingContext2D, x: number, y: number, bubble: SpeechBubble, alignLeft = false) => {
    const padding = 8
    const maxWidth = 180  // Increased from 120 to 180
    const fontSize = 10
    
    ctx.save()
    
    // Set font for measuring text
    ctx.font = `${fontSize}px 'Press Start 2P', monospace`
    ctx.textAlign = 'center'
    
    // Measure text and wrap if needed
    const words = bubble.message.split(' ')
    const lines: string[] = []
    let currentLine = ''
    
    for (const word of words) {
      const testLine = currentLine + (currentLine ? ' ' : '') + word
      const metrics = ctx.measureText(testLine)
      if (metrics.width > maxWidth && currentLine) {
        lines.push(currentLine)
        currentLine = word
      } else {
        currentLine = testLine
      }
    }
    if (currentLine) lines.push(currentLine)
    
    // Calculate bubble dimensions
    const lineHeight = fontSize + 2
    const bubbleWidth = Math.min(maxWidth + padding * 2, Math.max(...lines.map(line => ctx.measureText(line).width)) + padding * 2)
    const bubbleHeight = lines.length * lineHeight + padding * 2
    
    // Bubble position (adjusted for left/right alignment)
    // Y coordinate represents the center of the bubble
    let bubbleX: number
    if (alignLeft) {
      // For agent 1 (right side): bubble extends to the left
      bubbleX = x - bubbleWidth
    } else {
      // For agent 0 (left side): bubble extends to the right  
      bubbleX = x
    }
    const bubbleY = y - bubbleHeight / 2  // Center the bubble on the y coordinate
    
    // Determine colors based on bubble type
    let backgroundColor: string
    let borderColor: string
    let textColor: string
    
    switch (bubble.type) {
      case 'tool':
        backgroundColor = '#2563EB' // Blue
        borderColor = '#1D4ED8'
        textColor = '#FFFFFF'
        break
      case 'success':
        backgroundColor = '#16A34A' // Green
        borderColor = '#15803D'
        textColor = '#FFFFFF'
        break
      case 'failure':
        backgroundColor = '#DC2626' // Red
        borderColor = '#B91C1C'
        textColor = '#FFFFFF'
        break
      case 'detection':
        backgroundColor = '#F59E0B' // Yellow/Orange
        borderColor = '#D97706'
        textColor = '#000000'
        break
      default:
        backgroundColor = '#6B7280' // Gray
        borderColor = '#4B5563'
        textColor = '#FFFFFF'
    }
    
    // Draw bubble shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)'
    ctx.fillRect(bubbleX + 2, bubbleY + 2, bubbleWidth, bubbleHeight)
    
    // Draw bubble background
    ctx.fillStyle = backgroundColor
    ctx.fillRect(bubbleX, bubbleY, bubbleWidth, bubbleHeight)
    
    // Draw bubble border
    ctx.strokeStyle = borderColor
    ctx.lineWidth = 2
    ctx.strokeRect(bubbleX, bubbleY, bubbleWidth, bubbleHeight)
    
    // Draw pointer (triangle pointing toward logo at the edge)
    const pointerSize = 6
    let pointerX: number
    let pointerY: number
    
    if (alignLeft) {
      // For agent 1: pointer points right toward logo at x=700
      pointerX = bubbleX + bubbleWidth
      pointerY = bubbleY + bubbleHeight / 2
      
      ctx.beginPath()
      ctx.moveTo(pointerX, pointerY - pointerSize)
      ctx.lineTo(pointerX, pointerY + pointerSize)
      ctx.lineTo(pointerX + pointerSize, pointerY)
      ctx.closePath()
    } else {
      // For agent 0: pointer points left toward logo at x=100
      pointerX = bubbleX
      pointerY = bubbleY + bubbleHeight / 2
      
      ctx.beginPath()
      ctx.moveTo(pointerX, pointerY - pointerSize)
      ctx.lineTo(pointerX, pointerY + pointerSize)
      ctx.lineTo(pointerX - pointerSize, pointerY)
      ctx.closePath()
    }
    
    ctx.fillStyle = backgroundColor
    ctx.fill()
    ctx.strokeStyle = borderColor
    ctx.stroke()
    
    // Draw text
    ctx.fillStyle = textColor
    ctx.textAlign = 'center'
    const textX = bubbleX + bubbleWidth / 2
    const textStartY = bubbleY + padding + fontSize
    
    for (let i = 0; i < lines.length; i++) {
      ctx.fillText(lines[i], textX, textStartY + i * lineHeight)
    }
    
    ctx.restore()
  }

  const renderMentalArena = (ctx: CanvasRenderingContext2D, agents: [AgentState, AgentState]) => {
    ctx.clearRect(0, 0, 800, 200)  // Updated height

    // Background
    ctx.fillStyle = "#1e1e1e"
    ctx.fillRect(0, 0, 800, 200)  // Updated height

    // Grid pattern
    ctx.strokeStyle = "#333333"
    ctx.lineWidth = 1
    for (let i = 0; i < 800; i += 50) {
      ctx.beginPath()
      ctx.moveTo(i, 0)
      ctx.lineTo(i, 200)  // Updated height
      ctx.stroke()
    }
    for (let i = 0; i < 200; i += 50) {  // Updated height
      ctx.beginPath()
      ctx.moveTo(0, i)
      ctx.lineTo(800, i)
      ctx.stroke()
    }

    // Center divider
    ctx.strokeStyle = "#666666"
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(400, 0)
    ctx.lineTo(400, 200)  // Updated height
    ctx.stroke()

    // Draw agent logos - moved to outer sides
    const agentColors = ["#0066FF", "#FF6600"]
    const agentActive = [battleState.currentAgent === 0, battleState.currentAgent === 1]
    const basePositions = [{ x: 100, y: 100 }, { x: 700, y: 100 }]  // Moved to outer sides, centered at y=100
    const logoSize = 80

    for (let i = 0; i < 2; i++) {
      const baseX = basePositions[i].x
      const baseY = basePositions[i].y
      const isActive = agentActive[i] && battleState.isActive
      const animation = battleState.logoAnimations[i]

      // Calculate final position with animation offset
      const finalX = baseX + animation.x
      const finalY = baseY + animation.y

      // Get provider for current player
      const playerModel = i === 0 ? selectedPlayer1Model : selectedPlayer2Model
      const provider = getProviderFromModel(playerModel)
      const logoImage = logoImagesRef.current[provider]

      console.log(`Rendering agent ${i}: model=${playerModel}, provider=${provider}, hasLogo=${!!logoImage}, animation=${JSON.stringify(animation)}`)

      // Save context for transformations
      ctx.save()

      // Move to logo center
      ctx.translate(finalX, finalY)

      // Apply rotation
      if (animation.rotation !== 0) {
        ctx.rotate((animation.rotation * Math.PI) / 180)
      }

      // Apply scaling
      if (animation.scale !== 1) {
        ctx.scale(animation.scale, animation.scale)
      }

      // Draw glow effect if active
      if (isActive) {
        ctx.shadowColor = agentColors[i]
        ctx.shadowBlur = 30
        ctx.globalAlpha = 0.8
      }

      // Draw the logo image or fallback
      if (logoImage) {
        ctx.drawImage(logoImage, -logoSize/2, -logoSize/2, logoSize, logoSize)
      } else {
        // Fallback colored rectangle
        ctx.fillStyle = agentColors[i]
        ctx.fillRect(-logoSize/2, -logoSize/2, logoSize, logoSize)

        // Provider name
        ctx.fillStyle = "#FFFFFF"
        ctx.font = "10px 'Press Start 2P', monospace"
        ctx.textAlign = "center"
        ctx.fillText(provider.toUpperCase(), 0, 5)
      }

      // Reset shadow and alpha
      ctx.shadowBlur = 0
      ctx.globalAlpha = 1

      // Restore context
      ctx.restore()

      // Draw agent label below logo
      ctx.fillStyle = "#FFFFFF"
      ctx.font = "12px 'Press Start 2P', monospace"
      ctx.textAlign = "center"
      ctx.fillText(`AGENT ${i}`, baseX, baseY + logoSize/2 + 20)

      // Draw model name
      if (playerModel) {
        ctx.fillStyle = "#CCCCCC"
        ctx.font = "8px 'Press Start 2P', monospace"
        const modelName = playerModel.split(':')[1]
        ctx.fillText(modelName, baseX, baseY + logoSize/2 + 35)
      }

      // Draw active indicator
      if (isActive) {
        ctx.fillStyle = "#FFFF00"
        ctx.beginPath()
        ctx.arc(baseX + logoSize/2 - 10, baseY - logoSize/2 + 10, 6, 0, 2 * Math.PI)
        ctx.fill()

        // Pulsing effect for active indicator
        ctx.globalAlpha = 0.6 + 0.4 * Math.sin(Date.now() * 0.01)
        ctx.fillStyle = "#FFFF00"
        ctx.beginPath()
        ctx.arc(baseX + logoSize/2 - 10, baseY - logoSize/2 + 10, 10, 0, 2 * Math.PI)
        ctx.fill()
        ctx.globalAlpha = 1
      }

      // Draw speech bubbles if visible (stacked to the side of logos)
      const speechBubbles = battleState.speechBubbles[i]
      const visibleBubbles: SpeechBubble[] = []
      
      // Add bubbles to stack in order: tool (top), result (middle), detection (bottom)
      if (speechBubbles.tool.isVisible) visibleBubbles.push(speechBubbles.tool)
      if (speechBubbles.result.isVisible) visibleBubbles.push(speechBubbles.result)
      if (speechBubbles.detection.isVisible) visibleBubbles.push(speechBubbles.detection)
      
      // Position bubbles to the center of the arena (using full height)
      // Agent 0 (left side): bubbles in the center-left area
      // Agent 1 (right side): bubbles in the center-right area
      const bubbleBaseX = i === 0 ? 250 : 550  // Center-left and center-right positions
      
      // Distribute bubbles across the full height (200px)
      const totalBubbles = visibleBubbles.length
      if (totalBubbles > 0) {
        const estimatedBubbleHeight = 35  // Estimated height per bubble
        const topMargin = estimatedBubbleHeight + 10  // Ensure top bubble fits
        const bottomMargin = 20
        const availableHeight = 200 - topMargin - bottomMargin
        
        for (let bubbleIndex = 0; bubbleIndex < visibleBubbles.length; bubbleIndex++) {
          const bubble = visibleBubbles[bubbleIndex]
          
          let bubbleY: number
          if (totalBubbles === 1) {
            // Single bubble: center it
            bubbleY = 100
          } else {
            // Multiple bubbles: distribute evenly
            const spacing = availableHeight / (totalBubbles - 1)
            bubbleY = topMargin + (bubbleIndex * spacing)
          }
          
          drawSpeechBubble(ctx, bubbleBaseX, bubbleY, bubble, i === 1)
        }
      }
    }

    // Reset text alignment
    ctx.textAlign = "left"
  }

  const handleStreamEvent = (event: MessageEvent) => {
    const data = JSON.parse(event.data)

    setBattleState((prevState) => {
      let newLog = [...prevState.battleLog]
      const newAgents: [AgentState, AgentState] = [
        JSON.parse(JSON.stringify(prevState.agents[0])),
        JSON.parse(JSON.stringify(prevState.agents[1])),
      ]
      let newWinner = prevState.winner
      let newReason = prevState.reason
      let isStreaming = prevState.isStreaming
      let currentAgent = prevState.currentAgent
      let turn = prevState.turn

      const formatLog = (log: string) => `[T-${turn}] AGENT ${currentAgent}: ${log}`

      switch (data.type) {
        case "game_start":
          newLog = [`> Game ${data.game_id} started. Connecting to stream...`]
          break
        case "turn_start":
          currentAgent = data.agent
          turn = data.turn
          newLog.push(`\n--- TURN ${turn} (AGENT ${currentAgent}) ---`)
          // Trigger animation for every turn start
          if (currentAgent !== null) {
            setTimeout(() => startLogoAnimation(currentAgent!), 100)
          }
          break
        case "agent_thinking":
          newLog.push(formatLog("Thinking..."))
          break
        case "agent_response_chunk":
          const lastLog = newLog[newLog.length - 1]
          if (lastLog && lastLog.startsWith(formatLog(""))) {
            newLog[newLog.length - 1] += data.content
          } else {
            newLog.push(formatLog(data.content))
          }
          break
        case "tool_execution":
          if (data.tool_name && currentAgent !== null) {
            newLog.push(formatLog(`uses TOOL: ${data.tool_name}`))
            // Show speech bubble for tool usage
            setTimeout(() => {
              showSpeechBubble(currentAgent!, `Using ${data.tool_name}`, 'tool', data.tool_name)
            }, 200)
          }
          break
        case "tool_result":
          if (data.result && data.result.message && currentAgent !== null) {
            newLog.push(`> TOOL RESULT: ${data.result.message}`)
            
            // Determine bubble type and message based on result
            let bubbleType: SpeechBubble['type'] = 'success'
            let bubbleMessage = ''
            
            if (data.result.success === false) {
              bubbleType = 'failure'
              bubbleMessage = 'Failed!'
            } else if (data.result.success === true) {
              bubbleType = 'success'
              bubbleMessage = 'Success!'
            }
            
            // Show detection bubble if detected
            if (data.result.detection) {
              newLog.push(`> DETECTION! Opponent noticed the manipulation.`)
              setTimeout(() => {
                showSpeechBubble(currentAgent!, 'Detected!', 'detection')
              }, 500)
            } else if (bubbleMessage) {
              setTimeout(() => {
                showSpeechBubble(currentAgent!, bubbleMessage, bubbleType)
              }, 300)
            }
          }
          break
        case "game_state_update":
          newAgents[0].mentalState = {
            trust: data.game_state.agent_trust[0],
            memory: data.game_state.agent_memory_consistency[0],
            belief: data.game_state.agent_belief_integrity[0],
          }
          newAgents[1].mentalState = {
            trust: data.game_state.agent_trust[1],
            memory: data.game_state.agent_memory_consistency[1],
            belief: data.game_state.agent_belief_integrity[1],
          }
          break
        case "game_end":
        case "game_complete":
          isStreaming = false
          if (data.status?.startsWith("surrender")) {
            const loser = data.status.split("_")[1]
            newWinner = `Agent ${1 - Number.parseInt(loser, 10)}`
            newReason = "Opponent Surrendered"
          } else if (data.status?.startsWith("collapse")) {
            const loser = data.status.split("_")[1]
            newWinner = `Agent ${1 - Number.parseInt(loser, 10)}`
            newReason = "Cognitive Collapse"
          } else {
            newWinner = "Draw"
            newReason = "Max turns reached"
          }
          newLog.push(`\n--- BATTLE ENDED ---`)
          newLog.push(`> Winner: ${newWinner} (${newReason})`)
          eventSourceRef.current?.close()
          break
        case "error":
          newLog.push(`> ❌ SERVER ERROR: ${data.message}`)
          isStreaming = false
          eventSourceRef.current?.close()
          break
      }

      return {
        ...prevState,
        battleLog: newLog,
        agents: newAgents,
        isStreaming,
        winner: newWinner,
        reason: newReason,
        currentAgent,
        turn,
      }
    })
  }

  const startBattle = async () => {
    if (battleState.isStreaming) return

    // Validate inputs
    const requiredProviders = getRequiredApiKeys()
    const missingApiKeys = requiredProviders.filter(provider => !apiKeys[provider]?.trim())

    if (missingApiKeys.length > 0) {
      setBattleState(prev => ({
        ...prev,
        battleLog: [`> ❌ Please provide API keys for: ${missingApiKeys.map(p => getProviderDisplayName(p)).join(', ')}`]
      }))
      return
    }

    if (maxTurns < 1 || maxTurns > 20) {
      setBattleState(prev => ({
        ...prev,
        battleLog: ["> ❌ Max turns must be between 1 and 20."]
      }))
      return
    }

    setBattleState({
      ...initialBattleState,
      battleLog: ["> Initializing..."],
      logoAnimations: [JSON.parse(JSON.stringify(initialLogoAnimation)), JSON.parse(JSON.stringify(initialLogoAnimation))],
      speechBubbles: [JSON.parse(JSON.stringify(initialAgentSpeechBubbles)), JSON.parse(JSON.stringify(initialAgentSpeechBubbles))]
    })
    setShowConfig(false)

    const gameId = `battle-${Date.now()}`

    try {
      // Step 1: Health check to verify server is reachable and CORS is OK.
      console.log('DEBUG: Current window location:', {
        href: window.location.href,
        hostname: window.location.hostname,
        port: window.location.port,
        protocol: window.location.protocol
      })
      console.log('DEBUG: Resolved serverURL:', serverURL)
      setBattleState((prev) => ({ ...prev, battleLog: [`> Pinging server at ${serverURL}...`] }))
      const healthCheckResponse = await fetch(serverURL)
      if (!healthCheckResponse.ok) {
        throw new Error(`Server health check failed with status: ${healthCheckResponse.status}`)
      }
      const serverInfo = await healthCheckResponse.json()
      console.log("Server info:", serverInfo)
      setBattleState((prev) => ({ ...prev, battleLog: [...prev.battleLog, `> Server online: ${serverInfo.message}`] }))

      // Step 2: Start the game.
      setBattleState((prev) => ({ ...prev, battleLog: [...prev.battleLog, `> Starting game ${gameId}...`, `> Player 1: ${selectedPlayer1Model}`, `> Player 2: ${selectedPlayer2Model}`] }))
      const startResponse = await fetch(`${serverURL}/start-game`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          game_id: gameId,
          max_turns: maxTurns,
          player1_model: selectedPlayer1Model,
          player2_model: selectedPlayer2Model,
          api_keys: apiKeys
        }),
      })

      if (!startResponse.ok) {
        const errorData = await startResponse.json().catch(() => ({ detail: "Unknown server error." }))
        throw new Error(errorData.detail || "Failed to start game on server.")
      }

      // Step 3: Connect to the streaming endpoint.
      setBattleState((prev) => ({
        ...prev,
        isActive: true,
        isStreaming: true,
        battleLog: [`> Game started. Connecting to event stream...`],
      }))

      const es = new EventSource(`${serverURL}/stream-game/${gameId}`)
      eventSourceRef.current = es

      es.onmessage = handleStreamEvent
      es.onerror = (err) => {
        console.error("EventSource connection error:", err)
        setBattleState((prev) => ({
          ...prev,
          isStreaming: false,
          winner: prev.winner || "Error",
          reason: prev.reason || "Stream Disconnected",
          battleLog: [
            ...prev.battleLog,
            "\n--- CONNECTION LOST ---",
            "> ❌ Lost connection to the battle server.",
            "> The stream was interrupted. Please restart the battle.",
          ],
        }))
        es.close()
      }
    } catch (error: any) {
      console.error("Failed to start battle:", error)
      setBattleState((prev) => ({
        ...prev,
        isActive: false,
        isStreaming: false,
        battleLog: [
          "> ❌ Battle initialization failed.",
          `> URL: ${serverURL}`,
          `> Error: ${error.message}`,
          "> Possible causes:",
          "> 1. The backend server is down or has an error.",
          "> 2. A browser extension is blocking the request.",
          "> 3. The server's CORS policy is misconfigured.",
          "> 4. Invalid API key or unsupported model.",
        ],
      }))
    }
  }

  const restartBattle = () => {
    eventSourceRef.current?.close()
    setBattleState({
      ...initialBattleState,
      logoAnimations: [JSON.parse(JSON.stringify(initialLogoAnimation)), JSON.parse(JSON.stringify(initialLogoAnimation))],
      speechBubbles: [JSON.parse(JSON.stringify(initialAgentSpeechBubbles)), JSON.parse(JSON.stringify(initialAgentSpeechBubbles))]
    })
    setShowConfig(true)
  }

  return (
    <div
      className="pixel-ui min-h-screen p-4 bg-[#1e1e1e] text-white"
      style={{ fontFamily: "'Press Start 2P', monospace" }}
    >
      <style jsx>{`
        /* Custom range slider styling */
        input[type="range"]::-webkit-slider-thumb {
          appearance: none;
          height: 16px;
          width: 16px;
          border-radius: 50%;
          background: #9966FF;
          cursor: pointer;
          border: 2px solid #ffffff;
          box-shadow: 0 0 2px rgba(0,0,0,0.3);
        }
        input[type="range"]::-moz-range-thumb {
          height: 16px;
          width: 16px;
          border-radius: 50%;
          background: #9966FF;
          cursor: pointer;
          border: 2px solid #ffffff;
          box-shadow: 0 0 2px rgba(0,0,0,0.3);
        }
      `}</style>
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-6">
          <h1 className="text-4xl font-bold text-white tracking-wider" style={{ textShadow: "2px 2px 0px #000000" }}>
            Agent Battle Arena
          </h1>
          <p className="text-sm text-gray-400 mt-2">Mental Manipulation Battle</p>
        </div>

        {/* Configuration Panel */}
        {showConfig && (
          <Card className="mb-6 border border-black bg-[#2a2a2a]">
            <CardContent className="p-6">
              <h3 className="text-white text-sm mb-4 flex items-center">
                <Settings className="w-4 h-4 mr-2" /> BATTLE CONFIGURATION
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Player 1 Model Selection */}
                <div className="space-y-2">
                  <Label htmlFor="player1ModelSelect" className="text-white text-xs">
                    Player 1 Model * {availableModels?.providers ?
                      `(${Object.values(availableModels.providers).reduce((total: number, provider: any) => total + provider.models.length, 0)} models available)` :
                      '(Loading...)'}
                  </Label>
                  <Select value={selectedPlayer1Model} onValueChange={setSelectedPlayer1Model}>
                    <SelectTrigger className="bg-[#1a1a1a] border-gray-600 text-white text-xs">
                      <SelectValue placeholder="Select Player 1 model" />
                    </SelectTrigger>
                    <SelectContent className="bg-[#1a1a1a] border-gray-600">
                      {availableModels?.providers ? (
                        Object.entries(availableModels.providers).map(([providerId, providerInfo]: [string, any]) => (
                          <div key={`p1-${providerId}`}>
                            {providerInfo.models.map((model: string) => (
                              <SelectItem
                                key={`p1-${providerId}:${model}`}
                                value={`${providerId}:${model}`}
                                className="text-white hover:bg-gray-700"
                              >
                                <span className="text-blue-400">{providerInfo.name}</span>: {model}
                                {/* Add indicators for newest models */}
                                {(model.includes('2.5') || model.includes('2025') || model.includes('2505')) &&
                                  <span className="ml-2 text-xs bg-green-600 px-1 rounded">NEW</span>
                                }
                              </SelectItem>
                            ))}
                          </div>
                        ))
                      ) : (
                        <SelectItem value="loading" className="text-gray-400" disabled>
                          Loading models from server...
                        </SelectItem>
                      )}
                    </SelectContent>
                  </Select>
                  {selectedPlayer1Model && (
                    <div className="mt-1 p-2 bg-gray-800 rounded text-[10px] text-gray-300">
                      <strong>Player 1:</strong> {selectedPlayer1Model}
                    </div>
                  )}
                </div>

                {/* Player 2 Model Selection */}
                <div className="space-y-2">
                  <Label htmlFor="player2ModelSelect" className="text-white text-xs">
                    Player 2 Model *
                  </Label>
                  <Select value={selectedPlayer2Model} onValueChange={setSelectedPlayer2Model}>
                    <SelectTrigger className="bg-[#1a1a1a] border-gray-600 text-white text-xs">
                      <SelectValue placeholder="Select Player 2 model" />
                    </SelectTrigger>
                    <SelectContent className="bg-[#1a1a1a] border-gray-600">
                      {availableModels?.providers ? (
                        Object.entries(availableModels.providers).map(([providerId, providerInfo]: [string, any]) => (
                          <div key={`p2-${providerId}`}>
                            {providerInfo.models.map((model: string) => (
                              <SelectItem
                                key={`p2-${providerId}:${model}`}
                                value={`${providerId}:${model}`}
                                className="text-white hover:bg-gray-700"
                              >
                                <span className="text-orange-400">{providerInfo.name}</span>: {model}
                                {/* Add indicators for newest models */}
                                {(model.includes('2.5') || model.includes('2025') || model.includes('2505')) &&
                                  <span className="ml-2 text-xs bg-green-600 px-1 rounded">NEW</span>
                                }
                              </SelectItem>
                            ))}
                          </div>
                        ))
                      ) : (
                        <SelectItem value="loading" className="text-gray-400" disabled>
                          Loading models from server...
                        </SelectItem>
                      )}
                    </SelectContent>
                  </Select>
                  {selectedPlayer2Model && (
                    <div className="mt-1 p-2 bg-gray-800 rounded text-[10px] text-gray-300">
                      <strong>Player 2:</strong> {selectedPlayer2Model}
                    </div>
                  )}
                </div>

                {/* Model Summary */}
                {(selectedPlayer1Model || selectedPlayer2Model) && (
                  <div className="md:col-span-2 mt-2 p-3 bg-gray-900 rounded border border-gray-700">
                    <h4 className="text-white text-xs font-bold mb-2">BATTLE SETUP</h4>
                    <div className="grid grid-cols-2 gap-4 text-[10px]">
                      <div>
                        <span className="text-blue-400">PLAYER 1:</span>
                        <div className="text-white">{selectedPlayer1Model || 'Not selected'}</div>
                      </div>
                      <div>
                        <span className="text-orange-400">PLAYER 2:</span>
                        <div className="text-white">{selectedPlayer2Model || 'Not selected'}</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Dynamic API Key Inputs */}
                {getRequiredApiKeys().map((provider) => {
                  const providerInfo = availableModels?.providers[provider]
                  return (
                    <div key={provider} className="space-y-2 md:col-span-2">
                      <Label htmlFor={`apiKey-${provider}`} className="text-white text-xs">
                        {providerInfo?.name || provider} API Key *
                      </Label>
                      <Input
                        id={`apiKey-${provider}`}
                        type="password"
                        value={apiKeys[provider] || ""}
                        onChange={(e) => updateApiKey(provider, e.target.value)}
                        placeholder={`Enter your ${providerInfo?.name || provider} API key...`}
                        className="bg-[#1a1a1a] border-gray-600 text-white text-xs"
                      />
                      <p className="text-gray-400 text-[10px]">
                        Your API key is never stored and only used for this session
                      </p>
                    </div>
                  )
                })}

                <div className="space-y-2 md:col-span-2">
                  <Label htmlFor="maxTurns" className="text-white text-xs">
                    Max Turns (1-20)
                  </Label>
                  <Input
                    id="maxTurns"
                    type="number"
                    min="1"
                    max="20"
                    value={maxTurns}
                    onChange={(e) => setMaxTurns(parseInt(e.target.value) || 5)}
                    className="bg-[#1a1a1a] border-gray-600 text-white text-xs"
                  />
                  <p className="text-gray-400 text-[10px]">
                    Number of turns each agent gets before battle ends
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Battle Controls */}
        <div className="flex justify-center gap-4 mt-6 mb-6">
          {!battleState.isActive ? (
              <Button
                  onClick={startBattle}
                  className="bg-[#9966FF] hover:bg-[#8855EE] text-white disabled:bg-gray-600"
                  disabled={
                      battleState.isStreaming ||
                      !selectedPlayer1Model ||
                      !selectedPlayer2Model ||
                      getRequiredApiKeys().some(provider => !apiKeys[provider]?.trim()) ||
                      maxTurns < 1 ||
                      maxTurns > 20
                  }
              >
                <Zap className="w-4 h-4 mr-2" /> START BATTLE
              </Button>
          ) : (
              <Button onClick={restartBattle} variant="destructive">
                RESTART
              </Button>
          )}

          {/* Audio Controls */}
          <div className="flex items-center gap-2">
            <Button
              onClick={toggleAudio}
              variant="outline"
              size="sm"
              className="border-gray-600 text-gray-300 hover:bg-gray-700"
            >
              {audioState.isEnabled ? (
                <Volume2 className="w-4 h-4" />
              ) : (
                <VolumeX className="w-4 h-4" />
              )}
            </Button>
            {audioState.isEnabled && (
              <div className="flex items-center gap-2 min-w-[100px]">
                <span className="text-xs text-gray-400">VOL</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={audioState.volume}
                  onChange={(e) => setVolume(parseFloat(e.target.value))}
                  className="w-16 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, #9966FF 0%, #9966FF ${audioState.volume * 100}%, #374151 ${audioState.volume * 100}%, #374151 100%)`
                  }}
                />
                <span className="text-xs text-gray-400 w-8">{Math.round(audioState.volume * 100)}</span>
              </div>
            )}
          </div>

          {/* Test Sound Button */}
          <Button
            onClick={() => {
              console.log('Testing all sounds...')
              playSound('tool-use')
              setTimeout(() => playSound('success'), 300)
              setTimeout(() => playSound('failure'), 600)
              setTimeout(() => playSound('detection'), 900)
            }}
            variant="outline"
            size="sm"
            className="border-gray-600 text-gray-300 hover:bg-gray-700"
          >
            TEST SOUNDS
          </Button>

          {/* Test Animation Button */}
          <Button
              onClick={() => {
                console.log('Testing animation and stacked speech bubbles manually...')
                // Agent 0: Show all three bubble types in sequence
                startLogoAnimation(0)
                showSpeechBubble(0, 'Using prompt_manipulation', 'tool', 'prompt_manipulation')
                setTimeout(() => {
                  showSpeechBubble(0, 'Success!', 'success')
                }, 500)
                setTimeout(() => {
                  showSpeechBubble(0, 'Detected!', 'detection')
                }, 1000)

                // Agent 1: Show different combination
                setTimeout(() => {
                  startLogoAnimation(1)
                  showSpeechBubble(1, 'Using memory_alteration', 'tool', 'memory_alteration')
                  setTimeout(() => {
                    showSpeechBubble(1, 'Failed!', 'failure')
                  }, 800)
                }, 2000)
              }}
              variant="outline"
              className="border-gray-600 text-gray-300 hover:bg-gray-700"
          >
            TEST STACK
          </Button>

          {showConfig && (
              <Button
                  onClick={() => setShowConfig(false)}
                  variant="outline"
                  className="border-gray-600 text-gray-300 hover:bg-gray-700"
              >
                HIDE CONFIG
              </Button>
          )}

          {!showConfig && !battleState.isActive && (
              <Button
                  onClick={() => setShowConfig(true)}
                  variant="outline"
                  className="border-gray-600 text-gray-300 hover:bg-gray-700"
              >
                <Settings className="w-4 h-4 mr-2" /> CONFIG
              </Button>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
          {[0, 1].map((i) => {
            const playerModel = i === 0 ? selectedPlayer1Model : selectedPlayer2Model
            const playerColor = i === 0 ? 'blue' : 'orange'
            return (
              <Card
                key={i}
                className={`border-2 bg-[#333333] transition-all duration-300 ${
                  battleState.currentAgent === i && battleState.isActive 
                    ? 'border-yellow-400 shadow-lg shadow-yellow-400/20' 
                    : 'border-gray-600'
                }`}
              >
                <CardContent className="p-4">
                  <h3 className="text-white text-sm mb-3">
                    <Target className="w-4 h-4 inline mr-2" /> AGENT {i}
                  </h3>

                  {/* Model Information */}
                  {playerModel && (
                    <div className="mb-3 p-2 bg-gray-800 rounded">
                      <div className="text-[10px] text-gray-400">MODEL</div>
                      <div className={`text-xs ${i === 0 ? 'text-blue-400' : 'text-orange-400'}`}>
                        {playerModel.split(':')[1]}
                      </div>
                      <div className="text-[9px] text-gray-500">
                        {getProviderDisplayName(playerModel.split(':')[0])}
                      </div>
                    </div>
                  )}

                  <div className="space-y-2 text-white text-xs mb-4">
                    <div className="flex items-center gap-2">
                      <span>STATUS:</span>
                      {battleState.isActive ? (
                        battleState.currentAgent === i ? (
                          <span className="flex items-center gap-1 text-yellow-400">
                            <span className="animate-pulse">●</span> THINKING
                          </span>
                        ) : (
                          <span className="text-gray-400">WAITING</span>
                        )
                      ) : (
                        <span className="text-gray-500">IDLE</span>
                      )}
                    </div>
                  </div>

                {/* Mental State Bars */}
                <div className="space-y-2">
                  <MentalStateBar
                    label="TRUST"
                    value={battleState.agents[i].mentalState.trust}
                    color="#0066FF"
                  />
                  <MentalStateBar
                    label="MEMORY"
                    value={battleState.agents[i].mentalState.memory}
                    color="#00FF66"
                  />
                  <MentalStateBar
                    label="BELIEF"
                    value={battleState.agents[i].mentalState.belief}
                    color="#9966FF"
                  />
                </div>
              </CardContent>
            </Card>
            )
          })}
        </div>

        {/* Arena Canvas */}
        <Card className="mb-6 border-2 border-black bg-[#2a2a2a]">
          <CardContent className="p-2">
            <h3 className="text-white text-sm mb-4 text-center">BATTLE ARENA</h3>
            <canvas ref={canvasRef} className="w-full border border-black" style={{ imageRendering: "pixelated" }} />
            {battleState.winner && (
              <div className="mt-4 p-2 border border-gray-600 rounded bg-[#1a1a1a] text-center">
                <div className="text-white text-xs">
                  WINNER: {battleState.winner} | REASON: {battleState.reason.toUpperCase()}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="mt-6 border border-black bg-[#2a2a2a]">
          <CardContent className="p-4">
            <h3 className="text-white text-sm mb-4">BATTLE LOG</h3>
            <div
              ref={logContainerRef}
              className="p-4 border border-black h-60 overflow-y-auto bg-black text-xs text-green-400"
            >
              {battleState.battleLog.map((log, index) => (
                <p key={index} className="whitespace-pre-wrap leading-relaxed">
                  {log}
                </p>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
