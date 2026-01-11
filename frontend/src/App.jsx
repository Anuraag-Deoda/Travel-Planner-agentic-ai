import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Plane, MapPin, Calendar, DollarSign, Clock, Star, ChevronDown, ChevronRight,
  Send, Loader2, Check, X, Globe, Utensils, Hotel, Car, Lightbulb, AlertCircle,
  RefreshCw, Sparkles, Users, Compass, Map, Coffee, Camera, Ticket, ArrowRight,
  Navigation, Sun, Cloud, Wind, Thermometer, Heart, Share2, Download, ChevronUp,
  Building, Train, Bus, PlaneTakeoff, MapPinned, Route, Wallet, Info, Search
} from 'lucide-react'

// Agent configuration
const AGENTS = {
  clarification: { name: 'Understanding Request', icon: Compass, color: 'teal' },
  process_answers: { name: 'Processing Answers', icon: Loader2, color: 'blue' },
  planner: { name: 'Trip Planner', icon: Map, color: 'indigo' },
  geography: { name: 'Route Optimizer', icon: Route, color: 'cyan' },
  research: { name: 'Destination Research', icon: Camera, color: 'emerald' },
  food_culture: { name: 'Food & Culture', icon: Coffee, color: 'orange' },
  transport_scraper: { name: 'Price Finder', icon: Ticket, color: 'blue' },
  transport_budget: { name: 'Budget Calculator', icon: Wallet, color: 'green' },
  critic: { name: 'Plan Reviewer', icon: Check, color: 'violet' },
  finalize: { name: 'Finalizing', icon: Sparkles, color: 'teal' }
}

const AGENT_ORDER = [
  'clarification', 'process_answers', 'planner', 'geography',
  'research', 'food_culture', 'transport_scraper', 'transport_budget',
  'critic', 'finalize'
]

// Hero images for different destinations
const HERO_IMAGES = {
  india: 'https://images.unsplash.com/photo-1524492412937-b28074a5d7da?w=1600',
  japan: 'https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?w=1600',
  france: 'https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=1600',
  thailand: 'https://images.unsplash.com/photo-1528181304800-259b08848526?w=1600',
  italy: 'https://images.unsplash.com/photo-1523906834658-6e24ef2386f9?w=1600',
  default: 'https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=1600'
}

export default function App() {
  const [query, setQuery] = useState('')
  const [isConnected, setIsConnected] = useState(false)
  const [isPlanning, setIsPlanning] = useState(false)
  const [currentPhase, setCurrentPhase] = useState('input')
  const [completedAgents, setCompletedAgents] = useState(new Set())
  const [activeAgent, setActiveAgent] = useState(null)
  const [questions, setQuestions] = useState([])
  const [answers, setAnswers] = useState({})
  const [itinerary, setItinerary] = useState(null)
  const [error, setError] = useState(null)
  const [agentData, setAgentData] = useState({})
  const [startTime, setStartTime] = useState(null)
  const [elapsedTime, setElapsedTime] = useState(0)
  const wsRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)

  // Timer effect
  useEffect(() => {
    let interval
    if (isPlanning && startTime) {
      interval = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - startTime) / 1000))
      }, 1000)
    }
    return () => clearInterval(interval)
  }, [isPlanning, startTime])

  // WebSocket connection
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const isDev = window.location.port === '3000'
    const wsUrl = isDev
      ? 'ws://localhost:8000/ws/plan'
      : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/plan`

    try {
      wsRef.current = new WebSocket(wsUrl)

      wsRef.current.onopen = () => {
        setIsConnected(true)
        setError(null)
        const pingInterval = setInterval(() => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'ping' }))
          } else {
            clearInterval(pingInterval)
          }
        }, 30000)
        wsRef.current._pingInterval = pingInterval
      }

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'pong' || data.type === 'ping') return
          handleMessage(data)
        } catch (e) {
          console.error('Parse error:', e)
        }
      }

      wsRef.current.onerror = () => setIsConnected(false)

      wsRef.current.onclose = (event) => {
        setIsConnected(false)
        if (wsRef.current?._pingInterval) clearInterval(wsRef.current._pingInterval)
        if (event.code !== 1000) {
          reconnectTimeoutRef.current = setTimeout(connectWebSocket, 2000)
        }
      }
    } catch (e) {
      setError('Failed to connect')
    }
  }, [])

  useEffect(() => {
    connectWebSocket()
    return () => {
      clearTimeout(reconnectTimeoutRef.current)
      if (wsRef.current) {
        if (wsRef.current._pingInterval) clearInterval(wsRef.current._pingInterval)
        wsRef.current.close(1000)
      }
    }
  }, [connectWebSocket])

  const handleMessage = (data) => {
    switch (data.type) {
      case 'connected':
        break
      case 'agent_start':
        setActiveAgent(data.agent)
        if (currentPhase !== 'questions') setCurrentPhase('progress')
        break
      case 'agent_complete':
        setCompletedAgents(prev => new Set([...prev, data.agent]))
        if (data.data) setAgentData(prev => ({ ...prev, [data.agent]: data.data }))
        break
      case 'questions':
        setQuestions(data.questions || [])
        setCurrentPhase('questions')
        setActiveAgent(null)
        break
      case 'planning_complete':
        setItinerary(data.itinerary)
        setCurrentPhase('result')
        setIsPlanning(false)
        setActiveAgent(null)
        break
      case 'error':
        setError(data.error)
        setIsPlanning(false)
        setActiveAgent(null)
        break
    }
  }

  const startPlanning = () => {
    if (!query.trim() || query.length < 10) {
      setError('Please provide more details about your trip.')
      return
    }
    if (!isConnected) {
      setError('Connecting to server...')
      connectWebSocket()
      return
    }
    setError(null)
    setIsPlanning(true)
    setCompletedAgents(new Set())
    setActiveAgent(null)
    setAgentData({})
    setCurrentPhase('progress')
    setStartTime(Date.now())
    setElapsedTime(0)
    wsRef.current?.send(JSON.stringify({ type: 'start_planning', request: query }))
  }

  const submitAnswers = () => {
    if (Object.keys(answers).length === 0) {
      setError('Please answer at least one question.')
      return
    }
    setCurrentPhase('progress')
    setError(null)
    wsRef.current?.send(JSON.stringify({ type: 'answer_questions', answers }))
  }

  const resetPlanner = () => {
    setQuery('')
    setIsPlanning(false)
    setCurrentPhase('input')
    setCompletedAgents(new Set())
    setActiveAgent(null)
    setQuestions([])
    setAnswers({})
    setItinerary(null)
    setError(null)
    setAgentData({})
    setStartTime(null)
    setElapsedTime(0)
  }

  const progress = (completedAgents.size / AGENT_ORDER.length) * 100
  const formatTime = (s) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      {/* Navigation */}
      <nav className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-teal-500 to-cyan-600 rounded-xl flex items-center justify-center shadow-lg shadow-teal-500/20">
                <Plane className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">TravelAI</h1>
                <p className="text-xs text-slate-500">Intelligent Trip Planning</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${
                isConnected
                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                  : 'bg-amber-50 text-amber-700 border border-amber-200'
              }`}>
                <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-500' : 'bg-amber-500 animate-pulse'}`} />
                {isConnected ? 'Online' : 'Connecting...'}
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <AnimatePresence mode="wait">
          {/* Error Alert */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-center gap-3"
            >
              <AlertCircle className="w-5 h-5 text-red-500" />
              <p className="text-red-700 flex-1 text-sm">{error}</p>
              <button onClick={() => setError(null)} className="text-red-400 hover:text-red-600">
                <X className="w-5 h-5" />
              </button>
            </motion.div>
          )}

          {/* Input Phase */}
          {currentPhase === 'input' && (
            <motion.div
              key="input"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              {/* Hero Section */}
              <div className="text-center mb-10">
                <motion.h2
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="text-4xl sm:text-5xl font-bold text-slate-900 mb-4"
                >
                  Plan Your Perfect Trip
                </motion.h2>
                <p className="text-lg text-slate-600 max-w-2xl mx-auto">
                  Our AI-powered system uses 10 specialized agents to create personalized travel itineraries with real-time pricing.
                </p>
              </div>

              {/* Search Card */}
              <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/50 border border-slate-200 p-6 sm:p-8 max-w-4xl mx-auto">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-12 h-12 bg-teal-50 rounded-xl flex items-center justify-center">
                    <Search className="w-6 h-6 text-teal-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-slate-900">Describe Your Trip</h3>
                    <p className="text-sm text-slate-500">Tell us where you want to go and what you'd like to experience</p>
                  </div>
                </div>

                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Example: Plan a 7-day trip to Japan visiting Tokyo and Kyoto. I'm interested in culture, temples, and local food. Mid-range budget, traveling in April."
                  className="w-full min-h-[120px] px-4 py-4 bg-slate-50 border border-slate-200 rounded-xl text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent resize-none text-base transition-all"
                />

                {/* Quick Options */}
                <div className="mt-4 flex flex-wrap gap-2">
                  {[
                    { label: 'Rajasthan, India', query: 'Plan a 5-day trip to Rajasthan visiting Udaipur, Jodhpur, and Jaipur with cultural experiences' },
                    { label: 'Japan', query: 'Plan a 7-day trip to Japan visiting Tokyo and Kyoto with focus on culture and food' },
                    { label: 'Europe Tour', query: 'Plan a 10-day Europe trip covering Paris, Amsterdam, and Rome' },
                    { label: 'Thailand', query: 'Plan a 7-day Thailand trip to Bangkok and Phuket with beaches and temples' },
                  ].map((dest) => (
                    <button
                      key={dest.label}
                      onClick={() => setQuery(dest.query)}
                      className="px-4 py-2 bg-slate-100 hover:bg-teal-50 border border-slate-200 hover:border-teal-300 rounded-lg text-sm text-slate-600 hover:text-teal-700 transition-all"
                    >
                      {dest.label}
                    </button>
                  ))}
                </div>

                <button
                  onClick={startPlanning}
                  disabled={isPlanning || !query.trim() || !isConnected}
                  className="w-full mt-6 py-4 bg-gradient-to-r from-teal-500 to-cyan-600 hover:from-teal-600 hover:to-cyan-700 text-white font-semibold rounded-xl flex items-center justify-center gap-3 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-teal-500/30"
                >
                  {!isConnected ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Connecting...
                    </>
                  ) : isPlanning ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Planning...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-5 h-5" />
                      Create My Itinerary
                      <ArrowRight className="w-5 h-5" />
                    </>
                  )}
                </button>
              </div>

              {/* Features */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-12 max-w-4xl mx-auto">
                {[
                  { icon: Globe, title: 'Real-Time Prices', desc: 'Live transport costs', color: 'teal' },
                  { icon: Users, title: '10 AI Agents', desc: 'Specialized experts', color: 'blue' },
                  { icon: Route, title: 'Smart Routes', desc: 'Optimized travel', color: 'indigo' },
                  { icon: Wallet, title: 'Budget Control', desc: 'Cost breakdown', color: 'emerald' },
                ].map((f, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.1 }}
                    className="p-5 bg-white rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow"
                  >
                    <div className={`w-10 h-10 rounded-lg bg-${f.color}-50 flex items-center justify-center mb-3`}>
                      <f.icon className={`w-5 h-5 text-${f.color}-600`} />
                    </div>
                    <h4 className="font-semibold text-slate-900 text-sm">{f.title}</h4>
                    <p className="text-xs text-slate-500 mt-1">{f.desc}</p>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Progress Phase */}
          {currentPhase === 'progress' && (
            <motion.div
              key="progress"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="max-w-4xl mx-auto"
            >
              <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/50 border border-slate-200 p-6 sm:p-8">
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center gap-4">
                    <div className="w-14 h-14 bg-gradient-to-br from-teal-500 to-cyan-600 rounded-xl flex items-center justify-center shadow-lg shadow-teal-500/20">
                      <Loader2 className="w-7 h-7 text-white animate-spin" />
                    </div>
                    <div>
                      <h2 className="text-xl font-bold text-slate-900">Creating Your Itinerary</h2>
                      <p className="text-sm text-slate-500">
                        {activeAgent ? AGENTS[activeAgent]?.name : 'Initializing agents...'}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-3xl font-bold text-teal-600">{Math.round(progress)}%</p>
                    <p className="text-xs text-slate-500">{formatTime(elapsedTime)}</p>
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="h-3 bg-slate-100 rounded-full overflow-hidden mb-8">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    className="h-full bg-gradient-to-r from-teal-500 to-cyan-500 rounded-full"
                  />
                </div>

                {/* Agent Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                  {AGENT_ORDER.map((key) => {
                    const agent = AGENTS[key]
                    const isActive = activeAgent === key
                    const isComplete = completedAgents.has(key)
                    const Icon = agent.icon

                    return (
                      <div
                        key={key}
                        className={`p-4 rounded-xl border-2 transition-all ${
                          isActive
                            ? 'bg-teal-50 border-teal-500 shadow-lg shadow-teal-500/10'
                            : isComplete
                            ? 'bg-emerald-50 border-emerald-300'
                            : 'bg-slate-50 border-slate-200 opacity-50'
                        }`}
                      >
                        <div className="flex flex-col items-center text-center gap-2">
                          <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                            isComplete ? 'bg-emerald-500' : isActive ? 'bg-teal-500' : 'bg-slate-300'
                          }`}>
                            {isComplete ? (
                              <Check className="w-5 h-5 text-white" />
                            ) : (
                              <Icon className={`w-5 h-5 ${isActive ? 'text-white animate-pulse' : 'text-white'}`} />
                            )}
                          </div>
                          <span className={`text-xs font-medium ${
                            isActive ? 'text-teal-700' : isComplete ? 'text-emerald-700' : 'text-slate-500'
                          }`}>
                            {agent.name}
                          </span>
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Live Updates */}
                {Object.keys(agentData).length > 0 && (
                  <div className="mt-6 pt-6 border-t border-slate-200">
                    <h4 className="text-sm font-medium text-slate-700 mb-3">Progress Updates</h4>
                    <div className="space-y-2">
                      {agentData.planner?.cities && (
                        <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
                          <MapPin className="w-4 h-4 text-teal-600" />
                          <span className="text-sm text-slate-700">
                            Route: {agentData.planner.cities.map(c => c.name).join(' → ')}
                          </span>
                        </div>
                      )}
                      {agentData.research?.attractions_count > 0 && (
                        <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
                          <Camera className="w-4 h-4 text-teal-600" />
                          <span className="text-sm text-slate-700">
                            Found {agentData.research.attractions_count} attractions
                          </span>
                        </div>
                      )}
                      {agentData.transport_budget?.total > 0 && (
                        <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
                          <Wallet className="w-4 h-4 text-teal-600" />
                          <span className="text-sm text-slate-700">
                            Estimated budget: ${Math.round(agentData.transport_budget.total)}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {/* Questions Phase */}
          {currentPhase === 'questions' && (
            <motion.div
              key="questions"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="max-w-3xl mx-auto"
            >
              <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/50 border border-slate-200 p-6 sm:p-8">
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-12 h-12 bg-amber-50 rounded-xl flex items-center justify-center">
                    <Compass className="w-6 h-6 text-amber-600" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-slate-900">A Few Quick Questions</h2>
                    <p className="text-sm text-slate-500">Help us personalize your experience</p>
                  </div>
                </div>

                <div className="space-y-5">
                  {questions.map((q, idx) => {
                    const qId = q.question_id || `q${idx}`
                    const qText = q.question_text || q.question || ''
                    const opts = q.options || []

                    return (
                      <motion.div
                        key={qId}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.1 }}
                        className="p-5 bg-slate-50 rounded-xl border border-slate-200"
                      >
                        <p className="font-medium text-slate-900 mb-4">{qText}</p>
                        {opts.length > 0 && (
                          <div className="flex flex-wrap gap-2 mb-3">
                            {opts.map((opt, i) => (
                              <button
                                key={i}
                                onClick={() => setAnswers(prev => ({ ...prev, [qId]: opt }))}
                                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all border ${
                                  answers[qId] === opt
                                    ? 'bg-teal-500 text-white border-teal-500'
                                    : 'bg-white text-slate-700 border-slate-300 hover:border-teal-400 hover:text-teal-600'
                                }`}
                              >
                                {opt}
                              </button>
                            ))}
                          </div>
                        )}
                        <input
                          type="text"
                          placeholder="Or type your answer..."
                          className="w-full px-4 py-3 bg-white border border-slate-300 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent"
                          value={answers[qId] && !opts.includes(answers[qId]) ? answers[qId] : ''}
                          onChange={(e) => setAnswers(prev => ({ ...prev, [qId]: e.target.value }))}
                        />
                      </motion.div>
                    )
                  })}
                </div>

                <button
                  onClick={submitAnswers}
                  className="w-full mt-6 py-4 bg-gradient-to-r from-teal-500 to-cyan-600 text-white font-semibold rounded-xl flex items-center justify-center gap-2 shadow-lg shadow-teal-500/30"
                >
                  <Check className="w-5 h-5" />
                  Continue Planning
                </button>
              </div>
            </motion.div>
          )}

          {/* Result Phase */}
          {currentPhase === 'result' && itinerary && (
            <motion.div
              key="result"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <ItineraryResult itinerary={itinerary} onReset={resetPlanner} />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white mt-16">
        <div className="max-w-7xl mx-auto px-4 py-6 text-center text-sm text-slate-500">
          Powered by AI • 10 Specialized Agents • Real-Time Pricing
        </div>
      </footer>
    </div>
  )
}

// Itinerary Result Component
function ItineraryResult({ itinerary, onReset }) {
  const [expandedDays, setExpandedDays] = useState(new Set([1]))

  const toggleDay = (num) => {
    setExpandedDays(prev => {
      const s = new Set(prev)
      s.has(num) ? s.delete(num) : s.add(num)
      return s
    })
  }

  const cities = itinerary.cities_visited || []
  const days = itinerary.total_days || 0
  const budget = itinerary.budget_breakdown || {}
  const dailyPlans = itinerary.daily_plans || []
  const transport = itinerary.inter_city_transport || []
  const hotels = itinerary.hotels || []
  const tips = itinerary.cultural_tips || []
  const moneyTips = budget.money_saving_tips || []

  const getImage = () => {
    const c = (cities[0] || '').toLowerCase()
    for (const [k, v] of Object.entries(HERO_IMAGES)) {
      if (c.includes(k)) return v
    }
    return HERO_IMAGES.default
  }

  return (
    <div className="space-y-6">
      {/* Hero */}
      <div className="relative h-72 sm:h-96 rounded-2xl overflow-hidden shadow-xl">
        <img src={getImage()} alt="Destination" className="w-full h-full object-cover" />
        <div className="absolute inset-0 bg-gradient-to-t from-slate-900/90 via-slate-900/40 to-transparent" />
        <div className="absolute bottom-0 left-0 right-0 p-6 sm:p-8">
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-2">
            {itinerary.trip_title || 'Your Trip'}
          </h2>
          <p className="text-white/80 flex items-center gap-2">
            <MapPin className="w-4 h-4" />
            {cities.join(' → ')}
          </p>
        </div>
        <div className="absolute top-4 right-4 flex gap-2">
          <button className="p-2.5 bg-white/90 backdrop-blur rounded-lg text-slate-700 hover:bg-white transition shadow">
            <Heart className="w-5 h-5" />
          </button>
          <button className="p-2.5 bg-white/90 backdrop-blur rounded-lg text-slate-700 hover:bg-white transition shadow">
            <Share2 className="w-5 h-5" />
          </button>
          <button className="p-2.5 bg-white/90 backdrop-blur rounded-lg text-slate-700 hover:bg-white transition shadow">
            <Download className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          { icon: Calendar, label: 'Duration', value: `${days} Days`, color: 'teal' },
          { icon: MapPin, label: 'Cities', value: cities.length, color: 'blue' },
          { icon: DollarSign, label: 'Budget', value: `$${Math.round(budget.total || itinerary.total_estimated_cost_usd || 0)}`, color: 'emerald' },
          { icon: Star, label: 'Level', value: itinerary.budget_level || 'Mid-range', color: 'amber' },
        ].map((s, i) => (
          <div key={i} className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
            <div className={`w-10 h-10 rounded-lg bg-${s.color}-50 flex items-center justify-center mb-3`}>
              <s.icon className={`w-5 h-5 text-${s.color}-600`} />
            </div>
            <p className="text-2xl font-bold text-slate-900">{s.value}</p>
            <p className="text-xs text-slate-500">{s.label}</p>
          </div>
        ))}
      </div>

      {/* Daily Plans */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="p-5 border-b border-slate-200 bg-slate-50">
          <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
            <Route className="w-5 h-5 text-teal-600" />
            Daily Itinerary
          </h3>
        </div>

        <div className="divide-y divide-slate-200">
          {dailyPlans.map((day) => (
            <div key={day.day_number}>
              <button
                onClick={() => toggleDay(day.day_number)}
                className="w-full p-5 flex items-center justify-between hover:bg-slate-50 transition"
              >
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-teal-500 to-cyan-600 rounded-xl flex items-center justify-center text-white font-bold shadow">
                    {day.day_number}
                  </div>
                  <div className="text-left">
                    <p className="font-semibold text-slate-900">{day.city}</p>
                    <p className="text-sm text-slate-500">{(day.activities || []).length} activities planned</p>
                  </div>
                </div>
                <ChevronDown className={`w-5 h-5 text-slate-400 transition-transform ${expandedDays.has(day.day_number) ? 'rotate-180' : ''}`} />
              </button>

              <AnimatePresence>
                {expandedDays.has(day.day_number) && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="px-5 pb-5 space-y-3">
                      {(day.activities || []).map((act, idx) => {
                        const isMeal = act.activity_type === 'meal'
                        return (
                          <div key={idx} className="flex gap-4 p-4 bg-slate-50 rounded-xl border border-slate-100">
                            <div className="text-xs font-medium text-slate-500 w-20 pt-1">
                              {act.time_slot}
                            </div>
                            <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
                              isMeal ? 'bg-orange-100' : 'bg-teal-100'
                            }`}>
                              {isMeal ? (
                                <Utensils className="w-5 h-5 text-orange-600" />
                              ) : (
                                <Camera className="w-5 h-5 text-teal-600" />
                              )}
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="font-medium text-slate-900">{act.title}</p>
                              <div className="flex flex-wrap gap-3 mt-1 text-xs text-slate-500">
                                {act.attraction?.rating && (
                                  <span className="flex items-center gap-1">
                                    <Star className="w-3 h-3 text-amber-500" />
                                    {act.attraction.rating}
                                  </span>
                                )}
                                {act.attraction?.estimated_duration_hours && (
                                  <span className="flex items-center gap-1">
                                    <Clock className="w-3 h-3" />
                                    {act.attraction.estimated_duration_hours}h
                                  </span>
                                )}
                                {act.meal?.estimated_cost_usd && (
                                  <span className="flex items-center gap-1">
                                    <DollarSign className="w-3 h-3" />
                                    ${act.meal.estimated_cost_usd}
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          ))}
        </div>
      </div>

      {/* Transport */}
      {transport.length > 0 && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
          <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2 mb-4">
            <Car className="w-5 h-5 text-teal-600" />
            Transportation
          </h3>
          <div className="space-y-3">
            {transport.map((t, i) => (
              <div key={i} className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-100">
                <div className="flex items-center gap-3">
                  <span className="font-medium text-slate-900">{t.from_location}</span>
                  <ArrowRight className="w-4 h-4 text-teal-500" />
                  <span className="font-medium text-slate-900">{t.to_location}</span>
                </div>
                <div className="text-right">
                  <p className="font-medium text-slate-900 capitalize">{t.recommended?.mode || 'Various'}</p>
                  <p className="text-xs text-slate-500">
                    {t.recommended?.duration_hours || '?'}h • ~${t.recommended?.estimated_cost_usd || '?'}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Budget */}
      {budget.total > 0 && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
          <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2 mb-4">
            <Wallet className="w-5 h-5 text-emerald-600" />
            Budget Breakdown
          </h3>

          <div className="bg-gradient-to-r from-teal-500 to-cyan-600 rounded-xl p-6 text-center mb-4">
            <p className="text-4xl font-bold text-white">${Math.round(budget.total)}</p>
            <p className="text-white/80 text-sm">Estimated Total</p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { icon: Car, label: 'Transport', value: (budget.transport_inter_city || 0) + (budget.transport_local || 0) },
              { icon: Hotel, label: 'Hotels', value: budget.accommodation || 0 },
              { icon: Utensils, label: 'Food', value: budget.food || 0 },
              { icon: Ticket, label: 'Activities', value: budget.activities || 0 },
            ].filter(x => x.value > 0).map((x, i) => (
              <div key={i} className="bg-slate-50 rounded-xl p-4 text-center border border-slate-100">
                <x.icon className="w-5 h-5 text-slate-500 mx-auto mb-2" />
                <p className="text-lg font-semibold text-slate-900">${Math.round(x.value)}</p>
                <p className="text-xs text-slate-500">{x.label}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Hotels */}
      {hotels.length > 0 && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
          <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2 mb-4">
            <Hotel className="w-5 h-5 text-indigo-600" />
            Recommended Hotels
          </h3>
          <div className="grid sm:grid-cols-2 gap-3">
            {hotels.slice(0, 4).map((h, i) => (
              <div key={i} className="flex items-center gap-4 p-4 bg-slate-50 rounded-xl border border-slate-100">
                <div className="w-12 h-12 bg-indigo-100 rounded-xl flex items-center justify-center">
                  <Building className="w-6 h-6 text-indigo-600" />
                </div>
                <div className="flex-1">
                  <p className="font-medium text-slate-900">{h.name}</p>
                  <p className="text-xs text-slate-500">{h.city}</p>
                </div>
                {h.rating && (
                  <div className="flex items-center gap-1 px-2 py-1 bg-amber-100 rounded-lg">
                    <Star className="w-3 h-3 text-amber-600" />
                    <span className="text-sm font-medium text-amber-700">{h.rating}</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tips */}
      {(tips.length > 0 || moneyTips.length > 0) && (
        <div className="grid md:grid-cols-2 gap-4">
          {tips.length > 0 && (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
              <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2 mb-4">
                <Lightbulb className="w-5 h-5 text-amber-500" />
                Cultural Tips
              </h3>
              <ul className="space-y-2">
                {tips.slice(0, 5).map((t, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-slate-600">
                    <Check className="w-4 h-4 text-teal-500 mt-0.5 flex-shrink-0" />
                    {t}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {moneyTips.length > 0 && (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
              <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2 mb-4">
                <DollarSign className="w-5 h-5 text-emerald-500" />
                Money Saving Tips
              </h3>
              <ul className="space-y-2">
                {moneyTips.slice(0, 5).map((t, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-slate-600">
                    <Check className="w-4 h-4 text-teal-500 mt-0.5 flex-shrink-0" />
                    {t}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* New Plan Button */}
      <div className="flex justify-center pt-4">
        <button
          onClick={onReset}
          className="px-8 py-3 bg-slate-100 hover:bg-slate-200 border border-slate-300 rounded-xl text-slate-700 font-medium flex items-center gap-2 transition"
        >
          <RefreshCw className="w-5 h-5" />
          Plan Another Trip
        </button>
      </div>
    </div>
  )
}
