import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence, useScroll, useTransform } from 'framer-motion'
import '@fontsource/instrument-serif/400.css'
import '@fontsource/jetbrains-mono/400.css'
import '@fontsource/jetbrains-mono/500.css'
import './App.css'

// Types
interface Point {
  id: number
  x: number
  y: number
  matched?: boolean
  matchedTo?: number
}

interface Match {
  donorId: number
  recipientId: number
  distance: number
}

// Generate random points for demo
const generatePoints = (count: number, seed: number = 0): Point[] => {
  const points: Point[] = []
  for (let i = 0; i < count; i++) {
    const noise = Math.sin(seed + i * 0.5) * 0.3
    points.push({
      id: i,
      x: 0.1 + Math.random() * 0.8 + noise * 0.1,
      y: 0.1 + Math.random() * 0.8 + noise * 0.1,
      matched: false
    })
  }
  return points
}

// Distance calculations
const euclidean = (p1: Point, p2: Point): number =>
  Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

const manhattan = (p1: Point, p2: Point): number =>
  Math.abs(p1.x - p2.x) + Math.abs(p1.y - p2.y)

const chebyshev = (p1: Point, p2: Point): number =>
  Math.max(Math.abs(p1.x - p2.x), Math.abs(p1.y - p2.y))

const distanceFunctions: Record<string, (p1: Point, p2: Point) => number> = {
  euclidean,
  manhattan,
  mahalanobis: euclidean,
  chebyshev
}

// Matching Visualization Component
function MatchingDemo() {
  const [donors, setDonors] = useState<Point[]>(() => generatePoints(8, 1))
  const [recipients, setRecipients] = useState<Point[]>(() => generatePoints(6, 2))
  const [matches, setMatches] = useState<Match[]>([])
  const [distanceMetric, setDistanceMetric] = useState('euclidean')
  const [isAnimating, setIsAnimating] = useState(false)
  const [currentMatchIndex, setCurrentMatchIndex] = useState(-1)
  const [speed, setSpeed] = useState(1)

  const distFn = distanceFunctions[distanceMetric]

  const computeMatches = useCallback(() => {
    const newMatches: Match[] = []
    recipients.forEach((rec) => {
      let minDist = Infinity
      let bestDonor = 0
      donors.forEach((don) => {
        const dist = distFn(rec, don)
        if (dist < minDist) {
          minDist = dist
          bestDonor = don.id
        }
      })
      newMatches.push({ donorId: bestDonor, recipientId: rec.id, distance: minDist })
    })
    return newMatches
  }, [recipients, donors, distFn])

  const startAnimation = () => {
    setIsAnimating(true)
    setCurrentMatchIndex(-1)
    const newMatches = computeMatches()
    setMatches(newMatches)

    let index = 0
    const interval = setInterval(() => {
      setCurrentMatchIndex(index)
      index++
      if (index >= newMatches.length) {
        clearInterval(interval)
        setTimeout(() => setIsAnimating(false), 1000 / speed)
      }
    }, 800 / speed)
  }

  const resetDemo = () => {
    setDonors(generatePoints(8, Math.random() * 100))
    setRecipients(generatePoints(6, Math.random() * 100))
    setMatches([])
    setCurrentMatchIndex(-1)
    setIsAnimating(false)
  }

  return (
    <div className="matching-demo">
      <div className="demo-controls">
        <div className="control-group">
          <label>Distance Metric</label>
          <select
            value={distanceMetric}
            onChange={(e) => setDistanceMetric(e.target.value)}
            disabled={isAnimating}
          >
            <option value="euclidean">Euclidean (L2)</option>
            <option value="manhattan">Manhattan (L1)</option>
            <option value="chebyshev">Chebyshev (L-inf)</option>
          </select>
        </div>
        <div className="control-group">
          <label>Speed: {speed}x</label>
          <input
            type="range"
            min="0.5"
            max="3"
            step="0.5"
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
          />
        </div>
        <div className="button-group">
          <button onClick={startAnimation} disabled={isAnimating} className="primary-btn">
            {isAnimating ? 'Matching...' : 'Run Matching'}
          </button>
          <button onClick={resetDemo} disabled={isAnimating} className="secondary-btn">
            Reset
          </button>
        </div>
      </div>

      <div className="scatter-container">
        <div className="scatter-plot">
          <div className="plot-label">Donors (have Y variable)</div>
          <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
            <defs>
              <radialGradient id="donorGlow">
                <stop offset="0%" stopColor="#00d4ff" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#00d4ff" stopOpacity="0" />
              </radialGradient>
            </defs>
            {[20, 40, 60, 80].map(v => (
              <g key={v}>
                <line x1={v} y1="0" x2={v} y2="100" className="grid-line" />
                <line x1="0" y1={v} x2="100" y2={v} className="grid-line" />
              </g>
            ))}
            {donors.map((point) => {
              const isMatched = matches.slice(0, currentMatchIndex + 1)
                .some(m => m.donorId === point.id)
              return (
                <g key={point.id}>
                  {isMatched && (
                    <motion.circle
                      cx={point.x * 100}
                      cy={point.y * 100}
                      r="8"
                      fill="url(#donorGlow)"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1.5 }}
                      transition={{ duration: 0.3 }}
                    />
                  )}
                  <motion.circle
                    cx={point.x * 100}
                    cy={point.y * 100}
                    r="4"
                    className={`donor-point ${isMatched ? 'matched' : ''}`}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: point.id * 0.05 }}
                  />
                </g>
              )
            })}
          </svg>
        </div>

        <div className="match-lines-container">
          <svg viewBox="0 0 100 100" preserveAspectRatio="none">
            <AnimatePresence>
              {matches.slice(0, currentMatchIndex + 1).map((match, i) => {
                const donor = donors.find(d => d.id === match.donorId)!
                const recipient = recipients.find(r => r.id === match.recipientId)!
                return (
                  <motion.g key={`match-${i}`}>
                    <motion.line
                      x1={donor.x * 100}
                      y1={donor.y * 100}
                      x2={recipient.x * 100}
                      y2={recipient.y * 100}
                      className="match-line"
                      initial={{ pathLength: 0, opacity: 0 }}
                      animate={{ pathLength: 1, opacity: 1 }}
                      transition={{ duration: 0.5 / speed }}
                    />
                    <motion.circle
                      r="2"
                      fill="#ff9f43"
                      initial={{
                        cx: donor.x * 100,
                        cy: donor.y * 100,
                        opacity: 1
                      }}
                      animate={{
                        cx: recipient.x * 100,
                        cy: recipient.y * 100,
                        opacity: 0
                      }}
                      transition={{ duration: 0.5 / speed }}
                    />
                  </motion.g>
                )
              })}
            </AnimatePresence>
          </svg>
        </div>

        <div className="scatter-plot">
          <div className="plot-label">Recipients (need Y variable)</div>
          <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
            <defs>
              <radialGradient id="recipientGlow">
                <stop offset="0%" stopColor="#ff9f43" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#ff9f43" stopOpacity="0" />
              </radialGradient>
            </defs>
            {[20, 40, 60, 80].map(v => (
              <g key={v}>
                <line x1={v} y1="0" x2={v} y2="100" className="grid-line" />
                <line x1="0" y1={v} x2="100" y2={v} className="grid-line" />
              </g>
            ))}
            {recipients.map((point) => {
              const isMatched = matches.slice(0, currentMatchIndex + 1)
                .some(m => m.recipientId === point.id)
              return (
                <g key={point.id}>
                  {isMatched && (
                    <motion.circle
                      cx={point.x * 100}
                      cy={point.y * 100}
                      r="8"
                      fill="url(#recipientGlow)"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1.5 }}
                      transition={{ duration: 0.3 }}
                    />
                  )}
                  <motion.circle
                    cx={point.x * 100}
                    cy={point.y * 100}
                    r="4"
                    className={`recipient-point ${isMatched ? 'matched' : ''}`}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: point.id * 0.05 }}
                  />
                </g>
              )
            })}
          </svg>
        </div>
      </div>

      {matches.length > 0 && currentMatchIndex >= 0 && (
        <motion.div
          className="match-stats"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="stat">
            <span className="stat-value">{currentMatchIndex + 1}</span>
            <span className="stat-label">Matches Made</span>
          </div>
          <div className="stat">
            <span className="stat-value">
              {(matches.slice(0, currentMatchIndex + 1)
                .reduce((sum, m) => sum + m.distance, 0) / (currentMatchIndex + 1))
                .toFixed(3)}
            </span>
            <span className="stat-label">Avg Distance</span>
          </div>
        </motion.div>
      )}
    </div>
  )
}

// Optimal Transport Visualization
function OptimalTransportDemo() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isRunning, setIsRunning] = useState(false)
  const animationRef = useRef<number | undefined>(undefined)

  interface Particle {
    x: number
    y: number
    targetX: number
    targetY: number
    progress: number
    speed: number
    trail: { x: number; y: number }[]
  }

  const startAnimation = () => {
    setIsRunning(true)
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!

    const sources = Array.from({ length: 5 }, (_, i) => ({
      x: 80,
      y: 60 + i * 40
    }))

    const targets = Array.from({ length: 5 }, (_, i) => ({
      x: 320,
      y: 60 + i * 40
    }))

    const particles: Particle[] = []
    sources.forEach((src, i) => {
      for (let j = 0; j < 8; j++) {
        particles.push({
          x: src.x + (Math.random() - 0.5) * 20,
          y: src.y + (Math.random() - 0.5) * 20,
          targetX: targets[i].x + (Math.random() - 0.5) * 20,
          targetY: targets[i].y + (Math.random() - 0.5) * 20,
          progress: -j * 0.1 - Math.random() * 0.2,
          speed: 0.008 + Math.random() * 0.004,
          trail: []
        })
      }
    })

    const animate = () => {
      ctx.fillStyle = 'rgba(8, 8, 12, 0.1)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      sources.forEach(src => {
        ctx.beginPath()
        ctx.arc(src.x, src.y, 12, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(0, 212, 255, 0.3)'
        ctx.fill()
        ctx.beginPath()
        ctx.arc(src.x, src.y, 6, 0, Math.PI * 2)
        ctx.fillStyle = '#00d4ff'
        ctx.fill()
      })

      targets.forEach(tgt => {
        ctx.beginPath()
        ctx.arc(tgt.x, tgt.y, 12, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(255, 159, 67, 0.3)'
        ctx.fill()
        ctx.beginPath()
        ctx.arc(tgt.x, tgt.y, 6, 0, Math.PI * 2)
        ctx.fillStyle = '#ff9f43'
        ctx.fill()
      })

      let allDone = true
      particles.forEach(p => {
        if (p.progress < 1) {
          allDone = false
          p.progress += p.speed

          if (p.progress >= 0 && p.progress <= 1) {
            const t = p.progress
            const midX = (p.x + p.targetX) / 2
            const midY = (p.y + p.targetY) / 2 + (Math.random() - 0.5) * 30

            const currX = (1-t)**2 * p.x + 2*(1-t)*t * midX + t**2 * p.targetX
            const currY = (1-t)**2 * p.y + 2*(1-t)*t * midY + t**2 * p.targetY

            p.trail.push({ x: currX, y: currY })
            if (p.trail.length > 20) p.trail.shift()

            if (p.trail.length > 1) {
              ctx.beginPath()
              ctx.moveTo(p.trail[0].x, p.trail[0].y)
              p.trail.forEach((pt) => {
                ctx.lineTo(pt.x, pt.y)
              })
              ctx.strokeStyle = `rgba(0, 212, 255, ${0.5 * (1 - t)})`
              ctx.lineWidth = 1
              ctx.stroke()
            }

            ctx.beginPath()
            ctx.arc(currX, currY, 3, 0, Math.PI * 2)
            const gradient = ctx.createRadialGradient(currX, currY, 0, currX, currY, 6)
            gradient.addColorStop(0, '#00d4ff')
            gradient.addColorStop(1, 'rgba(0, 212, 255, 0)')
            ctx.fillStyle = gradient
            ctx.fill()
          }
        }
      })

      if (!allDone) {
        animationRef.current = requestAnimationFrame(animate)
      } else {
        setIsRunning(false)
      }
    }

    ctx.fillStyle = '#08080c'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    animate()
  }

  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [])

  return (
    <div className="ot-demo">
      <div className="ot-header">
        <h3>Optimal Transport: Earth Mover's Distance</h3>
        <p>Watch data "flow" from donors to recipients along optimal paths</p>
      </div>
      <div className="canvas-container">
        <canvas ref={canvasRef} width={400} height={300} />
        <div className="canvas-labels">
          <span className="source-label">Source</span>
          <span className="target-label">Target</span>
        </div>
      </div>
      <button
        onClick={startAnimation}
        disabled={isRunning}
        className="primary-btn"
      >
        {isRunning ? 'Transporting...' : 'Run Optimal Transport'}
      </button>
    </div>
  )
}

// Uncertainty Visualization
function UncertaintyDemo() {
  const [showMI, setShowMI] = useState(false)
  const [imputations, setImputations] = useState<number[]>([42])

  useEffect(() => {
    if (showMI) {
      const baseValue = 42
      const newImputations = Array.from({ length: 5 }, () =>
        baseValue + (Math.random() - 0.5) * 20
      )
      setImputations(newImputations)
    } else {
      setImputations([42])
    }
  }, [showMI])

  const mean = imputations.reduce((a, b) => a + b, 0) / imputations.length
  const variance = imputations.reduce((sum, v) => sum + (v - mean) ** 2, 0) / imputations.length
  const ci = showMI ? 1.96 * Math.sqrt(variance * (1 + 1/imputations.length)) : 0

  return (
    <div className="uncertainty-demo">
      <div className="toggle-container">
        <button
          className={`toggle-btn ${!showMI ? 'active' : ''}`}
          onClick={() => setShowMI(false)}
        >
          Single Imputation
        </button>
        <button
          className={`toggle-btn ${showMI ? 'active' : ''}`}
          onClick={() => setShowMI(true)}
        >
          Multiple Imputation (m=5)
        </button>
      </div>

      <div className="estimate-visualization">
        <svg viewBox="0 0 400 120" preserveAspectRatio="xMidYMid meet">
          <line x1="50" y1="80" x2="350" y2="80" stroke="#333" strokeWidth="2" />
          {[0, 25, 50, 75, 100].map((v, i) => (
            <g key={i}>
              <line x1={50 + i * 75} y1="75" x2={50 + i * 75} y2="85" stroke="#555" />
              <text x={50 + i * 75} y="100" textAnchor="middle" fill="#666" fontSize="10">
                {v}
              </text>
            </g>
          ))}

          <AnimatePresence mode="wait">
            {showMI && (
              <motion.rect
                key="ci"
                x={50 + (mean - ci) * 3}
                y="35"
                width={ci * 6}
                height="30"
                fill="rgba(0, 212, 255, 0.2)"
                rx="4"
                initial={{ scaleX: 0, opacity: 0 }}
                animate={{ scaleX: 1, opacity: 1 }}
                exit={{ scaleX: 0, opacity: 0 }}
                transition={{ duration: 0.5 }}
              />
            )}
          </AnimatePresence>

          <AnimatePresence>
            {imputations.map((val, i) => (
              <motion.circle
                key={`imp-${i}-${showMI}`}
                cx={50 + val * 3}
                cy="50"
                r={showMI ? 4 : 8}
                fill={showMI ? '#00d4ff' : '#ff9f43'}
                initial={{ scale: 0, y: -20 }}
                animate={{ scale: 1, y: 0 }}
                exit={{ scale: 0 }}
                transition={{ delay: i * 0.1, duration: 0.3 }}
              />
            ))}
          </AnimatePresence>

          {showMI && (
            <motion.line
              x1={50 + mean * 3}
              y1="30"
              x2={50 + mean * 3}
              y2="70"
              stroke="#00d4ff"
              strokeWidth="2"
              strokeDasharray="4 2"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
            />
          )}
        </svg>
      </div>

      <div className="uncertainty-stats">
        <div className="stat-card">
          <span className="stat-title">Estimate</span>
          <span className="stat-number">{mean.toFixed(1)}</span>
        </div>
        {showMI && (
          <motion.div
            className="stat-card"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <span className="stat-title">95% CI</span>
            <span className="stat-number">
              [{(mean - ci).toFixed(1)}, {(mean + ci).toFixed(1)}]
            </span>
          </motion.div>
        )}
      </div>

      <p className="uncertainty-explanation">
        {showMI
          ? "Multiple imputation captures uncertainty by generating several plausible matches, then combining results using Rubin's rules."
          : "Single imputation ignores matching uncertainty, potentially leading to overconfident conclusions."
        }
      </p>
    </div>
  )
}

// Feature Comparison Table
function FeatureComparison() {
  const features = [
    { name: 'NND Hot Deck Matching', r: true, py: true, category: 'core' },
    { name: 'Random Hot Deck', r: true, py: true, category: 'core' },
    { name: 'Rank-based Matching', r: true, py: true, category: 'core' },
    { name: 'Gower Distance', r: true, py: true, category: 'core' },
    { name: 'Frechet Bounds', r: true, py: true, category: 'core' },
    { name: 'Multiple Imputation', r: false, py: true, category: 'advanced' },
    { name: 'ML Propensity Matching', r: false, py: true, category: 'advanced' },
    { name: 'Optimal Transport', r: false, py: true, category: 'advanced' },
    { name: 'Bayesian Uncertainty', r: false, py: true, category: 'advanced' },
    { name: 'Embedding Distance', r: false, py: true, category: 'advanced' },
    { name: 'Survey Weight Calibration', r: false, py: true, category: 'advanced' },
    { name: 'Diagnostics Dashboard', r: false, py: true, category: 'advanced' },
  ]

  return (
    <div className="comparison-table">
      <table>
        <thead>
          <tr>
            <th>Feature</th>
            <th>R StatMatch</th>
            <th>py-statmatch</th>
          </tr>
        </thead>
        <tbody>
          {features.map((f, i) => (
            <motion.tr
              key={f.name}
              className={f.category}
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
              viewport={{ once: true }}
            >
              <td>{f.name}</td>
              <td>
                {f.r ? (
                  <span className="check">&#10003;</span>
                ) : (
                  <span className="cross">&#10007;</span>
                )}
              </td>
              <td>
                <span className="check">&#10003;</span>
                {!f.r && <span className="new-badge">NEW</span>}
              </td>
            </motion.tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// Code Example Component
function CodeExample() {
  const code = `from statmatch import nnd_hotdeck, create_fused, match_diagnostics

# Perform matching
result = nnd_hotdeck(
    data_rec=recipients,
    data_don=donors,
    match_vars=['age', 'income', 'education']
)

# Create fused dataset
fused = create_fused(
    data_rec=recipients,
    data_don=donors,
    mtc_ids=result['mtc.ids'],
    z_vars=['satisfaction', 'health_score']
)

# Check match quality
diag = match_diagnostics(result, recipients, donors, match_vars)
diag.to_html('report.html')  # Generate HTML report`

  return (
    <div className="code-example">
      <div className="code-header">
        <span className="dot red"></span>
        <span className="dot yellow"></span>
        <span className="dot green"></span>
        <span className="filename">example.py</span>
      </div>
      <pre>
        <code>{code}</code>
      </pre>
    </div>
  )
}

// Main App
function App() {
  const { scrollYProgress } = useScroll()
  const heroOpacity = useTransform(scrollYProgress, [0, 0.2], [1, 0])
  const heroScale = useTransform(scrollYProgress, [0, 0.2], [1, 0.95])

  return (
    <div className="app">
      <div className="grid-bg" />

      <motion.section
        className="hero"
        style={{ opacity: heroOpacity, scale: heroScale }}
      >
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <h1>
            <span className="highlight">Statistical Matching</span>
            <br />for Python
          </h1>
          <p className="tagline">
            Beyond R's StatMatch with ML propensity, optimal transport,
            <br />Bayesian uncertainty, and more.
          </p>

          <div className="install-box">
            <code>pip install py-statmatch</code>
            <button
              className="copy-btn"
              onClick={() => navigator.clipboard.writeText('pip install py-statmatch')}
            >
              Copy
            </button>
          </div>

          <div className="hero-buttons">
            <a href="https://github.com/PolicyEngine/py-statmatch" className="github-btn">
              <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
              </svg>
              View on GitHub
            </a>
            <a href="#demo" className="demo-btn">
              See Demo
            </a>
          </div>
        </motion.div>

        <div className="hero-stats">
          <div className="stat-item">
            <span className="stat-value">28</span>
            <span className="stat-label">Functions</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">255</span>
            <span className="stat-label">Tests</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">100%</span>
            <span className="stat-label">R Parity</span>
          </div>
        </div>
      </motion.section>

      <section id="demo" className="section">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2>Interactive Matching Demo</h2>
          <p className="section-subtitle">
            See how nearest neighbor distance matching connects records across datasets
          </p>
          <MatchingDemo />
        </motion.div>
      </section>

      <section className="section alt-bg">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          <h2>More Than a Port</h2>
          <p className="section-subtitle">
            All 21 R StatMatch functions, plus 7 advanced methods
          </p>
          <FeatureComparison />
        </motion.div>
      </section>

      <section className="section">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          <OptimalTransportDemo />
        </motion.div>
      </section>

      <section className="section alt-bg">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          <h2>Proper Uncertainty Quantification</h2>
          <p className="section-subtitle">
            Don't ignore matching uncertainty - use multiple imputation
          </p>
          <UncertaintyDemo />
        </motion.div>
      </section>

      <section className="section">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          <h2>Clean, Pythonic API</h2>
          <p className="section-subtitle">
            Familiar patterns for data scientists
          </p>
          <CodeExample />
        </motion.div>
      </section>

      <footer>
        <div className="footer-content">
          <p>
            Built by <a href="https://policyengine.org">PolicyEngine</a>
          </p>
          <p className="footer-links">
            <a href="https://github.com/PolicyEngine/py-statmatch">GitHub</a>
            <span>|</span>
            <a href="https://policyengine.github.io/py-statmatch/">Docs</a>
            <span>|</span>
            <a href="https://pypi.org/project/py-statmatch/">PyPI</a>
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
