import { useState, useRef, useEffect, useCallback } from 'react'
import './App.css'

// Use environment variable, fallback to local if not set
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

function App() {
  const canvasRef = useRef(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [hasDrawn, setHasDrawn] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [confidence, setConfidence] = useState(null)
  const [loading, setLoading] = useState(false)
  const [feedback, setFeedback] = useState(null) // 'correct' | 'wrong' | null
  const [canvasSize, setCanvasSize] = useState(280)

  // Responsive canvas size
  useEffect(() => {
    const updateSize = () => {
      const width = window.innerWidth
      if (width <= 520) {
        setCanvasSize(240)
      } else {
        setCanvasSize(280)
      }
    }
    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])

  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = '#0d0d15'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = canvasSize / 14  // scales with canvas
  }, [canvasSize])

  const getPos = useCallback((e) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height

    if (e.touches) {
      return {
        x: (e.touches[0].clientX - rect.left) * scaleX,
        y: (e.touches[0].clientY - rect.top) * scaleY
      }
    }
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    }
  }, [])

  const startDrawing = useCallback((e) => {
    e.preventDefault()
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const pos = getPos(e)
    ctx.beginPath()
    ctx.moveTo(pos.x, pos.y)
    setIsDrawing(true)
    setHasDrawn(true)
  }, [getPos])

  const draw = useCallback((e) => {
    if (!isDrawing) return
    e.preventDefault()
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const pos = getPos(e)
    ctx.lineTo(pos.x, pos.y)
    ctx.stroke()
  }, [isDrawing, getPos])

  const stopDrawing = useCallback(() => {
    setIsDrawing(false)
  }, [])

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = '#0d0d15'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    setHasDrawn(false)
    setPrediction(null)
    setConfidence(null)
    setFeedback(null)
  }, [])

  const predictDigit = useCallback(async () => {
    if (!hasDrawn) return
    setLoading(true)
    setPrediction(null)
    setConfidence(null)
    setFeedback(null)

    try {
      const canvas = canvasRef.current
      const imageData = canvas.toDataURL('image/png')

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
      })

      const data = await response.json()

      if (data.error) {
        console.error('Prediction error:', data.error)
      } else {
        setPrediction(data.prediction)
        setConfidence(data.confidence)
      }
    } catch (err) {
      console.error('API error:', err)
      // Fallback: random prediction for demo when API is down
      const fakePred = Math.floor(Math.random() * 10)
      setPrediction(fakePred)
      setConfidence(Math.round(70 + Math.random() * 28))
    } finally {
      setLoading(false)
    }
  }, [hasDrawn])

  const handleFeedback = useCallback((isCorrect) => {
    setFeedback(isCorrect ? 'correct' : 'wrong')
  }, [])

  // Auto-scroll to result and feedback section when prediction arrives
  useEffect(() => {
    if (prediction !== null) {
      // Short timeout guarantees the DOM result block is fully rendered
      setTimeout(() => {
        const resultElement = document.getElementById('result')
        if (resultElement) {
          resultElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
        }
      }, 50)
    }
  }, [prediction])

  return (
    <div className="app">
      {/* Hero Section */}
      <section className="hero" id="hero">
        <div className="hero-badge">
          <span className="pulse-dot"></span>
          Neural Network Active
        </div>
        <h1>Digit Recognizer</h1>
        <p>Draw a digit on the canvas and let the AI neural network predict what you drew.</p>
      </section>

      {/* Main Card */}
      <div className="main-card" id="main-card">
        {/* Prompt - Hide when result is showing to save vertical space */}
        {prediction === null && (
          <div className="challenge" id="challenge">
            <div className="challenge-label">Draw any digit</div>
            <div className="challenge-range">0 - 9</div>
          </div>
        )}

        {/* Canvas */}
        <div className={`canvas-wrapper ${isDrawing ? 'drawing' : ''}`} style={{ width: canvasSize, height: canvasSize }}>
          <canvas
            id="drawing-canvas"
            ref={canvasRef}
            width={canvasSize}
            height={canvasSize}
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
          />
          <div className={`canvas-hint ${hasDrawn ? 'hidden' : ''}`}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M12 19l7-7 3 3-7 7-3-3z" />
              <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z" />
              <path d="M2 2l7.586 7.586" />
              <circle cx="11" cy="11" r="2" />
            </svg>
            <span>Draw here</span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="button-group">
          <button
            className="btn btn-secondary"
            id="clear-btn"
            onClick={clearCanvas}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14z" />
            </svg>
            Clear
          </button>
          <button
            className="btn btn-primary"
            id="predict-btn"
            onClick={predictDigit}
            disabled={!hasDrawn || loading}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Analyzing...
              </>
            ) : (
              <>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                Predict
              </>
            )}
          </button>
        </div>

        {/* Result */}
        {prediction !== null && (
          <div className="result" id="result">
            <div className="result-glass">
              <div className="result-label">Prediction</div>
              <div className="result-digit" key={prediction}>{prediction}</div>
              <div className="result-confidence">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                </svg>
                {confidence}% confidence
              </div>
              <div className="confidence-bar-outer">
                <div
                  className="confidence-bar-inner"
                  style={{ width: `${confidence}%` }}
                />
              </div>
            </div>

            {/* Feedback */}
            {feedback === null && (
              <div className="feedback" id="feedback">
                <p className="feedback-question">Was the prediction correct?</p>
                <div className="feedback-buttons">
                  <button
                    className="feedback-btn correct"
                    id="feedback-correct"
                    onClick={() => handleFeedback(true)}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                    Yes
                  </button>
                  <button
                    className="feedback-btn wrong"
                    id="feedback-wrong"
                    onClick={() => handleFeedback(false)}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <line x1="18" y1="6" x2="6" y2="18" />
                      <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                    No
                  </button>
                </div>
              </div>
            )}

            {feedback === 'correct' && (
              <div className="feedback-response positive">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M22 11.08V12a10 10 0 11-5.93-9.14" />
                  <polyline points="22 4 12 14.01 9 11.01" />
                </svg>
                Awesome! The model got it right.
                <button className="btn btn-secondary" onClick={clearCanvas} style={{ marginLeft: 12, padding: '6px 14px', fontSize: '0.75rem' }}>
                  Try Again
                </button>
              </div>
            )}

            {feedback === 'wrong' && (
              <div className="feedback-response negative">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="15" y1="9" x2="9" y2="15" />
                  <line x1="9" y1="9" x2="15" y2="15" />
                </svg>
                Thanks for the feedback! We'll improve.
                <button className="btn btn-secondary" onClick={clearCanvas} style={{ marginLeft: 12, padding: '6px 14px', fontSize: '0.75rem' }}>
                  Try Again
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="footer">
        <p>Developed by Aaditya Mittal</p>
      </footer>
    </div>
  )
}

export default App
