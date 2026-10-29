import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function scoreToPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function riskClass(label) {
  const safe = String(label || "").toLowerCase();
  if (safe.includes("high")) {
    return "risk-high";
  }
  if (safe.includes("moderate")) {
    return "risk-moderate";
  }
  return "risk-low";
}

function emotionClass(intensity) {
  const safe = String(intensity || "none").toLowerCase();
  if (safe === "high") {
    return "emotion-high";
  }
  if (safe === "moderate") {
    return "emotion-moderate";
  }
  if (safe === "low") {
    return "emotion-low";
  }
  return "emotion-none";
}

function interpretationFromRisk(label) {
  const safe = String(label || "").toLowerCase();
  if (safe.includes("high")) {
    return {
      shortLabel: "High Concern",
      detail:
        "This indicates stronger signs of emotional distress and possible depressive symptoms. Consider professional mental health support promptly.",
    };
  }
  if (safe.includes("moderate")) {
    return {
      shortLabel: "Possible Concern",
      detail:
        "This suggests noticeable emotional strain. It may benefit from follow-up screening, self-care, and professional guidance if symptoms persist.",
    };
  }
  return {
    shortLabel: "Likely Normal / Low Concern",
    detail:
      "This indicates low current risk in the provided input. Continue healthy routines and monitor for any sustained mood changes.",
  };
}

function riskGradient(label) {
  const safe = String(label || "").toLowerCase();
  if (safe.includes("high")) {
    return "analysis-high";
  }
  if (safe.includes("moderate")) {
    return "analysis-moderate";
  }
  return "analysis-low";
}

function faceSignalBadge(state) {
  const safe = String(state || "").toLowerCase();
  if (safe === "used") {
    return { text: "Face signal used", className: "signal-used" };
  }
  if (safe === "ignored") {
    return { text: "Face signal ignored", className: "signal-ignored" };
  }
  if (safe === "skipped_crisis") {
    return { text: "Face skipped for safety", className: "signal-skipped" };
  }
  return { text: "Text only", className: "signal-text" };
}

export default function App() {
  const [text, setText] = useState("");
  const [modelName, setModelName] = useState("tfidf");
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState("");

  const canSubmit = useMemo(
    () => text.trim().length > 0 && !loading,
    [text, loading],
  );

  useEffect(() => {
    if (!image) {
      setImagePreviewUrl("");
      return;
    }

    const url = URL.createObjectURL(image);
    setImagePreviewUrl(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [image]);

  async function handleAnalyze(event) {
    event.preventDefault();
    setError("");
    setResult(null);

    if (!text.trim()) {
      setError("Please enter your text before analyzing.");
      return;
    }

    const formData = new FormData();
    formData.append("text", text);
    formData.append("model_name", modelName);
    if (image) {
      formData.append("image", image);
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to analyze input.");
      }

      setResult(payload);
    } catch (err) {
      setError(err.message || "Unexpected error while analyzing input.");
    } finally {
      setLoading(false);
    }
  }

  function downloadReport() {
    if (!result?.report_text) {
      return;
    }

    const blob = new Blob([result.report_text], {
      type: "text/plain;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `mind_aid_report_${Date.now()}.txt`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  const emotionLabel =
    result && typeof result.emotion === "string" && result.emotion.trim()
      ? result.emotion
      : "Not Available";

  const resultInterpretation = result
    ? interpretationFromRisk(result.risk_label)
    : null;

  const riskProbability = result
    ? Number(result.depression_probability ?? result.final_score ?? 0)
    : 0;
  const faceSignal = result ? faceSignalBadge(result.face_signal_state) : null;

  return (
    <div className="page">
      <div className="bg-orb orb-one" aria-hidden="true" />
      <div className="bg-orb orb-two" aria-hidden="true" />

      <header className="topbar">
        <div>
          <p className="eyebrow">AI Mental Health Screening</p>
          <h1>MIND-AID Dashboard</h1>
          <p>Clean, reliable multimodal emotional risk assessment.</p>
        </div>
        <div className="topbar-side">
          <div className="topbar-chip">Clinical Support Tool</div>
          <div className="topbar-mini-grid">
            <div className="mini-card">
              <span>Mode</span>
              <strong>{image ? "Multimodal" : "Text-first"}</strong>
            </div>
            <div className="mini-card">
              <span>Status</span>
              <strong>{loading ? "Analyzing" : "Ready"}</strong>
            </div>
          </div>
        </div>
      </header>

      <main className="layout">
        <section className="card gradient-card">
          <h2>Assessment Input</h2>
          <form onSubmit={handleAnalyze} className="form">
            <label htmlFor="thoughts">Your text</label>
            <textarea
              id="thoughts"
              rows={7}
              placeholder="Write how you are feeling today..."
              value={text}
              onChange={(event) => setText(event.target.value)}
            />

            <label htmlFor="model">Text model</label>
            <select
              id="model"
              value={modelName}
              onChange={(event) => setModelName(event.target.value)}
              disabled={loading}
            >
              <option value="tfidf">TF-IDF + LinearSVC (fast)</option>
              <option value="distilbert">Fine-tuned DistilBERT (semantic)</option>
            </select>
            <p className="upload-help">
              DistilBERT provides stronger semantic modeling and token-level
              SHAP explanations, but analysis takes longer.
            </p>

            <label htmlFor="image">Optional face image</label>
            <input
              id="image"
              type="file"
              accept=".jpg,.jpeg,.png"
              onChange={(event) => setImage(event.target.files?.[0] || null)}
            />

            <p className="upload-help">
              Tip: Use a clear, front-facing photo in good lighting for better
              facial emotion detection.
            </p>

            {imagePreviewUrl ? (
              <div className="upload-preview">
                <p>Image Preview</p>
                <img src={imagePreviewUrl} alt="Uploaded face preview" />
              </div>
            ) : null}

            <button type="submit" disabled={!canSubmit}>
              {loading ? "Analyzing..." : "Analyze"}
            </button>
          </form>

          {error ? <p className="error">{error}</p> : null}
        </section>

        <section className="card info-card">
          <h2>How it works</h2>
          <ul>
            <li>Text analysis always runs.</li>
            <li>Face analysis runs only if image is uploaded.</li>
            <li>Scores are fused into one final risk label.</li>
          </ul>
        </section>

        {!result ? (
          <section className="card full-width empty-state">
            <h2>Waiting for Analysis</h2>
            <p>
              Enter your text and optionally upload a face image to generate a
              fused emotional risk report with confidence and key factors.
            </p>
          </section>
        ) : null}

        {result ? (
          <section className="card full-width">
            <div className="results-header-row">
              <h2>Results</h2>
              <span className={`risk-pill ${riskClass(result.risk_label)}`}>
                {result.risk_label}
              </span>
            </div>
            <p className="muted">
              Model: {result.model_label || result.model || "Unknown"} ·
              Explanation: {result.explanation_method || "Model attribution"}
            </p>

            <div className="metrics">
              <article className="metric metric-risk">
                <span>Screening Label</span>
                <strong>{result.screening_label || result.risk_label}</strong>
              </article>
              <article className="metric metric-score">
                <span>Depression Probability</span>
                <strong>{scoreToPercent(result.final_score)}</strong>
              </article>
              <article className="metric metric-confidence">
                <span>Confidence</span>
                <strong>
                  {Number(result.confidence_pct || 0).toFixed(1)}%
                </strong>
              </article>
              <article
                className={`metric metric-emotion ${emotionClass(result.emotion_intensity)}`}
              >
                <span>Emotion</span>
                <strong>{emotionLabel}</strong>
              </article>
            </div>

            {faceSignal ? (
              <div className={`signal-pill ${faceSignal.className}`}>
                {faceSignal.text}
              </div>
            ) : null}

            {result.crisis_flag ? (
              <div className="crisis-banner">
                <strong>Urgent language detected.</strong>
                <p>{result.crisis_message}</p>
                <p>
                  Matched phrases: {result.crisis_phrases?.join(", ") || "none"}
                </p>
              </div>
            ) : null}

            <div className="bars">
              <div className="bar-row">
                <div className="bar-label-line">
                  <span>Text Depression Probability</span>
                  <span>
                    {scoreToPercent(
                      result.depression_probability ?? result.text_score,
                    )}
                  </span>
                </div>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${Math.min(
                        100,
                        Math.max(0, riskProbability * 100),
                      ).toFixed(1)}%`,
                    }}
                  />
                </div>
              </div>
              <div className="bar-row">
                <div className="bar-label-line">
                  <span>Final Screening Risk</span>
                  <span>{scoreToPercent(result.final_score)}</span>
                </div>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${Math.min(
                        100,
                        Math.max(0, Number(result.final_score || 0) * 100),
                      ).toFixed(1)}%`,
                    }}
                  />
                </div>
              </div>
            </div>

            <div className="subgrid">
              <div>
                <h3>Score Breakdown</h3>
                <p>
                  Text depression probability:{" "}
                  {scoreToPercent(
                    result.depression_probability ?? result.text_score,
                  )}
                </p>
                <p>
                  Final screening risk: {scoreToPercent(result.final_score)}
                </p>
                <p className="muted">{result.face_note}</p>
              </div>
              <div>
                <h3>Keywords</h3>
                {result.keywords?.length ? (
                  <div className="chips">
                    {result.keywords.map((keyword) => (
                      <span key={keyword} className="chip">
                        #{keyword}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="muted">No significant keywords identified.</p>
                )}
              </div>
            </div>

            <section className="interpretation-box">
              <h3>How to Understand This Result</h3>
              <div
                className={`interpretation-current ${riskClass(result.risk_label)} ${riskGradient(result.risk_label)}`}
              >
                <strong>
                  Current Category: {resultInterpretation.shortLabel}
                </strong>
                <p>{resultInterpretation.detail}</p>
              </div>
              <div className="interpretation-scale">
                <p>
                  <strong>Low Risk:</strong> Likely normal / low concern
                </p>
                <p>
                  <strong>Moderate Risk:</strong> Possible emotional concern
                </p>
                <p>
                  <strong>High Risk:</strong> High emotional concern, consider
                  professional support
                </p>
              </div>
              <p className="muted interpret-note">
                This dashboard is a screening aid, not a medical diagnosis. A
                high result means the text contains strong depression-related or
                crisis-like signals.
              </p>
            </section>

            <button
              type="button"
              className="secondary"
              onClick={downloadReport}
            >
              Download Report
            </button>
          </section>
        ) : null}
      </main>
    </div>
  );
}
