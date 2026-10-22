/**
 * ATLEE SLM LAB — Interactive Playground Logic
 * Connects directly to the real Keras/JAX FastAPI backend.
 */

// Application State
const state = {
  currentTab: 'playgroundTab',
  serverHealth: null,
  modelConfig: null,
  evaluationData: null,
  latestGeneration: null,
  currentStepIndex: 0,
};

// DOM Element Selectors
const elements = {
  // Navigation
  navTabs: document.querySelectorAll('.nav-tab'),
  tabPanes: document.querySelectorAll('.tab-pane'),

  // Header Status
  brandBadge: document.getElementById('brandBadge'),
  statusPill: document.getElementById('connectionStatusPill'),
  statusText: document.getElementById('connectionStatusText'),
  backendDeviceText: document.getElementById('backendDeviceText'),

  // Playground Form Controls
  modelSelect: document.getElementById('modelSelect'),
  modelParamCountTag: document.getElementById('modelParamCountTag'),
  promptInput: document.getElementById('promptInput'),
  promptCharCount: document.getElementById('promptCharCount'),
  maxTokensInput: document.getElementById('maxTokensInput'),
  maxTokensVal: document.getElementById('maxTokensVal'),
  samplingModeSelect: document.getElementById('samplingModeSelect'),
  tempControlRow: document.getElementById('tempControlRow'),
  temperatureInput: document.getElementById('temperatureInput'),
  temperatureVal: document.getElementById('temperatureVal'),
  seedControlRow: document.getElementById('seedControlRow'),
  seedInput: document.getElementById('seedInput'),
  generateBtn: document.getElementById('generateBtn'),
  generateBtnText: document.getElementById('generateBtnText'),
  clearBtn: document.getElementById('clearBtn'),
  presetBtns: document.querySelectorAll('.preset-btn'),

  // Output Elements
  textContentBox: document.getElementById('textContentBox'),
  latencyPill: document.getElementById('latencyPill'),
  throughputPill: document.getElementById('throughputPill'),
  copyTextBtn: document.getElementById('copyTextBtn'),
  copyToast: document.getElementById('copyToast'),
  playgroundError: document.getElementById('playgroundError'),
  playgroundErrorMsg: document.getElementById('playgroundErrorMsg'),

  // Probability Explorer
  probBarsContainer: document.getElementById('probBarsContainer'),
  stepNavControls: document.getElementById('stepNavControls'),
  stepChipsContainer: document.getElementById('stepChipsContainer'),
  stepIndicatorText: document.getElementById('stepIndicatorText'),
  prevStepBtn: document.getElementById('prevStepBtn'),
  nextStepBtn: document.getElementById('nextStepBtn'),

  // Tokenizer Inspector
  tokenizerTextInput: document.getElementById('tokenizerTextInput'),
  inspectTokenizeBtn: document.getElementById('inspectTokenizeBtn'),
  tokenCountMetric: document.getElementById('tokenCountMetric'),
  vocabSizeMetric: document.getElementById('vocabSizeMetric'),
  unkCountMetric: document.getElementById('unkCountMetric'),
  padIdMetric: document.getElementById('padIdMetric'),
  tokenChipsDisplay: document.getElementById('tokenChipsDisplay'),

  // Evaluation & Architecture
  lossChartContainer: document.getElementById('lossChartContainer'),
  archCheckpointBadge: document.getElementById('archCheckpointBadge'),
};

// ============================================================================
// Initialization & Backend Health Checking
// ============================================================================

async function checkHealthAndInitialize() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const health = await res.json();
    state.serverHealth = health;

    // Update Header Status
    elements.statusPill.className = 'status-pill status-ready';
    if (health.checkpoint_loaded) {
      elements.statusText.textContent = 'Model Ready (Checkpoint Loaded)';
    } else {
      elements.statusText.textContent = 'Model Ready (Untrained Baseline)';
    }

    elements.backendDeviceText.textContent = `${health.device.toUpperCase()}`;

    // Architecture Checkpoint Badge
    if (elements.archCheckpointBadge) {
      if (health.checkpoint_loaded) {
        elements.archCheckpointBadge.className = 'status-pill status-ready';
        elements.archCheckpointBadge.textContent = 'Checkpoint Loaded';
      } else {
        elements.archCheckpointBadge.className = 'status-pill status-warning';
        elements.archCheckpointBadge.textContent = 'Baseline (No Checkpoint)';
      }
    }

    // Fetch config and evaluation metrics
    fetchModelConfig();
    fetchEvaluationMetrics();
  } catch (err) {
    console.error('API Health check failed:', err);
    elements.statusPill.className = 'status-pill status-error';
    elements.statusText.textContent = 'API Offline';
    elements.backendDeviceText.textContent = 'Disconnected';
  }
}

async function fetchModelConfig() {
  try {
    const res = await fetch('/config');
    if (res.ok) {
      state.modelConfig = await res.json();
      const cfg = state.modelConfig;
      const totalParamsEl = document.getElementById('totalParamsMetric');
      if (totalParamsEl && cfg.total_parameters) {
        totalParamsEl.textContent = cfg.total_parameters.toLocaleString();
      }
      if (elements.vocabSizeMetric && cfg.vocabulary_size) {
        elements.vocabSizeMetric.textContent = cfg.vocabulary_size.toLocaleString();
      }
      const panelTag = document.getElementById('panelModelTag');
      if (panelTag) {
        panelTag.textContent = cfg.tokenizer_type === 'bpe' ? 'BPE Subword' : 'Word-Level';
      }
    }
  } catch (err) {
    console.warn('Could not fetch config:', err);
  }
}

async function fetchEvaluationMetrics() {
  try {
    const res = await fetch('/evaluation');
    if (res.ok) {
      const data = await res.json();
      state.evaluationData = data;
      renderLossChart(data.training_loss_history || []);
    }
  } catch (err) {
    console.warn('Could not fetch evaluation metrics:', err);
  }
}

// ============================================================================
// Navigation Tabs
// ============================================================================

elements.navTabs.forEach(tabBtn => {
  tabBtn.addEventListener('click', () => {
    const targetPaneId = tabBtn.getAttribute('data-tab');

    elements.navTabs.forEach(b => b.classList.remove('active'));
    elements.tabPanes.forEach(p => p.classList.remove('active'));

    tabBtn.classList.add('active');
    const targetPane = document.getElementById(targetPaneId);
    if (targetPane) {
      targetPane.classList.add('active');
      state.currentTab = targetPaneId;
    }

    // Auto-analyze tokenizer when switching to tokenizer tab if chips are empty
    if (targetPaneId === 'tokenizerTab' && elements.tokenChipsDisplay.children.length <= 1) {
      analyzeTokenizer();
    }
  });
});

// ============================================================================
// Playground Control Handlers
// ============================================================================

// Prompt Character Counter
elements.promptInput.addEventListener('input', () => {
  const len = elements.promptInput.value.length;
  elements.promptCharCount.textContent = `${len} / 2000`;
});

// Max Tokens Slider
elements.maxTokensInput.addEventListener('input', () => {
  elements.maxTokensVal.textContent = elements.maxTokensInput.value;
});

// Temperature Slider
elements.temperatureInput.addEventListener('input', () => {
  elements.temperatureVal.textContent = parseFloat(elements.temperatureInput.value).toFixed(2);
});

// Sampling Mode Switcher
elements.samplingModeSelect.addEventListener('change', () => {
  const isRandom = elements.samplingModeSelect.value === 'random';
  elements.tempControlRow.style.display = isRandom ? 'flex' : 'none';
  elements.seedControlRow.style.display = isRandom ? 'flex' : 'none';
});

// Model Architecture Selector
if (elements.modelSelect) {
  elements.modelSelect.addEventListener('change', async () => {
    const selectedId = elements.modelSelect.value;
    try {
      const res = await fetch('/models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: selectedId }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      // Update badge in top header
      if (elements.brandBadge) {
        if (selectedId === 'slm_bpe_val') {
          elements.brandBadge.textContent = 'BPE-1.68M';
        } else if (selectedId === 'slm_baseline_val') {
          elements.brandBadge.textContent = 'WORD-2.92M';
        } else {
          elements.brandBadge.textContent = 'SLM-3.36M';
        }
      }

      if (elements.modelParamCountTag) {
        elements.modelParamCountTag.textContent = `${(data.parameters / 1e6).toFixed(2)}M params`;
      }

      // Refresh config and token metrics
      await fetchModelConfig();
      if (state.currentTab === 'tokenizerTab') {
        analyzeTokenizer();
      }
    } catch (err) {
      console.error('Failed to switch model:', err);
      showPlaygroundError(`Model switch failed: ${err.message}`);
    }
  });
}

// Starter Presets
elements.presetBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const preset = btn.getAttribute('data-preset');
    if (preset) {
      elements.promptInput.value = preset;
      elements.promptCharCount.textContent = `${preset.length} / 2000`;
      elements.promptInput.focus();
    }
  });
});

// Clear Button
elements.clearBtn.addEventListener('click', () => {
  elements.promptInput.value = '';
  elements.promptCharCount.textContent = '0 / 2000';
  elements.textContentBox.innerHTML = '<span class="placeholder-text">Click "Generate Tokens" to run real inference...</span>';
  elements.latencyPill.textContent = 'Latency: -- ms';
  elements.throughputPill.textContent = 'Speed: -- tok/s';
  elements.copyTextBtn.disabled = true;
  elements.probBarsContainer.innerHTML = '<div class="empty-state"><p>Generate tokens to inspect step-by-step top-5 candidate probability distributions.</p></div>';
  elements.stepNavControls.style.display = 'none';
  elements.stepChipsContainer.style.display = 'none';
  elements.playgroundError.style.display = 'none';
  state.latestGeneration = null;
});

// Copy Output Text
elements.copyTextBtn.addEventListener('click', async () => {
  if (!state.latestGeneration) return;
  try {
    await navigator.clipboard.writeText(state.latestGeneration.full_text);
    elements.copyToast.style.display = 'inline';
    setTimeout(() => {
      elements.copyToast.style.display = 'none';
    }, 2000);
  } catch (err) {
    console.error('Clipboard copy failed:', err);
  }
});

// ============================================================================
// Autoregressive Generation Action
// ============================================================================

elements.generateBtn.addEventListener('click', handleGenerate);

async function handleGenerate() {
  const prompt = elements.promptInput.value.trim();
  if (!prompt) {
    showPlaygroundError('Please enter a prompt or select a starter preset.');
    elements.promptInput.focus();
    return;
  }

  // Clear previous errors
  elements.playgroundError.style.display = 'none';

  // Read Parameters
  const maxNewTokens = parseInt(elements.maxTokensInput.value, 10) || 15;
  const samplingMode = elements.samplingModeSelect.value;
  const temperature = parseFloat(elements.temperatureInput.value) || 1.0;
  const seedRaw = elements.seedInput.value.trim();
  const seed = seedRaw !== '' ? parseInt(seedRaw, 10) : null;

  // Build Payload
  const payload = {
    prompt: prompt,
    max_new_tokens: maxNewTokens,
    sampling_mode: samplingMode,
    temperature: temperature,
    seed: seed,
    top_k_probabilities: 5,
  };

  // UI Loading State
  setLoadingState(true);
  const clientStart = performance.now();

  try {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const clientDuration = performance.now() - clientStart;

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error (HTTP ${res.status})`);
    }

    const data = await res.json();
    state.latestGeneration = data;

    // Display Text in Output Viewer
    renderGenerationOutput(data);

    // Update Telemetry Indicators
    const serverLatency = data.latency_ms;
    const tokenCount = data.generated_token_ids ? data.generated_token_ids.length : maxNewTokens;
    const tokensPerSec = serverLatency > 0 ? ((tokenCount / (serverLatency / 1000))).toFixed(1) : '--';

    elements.latencyPill.textContent = `API: ${serverLatency.toFixed(0)}ms | Total: ${clientDuration.toFixed(0)}ms`;
    elements.throughputPill.textContent = `${tokensPerSec} tok/s`;
    elements.copyTextBtn.disabled = false;

    // Render Next-Token Probability Explorer
    renderProbabilityExplorer(data);
  } catch (err) {
    console.error('Generation failed:', err);
    showPlaygroundError(err.message || 'Generation failed. Check server connection.');
  } finally {
    setLoadingState(false);
  }
}

function setLoadingState(isLoading) {
  elements.generateBtn.disabled = isLoading;
  elements.generateBtnText.textContent = isLoading ? 'Generating Tokens...' : 'Generate Tokens';
  if (isLoading) {
    elements.textContentBox.innerHTML = '<span class="placeholder-text">Running Transformer causal attention and predicting tokens...</span>';
    elements.probBarsContainer.innerHTML = '<div class="empty-state"><p>Processing probability distribution...</p></div>';
  }
}

function showPlaygroundError(message) {
  elements.playgroundErrorMsg.textContent = message;
  elements.playgroundError.style.display = 'flex';
}

function renderGenerationOutput(data) {
  elements.textContentBox.innerHTML = '';

  const promptSpan = document.createElement('span');
  promptSpan.className = 'text-prompt';
  promptSpan.textContent = data.prompt + (data.prompt.endsWith(' ') ? '' : ' ');

  const genSpan = document.createElement('span');
  genSpan.className = 'text-continuation';
  genSpan.textContent = data.generated_text;

  elements.textContentBox.appendChild(promptSpan);
  elements.textContentBox.appendChild(genSpan);
}

// ============================================================================
// Probability Explorer (Step-by-Step Visualization)
// ============================================================================

function renderProbabilityExplorer(data) {
  const steps = data.step_top_candidates;
  if (!steps || steps.length === 0) {
    elements.probBarsContainer.innerHTML = '<div class="empty-state"><p>No step probability data returned by the server.</p></div>';
    elements.stepNavControls.style.display = 'none';
    elements.stepChipsContainer.style.display = 'none';
    return;
  }

  state.currentStepIndex = 0;
  elements.stepNavControls.style.display = 'flex';
  elements.stepChipsContainer.style.display = 'flex';

  renderStepChips(steps, data.generated_tokens);
  displayStepProbabilities(state.currentStepIndex);
}

function renderStepChips(steps, generatedTokens) {
  elements.stepChipsContainer.innerHTML = '';

  steps.forEach((step, idx) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = `step-chip ${idx === state.currentStepIndex ? 'active' : ''}`;
    const tokenLabel = generatedTokens && generatedTokens[idx] ? generatedTokens[idx] : `#${idx + 1}`;
    chip.textContent = `${idx + 1}: ${tokenLabel}`;
    chip.setAttribute('aria-label', `Inspect Generation Step ${idx + 1}`);

    chip.addEventListener('click', () => {
      state.currentStepIndex = idx;
      updateActiveStepChip();
      displayStepProbabilities(idx);
    });

    elements.stepChipsContainer.appendChild(chip);
  });
}

function updateActiveStepChip() {
  const chips = elements.stepChipsContainer.querySelectorAll('.step-chip');
  chips.forEach((c, idx) => {
    if (idx === state.currentStepIndex) {
      c.classList.add('active');
      c.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    } else {
      c.classList.remove('active');
    }
  });

  const totalSteps = state.latestGeneration.step_top_candidates.length;
  elements.stepIndicatorText.textContent = `Step ${state.currentStepIndex + 1} / ${totalSteps}`;
  elements.prevStepBtn.disabled = state.currentStepIndex === 0;
  elements.nextStepBtn.disabled = state.currentStepIndex === totalSteps - 1;
}

elements.prevStepBtn.addEventListener('click', () => {
  if (state.currentStepIndex > 0) {
    state.currentStepIndex--;
    updateActiveStepChip();
    displayStepProbabilities(state.currentStepIndex);
  }
});

elements.nextStepBtn.addEventListener('click', () => {
  const total = state.latestGeneration.step_top_candidates.length;
  if (state.currentStepIndex < total - 1) {
    state.currentStepIndex++;
    updateActiveStepChip();
    displayStepProbabilities(state.currentStepIndex);
  }
});

function displayStepProbabilities(stepIdx) {
  const data = state.latestGeneration;
  if (!data || !data.step_top_candidates || !data.step_top_candidates[stepIdx]) return;

  const candidates = data.step_top_candidates[stepIdx];
  const chosenToken = data.generated_tokens[stepIdx];

  elements.probBarsContainer.innerHTML = '';

  candidates.forEach(cand => {
    const row = document.createElement('div');
    row.className = 'prob-row';

    const isSelected = cand.token === chosenToken;

    // Token Label with badge if selected
    const labelSpan = document.createElement('div');
    labelSpan.className = 'prob-token-label';
    labelSpan.textContent = cand.token;

    if (isSelected) {
      const badge = document.createElement('span');
      badge.className = 'selected-badge';
      badge.textContent = 'SELECTED';
      labelSpan.appendChild(badge);
    }

    // Probability Bar
    const track = document.createElement('div');
    track.className = 'prob-bar-track';

    const fill = document.createElement('div');
    fill.className = `prob-bar-fill ${isSelected ? 'selected' : ''}`;
    const pct = Math.max(0, Math.min(100, cand.probability * 100));
    fill.style.width = `${pct.toFixed(2)}%`;
    track.appendChild(fill);

    // Probability Number
    const valSpan = document.createElement('div');
    valSpan.className = 'prob-val-text';
    valSpan.textContent = `${pct.toFixed(2)}%`;

    row.appendChild(labelSpan);
    row.appendChild(track);
    row.appendChild(valSpan);
    elements.probBarsContainer.appendChild(row);
  });
}

// ============================================================================
// Tokenizer Inspector
// ============================================================================

elements.inspectTokenizeBtn.addEventListener('click', analyzeTokenizer);

async function analyzeTokenizer() {
  const text = elements.tokenizerTextInput.value.trim();
  if (!text) {
    elements.tokenChipsDisplay.innerHTML = '<p class="placeholder-text">Enter text to tokenize.</p>';
    elements.tokenCountMetric.textContent = '0';
    elements.unkCountMetric.textContent = '0';
    return;
  }

  try {
    const res = await fetch('/tokenize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    elements.tokenCountMetric.textContent = data.token_count.toLocaleString();
    elements.vocabSizeMetric.textContent = data.vocabulary_size.toLocaleString();
    elements.unkCountMetric.textContent = data.unknown_token_count.toLocaleString();

    // Render Chips
    elements.tokenChipsDisplay.innerHTML = '';

    data.tokens.forEach((token, idx) => {
      const tid = data.token_ids[idx];
      const isUnk = tid === 5261 || token === '<UNK>';

      const chip = document.createElement('div');
      chip.className = `token-chip-item ${isUnk ? 'is-unk' : ''}`;

      const wordSpan = document.createElement('span');
      wordSpan.className = 'token-chip-word';
      wordSpan.textContent = token;

      const idSpan = document.createElement('span');
      idSpan.className = 'token-chip-id';
      idSpan.textContent = `#${tid}`;

      chip.appendChild(wordSpan);
      chip.appendChild(idSpan);

      if (isUnk) {
        const unkTag = document.createElement('span');
        unkTag.className = 'unk-tag';
        unkTag.textContent = '<UNK>';
        chip.appendChild(unkTag);
      }

      elements.tokenChipsDisplay.appendChild(chip);
    });
  } catch (err) {
    console.error('Tokenizer inspection error:', err);
    elements.tokenChipsDisplay.innerHTML = `<p class="placeholder-text">Tokenizer request failed: ${err.message}</p>`;
  }
}

// ============================================================================
// Training Loss Chart Renderer
// ============================================================================

function renderLossChart(lossList) {
  if (!elements.lossChartContainer || !lossList || lossList.length === 0) return;

  elements.lossChartContainer.innerHTML = '';

  const minLoss = Math.min(...lossList);
  const maxLoss = Math.max(...lossList);
  const range = maxLoss - minLoss || 1.0;

  lossList.forEach((lossVal, idx) => {
    const col = document.createElement('div');
    col.className = 'loss-bar-col';
    col.title = `Epoch ${idx + 1}: loss = ${lossVal.toFixed(4)}`;

    const bar = document.createElement('div');
    bar.className = 'loss-bar-rect';
    // Calculate normalized percentage height
    const normalizedHeight = 15 + ((lossVal - minLoss) / range) * 85;
    bar.style.height = `${normalizedHeight.toFixed(1)}%`;

    const label = document.createElement('span');
    label.className = 'loss-bar-label';
    label.textContent = (idx + 1) % 5 === 0 || idx === 0 || idx === lossList.length - 1 ? `${idx + 1}` : '';

    col.appendChild(bar);
    col.appendChild(label);
    elements.lossChartContainer.appendChild(col);
  });
}

// ============================================================================
// Startup Event Listener
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  // Initialize character count for default prefilled prompt
  if (elements.promptInput && elements.promptCharCount) {
    elements.promptCharCount.textContent = `${elements.promptInput.value.length} / 2000`;
  }
  checkHealthAndInitialize();
});
