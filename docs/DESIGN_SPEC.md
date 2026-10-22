# ATLEE SLM LAB — Visual Design & Frontend Architecture Specification

## 1. Product Personality & Aesthetic Direction
- **Identity:** High-density, professional AI research and developer laboratory for exploring, inspecting, and evaluating a causal Small Language Model (~3.36M parameters).
- **Tone:** Serious, deliberate, dark-mode technical workbench (reminiscent of deep research environments like Anthropic Workbench, Weights & Biases, and Google DeepMind).
- **Aesthetic Principles:**
  - Deep obsidian / charcoal backdrop with high-contrast text and crisp structural dividers.
  - Accent palette: Electric cyan (`#06b6d4` / `#22d3ee`) for primary actions and active generation states; muted violet/indigo (`#818cf8` / `#6366f1`) for model weights, token representations, and architectural metrics.
  - Functional data visualization: Horizontal probability bars with real percentages, clear token bounding boxes with `<UNK>` warning indicators, step-by-step token timeline scrubbers.
  - Restraint: Zero distracting neon glow, zero gratuitous 3D effects, no oversized hero sections pushing tools below the fold.

---

## 2. Information Architecture & Layout
The interface is structured into an intuitive 4-tab laboratory workbench:
1. **Interactive Playground (Core Tab):**
   - Left Column: Prompt editor with presets, generation parameters (Tokens, Mode, Temperature, Seed), primary action bar (`Generate`, `Clear`, `Copy`).
   - Right Column: Live Generation Output view (distinguishing prompt vs. continuation), generation telemetry (latency, token count, throughput), and the Next-Token Probability Explorer.
2. **Probability Explorer (Interactive Inspection View):**
   - Step selector timeline allowing scrubbing across each generated token step.
   - Top-K candidate probability distribution bars with exact values and selected token badge.
   - Transparency note explaining top-k coverage vs full 5,262-word vocabulary.
3. **Tokenizer Inspector Tab:**
   - Word-by-word token boundary visualization chips.
   - Integer token ID badges.
   - Distinctive warning styling for out-of-vocabulary `<UNK>` tokens.
   - Diagnostic panel with vocabulary size, sequence count, and whitespace tokenization explainer.
4. **Architecture & Diagnostics Tab:**
   - Parameter breakdown cards (Total parameters, Embedding matrix, Transformer block, Output dense layer).
   - Live system health metrics from `/health` and `/config`.
   - Real training loss progression chart (from `checkpoints/training_history.json`).
   - Real benchmark report (CPU latency, throughput, non-held-out perplexity clarification).

---

## 3. Design Tokens (CSS Custom Properties)

```css
:root {
  /* Color Palette */
  --bg-app: #090d16;
  --bg-surface: #0f172a;
  --bg-surface-elevated: #1e293b;
  --bg-surface-hover: #334155;
  --border-subtle: #1e293b;
  --border-strong: #334155;
  --border-focus: #06b6d4;

  /* Typography Colors */
  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --text-disabled: #475569;

  /* Accents */
  --cyan-primary: #06b6d4;
  --cyan-glow: rgba(6, 182, 212, 0.15);
  --cyan-text: #22d3ee;
  --violet-primary: #818cf8;
  --violet-glow: rgba(129, 140, 248, 0.15);
  --violet-text: #a5b4fc;

  /* Semantic Feedback */
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
  --info: #38bdf8;

  /* Typography Scale */
  --font-mono: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  
  /* Radii & Spacing */
  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
  --radius-xl: 12px;
}
```

---

## 4. Accessibility & Interaction Specifications
- **WCAG 2.1 AA Compliance:** Minimum 4.5:1 text-to-background contrast on all body elements.
- **Visible Keyboard Focus:** Distinct cyan outline (`2px solid var(--cyan-primary)`) with offset.
- **Form Controls:** Semantic `<label>` elements with explicit `for` attributes.
- **Safe Rendering:** All text insertions use `textContent` to prevent script execution vulnerabilities.
- **Reduced Motion:** `@media (prefers-reduced-motion: reduce)` disables all smooth transitions and animations.
