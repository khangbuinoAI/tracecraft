# AGENTS.md — TraceCraft Project Standards & Architecture Guidelines

Strict operational directives for AI agents operating inside the `TraceCraft` repository (`https://github.com/khangbuinoAI/tracecraft`).

---

## 1. Core Architectural Standard for All Lessons

Every lesson added to TraceCraft **MUST** follow the established unified layout, backend contract, and interactive design system. Never introduce fragmented or ad-hoc layouts.

### Canonical Stage Anatomy (Top to Bottom):
1. **Top Stage Bar**:
   - Left: `{Category} Stage` badge + `Step {current} / {total}` indicator.
   - Right: Floating Zoom Toolbar (`ZoomIn`, `ZoomOut`, `Reset`, `Maximize`).
2. **Interactive Lesson Control Card (`LessonControlHeader.tsx`)**:
   - Outer container: `rounded-2xl border overflow-hidden shadow-xs mb-2`.
   - **Left side**:
     - Rounded topic icon in soft container (`Zap`, `Binary`, `BrainCircuit`, `Grid`, `Home`, `Sparkles`, `Edit3`, `Bot`, `TrendingUp`, `Activity`, `Target`).
     - Bold title (`text-sm font-bold tracking-tight whitespace-nowrap`).
     - Concept badge in pill container (`text-[10px] font-mono px-2 py-0.5 rounded-full border`).
     - Dynamic subtitle displaying live mutated variables (e.g. current inputs, excitation $z$, weights $w$, bias $b$, predicted probabilities, or loss).
   - **Right side**:
     - Interactive Data Presets toolbar (`Sliders` icon + clean buttons mapped from `preset.inputVariants`).
     - Clicking any preset immediately invokes `onChangeInput(newVal)`, updating inputs and re-running traces live.
3. **Real-World Impact Banner (`Ý nghĩa thực tế:`)**:
   - `px-4 py-2 border-t flex items-center gap-2 text-xs`.
   - Info icon (`Info` in cyan/blue) followed by **`Ý nghĩa thực tế:`** (or **`Real-World Impact:`** in English).
   - Explains *why* the algorithm behaves this way, what engineering/domain problem it solves, and its real-world intuition (e.g., biological soma excitation, vanishing gradient in step functions, cancer screening recall asymmetry).
4. **Central Visual Canvas**:
   - Interactive SVG/Canvas visualization dedicated to the visual type (`regression-plot`, `logistic-regression`, `confusion-matrix`, `single-neuron`, `modern-neuron`, `neural-network`, `cnn-convolution`, `mnist-interactive`, `attention-matrix`).
5. **Bottom Invariant Strip**:
   - Displays active mathematical invariant formula using KaTeX (`MathFormula.tsx`).
6. **Floating DockPlayer**:
   - Bottom scrubber timeline, single-step buttons (`Prev`, `Next`), `Play`/`Pause`, and speed selector (`0.5x` to `2x`).

---

## 2. Backend Contract (`src/algorithms/presets.ts` & `src/types.ts`)

Every algorithm preset registered in `ALGORITHM_PRESETS` **MUST** implement the `AlgorithmPreset` interface with the following schema:

```typescript
export interface AlgorithmPreset {
  id: string;                                   // Unique slug (e.g. 'confusion-matrix')
  title: string;                                // e.g. '03. Confusion Matrix & Classification Metrics'
  category: AICategory;                         // 'Classical Machine Learning' | 'Deep Learning & Neural Networks' | 'Transformers & Modern LLMs'
  difficulty: DifficultyLevel;                  // 'Beginner' | 'Intermediate' | 'Advanced'
  badge: string;                                // Short domain pill (e.g. 'Evaluation Metrics')
  visualType: VisualType;                       // Stage router key
  formula: string;                              // KaTeX string for top card formula
  summary: string;                              // 1-2 sentence catalog summary
  description: string;                          // Comprehensive lesson overview
  sections: LessonSection[];                    // 4 detailed theory sections with formulas and key takeaways
  timeComplexity: string;                       // e.g. 'O(N)'
  spaceComplexity: string;                      // e.g. 'O(1)'
  defaultInput: string;                         // Default input configuration string
  inputVariants: { label: string; value: string }[]; // Selectable data presets
  tags: string[];                               // Search tags
  pythonCode: string;                           // Standard reference implementation
  code: string;                                 // Line-numbered execution code for 3-pane code runner
  vi?: LocalizedPresetInfo;                     // Complete Vietnamese translation (title, badge, summary, description, variants, sections)
  generateTraces: (inputStr?: string, lang?: Language) => TraceStep[]; // Step generator
}
```

### Trace Step Generator Guidelines:
- Return an array of `TraceStep` objects (typically 4–8 distinct steps: `init`, `forward`, `compute`, `backward`, `update`, `done`).
- Each step **MUST** set `line` matching the active line in `code`.
- Supply rich `explanation` and `invariant` strings in both Vietnamese and English (`lang === 'vi'`).
- Mutate state fields deterministically so the visual stage reflects real computations.

---

## 3. Bilingual Support Invariant (EN & VI)

- TraceCraft is strictly **100% bilingual**.
- Every label, title, subtitle, tooltip, theory section, and execution log message **MUST** be translated into Vietnamese (`vi`) and English (`en`).
- Helper functions: `getPresetTitle(preset, lang)`, `getPresetSummary(preset, lang)`, `getPresetBadge(preset, lang)`, `getPresetSections(preset, lang)`, `getPresetVariants(preset, lang)` in `presets.ts`.

---

## 4. Dual-Theme Support (Studio & Sketch)

- **Clean Pro Studio Mode (`studio`)**:
  - Dark blueprint background (`#090d16` / `#0b0f19`), borders (`border-[#1e293b]`), glowing SVG elements, high contrast text (`text-slate-100`, `text-cyan-300`, `text-emerald-400`).
- **Sketch Hand-Drawn Mode (`sketch`)**:
  - Organic parchment background (`#fbf7ee` / `#fdfbf7`), rough borders, Caveat typography, high-contrast dark text (`text-[#282828]`).
- All color classes in header cards and visual stages must provide high contrast in both themes (e.g. `text-cyan-600 dark:text-cyan-400`).

---

## 5. Verification & Deployment Protocol

Before completing any task:
1. Run `bun run build` in `/home/khang/tracecraft` — MUST pass with **0 errors and 0 warnings**.
2. Commit and push to `origin main` — this triggers GitHub Actions (`.github/workflows/deploy.yml`) to build and deploy to GitHub Pages (`https://khangbuinoai.github.io/tracecraft/`).
3. Verify live on GitHub Pages with browser or curl to confirm deployment success.
