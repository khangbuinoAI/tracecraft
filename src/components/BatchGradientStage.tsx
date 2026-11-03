import React, { useState, useMemo } from 'react';
import {
  MousePointerClick,
  CheckCircle2,
  GitCommit,
  Layers,
  Eye,
} from 'lucide-react';
import type {
  ThemeMode,
  Language,
  BatchGradientState,
  BatchTrajectoryPoint,
} from '../types';

interface BatchGradientStageProps {
  theme: ThemeMode;
  language: Language;
  batchGradientState?: BatchGradientState;
  onChangeInput?: (newInput: string) => void;
  isExpanded?: boolean;
}

const PADDING = { top: 35, right: 35, bottom: 45, left: 50 };
const CONTOUR_LEVELS = [0.3, 1.0, 2.5, 5.0, 9.0, 15.0, 24.0, 35.0];

export const BatchGradientStage: React.FC<BatchGradientStageProps> = ({
  theme,
  language,
  batchGradientState,
  onChangeInput,
  isExpanded = false,
}) => {
  const isDark = theme === 'studio';
  const isVi = language === 'vi';

  // Toggle visual overlays
  const [showSGD, setShowSGD] = useState(true);
  const [showMiniBatch, setShowMiniBatch] = useState(true);
  const [showVectors, setShowVectors] = useState(true);
  const [hoveredNode, setHoveredNode] = useState<{
    type: 'batch' | 'sgd' | 'minibatch';
    point: BatchTrajectoryPoint;
  } | null>(null);

  // Fallback defaults if no state provided
  const state: BatchGradientState = useMemo(() => {
    return (
      batchGradientState || {
        batchSize: 12,
        totalSamples: 12,
        learningRate: 0.15,
        currentW: [4.2, 2.2],
        loss: 18.5,
        gradient: [4.2, 8.8],
        batchPath: [],
        sgdPath: [],
        miniBatchPath: [],
        sampleGradients: [],
        iteration: 0,
        maxIterations: 14,
        converged: false,
      }
    );
  }, [batchGradientState]);

  const {
    batchSize,
    totalSamples,
    learningRate,
    currentW,
    loss,
    gradient,
    batchPath,
    sgdPath,
    miniBatchPath,
    sampleGradients,
    iteration,
    converged,
  } = state;

  // Coordinate Bounds: centered around (0, 0)
  const bounds = useMemo(() => {
    return {
      minW1: -5.5,
      maxW1: 5.5,
      minW2: -3.2,
      maxW2: 3.2,
    };
  }, []);

  const svgWidth = isExpanded ? 880 : 720;
  const svgHeight = isExpanded ? 520 : 420;
  const innerWidth = svgWidth - PADDING.left - PADDING.right;
  const innerHeight = svgHeight - PADDING.top - PADDING.bottom;

  const toSvgX = (w1: number) => {
    const ratio = (w1 - bounds.minW1) / (bounds.maxW1 - bounds.minW1 || 1);
    return PADDING.left + ratio * innerWidth;
  };

  const toSvgY = (w2: number) => {
    const ratio = (w2 - bounds.minW2) / (bounds.maxW2 - bounds.minW2 || 1);
    return PADDING.top + (1 - ratio) * innerHeight; // Cartesian Y invert
  };

  const toCartesianW1 = (svgX: number) => {
    const ratio = (svgX - PADDING.left) / innerWidth;
    return Number((bounds.minW1 + ratio * (bounds.maxW1 - bounds.minW1)).toFixed(2));
  };

  const toCartesianW2 = (svgY: number) => {
    const ratio = (svgHeight - PADDING.bottom - svgY) / innerHeight;
    return Number((bounds.minW2 + ratio * (bounds.maxW2 - bounds.minW2)).toFixed(2));
  };

  // Concentric Elliptical Contour Lines for Loss Surface: L(w1, w2) = 0.5 * (w1^2 + 4 * w2^2)
  const contourEllipses = useMemo(() => {
    const centerX = PADDING.left + ((0 - bounds.minW1) / (bounds.maxW1 - bounds.minW1)) * innerWidth;
    const centerY = PADDING.top + (1 - (0 - bounds.minW2) / (bounds.maxW2 - bounds.minW2)) * innerHeight;

    return CONTOUR_LEVELS.map((lvl) => {
      const radiusW1 = Math.sqrt(2 * lvl);
      const radiusW2 = Math.sqrt((2 * lvl) / 4);
      const rx = (radiusW1 / (bounds.maxW1 - bounds.minW1)) * innerWidth;
      const ry = (radiusW2 / (bounds.maxW2 - bounds.minW2)) * innerHeight;

      return {
        lvl,
        cx: centerX,
        cy: centerY,
        rx,
        ry,
      };
    });
  }, [bounds, innerWidth, innerHeight]);

  // Click on canvas to pick new initial starting point
  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!onChangeInput) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const clickSvgX = ((e.clientX - rect.left) / rect.width) * svgWidth;
    const clickSvgY = ((e.clientY - rect.top) / rect.height) * svgHeight;

    if (
      clickSvgX < PADDING.left ||
      clickSvgX > svgWidth - PADDING.right ||
      clickSvgY < PADDING.top ||
      clickSvgY > svgHeight - PADDING.bottom
    ) {
      return;
    }

    const newW1 = toCartesianW1(clickSvgX);
    const newW2 = toCartesianW2(clickSvgY);
    onChangeInput(`start=(${newW1}, ${newW2}), N=${totalSamples}, lr=${learningRate}`);
  };

  // Coordinates of Current Parameter Position
  const currentSvgX = toSvgX(currentW[0]);
  const currentSvgY = toSvgY(currentW[1]);

  // Vector scaling
  const gScale = 4.0;
  const batchGradVecX = -learningRate * gradient[0] * gScale;
  const batchGradVecY = learningRate * gradient[1] * gScale; // Cartesian to SVG invert
  const gradMag = Math.hypot(gradient[0], gradient[1]);

  return (
    <div className="w-full h-full flex flex-col items-center justify-between relative select-none p-2 sm:p-3 overflow-hidden">
      {/* ========================================================= */}
      {/* 1. TOP STATS BAR & INTERACTIVE TOGGLES                    */}
      {/* ========================================================= */}
      <div className="w-full max-w-4xl flex flex-wrap items-center justify-between gap-2.5 z-10 mb-2">
        {/* Status Pill */}
        <div className="flex items-center gap-2">
          <div
            className={`px-2.5 py-1 rounded-xl text-xs font-semibold flex items-center gap-1.5 border shadow-2xs ${
              converged
                ? 'text-emerald-500 bg-emerald-500/10 border-emerald-500/30'
                : 'text-blue-500 bg-blue-500/10 border-blue-500/30'
            }`}
          >
            {converged ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
            ) : (
              <Layers className="w-3.5 h-3.5 text-blue-500 animate-pulse" />
            )}
            <span>
              {converged
                ? isVi
                  ? 'Đã Hội tụ vào Cực tiểu'
                  : 'Converged to Minimum'
                : isVi
                ? `Vòng lặp (Epoch) t = ${iteration}`
                : `Epoch t = ${iteration}`}
            </span>
            <span className="font-mono text-[11px] opacity-75">
              • B = {batchSize} / {totalSamples} (100%)
            </span>
          </div>

          <div
            className={`hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-xl text-xs font-mono border ${
              isDark ? 'bg-slate-900/80 border-slate-800 text-slate-300' : 'bg-white border-slate-200 text-slate-700'
            }`}
          >
            <span>
              w: <strong className="text-blue-500 dark:text-blue-400">({currentW[0].toFixed(2)}, {currentW[1].toFixed(2)})</strong>
            </span>
            <span className="text-slate-400">•</span>
            <span>
              Loss: <strong className="text-rose-500 dark:text-rose-400">{loss.toFixed(3)}</strong>
            </span>
            <span className="text-slate-400">•</span>
            <span>
              ‖∇L‖: <strong className="text-cyan-500 dark:text-cyan-400">{gradMag.toFixed(3)}</strong>
            </span>
            <span className="text-slate-400">•</span>
            <span className="text-[11px] text-emerald-500">
              {isVi ? 'Độ lệch σ = 0 (Chính xác)' : 'Var σ = 0 (Exact)'}
            </span>
          </div>
        </div>

        {/* View Toggles */}
        <div className="flex items-center gap-1.5">
          {/* Mini-Batch Toggle */}
          <button
            onClick={() => setShowMiniBatch(!showMiniBatch)}
            className={`px-2.5 py-1 rounded-lg text-xs font-medium flex items-center gap-1.5 border transition-all ${
              showMiniBatch
                ? isDark
                  ? 'bg-purple-950/80 text-purple-300 border-purple-800'
                  : 'bg-purple-50 text-purple-700 border-purple-300'
                : isDark
                ? 'bg-slate-800/60 text-slate-400 border-slate-700'
                : 'bg-slate-100 text-slate-500 border-slate-200'
            }`}
            title={isVi ? 'So sánh với Mini-batch Gradient Descent (B=4)' : 'Toggle Mini-Batch Comparison Path (B=4)'}
          >
            <GitCommit className="w-3.5 h-3.5 text-purple-400" />
            <span className="hidden sm:inline">
              {isVi ? 'Mini-Batch (B=4)' : 'Mini-Batch'}
            </span>
          </button>

          {/* Stochastic GD Toggle */}
          <button
            onClick={() => setShowSGD(!showSGD)}
            className={`px-2.5 py-1 rounded-lg text-xs font-medium flex items-center gap-1.5 border transition-all ${
              showSGD
                ? isDark
                  ? 'bg-rose-950/80 text-rose-300 border-rose-800'
                  : 'bg-rose-50 text-rose-700 border-rose-300'
                : isDark
                ? 'bg-slate-800/60 text-slate-400 border-slate-700'
                : 'bg-slate-100 text-slate-500 border-slate-200'
            }`}
            title={isVi ? 'So sánh với Stochastic Gradient Descent thuần (B=1, Dao động ngẫu nhiên)' : 'Toggle Stochastic GD Path (B=1, High Variance)'}
          >
            <GitCommit className="w-3.5 h-3.5 text-rose-400" />
            <span className="hidden sm:inline">
              {isVi ? 'SGD (B=1, Nhiễu)' : 'SGD (B=1)'}
            </span>
          </button>

          {/* Sample Vectors Toggle */}
          <button
            onClick={() => setShowVectors(!showVectors)}
            className={`px-2.5 py-1 rounded-lg text-xs font-medium flex items-center gap-1.5 border transition-all ${
              showVectors
                ? isDark
                  ? 'bg-blue-950/80 text-blue-300 border-blue-800'
                  : 'bg-blue-50 text-blue-700 border-blue-300'
                : isDark
                ? 'bg-slate-800/60 text-slate-400 border-slate-700'
                : 'bg-slate-100 text-slate-500 border-slate-200'
            }`}
            title={isVi ? 'Bật/Tắt các vector gradient từng mẫu ∇ℓ_i tạo nên gradient toàn tập' : 'Toggle Individual Sample Gradient Vectors'}
          >
            <Eye className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">{isVi ? 'Vector Mẫu' : 'Sample Vectors'}</span>
          </button>
        </div>
      </div>

      {/* ========================================================= */}
      {/* 2. MAIN 2D CONTOUR LOSS SURFACE CANVAS                    */}
      {/* ========================================================= */}
      <div
        className={`w-full max-w-4xl flex-1 relative flex items-center justify-center rounded-2xl border overflow-hidden shadow-inner transition-colors ${
          isDark ? 'bg-[#080d1a] border-slate-800' : 'bg-slate-50/70 border-slate-200'
        }`}
      >
        {/* Click hint */}
        <div className="absolute top-2.5 left-3 z-10 flex items-center gap-1.5 text-[11px] font-sans opacity-70 pointer-events-none">
          <MousePointerClick className="w-3.5 h-3.5 text-blue-500" />
          <span>{isVi ? 'Nhấn lên bề mặt hàm mất mát để chọn điểm khởi đầu w₀' : 'Click contour to set new start parameters'}</span>
        </div>

        {/* Vector Decomposition Legend HUD */}
        {showVectors && (
          <div
            className={`absolute top-2.5 right-3 z-10 px-3 py-2 rounded-xl border text-xs font-mono shadow-md backdrop-blur-md transition-all ${
              isDark ? 'bg-slate-900/90 border-slate-700 text-slate-200' : 'bg-white/90 border-slate-200 text-slate-800'
            }`}
          >
            <div className="text-[10px] font-bold uppercase tracking-wider text-blue-400 mb-1 flex items-center gap-1">
              <Layers className="w-3 h-3" />
              <span>{isVi ? 'Phân tích Gradient Toàn Tập' : 'Full-Batch Averaging'}</span>
            </div>
            <div className="flex flex-col gap-1 text-[11px]">
              <div className="flex items-center gap-2">
                <span className="w-3 h-0.5 bg-slate-400 rounded-full border-t border-dashed opacity-60" />
                <span className="text-slate-400 font-semibold">{isVi ? 'Gradient từng mẫu:' : 'Sample Grads:'}</span>
                <span>∇ℓ_i (i=1..N)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-1 bg-blue-500 rounded-full" />
                <span className="text-blue-400 font-bold">{isVi ? 'Gradient Toàn Tập:' : 'Batch Gradient:'}</span>
                <span>(1/N)∑∇ℓ_i</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-blue-500/20 border border-blue-400" />
                <span className="text-blue-400">{isVi ? 'Độ lệch phương sai:' : 'Variance:'}</span>
                <span className="text-emerald-400 font-bold">σ² = 0 (Exact)</span>
              </div>
            </div>
          </div>
        )}

        {/* Hover Tooltip */}
        {hoveredNode && (
          <div
            className={`absolute bottom-3 left-3 z-20 px-3 py-2 rounded-xl border text-xs font-mono shadow-lg backdrop-blur-md transition-all ${
              isDark
                ? 'bg-slate-900/95 border-blue-500/50 text-slate-200 shadow-blue-950/50'
                : 'bg-white/95 border-blue-300 text-slate-800 shadow-slate-200'
            }`}
          >
            <div className="font-bold flex items-center justify-between gap-3 mb-1 pb-1 border-b border-inherit">
              <span className={hoveredNode.type === 'batch' ? 'text-blue-400' : hoveredNode.type === 'minibatch' ? 'text-purple-400' : 'text-rose-400'}>
                {hoveredNode.type === 'batch' ? 'Batch GD (B=N)' : hoveredNode.type === 'minibatch' ? 'Mini-Batch (B=4)' : 'Stochastic GD (B=1)'} #{hoveredNode.point.step}
              </span>
              <span className="text-slate-400">
                ({hoveredNode.point.w1.toFixed(2)}, {hoveredNode.point.w2.toFixed(2)})
              </span>
            </div>
            <div className="text-[11px] space-y-0.5">
              <div>Loss: <strong className="text-rose-500">{hoveredNode.point.loss.toFixed(4)}</strong></div>
              <div>
                ‖∇L‖: <strong className="text-cyan-500">{Math.hypot(hoveredNode.point.grad1, hoveredNode.point.grad2).toFixed(4)}</strong>
              </div>
            </div>
          </div>
        )}

        {/* SVG Canvas */}
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="w-full h-full cursor-crosshair overflow-hidden"
          onClick={handleSvgClick}
        >
          <defs>
            <marker
              id="arrow-batch-grad"
              viewBox="0 0 10 10"
              refX="6"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#3b82f6" />
            </marker>
            <marker
              id="arrow-sample-grad"
              viewBox="0 0 10 10"
              refX="6"
              refY="5"
              markerWidth="4"
              markerHeight="4"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#94a3b8" />
            </marker>
          </defs>

          {/* 1. Contour Ellipses & Shading */}
          <g className="contours pointer-events-none">
            {contourEllipses.map((c, idx) => {
              const opacity = 0.08 + idx * 0.035;
              const strokeColor = isDark
                ? `rgba(59, 130, 246, ${0.15 + idx * 0.07})`
                : `rgba(29, 78, 216, ${0.18 + idx * 0.07})`;

              return (
                <g key={`contour-${idx}`}>
                  <ellipse
                    cx={c.cx}
                    cy={c.cy}
                    rx={c.rx}
                    ry={c.ry}
                    fill="none"
                    stroke={strokeColor}
                    strokeWidth={idx === 0 ? 1.5 : 1}
                    strokeDasharray={idx % 2 === 1 ? '3 3' : undefined}
                    opacity={opacity}
                  />
                  {c.cx + c.rx < svgWidth - PADDING.right - 10 && (
                    <text
                      x={c.cx + c.rx - 4}
                      y={c.cy - 4}
                      fill={isDark ? '#93c5fd' : '#1d4ed8'}
                      fontSize="9"
                      fontFamily="monospace"
                      opacity="0.45"
                      textAnchor="end"
                    >
                      L={c.lvl}
                    </text>
                  )}
                </g>
              );
            })}
          </g>

          {/* 2. Coordinate Axes */}
          <g className="axes pointer-events-none" stroke={isDark ? '#334155' : '#cbd5e1'} strokeWidth="1">
            <line
              x1={PADDING.left}
              y1={toSvgY(0)}
              x2={svgWidth - PADDING.right}
              y2={toSvgY(0)}
              strokeDasharray="2 2"
              opacity="0.6"
            />
            <line
              x1={toSvgX(0)}
              y1={PADDING.top}
              x2={toSvgX(0)}
              y2={svgHeight - PADDING.bottom}
              strokeDasharray="2 2"
              opacity="0.6"
            />
            <circle cx={toSvgX(0)} cy={toSvgY(0)} r="4.5" fill="#3b82f6" />
            <text
              x={toSvgX(0) + 7}
              y={toSvgY(0) + 12}
              fill="#3b82f6"
              fontSize="10"
              fontWeight="bold"
              fontFamily="monospace"
            >
              w* = (0, 0)
            </text>
          </g>

          {/* 3. Pure Stochastic GD Trajectory (High Variance / Random Walk) */}
          {showSGD && sgdPath.length > 1 && (
            <g className="sgd-path transition-opacity duration-300">
              <polyline
                fill="none"
                stroke="#f43f5e"
                strokeWidth="1.8"
                strokeDasharray="3 3"
                opacity="0.8"
                points={sgdPath.map((p) => `${toSvgX(p.w1)},${toSvgY(p.w2)}`).join(' ')}
              />
              {sgdPath.map((p, idx) => (
                <circle
                  key={`sgd-${idx}`}
                  cx={toSvgX(p.w1)}
                  cy={toSvgY(p.w2)}
                  r="2.8"
                  fill="#f43f5e"
                  opacity="0.8"
                  className="cursor-pointer hover:r-4 transition-all"
                  onMouseEnter={() => setHoveredNode({ type: 'sgd', point: p })}
                  onMouseLeave={() => setHoveredNode(null)}
                />
              ))}
            </g>
          )}

          {/* 4. Mini-Batch GD Trajectory (Moderate Variance) */}
          {showMiniBatch && miniBatchPath.length > 1 && (
            <g className="minibatch-path transition-opacity duration-300">
              <polyline
                fill="none"
                stroke="#a855f7"
                strokeWidth="2.0"
                strokeDasharray="4 2"
                opacity="0.85"
                points={miniBatchPath.map((p) => `${toSvgX(p.w1)},${toSvgY(p.w2)}`).join(' ')}
              />
              {miniBatchPath.map((p, idx) => (
                <circle
                  key={`mb-${idx}`}
                  cx={toSvgX(p.w1)}
                  cy={toSvgY(p.w2)}
                  r="3.2"
                  fill="#a855f7"
                  opacity="0.85"
                  className="cursor-pointer hover:r-4 transition-all"
                  onMouseEnter={() => setHoveredNode({ type: 'minibatch', point: p })}
                  onMouseLeave={() => setHoveredNode(null)}
                />
              ))}
            </g>
          )}

          {/* 5. Full Batch GD Trajectory (Deterministic / Pure Smooth Line) */}
          {batchPath.length > 1 && (
            <g className="batch-path">
              {/* Outer Glow */}
              <polyline
                fill="none"
                stroke="#3b82f6"
                strokeWidth="5"
                opacity="0.25"
                strokeLinecap="round"
                strokeLinejoin="round"
                points={batchPath.map((p) => `${toSvgX(p.w1)},${toSvgY(p.w2)}`).join(' ')}
              />
              {/* Main Line */}
              <polyline
                fill="none"
                stroke="#3b82f6"
                strokeWidth="2.8"
                opacity="0.95"
                strokeLinecap="round"
                strokeLinejoin="round"
                points={batchPath.map((p) => `${toSvgX(p.w1)},${toSvgY(p.w2)}`).join(' ')}
              />
              {batchPath.map((p, idx) => {
                const isCurrent = idx === batchPath.length - 1;
                return (
                  <circle
                    key={`bgd-${idx}`}
                    cx={toSvgX(p.w1)}
                    cy={toSvgY(p.w2)}
                    r={isCurrent ? 6 : 4}
                    fill={isCurrent ? '#60a5fa' : '#3b82f6'}
                    stroke={isDark ? '#080d1a' : '#ffffff'}
                    strokeWidth="1.5"
                    className="cursor-pointer transition-all duration-150"
                    onMouseEnter={() => setHoveredNode({ type: 'batch', point: p })}
                    onMouseLeave={() => setHoveredNode(null)}
                  />
                );
              })}
            </g>
          )}

          {/* 6. Active Sample Gradients & Exact Batch Gradient Vector */}
          {showVectors && !converged && (
            <g className="batch-vectors pointer-events-none">
              {/* Fan of Individual Sample Gradient Vectors (∇ℓ_i) */}
              {sampleGradients.map((s, idx) => {
                const sVecX = -learningRate * s.gradW1 * gScale;
                const sVecY = learningRate * s.gradW2 * gScale;
                return (
                  <line
                    key={`sample-grad-${idx}`}
                    x1={currentSvgX}
                    y1={currentSvgY}
                    x2={currentSvgX + sVecX}
                    y2={currentSvgY + sVecY}
                    stroke="#94a3b8"
                    strokeWidth="1.2"
                    strokeDasharray="2 2"
                    opacity="0.55"
                    markerEnd="url(#arrow-sample-grad)"
                  />
                );
              })}

              {/* Exact Mean Batch Gradient Vector: -(alpha/N) * sum(grad_i) */}
              <line
                x1={currentSvgX}
                y1={currentSvgY}
                x2={currentSvgX + batchGradVecX}
                y2={currentSvgY + batchGradVecY}
                stroke="#3b82f6"
                strokeWidth="3.2"
                markerEnd="url(#arrow-batch-grad)"
              />
            </g>
          )}

          {/* 7. Current Position Marker */}
          <g className="current-pos pointer-events-none">
            <circle
              cx={currentSvgX}
              cy={currentSvgY}
              r="14"
              fill="rgba(59, 130, 246, 0.2)"
              stroke="#3b82f6"
              strokeWidth="1.5"
              strokeDasharray="4 3"
            />
            <circle cx={currentSvgX} cy={currentSvgY} r="5" fill="#3b82f6" stroke="#ffffff" strokeWidth="2" />
          </g>
        </svg>
      </div>

      {/* ========================================================= */}
      {/* 3. PHYSICAL INTUITION SUMMARY BANNER                       */}
      {/* ========================================================= */}
      <div
        className={`w-full max-w-4xl mt-2 px-3 py-2 rounded-xl border text-xs flex items-center justify-between gap-3 shadow-xs transition-colors ${
          isDark ? 'bg-slate-900/70 border-slate-800 text-slate-300' : 'bg-slate-100/80 border-slate-200 text-slate-700'
        }`}
      >
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-blue-400 shrink-0" />
          <span>
            <strong className="text-blue-400">
              {isVi ? 'Bản chất của Batch Gradient Descent:' : 'Nature of Batch Gradient Descent:'}
            </strong>{' '}
            {isVi
              ? 'Tính toán trung bình gradient trên toàn bộ N mẫu trong mỗi bước cập nhật. Nhờ đó, các nhiễu ngẫu nhiên từ từng mẫu triệt tiêu lẫn nhau (phương sai σ² = 0), tạo nên đường đi hoàn toàn trơn tru và hội tụ đơn điệu thẳng vào cực tiểu!'
              : 'Computes the true average gradient across all N samples per step. As a result, sample noises cancel out (variance σ² = 0), producing a perfectly smooth, monotonic trajectory directly descending into the minimum!'}
          </span>
        </div>

        <div className="hidden lg:flex items-center gap-2 shrink-0 font-mono text-[11px] text-slate-400">
          <span className="w-2 h-2 rounded-full bg-blue-400" />
          <span>O(T · N · D) Complexity</span>
        </div>
      </div>
    </div>
  );
};
