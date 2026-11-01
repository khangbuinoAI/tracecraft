import React, { useState, useMemo } from 'react';
import {
  TrendingDown,
  MousePointerClick,
  Activity,
  CheckCircle2,
  Info,
  GitCommit,
  Compass,
} from 'lucide-react';
import type { ThemeMode, Language, MomentumState, MomentumTrajectoryPoint } from '../types';

interface MomentumStageProps {
  theme: ThemeMode;
  language: Language;
  momentumState?: MomentumState;
  onChangeInput?: (newInput: string) => void;
  isExpanded?: boolean;
}

const PADDING = { top: 35, right: 35, bottom: 45, left: 50 };

const CONTOUR_LEVELS = [0.2, 0.8, 2.0, 4.5, 8.0, 13.0, 20.0, 30.0];

export const MomentumStage: React.FC<MomentumStageProps> = ({
  theme,
  language,
  momentumState,
  onChangeInput,
  isExpanded = false,
}) => {
  const isDark = theme === 'studio';
  const isVi = language === 'vi';

  // Toggle visual overlays
  const [showVanilla, setShowVanilla] = useState(true);
  const [showVectors, setShowVectors] = useState(true);
  const [hoveredNode, setHoveredNode] = useState<MomentumTrajectoryPoint | null>(null);

  // Fallback defaults if no state provided
  const state: MomentumState = useMemo(() => {
    return (
      momentumState || {
        beta: 0.9,
        learningRate: 0.1,
        currentW: [4.0, 1.8],
        loss: 18.2,
        velocity: [0, 0],
        gradient: [4.0, 18.0],
        momentumPath: [],
        vanillaPath: [],
        surfaceType: 'ravine',
        iteration: 0,
        maxIterations: 16,
        converged: false,
      }
    );
  }, [momentumState]);

  const {
    beta,
    learningRate,
    currentW,
    loss,
    velocity,
    gradient,
    momentumPath,
    vanillaPath,
    iteration,
    converged,
  } = state;

  // Coordinate Bounds: symmetric coordinate space centered around (0, 0)
  const bounds = useMemo(() => {
    return {
      minW1: -5.5,
      maxW1: 5.5,
      minW2: -3.0,
      maxW2: 3.0,
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

  // Concentric Elliptical Contour Lines for Ravine: L = 0.5 * (w1^2 + 10 * w2^2)
  // At level C: w1^2 / (2C) + w2^2 / (2C / 10) = 1
  // Semi-major axis (w1): sqrt(2C), Semi-minor axis (w2): sqrt(2C / 10)
  const contourEllipses = useMemo(() => {
    const centerX = PADDING.left + ((0 - bounds.minW1) / (bounds.maxW1 - bounds.minW1)) * innerWidth;
    const centerY = PADDING.top + (1 - (0 - bounds.minW2) / (bounds.maxW2 - bounds.minW2)) * innerHeight;

    return CONTOUR_LEVELS.map((lvl) => {
      const radiusW1 = Math.sqrt(2 * lvl);
      const radiusW2 = Math.sqrt((2 * lvl) / 10);
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
    onChangeInput(`start=(${newW1}, ${newW2}), beta=${beta}`);
  };

  // Vector Decomposition calculations at current position
  const currentPos = currentW;
  const currentSvgX = toSvgX(currentPos[0]);
  const currentSvgY = toSvgY(currentPos[1]);

  // Vector scale factor for visual clarity
  const vScale = 35;
  const gScale = 3.5;

  // 1. Negative Gradient vector: -alpha * grad
  const gradVecX = -learningRate * gradient[0] * gScale;
  const gradVecY = learningRate * gradient[1] * gScale; // Cartesian to SVG invert

  // 2. Momentum Velocity vector: beta * v_{t-1}
  const momVecX = beta * velocity[0] * vScale;
  const momVecY = -beta * velocity[1] * vScale;

  // 3. Resultant update vector: -v_t
  const updateVecX = gradVecX + momVecX;
  const updateVecY = gradVecY + momVecY;

  const velocityMag = Math.hypot(velocity[0], velocity[1]);
  const gradientMag = Math.hypot(gradient[0], gradient[1]);

  return (
    <div className="w-full h-full flex flex-col items-center justify-between relative select-none p-2 sm:p-3 overflow-hidden">
      {/* ========================================================= */}
      {/* 1. TOP STATS BAR & INTERACTIVE TOGGLES                    */}
      {/* ========================================================= */}
      <div className="w-full max-w-4xl flex flex-wrap items-center justify-between gap-2.5 z-10 mb-2">
        {/* Momentum Indicator Pill */}
        <div className="flex items-center gap-2">
          <div
            className={`px-2.5 py-1 rounded-xl text-xs font-semibold flex items-center gap-1.5 border shadow-2xs ${
              converged
                ? 'text-emerald-500 bg-emerald-500/10 border-emerald-500/30'
                : 'text-cyan-500 bg-cyan-500/10 border-cyan-500/30'
            }`}
          >
            {converged ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
            ) : (
              <Activity className="w-3.5 h-3.5 animate-pulse" />
            )}
            <span>
              {converged
                ? isVi
                  ? 'Đã Hội tụ vào Cực tiểu'
                  : 'Converged to Minimum'
                : isVi
                ? `Vòng lặp t = ${iteration}`
                : `Step t = ${iteration}`}
            </span>
            <span className="font-mono text-[11px] opacity-75">
              • β = {beta}
            </span>
          </div>

          <div
            className={`hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-xl text-xs font-mono border ${
              isDark ? 'bg-slate-900/80 border-slate-800 text-slate-300' : 'bg-white border-slate-200 text-slate-700'
            }`}
          >
            <span>
              {isVi ? 'Tọa độ:' : 'Pos:'}{' '}
              <strong className="text-cyan-500 dark:text-cyan-400">
                ({currentW[0].toFixed(2)}, {currentW[1].toFixed(2)})
              </strong>
            </span>
            <span className="text-slate-400">•</span>
            <span>
              Loss:{' '}
              <strong className="text-rose-500 dark:text-rose-400">{loss.toFixed(3)}</strong>
            </span>
            <span className="text-slate-400">•</span>
            <span>
              {isVi ? 'Vận tốc ‖v‖:' : 'Speed ‖v‖:'}{' '}
              <strong className="text-emerald-500 dark:text-emerald-400">{velocityMag.toFixed(3)}</strong>
            </span>
          </div>
        </div>

        {/* View Toggles */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setShowVanilla(!showVanilla)}
            className={`px-2.5 py-1 rounded-lg text-xs font-medium flex items-center gap-1.5 border transition-all ${
              showVanilla
                ? isDark
                  ? 'bg-rose-950/80 text-rose-300 border-rose-800'
                  : 'bg-rose-50 text-rose-700 border-rose-300'
                : isDark
                ? 'bg-slate-800/60 text-slate-400 border-slate-700'
                : 'bg-slate-100 text-slate-500 border-slate-200'
            }`}
            title={isVi ? 'So sánh với đường đi của Gradient Descent thuần' : 'Toggle Vanilla SGD Comparison Path'}
          >
            <GitCommit className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">
              {isVi ? 'SGD Thuần (Dao động)' : 'Vanilla SGD'}
            </span>
          </button>

          <button
            onClick={() => setShowVectors(!showVectors)}
            className={`px-2.5 py-1 rounded-lg text-xs font-medium flex items-center gap-1.5 border transition-all ${
              showVectors
                ? isDark
                  ? 'bg-cyan-950/80 text-cyan-300 border-cyan-800'
                  : 'bg-cyan-50 text-cyan-700 border-cyan-300'
                : isDark
                ? 'bg-slate-800/60 text-slate-400 border-slate-700'
                : 'bg-slate-100 text-slate-500 border-slate-200'
            }`}
            title={isVi ? 'Bật/Tắt phân tích vector Lực Gradient và Vận tốc Quán tính' : 'Toggle Vector Decomposition'}
          >
            <Compass className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">{isVi ? 'Vector Lực' : 'Vectors'}</span>
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
          <MousePointerClick className="w-3.5 h-3.5 text-cyan-500" />
          <span>{isVi ? 'Nhấn lên bề mặt hàm mất mát để chọn điểm xuất phát mới' : 'Click contour to set new start point'}</span>
        </div>

        {/* Vector Decomposition Legend HUD */}
        {showVectors && (
          <div
            className={`absolute top-2.5 right-3 z-10 px-3 py-2 rounded-xl border text-xs font-mono shadow-md backdrop-blur-md transition-all ${
              isDark ? 'bg-slate-900/90 border-slate-700 text-slate-200' : 'bg-white/90 border-slate-200 text-slate-800'
            }`}
          >
            <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1">
              {isVi ? 'Phân tích Vector Lực' : 'Vector Decomposition'}
            </div>
            <div className="flex flex-col gap-1 text-[11px]">
              <div className="flex items-center gap-2">
                <span className="w-3 h-0.5 bg-rose-500 rounded-full" />
                <span className="text-rose-400 font-semibold">{isVi ? 'Gradient thuần:' : 'Gradient:'}</span>
                <span>-α∇L</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-0.5 bg-purple-500 rounded-full" />
                <span className="text-purple-400 font-semibold">{isVi ? 'Quán tính cũ:' : 'Momentum:'}</span>
                <span>β·v_prev</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-0.5 bg-cyan-400 rounded-full" />
                <span className="text-cyan-400 font-semibold">{isVi ? 'Vận tốc tổng hợp:' : 'Net Update:'}</span>
                <span>v_t</span>
              </div>
            </div>
          </div>
        )}

        {/* Hover Tooltip */}
        {hoveredNode && (
          <div
            className={`absolute bottom-3 left-3 z-20 px-3 py-2 rounded-xl border text-xs font-mono shadow-lg backdrop-blur-md transition-all ${
              isDark
                ? 'bg-slate-900/95 border-cyan-500/50 text-slate-200 shadow-cyan-950/50'
                : 'bg-white/95 border-cyan-300 text-slate-800 shadow-slate-200'
            }`}
          >
            <div className="font-bold flex items-center justify-between gap-3 mb-1 pb-1 border-b border-inherit">
              <span>{isVi ? 'Bước #' : 'Step #'}{hoveredNode.step}</span>
              <span className="text-cyan-500">
                ({hoveredNode.w1.toFixed(2)}, {hoveredNode.w2.toFixed(2)})
              </span>
            </div>
            <div className="text-[11px] space-y-0.5">
              <div>Loss: <strong className="text-rose-500">{hoveredNode.loss.toFixed(4)}</strong></div>
              {hoveredNode.v1 !== undefined && (
                <div>
                  {isVi ? 'Vận tốc ‖v‖:' : 'Velocity ‖v‖:'}{' '}
                  <strong className="text-emerald-500">
                    {Math.hypot(hoveredNode.v1, hoveredNode.v2 || 0).toFixed(4)}
                  </strong>
                </div>
              )}
            </div>
          </div>
        )}

        {/* SVG Drawing Canvas */}
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="w-full h-full cursor-crosshair overflow-hidden"
          onClick={handleSvgClick}
        >
          <defs>
            {/* Arrow marker markers for vectors */}
            <marker
              id="arrow-grad"
              viewBox="0 0 10 10"
              refX="6"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#f43f5e" />
            </marker>
            <marker
              id="arrow-mom"
              viewBox="0 0 10 10"
              refX="6"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#a855f7" />
            </marker>
            <marker
              id="arrow-update"
              viewBox="0 0 10 10"
              refX="6"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#06b6d4" />
            </marker>
          </defs>

          {/* 1. Contour Ellipses & Gradient Wash */}
          <g className="contours pointer-events-none">
            {contourEllipses.map((c, idx) => {
              const opacity = 0.08 + idx * 0.035;
              const strokeColor = isDark
                ? `rgba(6, 182, 212, ${0.15 + idx * 0.08})`
                : `rgba(14, 116, 144, ${0.18 + idx * 0.08})`;

              return (
                <g key={`contour-${idx}`}>
                  {/* Shading fill between ellipses */}
                  <ellipse
                    cx={c.cx}
                    cy={c.cy}
                    rx={c.rx}
                    ry={c.ry}
                    fill={isDark ? `rgba(6, 182, 212, ${opacity})` : `rgba(14, 116, 144, ${opacity * 0.7})`}
                    stroke={strokeColor}
                    strokeWidth="1.2"
                    strokeDasharray={idx % 2 === 0 ? 'none' : '4 3'}
                  />
                  {/* Contour level text */}
                  {c.rx < innerWidth * 0.45 && (
                    <text
                      x={c.cx + c.rx - 14}
                      y={c.cy - 3}
                      className="text-[9px] font-mono fill-slate-400 opacity-60 select-none"
                    >
                      {c.lvl}
                    </text>
                  )}
                </g>
              );
            })}
          </g>

          {/* 2. Coordinate Axes */}
          <g className="axes text-[10px] font-mono select-none" stroke={isDark ? '#334155' : '#cbd5e1'}>
            {/* W1 Axis (horizontal through origin) */}
            <line
              x1={PADDING.left}
              y1={toSvgY(0)}
              x2={PADDING.left + innerWidth}
              y2={toSvgY(0)}
              strokeWidth="1.5"
            />
            {/* W2 Axis (vertical through origin) */}
            <line
              x1={toSvgX(0)}
              y1={PADDING.top}
              x2={toSvgX(0)}
              y2={PADDING.top + innerHeight}
              strokeWidth="1.5"
            />
          </g>

          {/* Axis Labels */}
          <g className="axis-labels text-[10px] font-mono fill-slate-400 select-none pointer-events-none">
            <text x={PADDING.left + innerWidth - 5} y={toSvgY(0) - 8} textAnchor="end">
              w₁ (Độ dốc thoải / Low curvature) →
            </text>
            <text x={toSvgX(0) + 8} y={PADDING.top + 12} textAnchor="start">
              ↑ w₂ (Hẻm dốc dốc / High curvature)
            </text>
          </g>

          {/* 3. Global Minimum Target (0, 0) */}
          <g className="global-min">
            <circle
              cx={toSvgX(0)}
              cy={toSvgY(0)}
              r="12"
              fill="rgba(16, 185, 129, 0.15)"
              stroke="#10b981"
              strokeWidth="1.5"
              strokeDasharray="3 2"
            />
            <circle cx={toSvgX(0)} cy={toSvgY(0)} r="3.5" fill="#10b981" />
            <text
              x={toSvgX(0) + 12}
              y={toSvgY(0) + 14}
              className="text-[10px] font-mono font-bold fill-emerald-500"
            >
              θ* = (0, 0)
            </text>
          </g>

          {/* 4. Vanilla SGD Trajectory (Oscillating Path in Red/Rose) */}
          {showVanilla && vanillaPath.length > 1 && (
            <g className="vanilla-path transition-opacity duration-300">
              {/* Path line */}
              <polyline
                fill="none"
                stroke="#f43f5e"
                strokeWidth="1.8"
                strokeDasharray="4 3"
                opacity="0.8"
                points={vanillaPath.map((p) => `${toSvgX(p.w1)},${toSvgY(p.w2)}`).join(' ')}
              />
              {/* Path nodes */}
              {vanillaPath.map((p, idx) => (
                <circle
                  key={`vanilla-${idx}`}
                  cx={toSvgX(p.w1)}
                  cy={toSvgY(p.w2)}
                  r="3.5"
                  fill="#f43f5e"
                  opacity={idx === vanillaPath.length - 1 ? 1 : 0.6}
                  onMouseEnter={() => setHoveredNode(p)}
                  onMouseLeave={() => setHoveredNode(null)}
                />
              ))}
            </g>
          )}

          {/* 5. Momentum Trajectory (Smooth Accelerated Path in Cyan/Emerald) */}
          {momentumPath.length > 1 && (
            <g className="momentum-path">
              {/* Path glow line */}
              <polyline
                fill="none"
                stroke="#06b6d4"
                strokeWidth="2.5"
                opacity="0.95"
                points={momentumPath.map((p) => `${toSvgX(p.w1)},${toSvgY(p.w2)}`).join(' ')}
              />
              {/* Nodes along momentum path */}
              {momentumPath.map((p, idx) => {
                const isCurrent = idx === momentumPath.length - 1;
                return (
                  <circle
                    key={`mom-${idx}`}
                    cx={toSvgX(p.w1)}
                    cy={toSvgY(p.w2)}
                    r={isCurrent ? 6 : 4}
                    fill={isCurrent ? '#10b981' : '#06b6d4'}
                    stroke={isDark ? '#080d1a' : '#ffffff'}
                    strokeWidth="1.5"
                    className="cursor-pointer transition-all duration-150"
                    onMouseEnter={() => setHoveredNode(p)}
                    onMouseLeave={() => setHoveredNode(null)}
                  />
                );
              })}
            </g>
          )}

          {/* 6. Active Physical Vectors Decomposition HUD at current position */}
          {showVectors && !converged && (
            <g className="force-vectors pointer-events-none">
              {/* Gradient Force Vector (-alpha * grad) */}
              <line
                x1={currentSvgX}
                y1={currentSvgY}
                x2={currentSvgX + gradVecX}
                y2={currentSvgY + gradVecY}
                stroke="#f43f5e"
                strokeWidth="2.2"
                markerEnd="url(#arrow-grad)"
              />
              {/* Momentum Velocity Vector (beta * v_prev) */}
              {velocityMag > 0.01 && (
                <line
                  x1={currentSvgX}
                  y1={currentSvgY}
                  x2={currentSvgX + momVecX}
                  y2={currentSvgY + momVecY}
                  stroke="#a855f7"
                  strokeWidth="2.2"
                  markerEnd="url(#arrow-mom)"
                />
              )}
              {/* Resultant Net Update Vector (v_t) */}
              <line
                x1={currentSvgX}
                y1={currentSvgY}
                x2={currentSvgX + updateVecX}
                y2={currentSvgY + updateVecY}
                stroke="#06b6d4"
                strokeWidth="3"
                markerEnd="url(#arrow-update)"
              />
            </g>
          )}

          {/* 7. Current Position Marker */}
          <g className="current-pos pointer-events-none">
            <circle
              cx={currentSvgX}
              cy={currentSvgY}
              r="14"
              fill="rgba(6, 182, 212, 0.2)"
              stroke="#06b6d4"
              strokeWidth="1.5"
              strokeDasharray="4 3"
            />
            <circle cx={currentSvgX} cy={currentSvgY} r="5" fill="#06b6d4" stroke="#ffffff" strokeWidth="2" />
          </g>
        </svg>
      </div>

      {/* ========================================================= */}
      {/* 3. PHYSICAL INTUITION SUMMARY BANNER                       */}
      {/* ========================================================= */}
      <div className="w-full max-w-4xl flex items-center justify-between gap-2 mt-2 pt-2 border-t border-slate-200 dark:border-slate-800 text-xs">
        <div className="flex items-center gap-2 overflow-x-auto max-w-full">
          <span className="text-slate-500 font-medium shrink-0 flex items-center gap-1">
            <TrendingDown className="w-3.5 h-3.5" />
            <span>{isVi ? 'Đặc tính Gia tốc:' : 'Dynamics:'}</span>
          </span>

          <div
            className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold flex items-center gap-1.5 border shrink-0 ${
              isDark ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-300' : 'bg-cyan-50 border-cyan-200 text-cyan-800'
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-cyan-400" />
            <span>
              {isVi ? 'Hệ số Quán tính:' : 'Momentum:'} β = {beta}
            </span>
            <span className="opacity-75 font-sans">
              (Gia tốc cực đại 1/(1-β) = {(1 / (1 - beta)).toFixed(0)}×)
            </span>
          </div>

          <div
            className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold flex items-center gap-1.5 border shrink-0 ${
              isDark ? 'bg-indigo-500/10 border-indigo-500/30 text-indigo-300' : 'bg-indigo-50 border-indigo-200 text-indigo-800'
            }`}
          >
            <span>
              α = {learningRate} • ‖∇L‖ = {gradientMag.toFixed(2)}
            </span>
          </div>
        </div>

        <div className="hidden lg:flex items-center gap-1 text-[11px] text-slate-500 shrink-0">
          <Info className="w-3.5 h-3.5 text-cyan-500" />
          <span>
            {isVi
              ? 'Triệt tiêu dao động ngang hẻm núi, tích lũy vận tốc dọc theo đáy thung lũng'
              : 'Dampens lateral oscillations, accumulates velocity along ravine floor'}
          </span>
        </div>
      </div>
    </div>
  );
};
