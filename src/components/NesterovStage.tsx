import React, { useState, useMemo } from 'react';
import {
  MousePointerClick,
  CheckCircle2,
  GitCommit,
  Navigation,
  Eye,
} from 'lucide-react';
import type { ThemeMode, Language, NesterovState, NesterovTrajectoryPoint, MomentumTrajectoryPoint } from '../types';

interface NesterovStageProps {
  theme: ThemeMode;
  language: Language;
  nesterovState?: NesterovState;
  onChangeInput?: (newInput: string) => void;
  isExpanded?: boolean;
}

const PADDING = { top: 35, right: 35, bottom: 45, left: 50 };
const CONTOUR_LEVELS = [0.2, 0.8, 2.0, 4.5, 8.0, 13.0, 20.0, 30.0];

export const NesterovStage: React.FC<NesterovStageProps> = ({
  theme,
  language,
  nesterovState,
  onChangeInput,
  isExpanded = false,
}) => {
  const isDark = theme === 'studio';
  const isVi = language === 'vi';

  // Toggle visual overlays
  const [showPolyak, setShowPolyak] = useState(true);
  const [showVanilla, setShowVanilla] = useState(false);
  const [showVectors, setShowVectors] = useState(true);
  const [hoveredNode, setHoveredNode] = useState<{
    type: 'nesterov' | 'polyak' | 'vanilla';
    point: NesterovTrajectoryPoint | MomentumTrajectoryPoint;
  } | null>(null);

  // Fallback defaults if no state provided
  const state: NesterovState = useMemo(() => {
    return (
      nesterovState || {
        beta: 0.9,
        learningRate: 0.1,
        currentW: [4.5, 1.8],
        loss: 26.325,
        lookaheadW: [4.5, 1.8],
        lookaheadLoss: 26.325,
        velocity: [0, 0],
        lookaheadGradient: [4.5, 18.0],
        nesterovPath: [],
        polyakPath: [],
        vanillaPath: [],
        surfaceType: 'ravine',
        iteration: 0,
        maxIterations: 14,
        converged: false,
      }
    );
  }, [nesterovState]);

  const {
    beta,
    learningRate,
    currentW,
    loss,
    lookaheadW,
    lookaheadLoss,
    velocity,
    lookaheadGradient,
    nesterovPath,
    polyakPath,
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
    onChangeInput(`start=(${newW1}, ${newW2}), beta=${beta}, lr=${learningRate}`);
  };

  // Coordinates of Current and Lookahead points
  const currentSvgX = toSvgX(currentW[0]);
  const currentSvgY = toSvgY(currentW[1]);
  const lookaheadSvgX = toSvgX(lookaheadW[0]);
  const lookaheadSvgY = toSvgY(lookaheadW[1]);

  // Vector scale factor for visual clarity
  const gScale = 3.5;

  // 1. Momentum jump vector: -beta * v_{prev}
  // From current position to lookahead position
  const jumpVecX = lookaheadSvgX - currentSvgX;
  const jumpVecY = lookaheadSvgY - currentSvgY;

  // 2. Lookahead gradient vector: -alpha * grad(lookahead)
  // Originates from lookahead position
  const lookaheadGradVecX = -learningRate * lookaheadGradient[0] * gScale;
  const lookaheadGradVecY = learningRate * lookaheadGradient[1] * gScale; // Cartesian to SVG invert

  // 3. Resultant update vector: -v_t
  // updateSvgX, updateSvgY from current position
  const updateVecX = jumpVecX + lookaheadGradVecX;
  const updateVecY = jumpVecY + lookaheadGradVecY;

  const velocityMag = Math.hypot(velocity[0], velocity[1]);
  const lookaheadDist = Math.hypot(lookaheadW[0] - currentW[0], lookaheadW[1] - currentW[1]);

  return (
    <div className="w-full h-full flex flex-col items-center justify-between relative select-none p-2 sm:p-3 overflow-hidden">
      {/* ========================================================= */}
      {/* 1. TOP STATS BAR & INTERACTIVE TOGGLES                    */}
      {/* ========================================================= */}
      <div className="w-full max-w-4xl flex flex-wrap items-center justify-between gap-2.5 z-10 mb-2">
        {/* Nesterov Status Pill */}
        <div className="flex items-center gap-2">
          <div
            className={`px-2.5 py-1 rounded-xl text-xs font-semibold flex items-center gap-1.5 border shadow-2xs ${
              converged
                ? 'text-emerald-500 bg-emerald-500/10 border-emerald-500/30'
                : 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30'
            }`}
          >
            {converged ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
            ) : (
              <Navigation className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
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
              θ: <strong className="text-emerald-500 dark:text-emerald-400">({currentW[0].toFixed(2)}, {currentW[1].toFixed(2)})</strong>
            </span>
            <span className="text-slate-400">•</span>
            <span>
              {isVi ? 'Nhìn trước:' : 'Lookahead:'}{' '}
              <strong className="text-amber-500 dark:text-amber-400">({lookaheadW[0].toFixed(2)}, {lookaheadW[1].toFixed(2)})</strong>
              <span className="text-[10px] opacity-75"> [L={lookaheadLoss.toFixed(2)}]</span>
            </span>
            <span className="text-slate-400">•</span>
            <span>
              Loss: <strong className="text-rose-500 dark:text-rose-400">{loss.toFixed(3)}</strong>
            </span>
            <span className="text-slate-400">•</span>
            <span>
              ‖v‖: <strong className="text-cyan-500 dark:text-cyan-400">{velocityMag.toFixed(3)}</strong>
            </span>
          </div>
        </div>

        {/* View Toggles */}
        <div className="flex items-center gap-1.5">
          {/* Polyak Momentum Toggle */}
          <button
            onClick={() => setShowPolyak(!showPolyak)}
            className={`px-2.5 py-1 rounded-lg text-xs font-medium flex items-center gap-1.5 border transition-all ${
              showPolyak
                ? isDark
                  ? 'bg-purple-950/80 text-purple-300 border-purple-800'
                  : 'bg-purple-50 text-purple-700 border-purple-300'
                : isDark
                ? 'bg-slate-800/60 text-slate-400 border-slate-700'
                : 'bg-slate-100 text-slate-500 border-slate-200'
            }`}
            title={isVi ? 'So sánh với Momentum cổ điển (Polyak) để thấy hiện tượng vượt ngưỡng' : 'Toggle Polyak Momentum Comparison Path'}
          >
            <GitCommit className="w-3.5 h-3.5 text-purple-400" />
            <span className="hidden sm:inline">
              {isVi ? 'Polyak (Vượt đích)' : 'Polyak Momentum'}
            </span>
          </button>

          {/* Vanilla SGD Toggle */}
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
            title={isVi ? 'So sánh với Gradient Descent thuần (Dao động mạnh)' : 'Toggle Vanilla SGD Comparison Path'}
          >
            <GitCommit className="w-3.5 h-3.5 text-rose-400" />
            <span className="hidden sm:inline">
              {isVi ? 'SGD Thuần' : 'Vanilla SGD'}
            </span>
          </button>

          {/* Lookahead Vectors Toggle */}
          <button
            onClick={() => setShowVectors(!showVectors)}
            className={`px-2.5 py-1 rounded-lg text-xs font-medium flex items-center gap-1.5 border transition-all ${
              showVectors
                ? isDark
                  ? 'bg-emerald-950/80 text-emerald-300 border-emerald-800'
                  : 'bg-emerald-50 text-emerald-700 border-emerald-300'
                : isDark
                ? 'bg-slate-800/60 text-slate-400 border-slate-700'
                : 'bg-slate-100 text-slate-500 border-slate-200'
            }`}
            title={isVi ? 'Bật/Tắt phân tích vector bước nhảy quán tính và gradient nhìn trước' : 'Toggle Lookahead Vector Decomposition'}
          >
            <Eye className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">{isVi ? 'Vector Nhìn trước' : 'Lookahead Vectors'}</span>
          </button>
        </div>
      </div>
      {/* Vector Decomposition Legend Strip (Positioned outside canvas to prevent covering visualization) */}
      {showVectors && (
        <div
          className={`w-full max-w-4xl mb-2 px-3 py-2 rounded-xl border text-xs font-mono shadow-xs flex flex-wrap items-center justify-between gap-2 transition-all animate-fade-in ${
            isDark
              ? 'bg-[#0f172a] border-emerald-900/60 text-slate-200'
              : 'bg-emerald-50/90 border-emerald-200 text-emerald-950'
          }`}
        >
          <div className="text-[10px] font-bold uppercase tracking-wider text-emerald-500 dark:text-emerald-400 flex items-center gap-1.5 shrink-0">
            <Eye className="w-3.5 h-3.5" />
            <span>{isVi ? 'Phân rã Vector Nhìn Trước:' : 'Look-Ahead Decomposition:'}</span>
          </div>
          <div className="flex flex-wrap items-center gap-x-3.5 gap-y-1 text-[11px]">
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-purple-400 rounded-full border-t border-dashed" />
              <span className="text-purple-400 font-semibold">{isVi ? 'Nhảy quán tính:' : 'Momentum Jump:'}</span>
              <span>-β·v_prev</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full border border-amber-400 bg-amber-400/40" />
              <span className="text-amber-500 dark:text-amber-400 font-semibold">{isVi ? 'Điểm nhìn trước:' : 'Look-Ahead Point:'}</span>
              <span>θ_ahead</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-rose-500 rounded-full" />
              <span className="text-rose-500 dark:text-rose-400 font-semibold">{isVi ? 'Gradient nhìn trước:' : 'Lookahead Grad:'}</span>
              <span>-α∇L(θ_ahead)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-emerald-500 rounded-full" />
              <span className="text-emerald-500 dark:text-emerald-400 font-semibold">{isVi ? 'Cập nhật tổng hợp:' : 'Net Update:'}</span>
              <span>-v_t</span>
            </div>
          </div>
        </div>
      )}

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
          <MousePointerClick className="w-3.5 h-3.5 text-emerald-500" />
          <span>{isVi ? 'Nhấn lên bề mặt hàm mất mát để chọn điểm xuất phát mới' : 'Click contour to set new start point'}</span>
        </div>


        {/* Hover Tooltip */}
        {hoveredNode && (
          <div
            className={`absolute bottom-3 left-3 z-20 px-3 py-2 rounded-xl border text-xs font-mono shadow-lg backdrop-blur-md transition-all ${
              isDark
                ? 'bg-slate-900/95 border-emerald-500/50 text-slate-200 shadow-emerald-950/50'
                : 'bg-white/95 border-emerald-300 text-slate-800 shadow-slate-200'
            }`}
          >
            <div className="font-bold flex items-center justify-between gap-3 mb-1 pb-1 border-b border-inherit">
              <span className={hoveredNode.type === 'nesterov' ? 'text-emerald-400' : hoveredNode.type === 'polyak' ? 'text-purple-400' : 'text-rose-400'}>
                {hoveredNode.type === 'nesterov' ? 'NAG' : hoveredNode.type === 'polyak' ? 'Polyak Momentum' : 'Vanilla SGD'} #{hoveredNode.point.step}
              </span>
              <span className="text-slate-400">
                ({hoveredNode.point.w1.toFixed(2)}, {hoveredNode.point.w2.toFixed(2)})
              </span>
            </div>
            <div className="text-[11px] space-y-0.5">
              <div>Loss: <strong className="text-rose-500">{hoveredNode.point.loss.toFixed(4)}</strong></div>
              {hoveredNode.point.v1 !== undefined && (
                <div>
                  {isVi ? 'Vận tốc ‖v‖:' : 'Speed ‖v‖:'}{' '}
                  <strong className="text-emerald-500">
                    {Math.hypot(hoveredNode.point.v1, hoveredNode.point.v2 || 0).toFixed(4)}
                  </strong>
                </div>
              )}
              {hoveredNode.type === 'nesterov' && 'lookaheadW1' in hoveredNode.point && hoveredNode.point.lookaheadW1 !== undefined && (
                <div>
                  {isVi ? 'Nhìn trước:' : 'Lookahead:'}{' '}
                  <strong className="text-amber-400">
                    ({hoveredNode.point.lookaheadW1.toFixed(2)}, {(hoveredNode.point.lookaheadW2 ?? 0).toFixed(2)})
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
            {/* Arrow markers for vector visualization */}
            <marker
              id="arrow-jump"
              viewBox="0 0 10 10"
              refX="6"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#c084fc" />
            </marker>
            <marker
              id="arrow-lookahead-grad"
              viewBox="0 0 10 10"
              refX="6"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#fb7185" />
            </marker>
            <marker
              id="arrow-nesterov-update"
              viewBox="0 0 10 10"
              refX="6"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#10b981" />
            </marker>
          </defs>

          {/* 1. Contour Ellipses & Gradient Wash */}
          <g className="contours pointer-events-none">
            {contourEllipses.map((c, idx) => {
              const opacity = 0.08 + idx * 0.035;
              const strokeColor = isDark
                ? `rgba(16, 185, 129, ${0.15 + idx * 0.07})`
                : `rgba(5, 150, 105, ${0.18 + idx * 0.07})`;

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
                  {/* Contour Level Tag on right side */}
                  {c.cx + c.rx < svgWidth - PADDING.right - 10 && (
                    <text
                      x={c.cx + c.rx - 4}
                      y={c.cy - 4}
                      fill={isDark ? '#6ee7b7' : '#047857'}
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
            {/* W1 Axis (Horizontal through w2 = 0) */}
            <line
              x1={PADDING.left}
              y1={toSvgY(0)}
              x2={svgWidth - PADDING.right}
              y2={toSvgY(0)}
              strokeDasharray="2 2"
              opacity="0.6"
            />
            {/* W2 Axis (Vertical through w1 = 0) */}
            <line
              x1={toSvgX(0)}
              y1={PADDING.top}
              x2={toSvgX(0)}
              y2={svgHeight - PADDING.bottom}
              strokeDasharray="2 2"
              opacity="0.6"
            />

            {/* Global Optimum Marker θ* = (0, 0) */}
            <circle cx={toSvgX(0)} cy={toSvgY(0)} r="4.5" fill="#10b981" />
            <text
              x={toSvgX(0) + 7}
              y={toSvgY(0) + 12}
              fill="#10b981"
              fontSize="10"
              fontWeight="bold"
              fontFamily="monospace"
            >
              θ* = (0, 0)
            </text>
          </g>

          {/* 3. Vanilla SGD Trajectory (Red Dashed Path) */}
          {showVanilla && vanillaPath.length > 1 && (
            <g className="vanilla-path transition-opacity duration-300">
              <polyline
                fill="none"
                stroke="#f43f5e"
                strokeWidth="1.8"
                strokeDasharray="4 3"
                opacity="0.75"
                points={vanillaPath.map((p) => `${toSvgX(p.w1)},${toSvgY(p.w2)}`).join(' ')}
              />
              {vanillaPath.map((p, idx) => (
                <circle
                  key={`vanilla-${idx}`}
                  cx={toSvgX(p.w1)}
                  cy={toSvgY(p.w2)}
                  r="3"
                  fill="#f43f5e"
                  opacity="0.8"
                  className="cursor-pointer hover:r-4 transition-all"
                  onMouseEnter={() => setHoveredNode({ type: 'vanilla', point: p })}
                  onMouseLeave={() => setHoveredNode(null)}
                />
              ))}
            </g>
          )}

          {/* 4. Classical Polyak Momentum Trajectory (Purple Overshoot Path) */}
          {showPolyak && polyakPath.length > 1 && (
            <g className="polyak-path transition-opacity duration-300">
              <polyline
                fill="none"
                stroke="#a855f7"
                strokeWidth="2.2"
                strokeDasharray="3 2"
                opacity="0.85"
                points={polyakPath.map((p) => `${toSvgX(p.w1)},${toSvgY(p.w2)}`).join(' ')}
              />
              {polyakPath.map((p, idx) => {
                const isCur = idx === polyakPath.length - 1;
                return (
                  <circle
                    key={`polyak-${idx}`}
                    cx={toSvgX(p.w1)}
                    cy={toSvgY(p.w2)}
                    r={isCur ? 5 : 3.5}
                    fill="#a855f7"
                    stroke={isDark ? '#080d1a' : '#ffffff'}
                    strokeWidth="1"
                    className="cursor-pointer hover:r-5 transition-all"
                    onMouseEnter={() => setHoveredNode({ type: 'polyak', point: p })}
                    onMouseLeave={() => setHoveredNode(null)}
                  />
                );
              })}
            </g>
          )}

          {/* 5. Nesterov Accelerated Gradient Trajectory (Solid Glowing Emerald Path) */}
          {nesterovPath.length > 1 && (
            <g className="nesterov-path">
              {/* Outer Glow */}
              <polyline
                fill="none"
                stroke="#10b981"
                strokeWidth="5"
                opacity="0.25"
                strokeLinecap="round"
                strokeLinejoin="round"
                points={nesterovPath.map((p) => `${toSvgX(p.w1)},${toSvgY(p.w2)}`).join(' ')}
              />
              {/* Main Line */}
              <polyline
                fill="none"
                stroke="#10b981"
                strokeWidth="2.5"
                opacity="0.95"
                strokeLinecap="round"
                strokeLinejoin="round"
                points={nesterovPath.map((p) => `${toSvgX(p.w1)},${toSvgY(p.w2)}`).join(' ')}
              />
              {/* Nodes along NAG path */}
              {nesterovPath.map((p, idx) => {
                const isCurrent = idx === nesterovPath.length - 1;
                return (
                  <circle
                    key={`nag-${idx}`}
                    cx={toSvgX(p.w1)}
                    cy={toSvgY(p.w2)}
                    r={isCurrent ? 6 : 4}
                    fill={isCurrent ? '#34d399' : '#10b981'}
                    stroke={isDark ? '#080d1a' : '#ffffff'}
                    strokeWidth="1.5"
                    className="cursor-pointer transition-all duration-150"
                    onMouseEnter={() => setHoveredNode({ type: 'nesterov', point: p })}
                    onMouseLeave={() => setHoveredNode(null)}
                  />
                );
              })}
            </g>
          )}

          {/* 6. Active Lookahead Vectors & Braking Action Visualization */}
          {showVectors && !converged && (
            <g className="lookahead-vectors pointer-events-none">
              {/* Vector 1: Momentum Jump from Current Position to Lookahead Position */}
              {lookaheadDist > 0.05 && (
                <line
                  x1={currentSvgX}
                  y1={currentSvgY}
                  x2={lookaheadSvgX}
                  y2={lookaheadSvgY}
                  stroke="#c084fc"
                  strokeWidth="2"
                  strokeDasharray="4 3"
                  markerEnd="url(#arrow-jump)"
                />
              )}

              {/* Lookahead Point Reticle / Halo */}
              {lookaheadDist > 0.05 && (
                <g className="lookahead-point">
                  <circle
                    cx={lookaheadSvgX}
                    cy={lookaheadSvgY}
                    r="9"
                    fill="rgba(245, 158, 11, 0.15)"
                    stroke="#f59e0b"
                    strokeWidth="1.2"
                    strokeDasharray="2 2"
                    className="animate-pulse"
                  />
                  <circle cx={lookaheadSvgX} cy={lookaheadSvgY} r="3.5" fill="#f59e0b" />
                  <text
                    x={lookaheadSvgX + 8}
                    y={lookaheadSvgY - 6}
                    fill="#f59e0b"
                    fontSize="9.5"
                    fontFamily="monospace"
                    fontWeight="bold"
                  >
                    θ_ahead ({lookaheadW[0].toFixed(2)}, {lookaheadW[1].toFixed(2)})
                  </text>
                </g>
              )}

              {/* Vector 2: Lookahead Gradient Force Vector from Lookahead Point */}
              <line
                x1={lookaheadSvgX}
                y1={lookaheadSvgY}
                x2={lookaheadSvgX + lookaheadGradVecX}
                y2={lookaheadSvgY + lookaheadGradVecY}
                stroke="#fb7185"
                strokeWidth="2.2"
                markerEnd="url(#arrow-lookahead-grad)"
              />

              {/* Vector 3: Resultant Net Update Vector from Current Position */}
              <line
                x1={currentSvgX}
                y1={currentSvgY}
                x2={currentSvgX + updateVecX}
                y2={currentSvgY + updateVecY}
                stroke="#10b981"
                strokeWidth="3"
                markerEnd="url(#arrow-nesterov-update)"
              />
            </g>
          )}

          {/* 7. Current Position Marker */}
          <g className="current-pos pointer-events-none">
            <circle
              cx={currentSvgX}
              cy={currentSvgY}
              r="14"
              fill="rgba(16, 185, 129, 0.2)"
              stroke="#10b981"
              strokeWidth="1.5"
              strokeDasharray="4 3"
            />
            <circle cx={currentSvgX} cy={currentSvgY} r="5" fill="#10b981" stroke="#ffffff" strokeWidth="2" />
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
          <Navigation className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>
            <strong className="text-emerald-400">
              {isVi ? 'Cơ chế Phanh Chủ động (Look-ahead Braking):' : 'Anticipatory Braking Mechanism:'}
            </strong>{' '}
            {isVi
              ? 'NAG thực hiện "bước nhảy quán tính" trước θ_ahead = θ - β·v, đo gradient tại điểm tương lai. Nếu đà quán tính đang đẩy hòn bi leo dốc bên kia, gradient nhìn trước sẽ ghìm phanh ngay lập tức, ngăn ngừa vọt lố (overshooting)!'
              : 'NAG takes the momentum leap first θ_ahead = θ - β·v, evaluating the gradient ahead of time. If momentum would carry the ball up the opposite wall, the look-ahead gradient applies immediate counter-braking, preventing wild overshoots!'}
          </span>
        </div>

        <div className="hidden lg:flex items-center gap-2 shrink-0 font-mono text-[11px] text-slate-400">
          <span className="w-2 h-2 rounded-full bg-emerald-400" />
          <span>O(1/k²) Convergence</span>
        </div>
      </div>
    </div>
  );
};
