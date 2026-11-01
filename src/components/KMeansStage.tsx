import React, { useState, useMemo } from 'react';
import {
  Target,
  MousePointerClick,
  Layers,
  Activity,
  CheckCircle2,
  Info,
  GitCommit,
} from 'lucide-react';
import type { ThemeMode, Language, KMeansState, KMeansPoint } from '../types';

interface KMeansStageProps {
  theme: ThemeMode;
  language: Language;
  kmeansState?: KMeansState;
  onChangeInput?: (newInput: string) => void;
  isExpanded?: boolean;
}

const CLUSTER_PALETTE = [
  {
    name: 'Cyan',
    main: '#06b6d4',
    bg: 'rgba(6, 182, 212, 0.14)',
    border: '#0891b2',
    glow: 'rgba(6, 182, 212, 0.4)',
    text: 'text-cyan-500 dark:text-cyan-400',
    badge: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40',
  },
  {
    name: 'Amber',
    main: '#f59e0b',
    bg: 'rgba(245, 158, 11, 0.14)',
    border: '#d97706',
    glow: 'rgba(245, 158, 11, 0.4)',
    text: 'text-amber-500 dark:text-amber-400',
    badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
  },
  {
    name: 'Rose',
    main: '#f43f5e',
    bg: 'rgba(244, 63, 94, 0.14)',
    border: '#e11d48',
    glow: 'rgba(244, 63, 94, 0.4)',
    text: 'text-rose-500 dark:text-rose-400',
    badge: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
  },
  {
    name: 'Emerald',
    main: '#10b981',
    bg: 'rgba(16, 185, 129, 0.14)',
    border: '#059669',
    glow: 'rgba(16, 185, 129, 0.4)',
    text: 'text-emerald-500 dark:text-emerald-400',
    badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
  },
  {
    name: 'Violet',
    main: '#8b5cf6',
    bg: 'rgba(139, 92, 246, 0.14)',
    border: '#7c3aed',
    glow: 'rgba(139, 92, 246, 0.4)',
    text: 'text-violet-500 dark:text-violet-400',
    badge: 'bg-violet-500/20 text-violet-300 border-violet-500/40',
  },
];

const PADDING = { top: 30, right: 30, bottom: 40, left: 45 };

export const KMeansStage: React.FC<KMeansStageProps> = ({
  theme,
  language,
  kmeansState,
  onChangeInput,
  isExpanded = false,
}) => {
  const isDark = theme === 'studio';
  const isVi = language === 'vi';

  // Toggle overlay states
  const [showVoronoi, setShowVoronoi] = useState(true);
  const [showLines, setShowLines] = useState(true);
  const [hoveredPoint, setHoveredPoint] = useState<KMeansPoint | null>(null);
  const [hoveredCentroid, setHoveredCentroid] = useState<number | null>(null);

  // Fallback defaults if no state provided
  const state: KMeansState = useMemo(() => {
    return (
      kmeansState || {
        k: 3,
        iteration: 0,
        maxIterations: 10,
        phase: 'init',
        points: [],
        centroids: [],
        inertia: 0,
        centroidShift: 0,
        converged: false,
      }
    );
  }, [kmeansState]);

  const { points, centroids, k, iteration, phase, inertia, centroidShift } = state;

  // Coordinate Bounds computation
  const bounds = useMemo(() => {
    let minX = 0;
    let maxX = 10;
    let minY = 0;
    let maxY = 10;

    const allPts = [
      ...points.map((p) => ({ x: p.x, y: p.y })),
      ...centroids.map((c) => ({ x: c.x, y: c.y })),
    ];

    if (allPts.length > 0) {
      const xs = allPts.map((p) => p.x);
      const ys = allPts.map((p) => p.y);
      minX = Math.floor(Math.min(...xs)) - 1;
      maxX = Math.ceil(Math.max(...xs)) + 1;
      minY = Math.floor(Math.min(...ys)) - 1;
      maxY = Math.ceil(Math.max(...ys)) + 1;
    }

    // Keep aspect ratio sane and minimum span
    const spanX = Math.max(10, maxX - minX);
    const spanY = Math.max(10, maxY - minY);
    return {
      minX: Math.min(minX, 0),
      maxX: Math.max(maxX, minX + spanX),
      minY: Math.min(minY, 0),
      maxY: Math.max(maxY, minY + spanY),
    };
  }, [points, centroids]);

  // SVG coordinate transformation
  const svgWidth = isExpanded ? 880 : 720;
  const svgHeight = isExpanded ? 520 : 420;
  const innerWidth = svgWidth - PADDING.left - PADDING.right;
  const innerHeight = svgHeight - PADDING.top - PADDING.bottom;

  const toSvgX = (x: number) => {
    const ratio = (x - bounds.minX) / (bounds.maxX - bounds.minX || 1);
    return PADDING.left + ratio * innerWidth;
  };

  const toSvgY = (y: number) => {
    const ratio = (y - bounds.minY) / (bounds.maxY - bounds.minY || 1);
    return PADDING.top + (1 - ratio) * innerHeight; // Invert for cartesian Y
  };

  const toCartesianX = (svgX: number) => {
    const ratio = (svgX - PADDING.left) / innerWidth;
    return Number((bounds.minX + ratio * (bounds.maxX - bounds.minX)).toFixed(2));
  };

  const toCartesianY = (svgY: number) => {
    const ratio = (svgHeight - PADDING.bottom - svgY) / innerHeight;
    return Number((bounds.minY + ratio * (bounds.maxY - bounds.minY)).toFixed(2));
  };

  // Discrete Voronoi background approximation grid
  const voronoiCells = useMemo(() => {
    if (!showVoronoi || centroids.length === 0) return [];
    const cols = 48;
    const rows = 32;
    const cellW = innerWidth / cols;
    const cellH = innerHeight / rows;

    const cells: { x: number; y: number; w: number; h: number; color: string }[] = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const ratioX = (c + 0.5) / cols;
        const ratioY = 1 - (r + 0.5) / rows;
        const cartX = bounds.minX + ratioX * (bounds.maxX - bounds.minX);
        const cartY = bounds.minY + ratioY * (bounds.maxY - bounds.minY);
        // Find closest centroid
        let bestDist = Infinity;
        let bestCluster = 0;
        centroids.forEach((cent, idx) => {
          const dx = cartX - cent.x;
          const dy = cartY - cent.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < bestDist) {
            bestDist = d2;
            bestCluster = idx;
          }
        });

        const palette = CLUSTER_PALETTE[bestCluster % CLUSTER_PALETTE.length];
        cells.push({
          x: PADDING.left + c * cellW,
          y: PADDING.top + r * cellH,
          w: cellW + 0.5,
          h: cellH + 0.5,
          color: palette.bg,
        });
      }
    }
    return cells;
  }, [showVoronoi, centroids, innerWidth, innerHeight, bounds]);

  // Click on canvas to add a new point
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

    const newX = toCartesianX(clickSvgX);
    const newY = toCartesianY(clickSvgY);

    // Format new points input string
    const currentPointsList = points.map((p) => `(${p.x}, ${p.y})`);
    currentPointsList.push(`(${newX}, ${newY})`);
    onChangeInput(currentPointsList.join(', '));
  };

  // Phase badge metadata
  const phaseMeta = useMemo(() => {
    switch (phase) {
      case 'init':
        return {
          title: isVi ? 'Giai đoạn 0: Khởi tạo Tâm Cụm (Seeds)' : 'Phase 0: K-Means++ Seeding',
          badge: isVi ? 'Khởi tạo' : 'Initialize',
          color: 'text-amber-500 bg-amber-500/10 border-amber-500/30',
          desc: isVi
            ? 'Khởi tạo K tâm cụm ngẫu nhiên phân tán khắp không gian dữ liệu.'
            : 'Initialize K initial centroids distributed across the feature space.',
        };
      case 'assign':
        return {
          title: isVi ? 'Giai đoạn 1: Gán Cụm (Phân hoạch Voronoi)' : 'Phase 1: Assignment Step',
          badge: isVi ? 'Gán Cụm' : 'Assignment',
          color: 'text-cyan-500 bg-cyan-500/10 border-cyan-500/30',
          desc: isVi
            ? 'Mỗi điểm dữ liệu được gán vào tâm cụm gần nhất theo khoảng cách Euclid.'
            : 'Every point is assigned to its closest centroid via Euclidean distance.',
        };
      case 'update':
        return {
          title: isVi ? 'Giai đoạn 2: Cập nhật Trọng tâm (Means)' : 'Phase 2: Centroid Update',
          badge: isVi ? 'Cập nhật Tâm' : 'Mean Update',
          color: 'text-indigo-500 bg-indigo-500/10 border-indigo-500/30',
          desc: isVi
            ? 'Tọa độ mỗi tâm cụm được tính lại bằng trung bình cộng tọa độ các điểm thuộc cụm.'
            : 'Each centroid shifts to the geometric center of mass of its assigned cluster.',
        };
      case 'converged':
        return {
          title: isVi ? 'Hội tụ Thành công! (Tâm cụm Cố định)' : 'Converged! Centroids Stabilized',
          badge: isVi ? 'Đã Hội tụ' : 'Converged',
          color: 'text-emerald-500 bg-emerald-500/10 border-emerald-500/30',
          desc: isVi
            ? 'Độ dịch chuyển tâm Δμ = 0 hoặc phân cụm không đổi. WCSS đạt cực tiểu cục bộ.'
            : 'Centroid shift Δμ = 0 or assignments unchanged. WCSS reached local optimum.',
        };
      default:
        return {
          title: 'K-Means Clustering',
          badge: 'Clustering',
          color: 'text-cyan-500 bg-cyan-500/10 border-cyan-500/30',
          desc: '',
        };
    }
  }, [phase, isVi]);

  // Points distribution per cluster
  const clusterCounts = useMemo(() => {
    const counts = new Array(k).fill(0);
    points.forEach((p) => {
      if (p.cluster >= 0 && p.cluster < k) {
        counts[p.cluster]++;
      }
    });
    return counts;
  }, [points, k]);

  return (
    <div className="w-full h-full flex flex-col items-center justify-between relative select-none p-2 sm:p-3 overflow-hidden">
      {/* ========================================================= */}
      {/* 1. TOP INTERACTIVE STATS & CONTROLS STRIP                */}
      {/* ========================================================= */}
      <div className="w-full max-w-4xl flex flex-wrap items-center justify-between gap-2.5 z-10 mb-2">
        {/* Phase Badge & Iteration */}
        <div className="flex items-center gap-2">
          <div
            className={`px-2.5 py-1 rounded-xl text-xs font-semibold flex items-center gap-1.5 border shadow-2xs ${phaseMeta.color}`}
          >
            {phase === 'converged' ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
            ) : (
              <Activity className="w-3.5 h-3.5 animate-pulse" />
            )}
            <span>{phaseMeta.badge}</span>
            <span className="font-mono text-[11px] opacity-75">
              • {isVi ? 'Vòng lặp' : 'Iter'} {iteration}
            </span>
          </div>

          <div
            className={`hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-xl text-xs font-mono border ${
              isDark ? 'bg-slate-900/80 border-slate-800 text-slate-300' : 'bg-white border-slate-200 text-slate-700'
            }`}
          >
            <span className="text-slate-500 font-bold">K = {k}</span>
            <span className="text-slate-400">•</span>
            <span>
              {isVi ? 'Quán tính WCSS:' : 'Inertia:'}{' '}
              <strong className="text-emerald-500 dark:text-emerald-400">{inertia.toFixed(2)}</strong>
            </span>
            {centroidShift > 0 && (
              <>
                <span className="text-slate-400">•</span>
                <span title={isVi ? 'Độ dịch chuyển tâm' : 'Centroid shift'}>
                  Δμ: <strong className="text-amber-500">{centroidShift.toFixed(2)}</strong>
                </span>
              </>
            )}
          </div>
        </div>

        {/* View Toggles */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setShowVoronoi(!showVoronoi)}
            className={`px-2 py-1 rounded-lg text-xs font-medium flex items-center gap-1 border transition-all ${
              showVoronoi
                ? isDark
                  ? 'bg-cyan-950/80 text-cyan-300 border-cyan-800'
                  : 'bg-cyan-50 text-cyan-700 border-cyan-300'
                : isDark
                ? 'bg-slate-800/60 text-slate-400 border-slate-700'
                : 'bg-slate-100 text-slate-500 border-slate-200'
            }`}
            title={isVi ? 'Bật/Tắt vùng phân hoạch Voronoi' : 'Toggle Voronoi Partitioning'}
          >
            <Layers className="w-3 h-3" />
            <span className="hidden md:inline">{isVi ? 'Vùng Voronoi' : 'Voronoi'}</span>
          </button>

          <button
            onClick={() => setShowLines(!showLines)}
            className={`px-2 py-1 rounded-lg text-xs font-medium flex items-center gap-1 border transition-all ${
              showLines
                ? isDark
                  ? 'bg-indigo-950/80 text-indigo-300 border-indigo-800'
                  : 'bg-indigo-50 text-indigo-700 border-indigo-300'
                : isDark
                ? 'bg-slate-800/60 text-slate-400 border-slate-700'
                : 'bg-slate-100 text-slate-500 border-slate-200'
            }`}
            title={isVi ? 'Bật/Tắt đường nối điểm đến tâm cụm' : 'Toggle Point-to-Centroid Lines'}
          >
            <GitCommit className="w-3 h-3" />
            <span className="hidden md:inline">{isVi ? 'Đường nối' : 'Distance Lines'}</span>
          </button>
        </div>
      </div>

      {/* ========================================================= */}
      {/* 2. MAIN 2D CLUSTERING CANVAS (SVG VORONOI + POINTS)        */}
      {/* ========================================================= */}
      <div
        className={`w-full max-w-4xl flex-1 relative flex items-center justify-center rounded-2xl border overflow-hidden shadow-inner transition-colors ${
          isDark ? 'bg-[#0a0f1d] border-slate-800' : 'bg-slate-50/70 border-slate-200'
        }`}
      >
        {/* Click to add point guide hint */}
        <div className="absolute top-2.5 left-3 z-10 flex items-center gap-1.5 text-[11px] font-sans opacity-70 pointer-events-none">
          <MousePointerClick className="w-3.5 h-3.5 text-cyan-500" />
          <span>{isVi ? 'Nhấn lên đồ thị để thêm điểm dữ liệu mới' : 'Click on plane to add data point'}</span>
        </div>

        {/* Hover Tooltip Card */}
        {hoveredPoint && (
          <div
            className={`absolute top-2.5 right-3 z-20 px-3 py-2 rounded-xl border text-xs font-mono shadow-lg backdrop-blur-md transition-all ${
              isDark
                ? 'bg-slate-900/95 border-cyan-500/50 text-slate-200 shadow-cyan-950/50'
                : 'bg-white/95 border-cyan-300 text-slate-800 shadow-slate-200'
            }`}
          >
            <div className="flex items-center justify-between gap-3 mb-1 pb-1 border-b border-inherit">
              <span className="font-bold flex items-center gap-1.5">
                <span
                  className="w-2.5 h-2.5 rounded-full"
                  style={{
                    backgroundColor:
                      hoveredPoint.cluster >= 0
                        ? CLUSTER_PALETTE[hoveredPoint.cluster % CLUSTER_PALETTE.length].main
                        : '#94a3b8',
                  }}
                />
                {isVi ? 'Điểm #' : 'Point #'}{hoveredPoint.id}
              </span>
              <span className="font-semibold text-cyan-500">
                ({hoveredPoint.x.toFixed(2)}, {hoveredPoint.y.toFixed(2)})
              </span>
            </div>
            <div className="text-[11px] space-y-0.5">
              <div>
                {isVi ? 'Thuộc cụm:' : 'Assigned:'}{' '}
                <strong
                  style={{
                    color:
                      hoveredPoint.cluster >= 0
                        ? CLUSTER_PALETTE[hoveredPoint.cluster % CLUSTER_PALETTE.length].main
                        : undefined,
                  }}
                >
                  {hoveredPoint.cluster >= 0
                    ? `Cluster ${hoveredPoint.cluster + 1} (${CLUSTER_PALETTE[hoveredPoint.cluster % CLUSTER_PALETTE.length].name})`
                    : isVi ? 'Chưa gán' : 'Unassigned'}
                </strong>
              </div>
              {hoveredPoint.distanceToCentroid !== undefined && (
                <div>
                  {isVi ? 'Khoảng cách tới tâm d:' : 'Distance to μ:'}{' '}
                  <strong className="text-amber-500 font-bold">{hoveredPoint.distanceToCentroid.toFixed(3)}</strong>
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
            {/* Glow filters for Centroids */}
            <filter id="glow-centroid" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            {/* Soft grid pattern */}
            <pattern id="grid-pattern" width="40" height="40" patternUnits="userSpaceOnUse">
              <path
                d="M 40 0 L 0 0 0 40"
                fill="none"
                stroke={isDark ? 'rgba(51, 65, 85, 0.25)' : 'rgba(203, 213, 225, 0.5)'}
                strokeWidth="1"
              />
            </pattern>
          </defs>

          {/* 1. Background Grid */}
          <rect
            x={PADDING.left}
            y={PADDING.top}
            width={innerWidth}
            height={innerHeight}
            fill="url(#grid-pattern)"
          />

          {/* 2. Voronoi Cells Shading */}
          {showVoronoi && (
            <g className="voronoi-regions pointer-events-none transition-opacity duration-300">
              {voronoiCells.map((c, i) => (
                <rect
                  key={i}
                  x={c.x}
                  y={c.y}
                  width={c.w}
                  height={c.h}
                  fill={c.color}
                />
              ))}
            </g>
          )}

          {/* 3. Coordinate Axes */}
          <g className="axes text-[10px] font-mono select-none" stroke={isDark ? '#334155' : '#cbd5e1'}>
            {/* X-axis */}
            <line
              x1={PADDING.left}
              y1={toSvgY(bounds.minY)}
              x2={PADDING.left + innerWidth}
              y2={toSvgY(bounds.minY)}
              strokeWidth="1.5"
            />
            {/* Y-axis */}
            <line
              x1={toSvgX(bounds.minX)}
              y1={PADDING.top}
              x2={toSvgX(bounds.minX)}
              y2={PADDING.top + innerHeight}
              strokeWidth="1.5"
            />
          </g>

          {/* 4. Axis Ticks & Labels */}
          <g className="ticks text-[9px] font-mono fill-slate-400 select-none pointer-events-none">
            {/* X Ticks */}
            {[0, 2, 4, 6, 8, 10].map((val) => {
              if (val < bounds.minX || val > bounds.maxX) return null;
              const sx = toSvgX(val);
              const sy = toSvgY(bounds.minY);
              return (
                <g key={`x-${val}`}>
                  <line x1={sx} y1={sy} x2={sx} y2={sy + 4} stroke={isDark ? '#475569' : '#94a3b8'} />
                  <text x={sx} y={sy + 14} textAnchor="middle">
                    {val}
                  </text>
                </g>
              );
            })}
            {/* Y Ticks */}
            {[0, 2, 4, 6, 8, 10].map((val) => {
              if (val < bounds.minY || val > bounds.maxY) return null;
              const sx = toSvgX(bounds.minX);
              const sy = toSvgY(val);
              return (
                <g key={`y-${val}`}>
                  <line x1={sx - 4} y1={sy} x2={sx} y2={sy} stroke={isDark ? '#475569' : '#94a3b8'} />
                  <text x={sx - 8} y={sy + 3} textAnchor="end">
                    {val}
                  </text>
                </g>
              );
            })}
          </g>

          {/* 5. Centroid Trajectory Trails (from prevX, prevY to x, y) */}
          <g className="centroid-trails pointer-events-none">
            {centroids.map((c, i) => {
              if (c.prevX === undefined || c.prevY === undefined) return null;
              const startX = toSvgX(c.prevX);
              const startY = toSvgY(c.prevY);
              const endX = toSvgX(c.x);
              const endY = toSvgY(c.y);
              const dist = Math.hypot(endX - startX, endY - startY);
              if (dist < 2) return null;

              const palette = CLUSTER_PALETTE[i % CLUSTER_PALETTE.length];
              return (
                <g key={`trail-${i}`}>
                  <line
                    x1={startX}
                    y1={startY}
                    x2={endX}
                    y2={endY}
                    stroke={palette.main}
                    strokeWidth="2"
                    strokeDasharray="4 3"
                    opacity="0.75"
                  />
                  <circle cx={startX} cy={startY} r="3" fill={palette.main} opacity="0.4" />
                </g>
              );
            })}
          </g>

          {/* 6. Distance Vectors (Lines from Points to their Centroid) */}
          {showLines && (
            <g className="point-lines pointer-events-none transition-opacity duration-300">
              {points.map((p) => {
                if (p.cluster < 0 || p.cluster >= centroids.length) return null;
                const cent = centroids[p.cluster];
                const px = toSvgX(p.x);
                const py = toSvgY(p.y);
                const cx = toSvgX(cent.x);
                const cy = toSvgY(cent.y);
                const palette = CLUSTER_PALETTE[p.cluster % CLUSTER_PALETTE.length];
                const isHovered = hoveredPoint?.id === p.id;

                return (
                  <line
                    key={`line-${p.id}`}
                    x1={px}
                    y1={py}
                    x2={cx}
                    y2={cy}
                    stroke={palette.main}
                    strokeWidth={isHovered ? '2' : '1'}
                    strokeDasharray={isHovered ? 'none' : '2 3'}
                    opacity={isHovered ? 0.95 : 0.28}
                  />
                );
              })}
            </g>
          )}

          {/* 7. Data Points */}
          <g className="points">
            {points.map((p) => {
              const cx = toSvgX(p.x);
              const cy = toSvgY(p.y);
              const isAssigned = p.cluster >= 0 && p.cluster < CLUSTER_PALETTE.length;
              const palette = isAssigned ? CLUSTER_PALETTE[p.cluster % CLUSTER_PALETTE.length] : null;
              const isHovered = hoveredPoint?.id === p.id;

              return (
                <g
                  key={`pt-${p.id}`}
                  className="cursor-pointer"
                  onMouseEnter={() => setHoveredPoint(p)}
                  onMouseLeave={() => setHoveredPoint(null)}
                >
                  {/* Outer selection ring on hover */}
                  {isHovered && (
                    <circle
                      cx={cx}
                      cy={cy}
                      r="10"
                      fill="none"
                      stroke={palette ? palette.main : '#38bdf8'}
                      strokeWidth="2"
                      opacity="0.8"
                    />
                  )}
                  {/* Point core */}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isHovered ? 6.5 : 5}
                    fill={palette ? palette.main : isDark ? '#94a3b8' : '#64748b'}
                    stroke={isDark ? '#0f172a' : '#ffffff'}
                    strokeWidth="1.5"
                    className="transition-all duration-200"
                  />
                </g>
              );
            })}
          </g>

          {/* 8. Cluster Centroids (μ_k) Prominent Markers */}
          <g className="centroids">
            {centroids.map((c, i) => {
              const cx = toSvgX(c.x);
              const cy = toSvgY(c.y);
              const palette = CLUSTER_PALETTE[i % CLUSTER_PALETTE.length];
              const isHovered = hoveredCentroid === i;

              return (
                <g
                  key={`centroid-${i}`}
                  className="cursor-pointer"
                  onMouseEnter={() => setHoveredCentroid(i)}
                  onMouseLeave={() => setHoveredCentroid(null)}
                >
                  {/* Outer halo centered on centroid */}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isHovered ? 20 : 16}
                    fill={palette.bg}
                    stroke={palette.main}
                    strokeWidth="1.5"
                    strokeDasharray="4 3"
                    opacity={0.85}
                  />

                  {/* Diamond / Cross marker */}
                  <path
                    d={`M ${cx} ${cy - 9} L ${cx + 9} ${cy} L ${cx} ${cy + 9} L ${cx - 9} ${cy} Z`}
                    fill={palette.main}
                    stroke={isDark ? '#090d16' : '#ffffff'}
                    strokeWidth="2"
                    filter="url(#glow-centroid)"
                  />

                  {/* Inner center dot */}
                  <circle cx={cx} cy={cy} r="2.5" fill={isDark ? '#090d16' : '#ffffff'} />

                  {/* Centroid Label (μ_1, μ_2, ...) */}
                  <g transform={`translate(${cx + 12}, ${cy - 10})`}>
                    <rect
                      x="-3"
                      y="-12"
                      width="34"
                      height="16"
                      rx="4"
                      fill={isDark ? '#090d16' : '#ffffff'}
                      stroke={palette.main}
                      strokeWidth="1"
                      opacity="0.92"
                    />
                    <text
                      x="14"
                      y="0"
                      textAnchor="middle"
                      className="text-[10px] font-mono font-bold"
                      fill={palette.main}
                    >
                      μ{i + 1}
                    </text>
                  </g>
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      {/* ========================================================= */}
      {/* 3. CLUSTER SUMMARY PILLS                                   */}
      {/* ========================================================= */}
      <div className="w-full max-w-4xl flex items-center justify-between gap-2 mt-2 pt-2 border-t border-slate-200 dark:border-slate-800 text-xs">
        <div className="flex items-center gap-2 overflow-x-auto max-w-full">
          <span className="text-slate-500 font-medium shrink-0 flex items-center gap-1">
            <Target className="w-3.5 h-3.5" />
            <span>{isVi ? 'Các Cụm:' : 'Clusters:'}</span>
          </span>
          {centroids.map((c, i) => {
            const palette = CLUSTER_PALETTE[i % CLUSTER_PALETTE.length];
            const count = clusterCounts[i] || 0;
            return (
              <div
                key={`badge-${i}`}
                className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold flex items-center gap-1.5 border shrink-0 ${palette.badge}`}
              >
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: palette.main }} />
                <span>
                  μ{i + 1} ({c.x.toFixed(1)}, {c.y.toFixed(1)})
                </span>
                <span className="opacity-70 font-sans">
                  • {count} {isVi ? 'điểm' : 'pts'}
                </span>
              </div>
            );
          })}
        </div>

        <div className="hidden lg:flex items-center gap-1 text-[11px] text-slate-500 shrink-0">
          <Info className="w-3.5 h-3.5 text-cyan-500" />
          <span>{phaseMeta.desc}</span>
        </div>
      </div>
    </div>
  );
};
