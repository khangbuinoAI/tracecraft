import React, { useState, useMemo, useEffect } from 'react';
import {
  Menu,
  X,
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  RotateCcw,
  Sliders,
  Shuffle,
  Code2,
  BookOpen,
  Layers,
  Search,
  Activity,
  Sparkles,
  Cpu,
  Monitor,
  Smartphone,
  Sun,
  Moon,
  Binary,
  BrainCircuit,
  Info,
  Check,
  Maximize2,
  Wifi,
  BatteryCharging
} from 'lucide-react';
import type {
  ThemeMode,
  TraceStep,
  AlgorithmPreset,
  Language,
  AndroidTab,
  AICategory,
  AlgorithmCategory
} from '../types';
import {
  ALGORITHM_PRESETS,
  getPresetTitle,
  getPresetCategory,
  getPresetBadge,
  getPresetSummary,
  getPresetVariants
} from '../algorithms/presets';
import { getLocalizedText } from '../i18n/translations';
import { VisualStage } from './VisualStage';
import { LessonGuide } from './LessonGuide';
import { LanguageSwitch } from './LanguageSwitch';

interface AndroidLayoutProps {
  theme: ThemeMode;
  onToggleTheme: () => void;
  selectedPreset: AlgorithmPreset;
  onSelectPreset: (preset: AlgorithmPreset) => void;
  step: TraceStep;
  currentStep: number;
  totalSteps: number;
  isPlaying: boolean;
  speed: number;
  onTogglePlay: () => void;
  onNextStep: () => void;
  onPrevStep: () => void;
  onReset: () => void;
  onSeek: (step: number) => void;
  onChangeSpeed: (speed: number) => void;
  inputStr: string;
  onChangeInput: (newInput: string) => void;
  language: Language;
  onSelectLanguage: (lang: Language) => void;
  onSwitchToDesktop?: () => void;
  isMobileScreen: boolean;
}

// Android Robot Icon SVG
export const AndroidIcon: React.FC<{ className?: string }> = ({ className = 'w-4 h-4' }) => (
  <svg viewBox="0 0 24 24" fill="currentColor" className={className}>
    {/* Android Head */}
    <path d="M6 10C6 6.686 8.686 4 12 4s6 2.686 6 6v1H6v-1z" />
    {/* Left Antenna */}
    <line x1="8" y1="4" x2="6.5" y2="1.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    {/* Right Antenna */}
    <line x1="16" y1="4" x2="17.5" y2="1.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    {/* Eyes */}
    <circle cx="9.5" cy="8" r="0.9" fill="var(--bg-eye, #090d16)" />
    <circle cx="14.5" cy="8" r="0.9" fill="var(--bg-eye, #090d16)" />
    {/* Body */}
    <path d="M6 12.5C6 11.67 6.67 11 7.5 11h9c.83 0 1.5.67 1.5 1.5V18c0 .83-.67 1.5-1.5 1.5H7.5C6.67 19.5 6 18.83 6 18v-5.5z" />
    {/* Left Arm */}
    <rect x="3.5" y="11.5" width="1.6" height="6" rx="0.8" />
    {/* Right Arm */}
    <rect x="18.9" y="11.5" width="1.6" height="6" rx="0.8" />
    {/* Left Leg */}
    <rect x="8.5" y="19.5" width="1.8" height="3" rx="0.9" />
    {/* Right Leg */}
    <rect x="13.7" y="19.5" width="1.8" height="3" rx="0.9" />
  </svg>
);

export const AndroidLayout: React.FC<AndroidLayoutProps> = ({
  theme,
  onToggleTheme,
  selectedPreset,
  onSelectPreset,
  step,
  currentStep,
  totalSteps,
  isPlaying,
  speed,
  onTogglePlay,
  onNextStep,
  onPrevStep,
  onReset,
  onSeek,
  onChangeSpeed,
  inputStr,
  onChangeInput,
  language,
  onSelectLanguage,
  onSwitchToDesktop,
  isMobileScreen,
}) => {
  const isDark = theme === 'studio';
  const t = getLocalizedText(language);

  // Android Navigation Tabs
  const [activeTab, setActiveTab] = useState<AndroidTab>('stage');
  const [drawerOpen, setDrawerOpen] = useState<boolean>(false);
  const [isDeviceFrame, setIsDeviceFrame] = useState<boolean>(!isMobileScreen);
  const [drawerSearch, setDrawerSearch] = useState<string>('');
  const [curriculumSearch, setCurriculumSearch] = useState<string>('');
  const [selectedCurriculumCategory, setSelectedCurriculumCategory] = useState<AlgorithmCategory | 'All'>('All');
  const [timeStr, setTimeStr] = useState<string>('09:41');

  // Live time for Android status bar
  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const hours = String(now.getHours()).padStart(2, '0');
      const mins = String(now.getMinutes()).padStart(2, '0');
      setTimeStr(`${hours}:${mins}`);
    };
    updateTime();
    const timer = setInterval(updateTime, 30000);
    return () => clearInterval(timer);
  }, []);

  const variants = useMemo(() => {
    return getPresetVariants(selectedPreset, language) || selectedPreset.inputVariants || [];
  }, [selectedPreset, language]);

  const speeds = [0.5, 1, 2];
  const cycleSpeed = () => {
    const idx = speeds.indexOf(speed);
    const next = speeds[(idx + 1) % speeds.length];
    onChangeSpeed(next);
  };

  const progressPct = ((currentStep - 1) / Math.max(1, totalSteps - 1)) * 100;

  // Handle data randomize
  const handleRandomize = () => {
    if (variants.length > 0) {
      const randomVariant = variants[Math.floor(Math.random() * variants.length)];
      onChangeInput(randomVariant.value);
    }
  };

  // Filtered presets for drawer
  const filteredDrawerPresets = useMemo(() => {
    const q = drawerSearch.toLowerCase().trim();
    if (!q) return ALGORITHM_PRESETS;
    return ALGORITHM_PRESETS.filter(
      (p) =>
        p.title.toLowerCase().includes(q) ||
        p.category.toLowerCase().includes(q) ||
        p.tags.some((tag) => tag.toLowerCase().includes(q))
    );
  }, [drawerSearch]);

  // Filtered presets for Curriculum tab
  const filteredCurriculumPresets = useMemo(() => {
    return ALGORITHM_PRESETS.filter((p) => {
      const matchesCat =
        selectedCurriculumCategory === 'All' || p.category === selectedCurriculumCategory;
      const q = curriculumSearch.toLowerCase().trim();
      const matchesQuery =
        !q ||
        p.title.toLowerCase().includes(q) ||
        p.summary.toLowerCase().includes(q) ||
        p.tags.some((t) => t.toLowerCase().includes(q));
      return matchesCat && matchesQuery;
    });
  }, [curriculumSearch, selectedCurriculumCategory]);

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Classical Machine Learning':
        return <Layers className="w-4 h-4 text-amber-500" />;
      case 'Deep Learning & Neural Networks':
        return <Binary className="w-4 h-4 text-indigo-400" />;
      case 'Transformers & Modern LLMs':
        return <BrainCircuit className="w-4 h-4 text-cyan-400" />;
      default:
        return <Cpu className="w-4 h-4 text-emerald-400" />;
    }
  };

  // Android Viewport Container
  return (
    <div
      className={`min-h-screen w-full flex items-center justify-center transition-colors duration-300 ${
        isDark ? 'bg-[#06080f] text-slate-100' : 'bg-[#ece8df] text-slate-800'
      }`}
    >
      {/* Desktop Controls Header (When on desktop with simulated Android chassis) */}
      {!isMobileScreen && (
        <div className="fixed top-3 left-4 right-4 z-50 flex items-center justify-between pointer-events-none">
          <div className="pointer-events-auto flex items-center gap-2 bg-[#0d1322]/90 backdrop-blur-md border border-[#1f2c47] px-3.5 py-1.5 rounded-full shadow-lg text-xs font-mono text-cyan-300">
            <AndroidIcon className="w-4 h-4 text-emerald-400 animate-pulse" />
            <span className="font-bold tracking-wide">TRACE CRAFT ANDROID EDITION</span>
            <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded-full border border-emerald-500/30">
              M3 UI
            </span>
          </div>

          <div className="pointer-events-auto flex items-center gap-2">
            <button
              onClick={() => setIsDeviceFrame((prev) => !prev)}
              className="flex items-center gap-1.5 bg-[#0d1322]/90 hover:bg-[#151f38] backdrop-blur-md border border-[#1f2c47] px-3 py-1.5 rounded-full shadow-lg text-xs font-semibold text-slate-200 transition-all active:scale-95"
              title={isDeviceFrame ? t.fullscreenView : t.phoneFrame}
            >
              {isDeviceFrame ? (
                <>
                  <Maximize2 className="w-3.5 h-3.5 text-cyan-400" />
                  <span>{t.fullscreenView}</span>
                </>
              ) : (
                <>
                  <Smartphone className="w-3.5 h-3.5 text-amber-400" />
                  <span>{t.phoneFrame}</span>
                </>
              )}
            </button>

            {onSwitchToDesktop && (
              <button
                onClick={onSwitchToDesktop}
                className="flex items-center gap-1.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white px-3.5 py-1.5 rounded-full shadow-lg text-xs font-bold transition-all active:scale-95"
                title={t.desktopStudio}
              >
                <Monitor className="w-3.5 h-3.5" />
                <span>{t.desktopStudio}</span>
              </button>
            )}
          </div>
        </div>
      )}

      {/* Android Device Shell (or Fullscreen View) */}
      <div
        className={`relative flex flex-col overflow-hidden transition-all duration-300 ${
          isDeviceFrame && !isMobileScreen
            ? 'w-[412px] h-[870px] max-h-[96vh] rounded-[48px] border-[10px] border-[#1e2330] shadow-[0_25px_60px_-15px_rgba(0,0,0,0.7),0_0_0_1px_rgba(255,255,255,0.1)] ring-1 ring-black/40'
            : 'w-full h-[100dvh] rounded-none border-none'
        } ${
          isDark ? 'bg-[#0a0e1a] text-slate-100' : 'bg-[#fbfbfb] text-slate-900'
        }`}
      >
        {/* ========================================================= */}
        {/* ANDROID STATUS BAR (Pixel / Material You Inspired)        */}
        {/* ========================================================= */}
        <div
          className={`h-7 px-6 shrink-0 flex items-center justify-between text-[11px] font-semibold select-none z-30 transition-colors ${
            isDark ? 'bg-[#0a0e1a] text-slate-300' : 'bg-[#fbfbfb] text-slate-700'
          }`}
        >
          {/* Left: Digital Time */}
          <span className="font-mono tracking-tight font-bold">{timeStr}</span>

          {/* Center: Android Camera Punch-Hole (when frame active) */}
          {isDeviceFrame && !isMobileScreen && (
            <div className="w-3.5 h-3.5 bg-black rounded-full ring-2 ring-[#1e2330] -mt-1 flex items-center justify-center">
              <div className="w-1 h-1 bg-cyan-900/60 rounded-full" />
            </div>
          )}

          {/* Right: Android Status Icons (5G, Wi-Fi, Battery) */}
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-mono tracking-tighter opacity-80">5G</span>
            <Wifi className="w-3 h-3 opacity-90" />
            <div className="flex items-center gap-0.5">
              <BatteryCharging className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-[9px] font-mono">100%</span>
            </div>
          </div>
        </div>

        {/* ========================================================= */}
        {/* MATERIAL 3 TOP APP BAR                                    */}
        {/* ========================================================= */}
        <header
          className={`h-14 px-3.5 border-b shrink-0 flex items-center justify-between select-none z-20 transition-colors ${
            isDark
              ? 'bg-[#0f1424] border-[#1c273e] text-slate-100'
              : 'bg-white border-slate-200 text-slate-800'
          }`}
        >
          {/* Left Hamburger Button -> Open Drawer */}
          <div className="flex items-center gap-2.5 min-w-0">
            <button
              onClick={() => setDrawerOpen(true)}
              className={`w-9 h-9 rounded-full flex items-center justify-center transition-all active:scale-90 ${
                isDark ? 'hover:bg-slate-800 text-slate-200' : 'hover:bg-slate-100 text-slate-700'
              }`}
              title={t.curriculumDrawer}
            >
              <Menu className="w-5 h-5" />
            </button>

            {/* App & Lesson Title */}
            <div className="flex flex-col min-w-0">
              <div className="flex items-center gap-1.5">
                <AndroidIcon className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                <span className="text-xs font-bold font-mono tracking-tight text-emerald-400 uppercase">
                  TraceCraft
                </span>
                <span
                  title={getPresetCategory(selectedPreset, language)}
                  className={`text-[9px] font-mono px-1.5 py-0.2 rounded-full border truncate max-w-[120px] ${
                    isDark
                      ? 'bg-cyan-950/60 text-cyan-300 border-cyan-800/60'
                      : 'bg-blue-50 text-blue-700 border-blue-200'
                  }`}
                >
                  {getPresetBadge(selectedPreset, language)}
                </span>
              </div>
              <h1 className="text-xs font-bold truncate max-w-[170px] sm:max-w-[210px] text-slate-900 dark:text-slate-100">
                {getPresetTitle(selectedPreset, language)}
              </h1>
            </div>
          </div>

          {/* Right Top Actions */}
          <div className="flex items-center gap-1">
            {/* Language Switch */}
            <LanguageSwitch language={language} onSelectLanguage={onSelectLanguage} theme={theme} />

            {/* Theme Toggle */}
            <button
              onClick={onToggleTheme}
              className={`w-8 h-8 rounded-full flex items-center justify-center transition-all active:scale-90 ${
                isDark ? 'text-amber-400 hover:bg-slate-800' : 'text-slate-600 hover:bg-slate-100'
              }`}
              title="Toggle Theme"
            >
              {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>

            {/* Step Reset */}
            <button
              onClick={onReset}
              className={`w-8 h-8 rounded-full flex items-center justify-center transition-all active:scale-90 ${
                isDark ? 'text-slate-400 hover:bg-slate-800' : 'text-slate-600 hover:bg-slate-100'
              }`}
              title={t.reset}
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </header>

        {/* ========================================================= */}
        {/* MAIN BODY CONTENT (Tab Switcher)                          */}
        {/* ========================================================= */}
        <main className="flex-1 overflow-y-auto overflow-x-hidden relative flex flex-col">
          {/* TAB 1: VISUAL STAGE & EXECUTION TRACKER */}
          {activeTab === 'stage' && (
            <div className="flex-1 flex flex-col p-3 space-y-3 pb-4 animate-tab-enter">
              {variants.length > 0 && (
                <div className="flex items-center gap-1.5 overflow-x-auto pb-1 shrink-0 no-scrollbar">
                  <div className="flex items-center gap-1 text-[11px] font-mono opacity-60 mr-1 shrink-0">
                    <Sliders className="w-3 h-3 text-cyan-400" />
                    <span>{t.variants}:</span>
                  </div>
                  {variants.map((v) => {
                    const isSelected =
                      inputStr.trim().replace(/\s+/g, '') === v.value.trim().replace(/\s+/g, '');
                    return (
                      <button
                        key={v.label}
                        onClick={() => {
                          onChangeInput(v.value);
                          onReset();
                        }}
                        className={`text-xs px-2.5 py-1 rounded-full whitespace-nowrap font-semibold border transition-all active:scale-95 ${
                          isSelected
                            ? isDark
                              ? 'bg-cyan-500 text-slate-950 border-cyan-400 font-bold shadow-sm shadow-cyan-500/30'
                              : 'bg-blue-600 text-white border-blue-600 font-bold shadow-sm'
                            : isDark
                              ? 'bg-[#121929] border-[#222f47] text-slate-300 hover:bg-[#1a253c]'
                              : 'bg-white border-slate-200 text-slate-700 hover:bg-slate-100'
                        }`}
                      >
                        {v.label}
                      </button>
                    );
                  })}
                  <button
                    onClick={handleRandomize}
                    className="p-1 rounded-full border border-dashed border-slate-600 hover:border-cyan-400 text-slate-400 hover:text-cyan-400 transition-colors shrink-0"
                    title="Randomize Variant"
                  >
                    <Shuffle className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}

              {/* Central Interactive Visual Stage */}
              <div
                className={`w-full rounded-2xl overflow-hidden border shadow-sm transition-colors shrink-0 ${
                  isDark ? 'bg-[#0d1222] border-[#1e2a42]' : 'bg-white border-slate-200'
                }`}
                style={{ minHeight: '340px' }}
              >
                <VisualStage
                  theme={theme}
                  step={step}
                  category={selectedPreset.category}
                  preset={selectedPreset}
                  inputStr={inputStr}
                  onChangeInput={onChangeInput}
                  language={language}
                />
              </div>

              {/* Real-World Impact Intuition Banner */}
              <div
                className={`p-3 rounded-2xl border text-xs flex items-start gap-2.5 transition-colors ${
                  isDark
                    ? 'bg-[#0e172a]/80 border-cyan-900/40 text-slate-200'
                    : 'bg-blue-50/70 border-blue-200 text-slate-800'
                }`}
              >
                <Info className="w-4 h-4 text-cyan-500 dark:text-cyan-400 shrink-0 mt-0.5" />
                <div className="space-y-0.5">
                  <span className="font-bold text-cyan-600 dark:text-cyan-300 block">
                    {language === 'vi' ? 'Ý nghĩa thực tế:' : 'Real-World Impact:'}
                  </span>
                  <p className="opacity-90 leading-relaxed text-[11px]">
                    {getPresetSummary(selectedPreset, language)}
                  </p>
                </div>
              </div>

              {/* Active Step Diagnostic & Math Invariant */}
              <div
                className={`p-3.5 rounded-2xl border flex flex-col gap-2 transition-colors ${
                  isDark ? 'bg-[#101728] border-[#1e2a42]' : 'bg-white border-slate-200'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <span
                      className={`text-[10px] font-mono font-bold uppercase px-2 py-0.5 rounded-full ${
                        isDark
                          ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                          : 'bg-blue-100 text-blue-800 border border-blue-200'
                      }`}
                    >
                      {step.action}
                    </span>
                    <span className="text-xs font-mono opacity-70">
                      {t.step} {currentStep} {t.of} {totalSteps}
                    </span>
                  </div>
                  {step.line > 0 && (
                    <span className="text-[11px] font-mono opacity-60">
                      Line {step.line}
                    </span>
                  )}
                </div>

                <p className="text-xs leading-relaxed font-medium text-slate-800 dark:text-slate-200">
                  {step.explanation}
                </p>

                {step.invariant && (
                  <div
                    className={`p-2 rounded-xl border-l-2 text-[11px] font-mono ${
                      isDark
                        ? 'bg-cyan-950/30 border-cyan-400 text-cyan-200'
                        : 'bg-blue-50 border-blue-600 text-blue-950'
                    }`}
                  >
                    <span className="font-bold uppercase text-[9px] block text-cyan-400 mb-0.5">
                      {t.invariant}:
                    </span>
                    <span>{step.invariant}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 2: CODE & LIVE STATE INSPECTOR */}
          {activeTab === 'code' && (
            <div className="flex-1 p-3.5 space-y-3 pb-24 animate-tab-enter">
              {/* Code Snippet Runner Card */}
              <div
                className={`rounded-2xl border overflow-hidden ${
                  isDark ? 'bg-[#101728] border-[#1e2a42]' : 'bg-white border-slate-200'
                }`}
              >
                <div className="px-3.5 py-2.5 border-b border-inherit flex items-center justify-between text-xs font-bold">
                  <div className="flex items-center gap-1.5">
                    <Code2 className="w-4 h-4 text-cyan-400" />
                    <span>{t.code}</span>
                  </div>
                  <span className="text-[10px] font-mono bg-slate-800 text-cyan-300 px-2 py-0.5 rounded-full">
                    Python 3.12
                  </span>
                </div>

                <div className="p-2 font-mono text-xs leading-relaxed max-h-72 overflow-y-auto">
                  {selectedPreset.code.split('\n').map((lineText, idx) => {
                    const lineNum = idx + 1;
                    const isActive = lineNum === step.line;
                    return (
                      <div
                        key={idx}
                        className={`flex items-center py-0.5 px-2 rounded-lg whitespace-pre ${
                          isActive
                            ? isDark
                              ? 'bg-cyan-950/80 text-cyan-300 font-bold border-l-2 border-cyan-400 shadow-sm'
                              : 'bg-blue-50 text-blue-900 font-bold border-l-2 border-blue-600'
                            : 'opacity-70'
                        }`}
                      >
                        <span className="w-6 text-right pr-2 text-[10px] opacity-40 select-none shrink-0">
                          {lineNum}
                        </span>
                        <span>{lineText || ' '}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Live State & Variables Inspector */}
              <div
                className={`p-3.5 rounded-2xl border ${
                  isDark ? 'bg-[#101728] border-[#1e2a42]' : 'bg-white border-slate-200'
                }`}
              >
                <div className="flex items-center gap-1.5 text-xs font-bold mb-2.5">
                  <Activity className="w-4 h-4 text-emerald-400" />
                  <span>{t.stateInspector}</span>
                </div>

                {Object.keys(step.state || {}).length === 0 ? (
                  <p className="text-xs opacity-60 font-mono italic">{t.noActiveState}</p>
                ) : (
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(step.state).map(([key, val]) => (
                      <div
                        key={key}
                        className={`p-2 rounded-xl border font-mono text-xs ${
                          isDark
                            ? 'bg-[#0a0e1a] border-slate-800 text-slate-200'
                            : 'bg-slate-50 border-slate-200 text-slate-800'
                        }`}
                      >
                        <span className="text-[10px] font-bold text-cyan-400 block truncate">
                          {key}
                        </span>
                        <span className="font-semibold truncate block mt-0.5">
                          {typeof val === 'number'
                            ? Number.isInteger(val)
                              ? val
                              : val.toFixed(4)
                            : typeof val === 'object'
                              ? JSON.stringify(val).slice(0, 30)
                              : String(val)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Execution Diagnostics */}
              <div
                className={`p-3.5 rounded-2xl border ${
                  isDark ? 'bg-[#101728] border-[#1e2a42]' : 'bg-white border-slate-200'
                }`}
              >
                <div className="flex items-center gap-1.5 text-xs font-bold mb-1.5 text-amber-500">
                  <Sparkles className="w-4 h-4" />
                  <span>{t.executionLog}</span>
                </div>
                <p className="text-xs leading-relaxed opacity-90">{step.explanation}</p>
              </div>
            </div>
          )}

          {/* TAB 3: LESSON GUIDE & THEORY */}
          {activeTab === 'theory' && (
            <div className="flex-1 p-2 pb-24 overflow-y-auto animate-tab-enter">
              <LessonGuide
                theme={theme}
                preset={selectedPreset}
                onLaunchVisualizer={() => setActiveTab('stage')}
                language={language}
              />
            </div>
          )}

          {/* TAB 4: CURRICULUM EXPLORER */}
          {activeTab === 'curriculum' && (
            <div className="flex-1 p-3.5 space-y-3 pb-24 overflow-y-auto animate-tab-enter">
              {/* Search Bar */}
              <div
                className={`flex items-center px-3 py-2 rounded-xl border text-xs font-mono ${
                  isDark
                    ? 'bg-[#101728] border-[#1e2a42] text-slate-200'
                    : 'bg-white border-slate-200 text-slate-800'
                }`}
              >
                <Search className="w-4 h-4 opacity-50 mr-2 shrink-0" />
                <input
                  type="text"
                  placeholder={t.searchLessons}
                  value={curriculumSearch}
                  onChange={(e) => setCurriculumSearch(e.target.value)}
                  className="w-full bg-transparent outline-none"
                />
              </div>

              {/* Category Filter Pills */}
              <div className="flex items-center gap-1.5 overflow-x-auto pb-1 no-scrollbar">
                {(['All', 'Classical Machine Learning', 'Deep Learning & Neural Networks', 'Transformers & Modern LLMs'] as const).map(
                  (cat) => (
                    <button
                      key={cat}
                      onClick={() => setSelectedCurriculumCategory(cat)}
                      className={`text-[11px] px-2.5 py-1 rounded-full whitespace-nowrap font-semibold border transition-all ${
                        selectedCurriculumCategory === cat
                          ? isDark
                            ? 'bg-cyan-500 text-slate-950 border-cyan-400 font-bold'
                            : 'bg-blue-600 text-white border-blue-600 font-bold'
                          : isDark
                            ? 'bg-[#101728] border-[#1e2a42] text-slate-400'
                            : 'bg-white border-slate-200 text-slate-700'
                      }`}
                    >
                      {cat === 'All' ? t.all : t.categories[cat as AICategory] || cat}
                    </button>
                  )
                )}
              </div>

              {/* Lessons List */}
              <div className="space-y-2">
                {filteredCurriculumPresets.map((preset) => {
                  const isSelected = preset.id === selectedPreset.id;
                  return (
                    <div
                      key={preset.id}
                      onClick={() => {
                        onSelectPreset(preset);
                        setActiveTab('stage');
                      }}
                      className={`p-3 rounded-2xl border cursor-pointer transition-all active:scale-[0.98] ${
                        isSelected
                          ? isDark
                            ? 'bg-[#131f38] border-cyan-500/60 shadow-md shadow-cyan-500/10'
                            : 'bg-blue-50/90 border-blue-400 shadow-sm'
                          : isDark
                            ? 'bg-[#101728] border-[#1e2a42] hover:bg-[#151f35]'
                            : 'bg-white border-slate-200 hover:bg-slate-50'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2 mb-1.5">
                        <div className="flex items-center gap-1.5">
                          {getCategoryIcon(preset.category)}
                          <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full font-bold bg-slate-800 text-slate-300">
                            {getPresetBadge(preset, language)}
                          </span>
                        </div>
                        {isSelected && (
                          <span className="flex items-center gap-1 text-[10px] font-bold text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded-full border border-cyan-800/40">
                            <Check className="w-3 h-3" />
                            {language === 'vi' ? 'Đang mở' : 'Active'}
                          </span>
                        )}
                      </div>
                      <h3 className="text-xs font-bold text-slate-900 dark:text-slate-100">
                        {getPresetTitle(preset, language)}
                      </h3>
                      <p className="text-[11px] opacity-70 line-clamp-2 mt-1 leading-relaxed">
                        {getPresetSummary(preset, language)}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ========================================================= */}
          {/* ANDROID FLOATING PLAYER DOCK (Visible on 'stage' tab)     */}
          {/* ========================================================= */}
        </main>
        {/* ========================================================= */}
        {/* ANDROID BOTTOM PLAYER DOCK (Docked above Bottom Nav)      */}
        {/* ========================================================= */}
        {activeTab === 'stage' && (
          <div
            className={`shrink-0 px-3 py-2 border-t z-20 flex flex-col gap-1.5 transition-colors select-none ${
              isDark
                ? 'bg-[#0e1628] border-[#1e2b45] text-slate-100'
                : 'bg-white border-slate-200 text-slate-900 shadow-sm'
            }`}
          >
            {/* Touch Scrubber */}
            <div className="w-full relative flex items-center h-3">
              <div className="absolute inset-x-0 h-1.5 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden pointer-events-none">
                <div
                  style={{ width: `${progressPct}%` }}
                  className="h-full bg-cyan-500 rounded-full transition-all duration-100 ease-out"
                />
              </div>
              <input
                type="range"
                min={1}
                max={Math.max(1, totalSteps)}
                value={currentStep}
                onChange={(e) => onSeek(Number(e.target.value))}
                className="w-full h-3 opacity-0 cursor-pointer absolute inset-0 z-10"
              />
              <div
                style={{ left: `calc(${progressPct}% - 6px)` }}
                className="w-3 h-3 rounded-full bg-cyan-400 shadow pointer-events-none absolute transition-all duration-100 ease-out border-2 border-white dark:border-slate-950"
              />
            </div>

            {/* Player Controls */}
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono opacity-70">
                {currentStep} / {totalSteps}
              </span>

              <div className="flex items-center gap-2">
                <button
                  onClick={onPrevStep}
                  disabled={currentStep <= 1}
                  className="w-8 h-8 rounded-full flex items-center justify-center transition-all active:scale-90 disabled:opacity-30 bg-slate-200/60 dark:bg-slate-800/80"
                  title={t.previous}
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>

                {/* Circular FAB Play/Pause */}
                <button
                  onClick={onTogglePlay}
                  className="w-10 h-10 rounded-full flex items-center justify-center bg-gradient-to-r from-cyan-500 to-blue-600 text-slate-950 font-bold shadow-md shadow-cyan-500/25 active:scale-90 transition-transform"
                  title={isPlaying ? t.pause : t.play}
                >
                  {isPlaying ? (
                    <Pause className="w-4 h-4 fill-current" />
                  ) : (
                    <Play className="w-4 h-4 fill-current ml-0.5" />
                  )}
                </button>

                <button
                  onClick={onNextStep}
                  disabled={currentStep >= totalSteps}
                  className="w-8 h-8 rounded-full flex items-center justify-center transition-all active:scale-90 disabled:opacity-30 bg-slate-200/60 dark:bg-slate-800/80"
                  title={t.next}
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>

              <button
                onClick={cycleSpeed}
                className="text-[10px] font-mono px-2 py-0.5 rounded-full font-bold bg-slate-200 dark:bg-slate-800 text-cyan-400"
                title={t.speed}
              >
                {speed}x
              </button>
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* ANDROID CURRICULUM DRAWER (Slide from Left)               */}
        {/* ========================================================= */}
        {drawerOpen && (
          <div className="absolute inset-0 z-40 flex">
            {/* Backdrop */}
            <div
              onClick={() => setDrawerOpen(false)}
              className="absolute inset-0 bg-black/60 backdrop-blur-xs transition-opacity animate-fade-in"
            />

            {/* Slide-in Drawer Container */}
            <div
              className={`relative w-[300px] h-full flex flex-col shadow-2xl z-50 transition-colors animate-slide-right ${
                isDark ? 'bg-[#0d1322] border-r border-[#1c273e] text-slate-100' : 'bg-white border-r border-slate-200 text-slate-800'
              }`}
            >
              {/* Drawer Header */}
              <div className="p-4 border-b border-inherit flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center text-slate-950 shadow-md">
                    <AndroidIcon className="w-5 h-5 text-slate-950" />
                  </div>
                  <div>
                    <h2 className="text-sm font-bold font-mono tracking-tight">TraceCraft Android</h2>
                    <p className="text-[10px] opacity-60">{t.curriculum}</p>
                  </div>
                </div>

                <button
                  onClick={() => setDrawerOpen(false)}
                  className="w-8 h-8 rounded-full flex items-center justify-center hover:bg-slate-800/60 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Drawer Search */}
              <div className="p-3 border-b border-inherit">
                <div
                  className={`flex items-center px-3 py-1.5 rounded-xl border text-xs font-mono ${
                    isDark ? 'bg-[#080c16] border-[#1c273e]' : 'bg-slate-50 border-slate-200'
                  }`}
                >
                  <Search className="w-3.5 h-3.5 opacity-40 mr-2 shrink-0" />
                  <input
                    type="text"
                    placeholder={t.searchLessons}
                    value={drawerSearch}
                    onChange={(e) => setDrawerSearch(e.target.value)}
                    className="w-full bg-transparent outline-none text-xs"
                  />
                </div>
              </div>

              {/* Drawer Lessons List */}
              <div className="flex-1 overflow-y-auto p-3 space-y-1.5">
                {filteredDrawerPresets.map((preset) => {
                  const isSelected = preset.id === selectedPreset.id;
                  return (
                    <button
                      key={preset.id}
                      onClick={() => {
                        onSelectPreset(preset);
                        setDrawerOpen(false);
                        setActiveTab('stage');
                      }}
                      className={`w-full text-left p-2.5 rounded-xl flex items-center gap-2.5 transition-all text-xs font-semibold ${
                        isSelected
                          ? isDark
                            ? 'bg-cyan-950/70 text-cyan-300 border border-cyan-800/50 shadow-xs'
                            : 'bg-blue-50 text-blue-900 border border-blue-200 shadow-xs'
                          : isDark
                            ? 'hover:bg-slate-800/50 text-slate-300'
                            : 'hover:bg-slate-100 text-slate-700'
                      }`}
                    >
                      <div className="shrink-0">{getCategoryIcon(preset.category)}</div>
                      <div className="min-w-0 flex-1">
                        <span className="block truncate">{getPresetTitle(preset, language)}</span>
                        <span className="text-[10px] opacity-60 font-mono block truncate">
                          {getPresetBadge(preset, language)}
                        </span>
                      </div>
                      {isSelected && <Check className="w-4 h-4 text-cyan-400 shrink-0" />}
                    </button>
                  );
                })}
              </div>

              {/* Drawer Footer */}
              <div className="p-3 border-t border-inherit flex items-center justify-between text-[11px] opacity-70">
                <span>12 {language === 'vi' ? 'Bài học' : 'Lessons'}</span>
                <span className="font-mono text-[10px]">v1.0.0 M3</span>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* ANDROID MATERIAL 3 BOTTOM NAVIGATION BAR                  */}
        {/* ========================================================= */}
        <nav
          className={`h-16 shrink-0 border-t flex items-center justify-around px-2 select-none z-30 transition-colors ${
            isDark
              ? 'bg-[#0c1120] border-[#1c273e] text-slate-300'
              : 'bg-white border-slate-200 text-slate-600'
          }`}
        >
          {/* Tab 1: Stage */}
          <button
            onClick={() => setActiveTab('stage')}
            className="flex-1 flex flex-col items-center justify-center gap-1 transition-all active:scale-95 py-1"
          >
            <div
              className={`px-4 py-0.5 rounded-full flex items-center justify-center transition-all ${
                activeTab === 'stage'
                  ? isDark
                    ? 'bg-cyan-500/20 text-cyan-300 font-bold'
                    : 'bg-blue-100 text-blue-800 font-bold'
                  : 'opacity-70'
              }`}
            >
              <Activity className="w-4 h-4" />
            </div>
            <span
              className={`text-[10px] tracking-tight ${
                activeTab === 'stage' ? 'font-bold text-cyan-400 dark:text-cyan-300' : 'opacity-70'
              }`}
            >
              {t.visualize}
            </span>
          </button>

          {/* Tab 2: Code */}
          <button
            onClick={() => setActiveTab('code')}
            className="flex-1 flex flex-col items-center justify-center gap-1 transition-all active:scale-95 py-1"
          >
            <div
              className={`px-4 py-0.5 rounded-full flex items-center justify-center transition-all ${
                activeTab === 'code'
                  ? isDark
                    ? 'bg-cyan-500/20 text-cyan-300 font-bold'
                    : 'bg-blue-100 text-blue-800 font-bold'
                  : 'opacity-70'
              }`}
            >
              <Code2 className="w-4 h-4" />
            </div>
            <span
              className={`text-[10px] tracking-tight ${
                activeTab === 'code' ? 'font-bold text-cyan-400 dark:text-cyan-300' : 'opacity-70'
              }`}
            >
              {t.code}
            </span>
          </button>

          {/* Tab 3: Theory */}
          <button
            onClick={() => setActiveTab('theory')}
            className="flex-1 flex flex-col items-center justify-center gap-1 transition-all active:scale-95 py-1"
          >
            <div
              className={`px-4 py-0.5 rounded-full flex items-center justify-center transition-all ${
                activeTab === 'theory'
                  ? isDark
                    ? 'bg-cyan-500/20 text-cyan-300 font-bold'
                    : 'bg-blue-100 text-blue-800 font-bold'
                  : 'opacity-70'
              }`}
            >
              <BookOpen className="w-4 h-4" />
            </div>
            <span
              className={`text-[10px] tracking-tight ${
                activeTab === 'theory' ? 'font-bold text-cyan-400 dark:text-cyan-300' : 'opacity-70'
              }`}
            >
              {t.theory}
            </span>
          </button>

          {/* Tab 4: Curriculum */}
          <button
            onClick={() => setActiveTab('curriculum')}
            className="flex-1 flex flex-col items-center justify-center gap-1 transition-all active:scale-95 py-1"
          >
            <div
              className={`px-4 py-0.5 rounded-full flex items-center justify-center transition-all ${
                activeTab === 'curriculum'
                  ? isDark
                    ? 'bg-cyan-500/20 text-cyan-300 font-bold'
                    : 'bg-blue-100 text-blue-800 font-bold'
                  : 'opacity-70'
              }`}
            >
              <Layers className="w-4 h-4" />
            </div>
            <span
              className={`text-[10px] tracking-tight ${
                activeTab === 'curriculum'
                  ? 'font-bold text-cyan-400 dark:text-cyan-300'
                  : 'opacity-70'
              }`}
            >
              {t.curriculum}
            </span>
          </button>
        </nav>
      </div>
    </div>
  );
};
export default AndroidLayout;
