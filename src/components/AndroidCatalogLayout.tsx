import React, { useState, useMemo, useEffect } from 'react';
import {
  Home,
  Layers,
  Search,
  Sparkles,
  Binary,
  BrainCircuit,
  ArrowRight,
  Menu,
  X,
  Monitor,
  Smartphone,
  Sun,
  Moon,
  Maximize2,
  Wifi,
  BatteryCharging,
  Settings,
  Activity,
  Cpu
} from 'lucide-react';
import type {
  ThemeMode,
  AlgorithmPreset,
  Language,
  AICategory,
  AlgorithmCategory,
  DifficultyLevel
} from '../types';
import {
  ALGORITHM_PRESETS,
  getPresetTitle,
  getPresetSummary,
  getPresetBadge
} from '../algorithms/presets';
import { getLocalizedText } from '../i18n/translations';
import { MathFormula } from './MathFormula';
import { LanguageSwitch } from './LanguageSwitch';
import { AndroidIcon } from './AndroidLayout';

interface AndroidCatalogLayoutProps {
  theme: ThemeMode;
  onToggleTheme: () => void;
  onSelectAlgorithm: (preset: AlgorithmPreset) => void;
  language: Language;
  onSelectLanguage: (lang: Language) => void;
  onSwitchToDesktop?: () => void;
  onLaunchVisualizer?: () => void;
  isMobileScreen: boolean;
}

type MainTab = 'home' | 'categories' | 'settings';

export const AndroidCatalogLayout: React.FC<AndroidCatalogLayoutProps> = ({
  theme,
  onToggleTheme,
  onSelectAlgorithm,
  language,
  onSelectLanguage,
  onSwitchToDesktop,
  onLaunchVisualizer,
  isMobileScreen,
}) => {
  const isDark = theme === 'studio';
  const t = getLocalizedText(language);

  const [activeTab, setActiveTab] = useState<MainTab>('home');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<AlgorithmCategory | 'All'>('All');
  const [drawerOpen, setDrawerOpen] = useState<boolean>(false);
  const [isDeviceFrame, setIsDeviceFrame] = useState<boolean>(!isMobileScreen);
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

  const categories: (AlgorithmCategory | 'All')[] = [
    'All',
    'Classical Machine Learning',
    'Deep Learning & Neural Networks',
    'Transformers & Modern LLMs',
  ];

  const filteredPresets = useMemo(() => {
    return ALGORITHM_PRESETS.filter((p) => {
      const matchesCategory =
        selectedCategory === 'All' || p.category === selectedCategory;
      const q = searchQuery.toLowerCase().trim();
      const matchesSearch =
        !q ||
        p.title.toLowerCase().includes(q) ||
        p.summary.toLowerCase().includes(q) ||
        p.description.toLowerCase().includes(q) ||
        p.tags.some((tag) => tag.toLowerCase().includes(tag));
      return matchesCategory && matchesSearch;
    });
  }, [searchQuery, selectedCategory]);

  // Group by category for 'categories' tab
  const groupedPresets = useMemo(() => {
    const map: Partial<Record<AlgorithmCategory, AlgorithmPreset[]>> = {};
    for (const p of ALGORITHM_PRESETS) {
      if (!map[p.category]) map[p.category] = [];
      map[p.category]?.push(p);
    }
    return map;
  }, []);

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

  const getDifficultyBadge = (difficulty: DifficultyLevel) => {
    switch (difficulty) {
      case 'Beginner':
        return isDark
          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
          : 'bg-emerald-50 text-emerald-700 border border-emerald-200';
      case 'Intermediate':
        return isDark
          ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
          : 'bg-amber-50 text-amber-700 border border-amber-200';
      case 'Advanced':
        return isDark
          ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30'
          : 'bg-rose-50 text-rose-700 border border-rose-200';
    }
  };

  // Featured Spotlight Lesson
  const spotlightPreset = ALGORITHM_PRESETS[0];

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
              Home M3
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
          <span className="font-mono tracking-tight font-bold">{timeStr}</span>

          {/* Android Punch Hole Notch */}
          {isDeviceFrame && !isMobileScreen && (
            <div className="w-3.5 h-3.5 bg-black rounded-full ring-2 ring-[#1e2330] -mt-1 flex items-center justify-center">
              <div className="w-1 h-1 bg-cyan-900/60 rounded-full" />
            </div>
          )}

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
          {/* Hamburger Drawer Toggle */}
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

            {/* App Brand */}
            <div className="flex items-center gap-1.5">
              <AndroidIcon className="w-4 h-4 text-emerald-400 shrink-0" />
              <span className="text-sm font-extrabold font-mono tracking-tight text-slate-900 dark:text-white">
                TraceCraft
              </span>
              <span className="text-[9px] font-mono bg-emerald-500/20 text-emerald-300 px-1.5 py-0.2 rounded-full border border-emerald-500/30">
                M3
              </span>
            </div>
          </div>

          {/* Right Top Actions */}
          <div className="flex items-center gap-1">
            <LanguageSwitch language={language} onSelectLanguage={onSelectLanguage} theme={theme} />

            <button
              onClick={onToggleTheme}
              className={`w-8 h-8 rounded-full flex items-center justify-center transition-all active:scale-90 ${
                isDark ? 'text-amber-400 hover:bg-slate-800' : 'text-slate-600 hover:bg-slate-100'
              }`}
              title="Toggle Theme"
            >
              {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
        </header>

        {/* ========================================================= */}
        {/* MAIN BODY CONTENT                                         */}
        {/* ========================================================= */}
        <main className="flex-1 overflow-y-auto overflow-x-hidden relative flex flex-col pb-20">
          {/* TAB 1: HOME FEED */}
          {activeTab === 'home' && (
            <div className="p-3.5 space-y-4 animate-tab-enter">
              {/* Featured Lesson Hero Card */}
              <div
                onClick={() => onSelectAlgorithm(spotlightPreset)}
                className={`p-4 rounded-3xl border cursor-pointer relative overflow-hidden transition-all active:scale-[0.98] shadow-md ${
                  isDark
                    ? 'bg-gradient-to-br from-[#121c33] via-[#0e1628] to-[#152342] border-[#223356] shadow-cyan-500/10'
                    : 'bg-gradient-to-br from-blue-500 via-indigo-600 to-cyan-500 text-white shadow-blue-500/20'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-1.5">
                    <Sparkles className="w-4 h-4 text-cyan-300" />
                    <span className="text-[10px] font-mono uppercase tracking-wider font-bold text-cyan-300">
                      {language === 'vi' ? 'Bài học Nổi bật' : 'Featured Lesson'}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-white/20 text-white font-bold backdrop-blur-xs">
                    01 • {getPresetBadge(spotlightPreset, language)}
                  </span>
                </div>

                <h2 className="text-sm font-bold text-white mb-1">
                  {getPresetTitle(spotlightPreset, language)}
                </h2>

                <p className="text-xs text-white/80 line-clamp-2 leading-relaxed mb-3">
                  {getPresetSummary(spotlightPreset, language)}
                </p>

                <div className="flex items-center justify-between pt-1">
                  <span className="text-xs font-bold text-white flex items-center gap-1">
                    <span>{language === 'vi' ? 'Bắt đầu học ngay' : 'Start Lesson Now'}</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </span>
                  <div className="w-7 h-7 rounded-full bg-white text-slate-950 flex items-center justify-center shadow">
                    <ArrowRight className="w-4 h-4" />
                  </div>
                </div>
              </div>

              {/* Material 3 Search Bar */}
              <div
                className={`flex items-center px-3.5 py-2.5 rounded-2xl border transition-colors ${
                  isDark
                    ? 'bg-[#101728] border-[#1e2a42] text-slate-200 focus-within:border-cyan-400'
                    : 'bg-white border-slate-200 text-slate-800 shadow-xs focus-within:border-blue-500'
                }`}
              >
                <Search className="w-4 h-4 opacity-50 mr-2.5 shrink-0" />
                <input
                  type="text"
                  placeholder={t.searchPlaceholder}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-transparent text-xs font-mono outline-none"
                />
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery('')}
                    className="p-1 rounded-full hover:bg-slate-700/40 text-slate-400"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>

              {/* Horizontally Scrollable Category Filter Chips */}
              <div className="flex items-center gap-1.5 overflow-x-auto pb-1 no-scrollbar shrink-0">
                {categories.map((cat) => (
                  <button
                    key={cat}
                    onClick={() => setSelectedCategory(cat)}
                    className={`text-xs px-3 py-1.5 rounded-full whitespace-nowrap font-semibold border transition-all active:scale-95 ${
                      selectedCategory === cat
                        ? isDark
                          ? 'bg-cyan-500 text-slate-950 border-cyan-400 font-bold shadow-sm shadow-cyan-500/20'
                          : 'bg-blue-600 text-white border-blue-600 font-bold shadow-sm'
                        : isDark
                          ? 'bg-[#101728] border-[#1e2a42] text-slate-300 hover:bg-[#162138]'
                          : 'bg-white border-slate-200 text-slate-700 hover:bg-slate-100'
                    }`}
                  >
                    {cat === 'All' ? t.all : t.categories[cat as AICategory] || cat}
                  </button>
                ))}
              </div>

              {/* Lessons Feed Header */}
              <div className="flex items-center justify-between text-xs pt-1">
                <span className="font-bold opacity-80 uppercase tracking-wider text-[10px] font-mono">
                  {filteredPresets.length} {language === 'vi' ? 'BÀI HỌC KHẢ DỤNG' : 'AVAILABLE LESSONS'}
                </span>
              </div>

              {/* Lesson Cards Feed */}
              <div className="space-y-3">
                {filteredPresets.map((preset) => (
                  <div
                    key={preset.id}
                    onClick={() => onSelectAlgorithm(preset)}
                    className={`p-3.5 rounded-2xl border cursor-pointer transition-all active:scale-[0.98] ${
                      isDark
                        ? 'bg-[#0f1526] hover:bg-[#141d33] border-[#1c273e] hover:border-cyan-500/40 shadow-sm'
                        : 'bg-white hover:bg-slate-50 border-slate-200 shadow-xs'
                    }`}
                  >
                    {/* Card Header */}
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <div className="flex items-center gap-1.5">
                        {getCategoryIcon(preset.category)}
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full font-bold bg-slate-800 text-slate-300">
                          {getPresetBadge(preset, language)}
                        </span>
                      </div>
                      <span
                        className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded-full font-bold ${getDifficultyBadge(
                          preset.difficulty
                        )}`}
                      >
                        {t.difficulties[preset.difficulty] || preset.difficulty}
                      </span>
                    </div>

                    {/* Title */}
                    <h3 className="text-xs font-bold text-slate-900 dark:text-slate-100 leading-snug">
                      {getPresetTitle(preset, language)}
                    </h3>

                    {/* Summary */}
                    <p className="text-[11px] opacity-75 line-clamp-2 leading-relaxed mt-1">
                      {getPresetSummary(preset, language)}
                    </p>

                    {/* Formula Pill preview */}
                    {preset.formula && (
                      <div className="mt-2.5 p-1.5 rounded-xl bg-slate-100 dark:bg-[#070b14] border border-slate-200 dark:border-slate-800 overflow-x-auto no-scrollbar">
                        <div className="text-[11px] scale-90 origin-left">
                          <MathFormula math={preset.formula} />
                        </div>
                      </div>
                    )}

                    {/* Card Action */}
                    <div className="mt-3 pt-2.5 border-t border-inherit flex items-center justify-between text-xs">
                      <span className="text-[11px] font-mono text-cyan-600 dark:text-cyan-400 font-semibold flex items-center gap-1">
                        <span>{language === 'vi' ? 'Mở mô phỏng' : 'Launch Visualizer'}</span>
                        <ArrowRight className="w-3 h-3" />
                      </span>
                      <span className="text-[10px] font-mono opacity-50">{preset.timeComplexity}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 2: CATEGORIES BREAKDOWN */}
          {activeTab === 'categories' && (
            <div className="p-3.5 space-y-4 animate-tab-enter">
              <div className="space-y-1">
                <h2 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  {language === 'vi' ? 'Phân loại Mô hình AI' : 'AI Curriculum Tracks'}
                </h2>
                <p className="text-xs opacity-70">
                  {language === 'vi'
                    ? 'Lộ trình từ Machine Learning Cổ điển đến Transformers & LLMs hiện đại.'
                    : 'From classical foundations to deep learning and modern attention mechanisms.'}
                </p>
              </div>

              {(Object.entries(groupedPresets) as [AlgorithmCategory, AlgorithmPreset[]][]).map(
                ([cat, presets]) => (
                  <div
                    key={cat}
                    className={`rounded-2xl border p-3.5 space-y-2.5 ${
                      isDark ? 'bg-[#0f1526] border-[#1c273e]' : 'bg-white border-slate-200 shadow-xs'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {getCategoryIcon(cat)}
                        <h3 className="text-xs font-bold text-slate-900 dark:text-slate-100">
                          {t.categories[cat as AICategory] || cat}
                        </h3>
                      </div>
                      <span className="text-[10px] font-mono bg-slate-800 text-slate-300 px-2 py-0.5 rounded-full font-bold">
                        {presets.length} {language === 'vi' ? 'bài' : 'topics'}
                      </span>
                    </div>

                    <div className="space-y-1.5 pt-1">
                      {presets.map((preset) => (
                        <div
                          key={preset.id}
                          onClick={() => onSelectAlgorithm(preset)}
                          className={`p-2 rounded-xl text-xs flex items-center justify-between cursor-pointer transition-all active:scale-[0.98] ${
                            isDark
                              ? 'hover:bg-slate-800/60 text-slate-200'
                              : 'hover:bg-slate-100 text-slate-800'
                          }`}
                        >
                          <span className="truncate pr-2 font-medium">
                            {getPresetTitle(preset, language)}
                          </span>
                          <ArrowRight className="w-3.5 h-3.5 text-cyan-400 shrink-0 opacity-70" />
                        </div>
                      ))}
                    </div>
                  </div>
                )
              )}
            </div>
          )}

          {/* TAB 3: SETTINGS & ABOUT */}
          {activeTab === 'settings' && (
            <div className="p-3.5 space-y-4 animate-tab-enter">
              <div className="space-y-1">
                <h2 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  {language === 'vi' ? 'Cài đặt & Giới thiệu' : 'Settings & About'}
                </h2>
                <p className="text-xs opacity-70">TraceCraft Android Edition — v1.0.0 M3</p>
              </div>

              {/* Preferences Card */}
              <div
                className={`rounded-2xl border p-3.5 space-y-3 ${
                  isDark ? 'bg-[#0f1526] border-[#1c273e]' : 'bg-white border-slate-200'
                }`}
              >
                <div className="flex items-center justify-between text-xs">
                  <div>
                    <span className="font-bold block">{language === 'vi' ? 'Ngôn ngữ' : 'Language'}</span>
                    <span className="text-[10px] opacity-60">Tiếng Việt & English</span>
                  </div>
                  <LanguageSwitch language={language} onSelectLanguage={onSelectLanguage} theme={theme} />
                </div>

                <div className="w-full h-px bg-inherit opacity-40" />

                <div className="flex items-center justify-between text-xs">
                  <div>
                    <span className="font-bold block">{language === 'vi' ? 'Giao diện' : 'Theme Mode'}</span>
                    <span className="text-[10px] opacity-60">
                      {isDark ? 'Clean Studio (Dark)' : 'Organic Sketch (Light)'}
                    </span>
                  </div>
                  <button
                    onClick={onToggleTheme}
                    className="p-2 rounded-xl border border-inherit text-xs font-bold"
                  >
                    {isDark ? '🌙 Studio' : '☀️ Sketch'}
                  </button>
                </div>
              </div>

              {/* About TraceCraft */}
              <div
                className={`rounded-2xl border p-3.5 space-y-2 text-xs leading-relaxed ${
                  isDark ? 'bg-[#0f1526] border-[#1c273e]' : 'bg-white border-slate-200'
                }`}
              >
                <div className="flex items-center gap-1.5 font-bold text-emerald-400">
                  <AndroidIcon className="w-4 h-4" />
                  <span>TraceCraft Android Platform</span>
                </div>
                <p className="opacity-80 text-[11px]">
                  {language === 'vi'
                    ? 'Nền tảng mô phỏng thực thi AI theo nguyên lý toán học và mã nguồn Python. Được tối ưu hóa cho màn hình cảm ứng di động với chuẩn Material Design 3.'
                    : 'Interactive AI & Machine Learning execution tracing platform built with React 19, TypeScript, and Material Design 3.'}
                </p>
                <div className="pt-2 flex items-center justify-between text-[10px] font-mono opacity-60">
                  <span>Author: @khangbuinoAI</span>
                  <span>100% Bilingual</span>
                </div>
              </div>
            </div>
          )}
        </main>

        {/* ========================================================= */}
        {/* ANDROID NAVIGATION DRAWER                                 */}
        {/* ========================================================= */}
        {drawerOpen && (
          <div className="absolute inset-0 z-40 flex">
            <div
              onClick={() => setDrawerOpen(false)}
              className="absolute inset-0 bg-black/60 backdrop-blur-xs transition-opacity animate-fade-in"
            />

            <div
              className={`relative w-[300px] h-full flex flex-col shadow-2xl z-50 transition-colors animate-slide-right ${
                isDark ? 'bg-[#0d1322] border-r border-[#1c273e] text-slate-100' : 'bg-white border-r border-slate-200 text-slate-800'
              }`}
            >
              <div className="p-4 border-b border-inherit flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center text-slate-950 shadow">
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

              <div className="flex-1 overflow-y-auto p-3 space-y-1.5">
                {ALGORITHM_PRESETS.map((preset) => (
                  <button
                    key={preset.id}
                    onClick={() => {
                      onSelectAlgorithm(preset);
                      setDrawerOpen(false);
                    }}
                    className={`w-full text-left p-2.5 rounded-xl flex items-center gap-2.5 transition-all text-xs font-semibold ${
                      isDark ? 'hover:bg-slate-800/50 text-slate-300' : 'hover:bg-slate-100 text-slate-700'
                    }`}
                  >
                    <div className="shrink-0">{getCategoryIcon(preset.category)}</div>
                    <div className="min-w-0 flex-1">
                      <span className="block truncate">{getPresetTitle(preset, language)}</span>
                      <span className="text-[10px] opacity-60 font-mono block truncate">
                        {getPresetBadge(preset, language)}
                      </span>
                    </div>
                  </button>
                ))}
              </div>

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
          {/* Tab 1: Home */}
          <button
            onClick={() => setActiveTab('home')}
            className="flex-1 flex flex-col items-center justify-center gap-1 transition-all active:scale-95 py-1"
          >
            <div
              className={`px-4 py-0.5 rounded-full flex items-center justify-center transition-all ${
                activeTab === 'home'
                  ? isDark
                    ? 'bg-cyan-500/20 text-cyan-300 font-bold'
                    : 'bg-blue-100 text-blue-800 font-bold'
                  : 'opacity-70'
              }`}
            >
              <Home className="w-4 h-4" />
            </div>
            <span
              className={`text-[10px] tracking-tight ${
                activeTab === 'home' ? 'font-bold text-cyan-400 dark:text-cyan-300' : 'opacity-70'
              }`}
            >
              {language === 'vi' ? 'Trang chủ' : 'Home'}
            </span>
          </button>

          {/* Tab 2: Categories */}
          <button
            onClick={() => setActiveTab('categories')}
            className="flex-1 flex flex-col items-center justify-center gap-1 transition-all active:scale-95 py-1"
          >
            <div
              className={`px-4 py-0.5 rounded-full flex items-center justify-center transition-all ${
                activeTab === 'categories'
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
                activeTab === 'categories' ? 'font-bold text-cyan-400 dark:text-cyan-300' : 'opacity-70'
              }`}
            >
              {language === 'vi' ? 'Danh mục' : 'Categories'}
            </span>
          </button>

          {/* Tab 3: Launch Visualizer */}
          <button
            onClick={() => {
              if (onLaunchVisualizer) onLaunchVisualizer();
              else onSelectAlgorithm(spotlightPreset);
            }}
            className="flex-1 flex flex-col items-center justify-center gap-1 transition-all active:scale-95 py-1"
          >
            <div className="px-4 py-0.5 rounded-full flex items-center justify-center opacity-70 hover:opacity-100">
              <Activity className="w-4 h-4 text-emerald-400" />
            </div>
            <span className="text-[10px] tracking-tight opacity-70">
              {t.visualize}
            </span>
          </button>

          {/* Tab 4: Settings */}
          <button
            onClick={() => setActiveTab('settings')}
            className="flex-1 flex flex-col items-center justify-center gap-1 transition-all active:scale-95 py-1"
          >
            <div
              className={`px-4 py-0.5 rounded-full flex items-center justify-center transition-all ${
                activeTab === 'settings'
                  ? isDark
                    ? 'bg-cyan-500/20 text-cyan-300 font-bold'
                    : 'bg-blue-100 text-blue-800 font-bold'
                  : 'opacity-70'
              }`}
            >
              <Settings className="w-4 h-4" />
            </div>
            <span
              className={`text-[10px] tracking-tight ${
                activeTab === 'settings' ? 'font-bold text-cyan-400 dark:text-cyan-300' : 'opacity-70'
              }`}
            >
              {language === 'vi' ? 'Cài đặt' : 'Settings'}
            </span>
          </button>
        </nav>
      </div>
    </div>
  );
};
export default AndroidCatalogLayout;
