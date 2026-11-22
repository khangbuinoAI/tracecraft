import React from 'react';
import { 
  Sparkles, 
  Paintbrush, 
  Sliders, 
  Smartphone, 
  Monitor, 
  RotateCcw,
  ArrowLeft,
  BookOpen,
  PanelLeft,
  PanelRight
} from 'lucide-react';
import type { ThemeMode, ViewMode, AlgorithmPreset, StudioTab } from '../types';
import cbtLogo from '../assets/cbt-logo.png';

interface HeaderProps {
  theme: ThemeMode;
  onToggleTheme: () => void;
  viewMode: ViewMode;
  onToggleViewMode: () => void;
  selectedPreset: AlgorithmPreset;
  inputStr: string;
  onChangeInput: (newInput: string) => void;
  onReset: () => void;
  onBackToCatalog: () => void;
  onToggleSidebar: () => void;
  showCodePane: boolean;
  onToggleCodePane: () => void;
  showDiagnosticsPane: boolean;
  onToggleDiagnosticsPane: () => void;
  studioTab: StudioTab;
  onSelectStudioTab: (tab: StudioTab) => void;
}

export const Header: React.FC<HeaderProps> = ({
  theme: _theme,
  onToggleTheme,
  viewMode,
  onToggleViewMode,
  selectedPreset,
  inputStr,
  onChangeInput,
  onReset,
  onBackToCatalog,
  onToggleSidebar,
  showCodePane,
  onToggleCodePane,
  showDiagnosticsPane,
  onToggleDiagnosticsPane,
  studioTab,
  onSelectStudioTab,
}) => {
  const isSketch = false;

  return (
    <header
      className={`w-full border-b transition-colors duration-200 select-none ${
        isSketch
          ? 'bg-[#fcfaf5] border-[#3c3836]/30 text-[#282828]'
          : 'bg-[#0b0f19] border-[#222f46] text-white'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 py-2.5 flex flex-wrap items-center justify-between gap-3">
        {/* Brand & Navigation */}
        <div className="flex items-center gap-2 sm:gap-3">
          <img
            src={cbtLogo}
            alt="Chuyen Ben Tre AIO"
            className="w-9 h-9 sm:w-10 sm:h-10 object-contain shrink-0 drop-shadow-[0_1px_2px_rgba(15,3,69,0.18)]"
          />

          <div className="hidden sm:flex flex-col leading-none">
            <span
              className={`font-extrabold tracking-tight text-base leading-none ${
                isSketch ? 'text-[#282828]' : 'text-white'
              }`}
            >
              Chuyen Ben Tre
            </span>
            <span
              className={`text-[10px] font-mono tracking-[0.26em] uppercase mt-0.5 ${
                isSketch ? 'text-[#504945]' : 'text-slate-300/80'
              }`}
            >
              AIO
            </span>
          </div>

          {/* Back to Catalog */}
          <button
            onClick={onBackToCatalog}
            title="Return to Catalog Homepage"
            className={`flex items-center gap-1 px-2.5 py-1.5 rounded-md text-xs font-semibold transition-all ${
              isSketch
                ? 'bg-[#ebdbb2] border border-[#3c3836]/40 hover:bg-[#d5c4a1] text-[#282828] font-sketch text-sm'
                : 'bg-[#151c2d] border border-[#2a3752] text-slate-200 hover:text-cyan-400 hover:border-cyan-500/50'
            }`}
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span className="hidden md:inline">Catalog</span>
          </button>

          {/* Dynamic Curriculum Sidebar Trigger */}
          <button
            onClick={onToggleSidebar}
            title="Open Dynamic Curriculum Sidebar (Ctrl+B)"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-bold transition-all shadow-sm ${
              isSketch
                ? 'bg-[#d65d0e] text-white border-2 border-[#3c3836] font-sketch text-sm'
                : 'bg-gradient-to-r from-indigo-600 to-cyan-500 text-white shadow-cyan-500/20 hover:opacity-95'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span>Topics</span>
          </button>
        </div>

        {/* Current Active Topic Display & Input Tweaker */}
        <div className="flex items-center gap-2 flex-wrap">
          {/* Active Topic Chip */}
          <button
            onClick={onToggleSidebar}
            title="Click to switch topic"
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-mono font-medium border transition-colors ${
              isSketch
                ? 'bg-[#fbf7ee] border-2 border-[#3c3836] text-[#282828]'
                : 'bg-[#131b2d] border-[#222f46] text-slate-200 hover:border-cyan-500/50'
            }`}
          >
            <span className="opacity-60 hidden lg:inline">[{selectedPreset.category}]</span>
            <span className="font-bold">{selectedPreset.title}</span>
            <span
              className={`text-[10px] px-1.5 py-0.2 rounded font-semibold ${
                isSketch ? 'bg-[#ebdbb2] text-[#3c3836]' : 'bg-cyan-950 text-cyan-300'
              }`}
            >
              {selectedPreset.badge}
            </span>
          </button>

          {/* Trace vs Theory Tab Switcher */}
          <div
            className={`flex items-center p-0.5 rounded-lg border text-xs font-mono ${
              isSketch
                ? 'bg-[#ebdbb2] border-[#3c3836]/30'
                : 'bg-[#121929] border-[#222f46]'
            }`}
          >
            <button
              onClick={() => onSelectStudioTab('trace')}
              className={`px-2.5 py-1 rounded-md font-bold transition-all flex items-center gap-1.5 ${
                studioTab === 'trace'
                  ? isSketch
                    ? 'bg-[#d65d0e] text-white'
                    : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'opacity-60 hover:opacity-100'
              }`}
            >
              <Sparkles className="w-3 h-3" />
              <span>Trace</span>
            </button>
            <button
              onClick={() => onSelectStudioTab('theory')}
              className={`px-2.5 py-1 rounded-md font-bold transition-all flex items-center gap-1.5 ${
                studioTab === 'theory'
                  ? isSketch
                    ? 'bg-[#d65d0e] text-white'
                    : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'opacity-60 hover:opacity-100'
              }`}
            >
              <BookOpen className="w-3 h-3" />
              <span>Theory</span>
            </button>
          </div>
          {/* Quick Input Tweaker */}
          <div className="hidden xl:flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 opacity-60" />
            <input
              type="text"
              value={inputStr}
              onChange={(e) => onChangeInput(e.target.value)}
              title="Edit comma-separated values to trace custom inputs"
              placeholder="e.g. 5, 2, 8, 1, 4"
              className={`text-xs px-2.5 py-1.5 rounded w-44 font-mono transition-colors outline-none ${
                isSketch
                  ? 'bg-[#fbf7ee] border border-[#3c3836]/40 text-[#282828]'
                  : 'bg-[#151c2d] border border-[#2a3752] text-slate-200 focus:border-cyan-500/70'
              }`}
            />
          </div>
        </div>

        {/* Dynamic Panel Toggles & View Controls */}
        <div className="flex items-center gap-1.5 sm:gap-2">
          {/* Panel Visibility Toggles (Desktop only) */}
          {viewMode === 'desktop-3pane' && (
            <div
              className={`hidden md:flex items-center p-0.5 rounded-lg border ${
                isSketch ? 'bg-[#ebdbb2] border-[#3c3836]/30' : 'bg-[#121929] border-[#222f46]'
              }`}
            >
              <button
                onClick={onToggleCodePane}
                title={showCodePane ? 'Hide Code Panel' : 'Show Code Panel'}
                className={`p-1.5 rounded text-xs transition-colors ${
                  showCodePane
                    ? isSketch
                      ? 'bg-[#fbf7ee] text-[#d65d0e] font-bold shadow-sm'
                      : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                    : 'opacity-50 hover:opacity-100'
                }`}
              >
                <PanelLeft className="w-4 h-4" />
              </button>
              <button
                onClick={onToggleDiagnosticsPane}
                title={showDiagnosticsPane ? 'Hide Diagnostics Panel' : 'Show Diagnostics Panel'}
                className={`p-1.5 rounded text-xs transition-colors ${
                  showDiagnosticsPane
                    ? isSketch
                      ? 'bg-[#fbf7ee] text-[#d65d0e] font-bold shadow-sm'
                      : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                    : 'opacity-50 hover:opacity-100'
                }`}
              >
                <PanelRight className="w-4 h-4" />
              </button>
            </div>
          )}

          {/* Mobile Story vs 3-Pane Desktop switch */}
          <button
            onClick={onToggleViewMode}
            title={
              viewMode === 'desktop-3pane'
                ? 'Switch to Mobile Story Card Mode'
                : 'Switch to 3-Pane Studio Layout'
            }
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-all ${
              viewMode === 'mobile-story'
                ? isSketch
                  ? 'bg-[#d65d0e] text-white border-2 border-[#3c3836]'
                  : 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : isSketch
                  ? 'bg-[#f6f2e9] border border-[#3c3836]/30 hover:bg-[#ebdbb2]'
                  : 'bg-[#151c2d] border border-[#2a3752] text-slate-300 hover:bg-[#1f293d]'
            }`}
          >
            {viewMode === 'mobile-story' ? (
              <>
                <Smartphone className="w-3.5 h-3.5 text-amber-300" />
                <span className="hidden sm:inline">Story Mode</span>
              </>
            ) : (
              <>
                <Monitor className="w-3.5 h-3.5 opacity-70" />
                <span className="hidden sm:inline">3-Pane</span>
              </>
            )}
          </button>

          {/* Theme Mode Toggle */}
          <button
            onClick={onToggleTheme}
            title={`Switch to ${isSketch ? 'Clean Pro Studio' : 'Hand-Drawn Sketch'} theme`}
            className={`flex items-center gap-1 px-2.5 py-1.5 rounded-md text-xs font-semibold transition-all ${
              isSketch
                ? 'bg-[#ebdbb2] border-2 border-[#3c3836] text-[#282828] hover:bg-[#d5c4a1] font-sketch text-sm'
                : 'bg-[#151c2d] border border-[#2a3752] text-slate-200 hover:border-cyan-500/50 hover:bg-[#1a233a]'
            }`}
          >
            {isSketch ? (
              <>
                <Paintbrush className="w-3.5 h-3.5 text-[#d65d0e]" />
                <span className="hidden sm:inline">Sketch</span>
              </>
            ) : (
              <>
                <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                <span className="hidden sm:inline">Clean</span>
              </>
            )}
          </button>

          {/* Reset Button */}
          <button
            onClick={onReset}
            title="Reset trace to step 1"
            className={`p-1.5 rounded-md transition-colors ${
              isSketch
                ? 'hover:bg-[#ebdbb2] text-[#3c3836]'
                : 'hover:bg-[#1f293d] text-slate-400 hover:text-white'
            }`}
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>
    </header>
  );
};
