import React, { useState, useMemo, useEffect } from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ShieldCheck,
  Sliders,
  Sparkles,
  Info,
  HeartPulse,
  Mail,
  CreditCard,
  Scale,
  Target,
  Activity,
  TrendingUp,
  HelpCircle,
} from 'lucide-react';
import type {
  ThemeMode,
  Language,
  ConfusionMatrixData,
  ConfusionMatrixSample,
  QuadrantType,
} from '../types';

interface ConfusionMatrixStageProps {
  theme: ThemeMode;
  language: Language;
  confusionMatrixState?: ConfusionMatrixData;
  onChangeInput?: (newInput: string) => void;
  isExpanded?: boolean;
}

export const ConfusionMatrixStage: React.FC<ConfusionMatrixStageProps> = ({
  theme,
  language,
  confusionMatrixState,
  onChangeInput,
  isExpanded = false,
}) => {
  const isDark = theme === 'studio';
  const isVi = language === 'vi';

  // Base data from trace step
  const defaultState: ConfusionMatrixData = useMemo(() => {
    return (
      confusionMatrixState || {
        tp: 12,
        fp: 3,
        fn: 2,
        tn: 83,
        total: 100,
        threshold: 0.5,
        accuracy: 0.95,
        precision: 0.8,
        recall: 0.857,
        specificity: 0.965,
        f1Score: 0.828,
        scenario: 'medical',
        samples: [],
      }
    );
  }, [confusionMatrixState]);

  // Interactive threshold override (controlled by slider or synchronized from step)
  const [localThreshold, setLocalThreshold] = useState<number>(defaultState.threshold);
  const [hoveredSample, setHoveredSample] = useState<ConfusionMatrixSample | null>(null);
  const [selectedQuadrant, setSelectedQuadrant] = useState<QuadrantType | null>(null);

  // Sync when trace step changes threshold
  useEffect(() => {
    if (confusionMatrixState?.threshold !== undefined) {
      setLocalThreshold(confusionMatrixState.threshold);
    }
    if (confusionMatrixState?.activeQuadrant !== undefined) {
      setSelectedQuadrant(confusionMatrixState.activeQuadrant);
    }
  }, [confusionMatrixState?.threshold, confusionMatrixState?.activeQuadrant]);

  const rawSamples: ConfusionMatrixSample[] = useMemo(() => {
    if (confusionMatrixState?.samples && confusionMatrixState.samples.length > 0) {
      return confusionMatrixState.samples;
    }
    // Realistic fallback samples if none provided
    const base: ConfusionMatrixSample[] = [
      { id: 1, prob: 0.94, actual: 1, label: isVi ? 'Bệnh nhân #1' : 'Patient #1' },
      { id: 2, prob: 0.88, actual: 1, label: isVi ? 'Bệnh nhân #2' : 'Patient #2' },
      { id: 3, prob: 0.82, actual: 1, label: isVi ? 'Bệnh nhân #3' : 'Patient #3' },
      { id: 4, prob: 0.77, actual: 1, label: isVi ? 'Bệnh nhân #4' : 'Patient #4' },
      { id: 5, prob: 0.71, actual: 1, label: isVi ? 'Bệnh nhân #5' : 'Patient #5' },
      { id: 6, prob: 0.65, actual: 1, label: isVi ? 'Bệnh nhân #6' : 'Patient #6' },
      { id: 7, prob: 0.58, actual: 1, label: isVi ? 'Bệnh nhân #7' : 'Patient #7' },
      { id: 8, prob: 0.52, actual: 1, label: isVi ? 'Bệnh nhân #8' : 'Patient #8' },
      { id: 9, prob: 0.44, actual: 1, label: isVi ? 'Bệnh nhân #9' : 'Patient #9' },
      { id: 10, prob: 0.36, actual: 1, label: isVi ? 'Bệnh nhân #10' : 'Patient #10' },
      { id: 11, prob: 0.69, actual: 0, label: isVi ? 'Người khỏe #11' : 'Healthy #11' },
      { id: 12, prob: 0.62, actual: 0, label: isVi ? 'Người khỏe #12' : 'Healthy #12' },
      { id: 13, prob: 0.54, actual: 0, label: isVi ? 'Người khỏe #13' : 'Healthy #13' },
      { id: 14, prob: 0.45, actual: 0, label: isVi ? 'Người khỏe #14' : 'Healthy #14' },
      { id: 15, prob: 0.38, actual: 0, label: isVi ? 'Người khỏe #15' : 'Healthy #15' },
      { id: 16, prob: 0.31, actual: 0, label: isVi ? 'Người khỏe #16' : 'Healthy #16' },
      { id: 17, prob: 0.24, actual: 0, label: isVi ? 'Người khỏe #17' : 'Healthy #17' },
      { id: 18, prob: 0.18, actual: 0, label: isVi ? 'Người khỏe #18' : 'Healthy #18' },
      { id: 19, prob: 0.12, actual: 0, label: isVi ? 'Người khỏe #19' : 'Healthy #19' },
      { id: 20, prob: 0.05, actual: 0, label: isVi ? 'Người khỏe #20' : 'Healthy #20' },
    ];
    return base;
  }, [confusionMatrixState?.samples, isVi]);

  // Recalculate dynamic confusion matrix & metrics whenever threshold changes
  const computed = useMemo(() => {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    let tn = 0;

    const evaluatedSamples: ConfusionMatrixSample[] = rawSamples.map((s) => {
      const pred: 0 | 1 = s.prob >= localThreshold ? 1 : 0;
      let status: QuadrantType;
      if (s.actual === 1 && pred === 1) {
        status = 'tp';
        tp++;
      } else if (s.actual === 0 && pred === 1) {
        status = 'fp';
        fp++;
      } else if (s.actual === 1 && pred === 0) {
        status = 'fn';
        fn++;
      } else {
        status = 'tn';
        tn++;
      }
      return { ...s, pred, status };
    });

    const total = evaluatedSamples.length;
    const accuracy = total > 0 ? (tp + tn) / total : 0;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const specificity = tn + fp > 0 ? tn / (tn + fp) : 0;
    const f1Score =
      precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    return {
      tp,
      fp,
      fn,
      tn,
      total,
      accuracy,
      precision,
      recall,
      specificity,
      f1Score,
      samples: evaluatedSamples,
      actualPositives: tp + fn,
      actualNegatives: tn + fp,
      predictedPositives: tp + fp,
      predictedNegatives: tn + fn,
    };
  }, [rawSamples, localThreshold]);

  const scenario = confusionMatrixState?.scenario || 'medical';

  const getScenarioMeta = () => {
    switch (scenario) {
      case 'medical':
        return {
          icon: <HeartPulse className="w-4 h-4 text-rose-500 animate-pulse" />,
          title: isVi ? 'Sàng lọc Ung thư & Chẩn đoán Bệnh' : 'Cancer Screening & Medical Diagnostic',
          badge: isVi ? 'Ưu tiên Recall' : 'Recall Priority',
          badgeColor: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
          costExplanation: isVi
            ? 'Âm tính giả (FN - Bỏ sót bệnh) là thảm họa! Bệnh nhân không được điều trị kịp thời có thể nguy hiểm tính mạng. Vì vậy, ta cần hạ ngưỡng τ (ví dụ τ = 0.30) để đạt Recall tối đa.'
            : 'False Negatives (FN - Missed Disease) can be fatal! Patients missing early intervention face severe risk. We must lower threshold τ (e.g. τ = 0.30) to maximize Recall.',
          posLabel: isVi ? 'Dương tính (Nhiễm bệnh)' : 'Positive (Has Disease)',
          negLabel: isVi ? 'Âm tính (Khỏe mạnh)' : 'Negative (Healthy)',
          tpTag: isVi ? 'Phát hiện đúng bệnh' : 'Correct Diagnosis',
          fpTag: isVi ? 'Báo động nhầm (Xét nghiệm lại)' : 'False Alarm (Biopsy)',
          fnTag: isVi ? 'Bỏ sót bệnh (Cực kỳ nguy hiểm)' : 'Missed Disease (Fatal)',
          tnTag: isVi ? 'Loại trừ đúng (An tâm)' : 'Correctly Cleared',
        };
      case 'spam':
        return {
          icon: <Mail className="w-4 h-4 text-amber-500" />,
          title: isVi ? 'Bộ Lọc Thư Rác (Email Spam Filter)' : 'Email Spam Filter Classifier',
          badge: isVi ? 'Ưu tiên Precision' : 'Precision Priority',
          badgeColor: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
          costExplanation: isVi
            ? 'Dương tính giả (FP - Đánh dấu nhầm thư quan trọng vào Spam) là tối kỵ! Mất hợp đồng công việc hoặc thư tuyển dụng gây thiệt hại lớn. Ta cần tăng ngưỡng τ (ví dụ τ = 0.70) để đạt Precision tối đa.'
            : 'False Positives (FP - Misclassifying urgent contracts as Spam) is disastrous! Important emails get lost in the junk folder. We must raise threshold τ (e.g. τ = 0.70) to ensure high Precision.',
          posLabel: isVi ? 'Spam (Thư rác)' : 'Spam (Junk)',
          negLabel: isVi ? 'Ham (Thư quan trọng)' : 'Ham (Inbox)',
          tpTag: isVi ? 'Chặn đúng thư rác' : 'Spam Blocked',
          fpTag: isVi ? 'Mất thư quan trọng (Thảm họa)' : 'Urgent Mail Lost (Costly)',
          fnTag: isVi ? 'Lọt thư rác vào hộp thư đến' : 'Spam Leaked to Inbox',
          tnTag: isVi ? 'Gửi đúng vào hộp thư đến' : 'Clean Mail in Inbox',
        };
      case 'fraud':
        return {
          icon: <CreditCard className="w-4 h-4 text-emerald-500" />,
          title: isVi ? 'Phát hiện Gian lận Thẻ Tín dụng (Imbalanced)' : 'Credit Card Fraud Detection',
          badge: isVi ? 'Nghịch lý Accuracy' : 'Accuracy Paradox',
          badgeColor: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40',
          costExplanation: isVi
            ? 'Dữ liệu mất cân bằng nghiêm trọng (95% hợp lệ, 5% gian lận). Một mô hình ngây thơ đoán 100% hợp lệ vẫn đạt 95% Accuracy nhưng bắt được 0 vụ gian lận (Recall = 0%)! Ma trận Nhầm lẫn bóc trần sự thật này.'
            : 'Heavy class imbalance (95% normal, 5% fraud). A naive dummy model predicting "no fraud" achieves 95% Accuracy yet catches ZERO fraud (Recall = 0%)! The Confusion Matrix exposes this illusion.',
          posLabel: isVi ? 'Gian lận (Fraud)' : 'Fraudulent',
          negLabel: isVi ? 'Hợp lệ (Legitimate)' : 'Legitimate',
          tpTag: isVi ? 'Bắt đúng giao dịch gian lận' : 'Fraud Intercepted',
          fpTag: isVi ? 'Khóa nhầm thẻ của khách' : 'Card Blocked Inconvenience',
          fnTag: isVi ? 'Để lọt tiền bị đánh cắp' : 'Stolen Funds (Direct Loss)',
          tnTag: isVi ? 'Giao dịch an toàn thông suốt' : 'Smooth Valid Transaction',
        };
      default:
        return {
          icon: <Scale className="w-4 h-4 text-cyan-400" />,
          title: isVi ? 'Phân loại Nhị phân Chuẩn' : 'Standard Binary Classification',
          badge: isVi ? 'Cân bằng F1' : 'Balanced F1',
          badgeColor: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40',
          costExplanation: isVi
            ? 'Điều chỉnh ngưỡng quyết định τ để cân bằng giữa Độ chính xác (Precision) và Độ nhạy (Recall), tối đa hóa điểm F1 hài hòa.'
            : 'Adjust the decision threshold τ to balance Precision and Recall, optimizing the harmonic F1-Score.',
          posLabel: isVi ? 'Lớp 1 (Positive)' : 'Class 1 (Positive)',
          negLabel: isVi ? 'Lớp 0 (Negative)' : 'Class 0 (Negative)',
          tpTag: isVi ? 'Dương tính Thật (TP)' : 'True Positive (TP)',
          fpTag: isVi ? 'Dương tính Giả (FP - Sai loại I)' : 'False Positive (FP - Type I)',
          fnTag: isVi ? 'Âm tính Giả (FN - Sai loại II)' : 'False Negative (FN - Type II)',
          tnTag: isVi ? 'Âm tính Thật (TN)' : 'True Negative (TN)',
        };
    }
  };

  const meta = getScenarioMeta();

  // SVG dimensions for Sample Probability Distribution strip
  const STRIP_W = 720;
  const STRIP_H = isExpanded ? 140 : 110;
  const STRIP_PADDING = { left: 45, right: 45, top: 25, bottom: 35 };
  const innerW = STRIP_W - STRIP_PADDING.left - STRIP_PADDING.right;

  const toStripX = (prob: number) => {
    return STRIP_PADDING.left + Math.max(0, Math.min(1, prob)) * innerW;
  };

  const thresholdSvgX = toStripX(localThreshold);

  return (
    <div className="w-full h-full flex flex-col relative select-none overflow-y-auto overflow-x-hidden font-sans">
      {/* ========================================================= */}
      {/* 1. SCENARIO CONTEXT & INTERACTIVE THRESHOLD CONTROLS      */}
      {/* ========================================================= */}
      <div
        className={`px-4 py-3 border-b flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 transition-colors ${
          isDark ? 'border-slate-800 bg-[#0d131f]/95 text-slate-200' : 'border-slate-200 bg-white/95 text-slate-800'
        }`}
      >
        {/* Scenario Header with Badge */}
        <div className="flex items-center gap-3">
          <div
            className={`p-2 rounded-xl flex items-center justify-center border ${
              isDark ? 'bg-slate-800/80 border-slate-700' : 'bg-slate-100 border-slate-200'
            }`}
          >
            {meta.icon}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold tracking-tight">{meta.title}</span>
              <span
                className={`text-[10px] font-mono px-2 py-0.5 rounded-full border font-semibold ${meta.badgeColor}`}
              >
                {meta.badge}
              </span>
            </div>
            <div className="text-[11px] text-slate-500 dark:text-slate-400 flex items-center gap-2 mt-0.5">
              <span>{isVi ? 'Tổng mẫu:' : 'Total Samples:'} <strong className="font-mono text-slate-700 dark:text-slate-200">{computed.total}</strong></span>
              <span>•</span>
              <span className="text-emerald-500 font-semibold">{isVi ? 'Thực tế (+):' : 'Actual (+):'} {computed.actualPositives}</span>
              <span>•</span>
              <span className="text-sky-500 font-semibold">{isVi ? 'Thực tế (-):' : 'Actual (-):'} {computed.actualNegatives}</span>
            </div>
          </div>
        </div>

        {/* Interactive Threshold Slider (τ) */}
        <div
          className={`flex items-center gap-3 px-3 py-1.5 rounded-xl border ${
            isDark ? 'bg-[#090d16] border-slate-800' : 'bg-slate-50 border-slate-200'
          }`}
        >
          <div className="flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-xs font-mono font-bold text-slate-600 dark:text-slate-300">
              {isVi ? 'Ngưỡng' : 'Threshold'} <strong className="text-cyan-400">τ = {localThreshold.toFixed(2)}</strong>
            </span>
          </div>

          <input
            type="range"
            min="0.05"
            max="0.95"
            step="0.05"
            value={localThreshold}
            onChange={(e) => {
              const val = Number(e.target.value);
              setLocalThreshold(val);
              if (onChangeInput) {
                // If user touches threshold, notify parent
                onChangeInput(`threshold=${val}`);
              }
            }}
            className="w-24 sm:w-32 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
          />

          <div className="flex items-center gap-1">
            <button
              onClick={() => setLocalThreshold(0.3)}
              title="Medical preset (Low threshold)"
              className={`text-[10px] font-mono px-1.5 py-0.5 rounded border transition-colors ${
                Math.abs(localThreshold - 0.3) < 0.02
                  ? 'bg-rose-500/20 text-rose-300 border-rose-500/50 font-bold'
                  : 'text-slate-400 hover:text-slate-200 border-slate-700'
              }`}
            >
              0.30
            </button>
            <button
              onClick={() => setLocalThreshold(0.5)}
              title="Default balanced threshold"
              className={`text-[10px] font-mono px-1.5 py-0.5 rounded border transition-colors ${
                Math.abs(localThreshold - 0.5) < 0.02
                  ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50 font-bold'
                  : 'text-slate-400 hover:text-slate-200 border-slate-700'
              }`}
            >
              0.50
            </button>
            <button
              onClick={() => setLocalThreshold(0.7)}
              title="Spam filter preset (High threshold)"
              className={`text-[10px] font-mono px-1.5 py-0.5 rounded border transition-colors ${
                Math.abs(localThreshold - 0.7) < 0.02
                  ? 'bg-amber-500/20 text-amber-300 border-amber-500/50 font-bold'
                  : 'text-slate-400 hover:text-slate-200 border-slate-700'
              }`}
            >
              0.70
            </button>
          </div>
        </div>
      </div>

      {/* ========================================================= */}
      {/* 2. REALISTIC CONTEXT BANNER                               */}
      {/* ========================================================= */}
      <div
        className={`px-4 py-2 border-b flex items-center gap-2 text-xs transition-colors ${
          isDark ? 'border-slate-800/80 bg-[#0f172a]/60 text-slate-300' : 'border-slate-200/80 bg-amber-50/60 text-slate-700'
        }`}
      >
        <Info className="w-4 h-4 text-cyan-400 shrink-0" />
        <span className="leading-relaxed">
          <strong>{isVi ? 'Ý nghĩa thực tế:' : 'Real-World Impact:'}</strong> {meta.costExplanation}
        </span>
      </div>

      {/* ========================================================= */}
      {/* 3. MAIN STAGE BODY: 2x2 CONFUSION MATRIX + SAMPLE STRIP   */}
      {/* ========================================================= */}
      <div className="flex-1 p-3 sm:p-5 flex flex-col gap-4">
        {/* Top Grid Section: The 2x2 Confusion Matrix Heatmap */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          {/* LEFT 7 COLS: THE 2x2 MATRIX */}
          <div
            className={`lg:col-span-7 rounded-2xl border p-4 sm:p-5 flex flex-col justify-between shadow-sm transition-all ${
              isDark ? 'bg-[#111827] border-[#1e293b]' : 'bg-white border-slate-200'
            }`}
          >
            <div className="flex items-center justify-between mb-3 border-b pb-2 border-inherit">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 text-indigo-400" />
                <span className="font-bold text-sm tracking-tight">
                  {isVi ? 'Ma trận Nhầm lẫn (2×2 Matrix)' : 'Confusion Matrix (2×2 Table)'}
                </span>
              </div>
              <span className="text-[11px] text-slate-400 font-mono">
                {isVi ? 'Nhấp ô để lọc mẫu' : 'Click a quadrant to filter samples'}
              </span>
            </div>

            {/* The 2x2 Matrix Grid */}
            <div className="w-full flex flex-col gap-2">
              {/* Column Dimension Label: PREDICTED CLASS */}
              <div className="grid grid-cols-12 text-center text-xs font-mono font-bold text-slate-400">
                <div className="col-span-3"></div>
                <div className="col-span-9 flex flex-col">
                  <span className="text-[11px] uppercase tracking-wider text-indigo-400 font-semibold mb-1">
                    {isVi ? 'Dự đoán của Mô hình (Predicted)' : 'Model Prediction (Threshold τ)'}
                  </span>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <span className="text-emerald-500 dark:text-emerald-400 font-bold">
                      {isVi ? 'Dự đoán Dương (+) [ŷ=1]' : 'Pred Positive (+) [ŷ=1]'}
                    </span>
                    <span className="text-sky-500 dark:text-sky-400 font-bold">
                      {isVi ? 'Dự đoán Âm (-) [ŷ=0]' : 'Pred Negative (-) [ŷ=0]'}
                    </span>
                  </div>
                </div>
              </div>

              {/* ROW 1: ACTUAL POSITIVE (y = 1) */}
              <div className="grid grid-cols-12 gap-2 items-stretch">
                {/* Row Header Label */}
                <div
                  className={`col-span-3 rounded-xl p-2.5 flex flex-col justify-center items-center text-center border font-mono text-xs ${
                    isDark ? 'bg-[#0f172a] border-emerald-950/60 text-emerald-400' : 'bg-emerald-50/70 border-emerald-200 text-emerald-700'
                  }`}
                >
                  <span className="text-[9px] uppercase font-bold tracking-wider opacity-75">
                    {isVi ? 'Thực tế' : 'Actual'}
                  </span>
                  <strong className="text-sm">y = 1</strong>
                  <span className="text-[10px] opacity-80 mt-0.5">{meta.posLabel}</span>
                  <span className="text-[9px] mt-1 font-bold text-emerald-500">
                    Σ = {computed.actualPositives}
                  </span>
                </div>

                {/* Quadrant 1: TRUE POSITIVE (TP) */}
                <button
                  onClick={() =>
                    setSelectedQuadrant(selectedQuadrant === 'tp' ? null : 'tp')
                  }
                  className={`col-span-4 sm:col-span-4 rounded-xl p-3 border text-left flex flex-col justify-between transition-all cursor-pointer relative overflow-hidden group ${
                    selectedQuadrant === 'tp'
                      ? 'ring-2 ring-emerald-500 scale-[1.02] shadow-lg'
                      : ''
                  } ${
                    isDark
                      ? 'bg-gradient-to-br from-emerald-950/40 to-[#0c1f18] border-emerald-800/60 hover:border-emerald-500'
                      : 'bg-emerald-50/60 border-emerald-300 hover:border-emerald-500'
                  }`}
                >
                  <div className="flex items-center justify-between w-full">
                    <span className="text-xs font-mono font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      TP
                    </span>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-bold">
                      {computed.total > 0
                        ? `${((computed.tp / computed.total) * 100).toFixed(0)}%`
                        : '0%'}
                    </span>
                  </div>

                  <div className="my-1.5">
                    <div className="text-2xl sm:text-3xl font-extrabold font-mono text-emerald-600 dark:text-emerald-300">
                      {computed.tp}
                    </div>
                    <div className="text-[11px] font-bold text-emerald-700 dark:text-emerald-200 mt-0.5">
                      {isVi ? 'Dương tính Thật' : 'True Positive'}
                    </div>
                  </div>

                  <div className="text-[10px] text-emerald-800/70 dark:text-emerald-400/80 font-medium line-clamp-1">
                    {meta.tpTag}
                  </div>
                </button>

                {/* Quadrant 2: FALSE NEGATIVE (FN) */}
                <button
                  onClick={() =>
                    setSelectedQuadrant(selectedQuadrant === 'fn' ? null : 'fn')
                  }
                  className={`col-span-5 sm:col-span-5 rounded-xl p-3 border text-left flex flex-col justify-between transition-all cursor-pointer relative overflow-hidden group ${
                    selectedQuadrant === 'fn'
                      ? 'ring-2 ring-rose-500 scale-[1.02] shadow-lg'
                      : ''
                  } ${
                    isDark
                      ? 'bg-gradient-to-br from-rose-950/40 to-[#1f0c13] border-rose-800/60 hover:border-rose-500'
                      : 'bg-rose-50/60 border-rose-300 hover:border-rose-500'
                  }`}
                >
                  <div className="flex items-center justify-between w-full">
                    <span className="text-xs font-mono font-bold text-rose-600 dark:text-rose-400 flex items-center gap-1.5">
                      <XCircle className="w-3.5 h-3.5" />
                      FN (Type II)
                    </span>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 font-bold">
                      {computed.total > 0
                        ? `${((computed.fn / computed.total) * 100).toFixed(0)}%`
                        : '0%'}
                    </span>
                  </div>

                  <div className="my-1.5">
                    <div className="text-2xl sm:text-3xl font-extrabold font-mono text-rose-600 dark:text-rose-300">
                      {computed.fn}
                    </div>
                    <div className="text-[11px] font-bold text-rose-700 dark:text-rose-200 mt-0.5">
                      {isVi ? 'Âm tính Giả' : 'False Negative'}
                    </div>
                  </div>

                  <div className="text-[10px] text-rose-800/70 dark:text-rose-400/80 font-medium line-clamp-1">
                    {meta.fnTag}
                  </div>
                </button>
              </div>

              {/* ROW 2: ACTUAL NEGATIVE (y = 0) */}
              <div className="grid grid-cols-12 gap-2 items-stretch">
                {/* Row Header Label */}
                <div
                  className={`col-span-3 rounded-xl p-2.5 flex flex-col justify-center items-center text-center border font-mono text-xs ${
                    isDark ? 'bg-[#0f172a] border-sky-950/60 text-sky-400' : 'bg-sky-50/70 border-sky-200 text-sky-700'
                  }`}
                >
                  <span className="text-[9px] uppercase font-bold tracking-wider opacity-75">
                    {isVi ? 'Thực tế' : 'Actual'}
                  </span>
                  <strong className="text-sm">y = 0</strong>
                  <span className="text-[10px] opacity-80 mt-0.5">{meta.negLabel}</span>
                  <span className="text-[9px] mt-1 font-bold text-sky-500">
                    Σ = {computed.actualNegatives}
                  </span>
                </div>

                {/* Quadrant 3: FALSE POSITIVE (FP) */}
                <button
                  onClick={() =>
                    setSelectedQuadrant(selectedQuadrant === 'fp' ? null : 'fp')
                  }
                  className={`col-span-4 sm:col-span-4 rounded-xl p-3 border text-left flex flex-col justify-between transition-all cursor-pointer relative overflow-hidden group ${
                    selectedQuadrant === 'fp'
                      ? 'ring-2 ring-amber-500 scale-[1.02] shadow-lg'
                      : ''
                  } ${
                    isDark
                      ? 'bg-gradient-to-br from-amber-950/40 to-[#1f170c] border-amber-800/60 hover:border-amber-500'
                      : 'bg-amber-50/60 border-amber-300 hover:border-amber-500'
                  }`}
                >
                  <div className="flex items-center justify-between w-full">
                    <span className="text-xs font-mono font-bold text-amber-600 dark:text-amber-400 flex items-center gap-1.5">
                      <AlertTriangle className="w-3.5 h-3.5" />
                      FP (Type I)
                    </span>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 font-bold">
                      {computed.total > 0
                        ? `${((computed.fp / computed.total) * 100).toFixed(0)}%`
                        : '0%'}
                    </span>
                  </div>

                  <div className="my-1.5">
                    <div className="text-2xl sm:text-3xl font-extrabold font-mono text-amber-600 dark:text-amber-300">
                      {computed.fp}
                    </div>
                    <div className="text-[11px] font-bold text-amber-700 dark:text-amber-200 mt-0.5">
                      {isVi ? 'Dương tính Giả' : 'False Positive'}
                    </div>
                  </div>

                  <div className="text-[10px] text-amber-800/70 dark:text-amber-400/80 font-medium line-clamp-1">
                    {meta.fpTag}
                  </div>
                </button>

                {/* Quadrant 4: TRUE NEGATIVE (TN) */}
                <button
                  onClick={() =>
                    setSelectedQuadrant(selectedQuadrant === 'tn' ? null : 'tn')
                  }
                  className={`col-span-5 sm:col-span-5 rounded-xl p-3 border text-left flex flex-col justify-between transition-all cursor-pointer relative overflow-hidden group ${
                    selectedQuadrant === 'tn'
                      ? 'ring-2 ring-sky-500 scale-[1.02] shadow-lg'
                      : ''
                  } ${
                    isDark
                      ? 'bg-gradient-to-br from-sky-950/40 to-[#0c1824] border-sky-800/60 hover:border-sky-500'
                      : 'bg-sky-50/60 border-sky-300 hover:border-sky-500'
                  }`}
                >
                  <div className="flex items-center justify-between w-full">
                    <span className="text-xs font-mono font-bold text-sky-600 dark:text-sky-400 flex items-center gap-1.5">
                      <ShieldCheck className="w-3.5 h-3.5" />
                      TN
                    </span>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-400 font-bold">
                      {computed.total > 0
                        ? `${((computed.tn / computed.total) * 100).toFixed(0)}%`
                        : '0%'}
                    </span>
                  </div>

                  <div className="my-1.5">
                    <div className="text-2xl sm:text-3xl font-extrabold font-mono text-sky-600 dark:text-sky-300">
                      {computed.tn}
                    </div>
                    <div className="text-[11px] font-bold text-sky-700 dark:text-sky-200 mt-0.5">
                      {isVi ? 'Âm tính Thật' : 'True Negative'}
                    </div>
                  </div>

                  <div className="text-[10px] text-sky-800/70 dark:text-sky-400/80 font-medium line-clamp-1">
                    {meta.tnTag}
                  </div>
                </button>
              </div>

              {/* Column Marginal Totals */}
              <div className="grid grid-cols-12 gap-2 text-center text-[11px] font-mono text-slate-400 pt-1">
                <div className="col-span-3 text-right pr-2 font-semibold">
                  {isVi ? 'Tổng dự đoán:' : 'Pred Totals:'}
                </div>
                <div className="col-span-4 font-bold text-emerald-500">
                  Σ = {computed.predictedPositives}
                </div>
                <div className="col-span-5 font-bold text-sky-500">
                  Σ = {computed.predictedNegatives}
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT 5 COLS: KEY PERFORMANCE METRICS DASHBOARD */}
          <div
            className={`lg:col-span-5 rounded-2xl border p-4 sm:p-5 flex flex-col justify-between shadow-sm transition-all ${
              isDark ? 'bg-[#111827] border-[#1e293b]' : 'bg-white border-slate-200'
            }`}
          >
            <div className="flex items-center justify-between mb-3 border-b pb-2 border-inherit">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-cyan-400" />
                <span className="font-bold text-sm tracking-tight">
                  {isVi ? 'Chỉ số Đánh giá Mô hình' : 'Classification Metrics'}
                </span>
              </div>
              <span className="text-[11px] font-mono text-cyan-400 font-bold">
                F1 = {computed.f1Score.toFixed(3)}
              </span>
            </div>

            {/* Metric Bars List */}
            <div className="space-y-3">
              {/* 1. Accuracy */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-slate-600 dark:text-slate-300 flex items-center gap-1.5">
                    <strong>Accuracy</strong>
                    <span className="text-[10px] font-mono text-slate-400">(TP+TN)/Total</span>
                  </span>
                  <span className="font-mono font-bold text-indigo-500">
                    {(computed.accuracy * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                  <div
                    className="bg-indigo-500 h-full rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(100, computed.accuracy * 100)}%` }}
                  />
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5">
                  {isVi
                    ? 'Tỷ lệ dự đoán đúng toàn diện. Cẩn thận với tập dữ liệu mất cân bằng!'
                    : 'Overall correct predictions. Beware on imbalanced datasets!'}
                </div>
              </div>

              {/* 2. Precision */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-slate-600 dark:text-slate-300 flex items-center gap-1.5">
                    <strong>Precision</strong>
                    <span className="text-[10px] font-mono text-slate-400">TP/(TP+FP)</span>
                  </span>
                  <span className="font-mono font-bold text-amber-500">
                    {(computed.precision * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                  <div
                    className="bg-amber-500 h-full rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(100, computed.precision * 100)}%` }}
                  />
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5">
                  {isVi
                    ? 'Độ chuẩn xác: Khi mô hình báo Dương tính, tỷ lệ thật sự đúng là bao nhiêu?'
                    : 'Reliability: Out of all predicted positives, how many are truly positive?'}
                </div>
              </div>

              {/* 3. Recall / Sensitivity */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-slate-600 dark:text-slate-300 flex items-center gap-1.5">
                    <strong>Recall (Sensitivity)</strong>
                    <span className="text-[10px] font-mono text-slate-400">TP/(TP+FN)</span>
                  </span>
                  <span className="font-mono font-bold text-emerald-500">
                    {(computed.recall * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                  <div
                    className="bg-emerald-500 h-full rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(100, computed.recall * 100)}%` }}
                  />
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5">
                  {isVi
                    ? 'Độ nhạy: Trong tất cả ca bệnh thực tế, mô hình đã tóm được bao nhiêu ca?'
                    : 'Coverage: Out of all actual positives, how many did the model catch?'}
                </div>
              </div>

              {/* 4. Specificity */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-slate-600 dark:text-slate-300 flex items-center gap-1.5">
                    <strong>Specificity</strong>
                    <span className="text-[10px] font-mono text-slate-400">TN/(TN+FP)</span>
                  </span>
                  <span className="font-mono font-bold text-sky-500">
                    {(computed.specificity * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                  <div
                    className="bg-sky-500 h-full rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(100, computed.specificity * 100)}%` }}
                  />
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5">
                  {isVi
                    ? 'Độ đặc hiệu: Khả năng loại trừ người khỏe mạnh / ca âm tính.'
                    : 'True Negative Rate: Ability to correctly clear healthy/negative cases.'}
                </div>
              </div>

              {/* 5. F1-Score */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-slate-600 dark:text-slate-300 flex items-center gap-1.5">
                    <strong>F1-Score</strong>
                    <span className="text-[10px] font-mono text-slate-400">2·(P·R)/(P+R)</span>
                  </span>
                  <span className="font-mono font-bold text-cyan-400">
                    {(computed.f1Score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-amber-500 to-emerald-500 h-full rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(100, computed.f1Score * 100)}%` }}
                  />
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5">
                  {isVi
                    ? 'Trung bình điều hòa giữa Precision & Recall; trừng phạt nặng khi một trong hai bị yếu.'
                    : 'Harmonic mean of Precision and Recall; penalizes severe imbalance.'}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* BOTTOM SECTION: INTERACTIVE SAMPLE PROBABILITY STRIP & THRESHOLD DIVISION */}
        <div
          className={`rounded-2xl border p-4 shadow-sm transition-all ${
            isDark ? 'bg-[#111827] border-[#1e293b]' : 'bg-white border-slate-200'
          }`}
        >
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-cyan-400" />
              <span className="font-bold text-xs sm:text-sm">
                {isVi
                  ? 'Phân bố Xác suất Đầu ra P(y=1) & Ngưỡng Quyết định τ'
                  : 'Output Probability Distribution P(y=1) & Decision Threshold τ'}
              </span>
            </div>
            <div className="flex items-center gap-3 text-[11px] font-mono">
              <span className="flex items-center gap-1 text-emerald-500">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block" />
                {isVi ? 'Thực tế (+)' : 'Actual Positive'}
              </span>
              <span className="flex items-center gap-1 text-sky-500">
                <span className="w-2.5 h-2.5 rounded-full bg-sky-500 inline-block" />
                {isVi ? 'Thực tế (-)' : 'Actual Negative'}
              </span>
            </div>
          </div>

          {/* SVG Dot Plot along Probability Axis */}
          <div className="w-full overflow-x-auto py-1">
            <svg
              viewBox={`0 0 ${STRIP_W} ${STRIP_H}`}
              className="w-full max-h-32 overflow-visible select-none"
            >
              <defs>
                {/* Gradient for Left Negative Zone */}
                <linearGradient id="negZoneGrad" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor={isDark ? '#0284c7' : '#bae6fd'} stopOpacity="0.08" />
                  <stop offset="100%" stopColor={isDark ? '#0284c7' : '#bae6fd'} stopOpacity="0.25" />
                </linearGradient>
                {/* Gradient for Right Positive Zone */}
                <linearGradient id="posZoneGrad" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor={isDark ? '#10b981' : '#a7f3d0'} stopOpacity="0.25" />
                  <stop offset="100%" stopColor={isDark ? '#10b981' : '#a7f3d0'} stopOpacity="0.08" />
                </linearGradient>
              </defs>

              {/* Shaded Left Region: Predicted Negative (P < threshold) */}
              <rect
                x={STRIP_PADDING.left}
                y={STRIP_PADDING.top - 10}
                width={Math.max(0, thresholdSvgX - STRIP_PADDING.left)}
                height={STRIP_H - STRIP_PADDING.top - STRIP_PADDING.bottom + 20}
                fill="url(#negZoneGrad)"
                rx="6"
              />
              <text
                x={STRIP_PADDING.left + 8}
                y={STRIP_PADDING.top + 6}
                fill={isDark ? '#38bdf8' : '#0284c7'}
                fontSize="9"
                fontFamily="monospace"
                fontWeight="bold"
              >
                {isVi ? '← Dự đoán Âm tính (ŷ = 0)' : '← Predicted Negative (ŷ = 0)'}
              </text>

              {/* Shaded Right Region: Predicted Positive (P >= threshold) */}
              <rect
                x={thresholdSvgX}
                y={STRIP_PADDING.top - 10}
                width={Math.max(0, STRIP_W - STRIP_PADDING.right - thresholdSvgX)}
                height={STRIP_H - STRIP_PADDING.top - STRIP_PADDING.bottom + 20}
                fill="url(#posZoneGrad)"
                rx="6"
              />
              <text
                x={STRIP_W - STRIP_PADDING.right - 8}
                y={STRIP_PADDING.top + 6}
                textAnchor="end"
                fill={isDark ? '#34d399' : '#059669'}
                fontSize="9"
                fontFamily="monospace"
                fontWeight="bold"
              >
                {isVi ? 'Dự đoán Dương tính (ŷ = 1) →' : 'Predicted Positive (ŷ = 1) →'}
              </text>

              {/* Main Probability Axis Line */}
              <line
                x1={STRIP_PADDING.left}
                y1={STRIP_H - STRIP_PADDING.bottom}
                x2={STRIP_W - STRIP_PADDING.right}
                y2={STRIP_H - STRIP_PADDING.bottom}
                stroke={isDark ? '#334155' : '#cbd5e1'}
                strokeWidth="2"
              />

              {/* Probability Axis Ticks */}
              {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((tVal) => {
                const tx = toStripX(tVal);
                return (
                  <g key={tVal}>
                    <line
                      x1={tx}
                      y1={STRIP_H - STRIP_PADDING.bottom}
                      x2={tx}
                      y2={STRIP_H - STRIP_PADDING.bottom + 6}
                      stroke={isDark ? '#64748b' : '#94a3b8'}
                      strokeWidth="1.5"
                    />
                    <text
                      x={tx}
                      y={STRIP_H - STRIP_PADDING.bottom + 18}
                      textAnchor="middle"
                      fill={isDark ? '#94a3b8' : '#64748b'}
                      fontSize="9"
                      fontFamily="monospace"
                    >
                      {tVal.toFixed(1)}
                    </text>
                  </g>
                );
              })}

              {/* Vertical Neon Threshold Divider Bar */}
              <line
                x1={thresholdSvgX}
                y1={STRIP_PADDING.top - 15}
                x2={thresholdSvgX}
                y2={STRIP_H - STRIP_PADDING.bottom + 8}
                stroke="#06b6d4"
                strokeWidth="2.5"
                strokeDasharray="4,3"
                className="transition-all duration-150"
              />
              {/* Threshold Badge Pill */}
              <g className="transition-all duration-150">
                <rect
                  x={thresholdSvgX - 28}
                  y={STRIP_PADDING.top - 24}
                  width="56"
                  height="16"
                  rx="4"
                  fill="#0891b2"
                />
                <text
                  x={thresholdSvgX}
                  y={STRIP_PADDING.top - 12}
                  textAnchor="middle"
                  fill="#ffffff"
                  fontSize="8.5"
                  fontFamily="monospace"
                  fontWeight="bold"
                >
                  τ = {localThreshold.toFixed(2)}
                </text>
              </g>

              {/* Sample Dots Scatter */}
              {computed.samples.map((s, idx) => {
                const sx = toStripX(s.prob);
                // Jitter vertical position slightly to prevent overlap of identical probabilities
                const sy =
                  STRIP_H -
                  STRIP_PADDING.bottom -
                  16 -
                  ((idx * 17) % 36);

                const isActualPos = s.actual === 1;
                const isPredPos = s.prob >= localThreshold;
                let dotColor = '#10b981'; // Green for TP
                if (isActualPos && !isPredPos) dotColor = '#f43f5e'; // Red for FN
                if (!isActualPos && isPredPos) dotColor = '#f59e0b'; // Amber for FP
                if (!isActualPos && !isPredPos) dotColor = '#0284c7'; // Blue for TN

                const isHovered = hoveredSample?.id === s.id;
                const isFiltered =
                  !selectedQuadrant || s.status === selectedQuadrant;

                return (
                  <g
                    key={s.id}
                    className="cursor-pointer transition-all duration-200"
                    opacity={isFiltered ? (isHovered ? 1 : 0.85) : 0.2}
                    onMouseEnter={() => setHoveredSample(s)}
                    onMouseLeave={() => setHoveredSample(null)}
                  >
                    {/* Outer Glow Halo if hovered */}
                    {isHovered && (
                      <circle
                        cx={sx}
                        cy={sy}
                        r="14"
                        fill={dotColor}
                        fillOpacity="0.25"
                      />
                    )}

                    {/* Main Dot */}
                    <circle
                      cx={sx}
                      cy={sy}
                      r={isHovered ? 7 : 5.5}
                      fill={isDark ? '#0f172a' : '#ffffff'}
                      stroke={dotColor}
                      strokeWidth="2.5"
                    />

                    {/* Symbol inside node: + for actual 1, - for actual 0 */}
                    <text
                      x={sx}
                      y={sy + 2.5}
                      textAnchor="middle"
                      fill={dotColor}
                      fontSize="7.5"
                      fontFamily="monospace"
                      fontWeight="bold"
                    >
                      {s.actual === 1 ? '+' : '–'}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Hovered Sample Detail Tooltip / Invariant */}
          <div
            className={`mt-2 p-2.5 rounded-xl border flex flex-wrap items-center justify-between text-xs font-mono transition-colors ${
              isDark ? 'bg-[#090d16] border-slate-800' : 'bg-slate-50 border-slate-200'
            }`}
          >
            {hoveredSample ? (
              <div className="flex flex-wrap items-center gap-3">
                <span className="font-bold text-slate-200">
                  {hoveredSample.label || `Sample #${hoveredSample.id}`}:
                </span>
                <span className="text-slate-400">
                  {isVi ? 'Xác suất P:' : 'Prob P:'}{' '}
                  <strong className="text-cyan-400 font-bold">
                    {hoveredSample.prob.toFixed(2)}
                  </strong>
                </span>
                <span className="text-slate-400">
                  {isVi ? 'Thực tế y:' : 'Actual y:'}{' '}
                  <strong
                    className={
                      hoveredSample.actual === 1 ? 'text-emerald-400' : 'text-sky-400'
                    }
                  >
                    {hoveredSample.actual}
                  </strong>
                </span>
                <span className="text-slate-400">
                  {isVi ? 'Dự đoán ŷ:' : 'Predicted ŷ:'}{' '}
                  <strong
                    className={
                      hoveredSample.pred === 1 ? 'text-emerald-400' : 'text-sky-400'
                    }
                  >
                    {hoveredSample.pred}
                  </strong>
                </span>
                <span
                  className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase ${
                    hoveredSample.status === 'tp'
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                      : hoveredSample.status === 'fn'
                      ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                      : hoveredSample.status === 'fp'
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                      : 'bg-sky-500/20 text-sky-300 border border-sky-500/40'
                  }`}
                >
                  {hoveredSample.status?.toUpperCase()}
                </span>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-slate-400">
                <HelpCircle className="w-3.5 h-3.5 text-slate-500" />
                <span>
                  {isVi
                    ? 'Rê chuột qua từng điểm mẫu trên thanh xác suất để kiểm tra giá trị và phân loại!'
                    : 'Hover over any dot on the probability strip to inspect its class and prediction!'}
                </span>
              </div>
            )}

            {selectedQuadrant && (
              <button
                onClick={() => setSelectedQuadrant(null)}
                className="text-[10px] text-cyan-400 hover:underline font-bold"
              >
                {isVi ? '✕ Bỏ lọc góc phần tư' : '✕ Clear Quadrant Filter'}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ========================================================= */}
      {/* 4. BOTTOM FORMULA & TRADE-OFF SUMMARY STRIP               */}
      {/* ========================================================= */}
      <div
        className={`px-4 py-2 border-t flex flex-wrap items-center justify-between text-xs font-mono transition-colors ${
          isDark ? 'border-slate-800 bg-[#0d131f]/95' : 'border-slate-200 bg-slate-50/95'
        }`}
      >
        <div className="flex items-center gap-2">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          <span className="font-semibold text-slate-500 dark:text-slate-400">
            {isVi ? 'Quy tắc Quyết định:' : 'Decision Rule:'}
          </span>
          <span className="font-bold text-cyan-400">
            ŷ = (P(y=1) ≥ {localThreshold.toFixed(2)}) ? 1 : 0
          </span>
          <span className="text-slate-600">|</span>
          <span className="font-bold text-amber-400">
            Precision = {computed.precision.toFixed(2)}
          </span>
          <span className="text-slate-600">•</span>
          <span className="font-bold text-emerald-400">
            Recall = {computed.recall.toFixed(2)}
          </span>
        </div>

        <div className="flex items-center gap-3 text-[11px] text-slate-400">
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />
            <span>TP ({computed.tp})</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-rose-500 inline-block" />
            <span>FN ({computed.fn})</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-amber-500 inline-block" />
            <span>FP ({computed.fp})</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-sky-500 inline-block" />
            <span>TN ({computed.tn})</span>
          </div>
        </div>
      </div>
    </div>
  );
};
