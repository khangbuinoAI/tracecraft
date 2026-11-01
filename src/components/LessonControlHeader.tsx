import React from 'react';
import {
  Zap,
  Binary,
  Grid,
  Home,
  Sparkles,
  Edit3,
  Bot,
  TrendingUp,
  Activity,
  Info,
  Sliders,
  Cpu,
  BrainCircuit,
  Target,
  Compass,
} from 'lucide-react';
import type {
  ThemeMode,
  Language,
  AlgorithmPreset,
  TraceStep,
} from '../types';
import { getPresetVariants } from '../algorithms/presets';

interface LessonControlHeaderProps {
  preset: AlgorithmPreset;
  step: TraceStep;
  theme: ThemeMode;
  language: Language;
  onChangeInput?: (newInput: string) => void;
  inputStr?: string;
}

export const LessonControlHeader: React.FC<LessonControlHeaderProps> = ({
  preset,
  step,
  theme,
  language,
  onChangeInput,
  inputStr = '',
}) => {
  const isDark = theme === 'studio';
  const isVi = language === 'vi';
  const id = preset.id;
  const variants = getPresetVariants(preset, language) || preset.inputVariants || [];

  // Match active variant based on input string
  const activeVariant = variants.find(
    (v) =>
      inputStr.trim().replace(/\s+/g, '') === v.value.trim().replace(/\s+/g, '')
  ) || variants[0];

  // Specific metadata per lesson
  const getHeaderMeta = () => {
    switch (id) {
      case 'single-neuron': {
        const neuron = step.state.neuronState;
        const z = neuron?.weightedSum !== undefined ? Number(neuron.weightedSum) : 0;
        const a = neuron?.outputVal !== undefined ? Number(neuron.outputVal) : 0;
        const isFired = a === 1;

        return {
          icon: <Zap className="w-4 h-4 text-amber-500 animate-pulse" />,
          title: isVi ? 'Khớp Thần Kinh (Perceptron)' : 'Synapse & Perceptron',
          badge: isVi ? 'Phần tử Cơ bản' : 'Atomic Unit',
          badgeColor: isDark
            ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
            : 'bg-amber-100 text-amber-800 border-amber-300',
          subtitle: (
            <span>
              {isVi ? 'Tổng kích hoạt:' : 'Sum excitation:'}{' '}
              <strong className="font-mono text-cyan-600 dark:text-cyan-400">z = {z.toFixed(2)}</strong>
              {' • '}
              {isVi ? 'Đầu ra:' : 'Output:'}{' '}
              <strong className={`font-mono ${isFired ? 'text-emerald-600 dark:text-emerald-400' : 'text-slate-500 dark:text-slate-400'}`}>
                a = {a} ({isFired ? (isVi ? 'Phát xung' : 'Fired') : (isVi ? 'Nghỉ' : 'Silent')})
              </strong>
            </span>
          ),
          impact: isVi
            ? 'Mô phỏng nơ-ron sinh học Frank Rosenblatt (1957). Tín hiệu đầu vào x qua khớp thần kinh (synapse) w được nhân tỷ lệ và cộng dồn tại thân tế bào (soma). Nếu tổng z = w·x + b vượt ngưỡng, nơ-ron sẽ phát xung thần kinh (fire = 1)!'
            : 'Models biological synapses & soma (Rosenblatt 1957). Synaptic weights w scale incoming signals x into total excitation z = w·x + b. Once z exceeds the threshold bias, the neuron fires an action potential (fire = 1)!',
        };
      }

      case 'modern-neuron': {
        const modern = step.state.modernNeuronState;
        const actType = modern?.activationType || 'sigmoid';
        const z = modern?.zVal ?? 0;
        const a = modern?.outputVal ?? 0;
        const grad = modern?.gradientVal ?? 0;

        return {
          icon: <Binary className="w-4 h-4 text-cyan-500 dark:text-cyan-400" />,
          title: isVi ? 'Hàm Kích hoạt Khả vi' : 'Differentiable Activations',
          badge: isVi ? 'Tính Khả vi (Gradient)' : 'Differentiable Calculus',
          badgeColor: isDark
            ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
            : 'bg-cyan-100 text-cyan-800 border-cyan-300',
          subtitle: (
            <span>
              {isVi ? 'Hàm kích hoạt:' : 'Activation:'}{' '}
              <strong className="font-mono text-indigo-600 dark:text-indigo-400 uppercase">{actType}</strong>
              {' • '}
              z = <strong className="font-mono text-cyan-600 dark:text-cyan-400">{z.toFixed(2)}</strong>
              {' • '}
              a = <strong className="font-mono text-emerald-600 dark:text-emerald-400">{a.toFixed(3)}</strong>
              {' • '}
              {isVi ? 'Đạo hàm ∇:' : 'Gradient ∇:'}{' '}
              <strong className={`font-mono ${grad === 0 ? 'text-rose-600 dark:text-rose-400' : 'text-amber-600 dark:text-amber-400'}`}>
                {grad.toFixed(3)}
              </strong>
            </span>
          ),
          impact: isVi
            ? 'Hàm bước nhị phân có đạo hàm bằng 0 ở khắp mọi nơi (da/dz = 0), triệt tiêu hoàn toàn Gradient Descent. Đột phá của AI hiện đại là thay thế bằng các hàm khả vi trơn (Sigmoid & ReLU) với đạo hàm liên tục, mở đường cho thuật toán Lan truyền ngược (Backpropagation)!'
            : 'The step function has zero derivative almost everywhere (da/dz = 0), freezing Gradient Descent. Modern deep learning triumphed by switching to smooth, differentiable curves (Sigmoid & ReLU), providing continuous gradients that guide Backpropagation downhill!',
        };
      }

      case 'neural-network-mlp': {
        const layers = step.state.layers;
        const outputNode = layers && layers.length > 0 ? layers[layers.length - 1].nodes[0] : null;
        const hasComputed = outputNode && typeof outputNode.value === 'number' && outputNode.value > 0;
        const outProb = hasComputed ? Number(outputNode.value) : 0.82;
        return {
          icon: <BrainCircuit className="w-4 h-4 text-indigo-500 dark:text-indigo-400" />,
          title: isVi ? 'Mạng Nơ-ron Đa tầng (MLP)' : 'Multi-Layer Perceptron (MLP)',
          badge: isVi ? 'Lan truyền Xuôi' : 'Feedforward Network',
          badgeColor: isDark
            ? 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40'
            : 'bg-indigo-100 text-indigo-800 border-indigo-300',
          subtitle: (
            <span>
              {isVi ? 'Cấu trúc:' : 'Arch:'}{' '}
              <strong className="font-mono text-cyan-600 dark:text-cyan-400">2 → 4 → 1</strong>
              {' • '}
              {isVi ? 'Tầng ẩn:' : 'Hidden:'}{' '}
              <strong className="font-mono text-indigo-600 dark:text-indigo-400">4 ReLU</strong>
              {' • '}
              {isVi ? 'Dự đoán:' : 'Prediction:'}{' '}
              <strong className="font-mono text-emerald-600 dark:text-emerald-400">{(outProb * 100).toFixed(1)}%</strong>
            </span>
          ),
          impact: isVi
            ? 'Định lý Xấp xỉ Phổ quát (Universal Approximation Theorem) chứng minh mạng MLP chỉ với một tầng ẩn phi tuyến có thể xấp xỉ bất kỳ hàm số liên tục phức tạp nào. Tầng ẩn tạo ra không gian biểu diễn đặc trưng trừu tượng trước khi đưa ra quyết định cuối cùng.'
            : 'The Universal Approximation Theorem proves that an MLP with non-linear hidden layers can approximate any continuous function. Hidden layers transform raw coordinates into high-level latent representations to untangle complex non-linear patterns.',
        };
      }

      case 'cnn-feature-map': {
        const kernelPos = (step.state.kernelPos as { r: number; c: number } | undefined) || { r: 0, c: 0 };
        const kernelName = activeVariant?.label || 'Sobel Edge';

        return {
          icon: <Grid className="w-4 h-4 text-emerald-500 dark:text-emerald-400" />,
          title: isVi ? 'Trích xuất Đặc trưng 2D (CNN)' : '2D Feature Map (CNN)',
          badge: isVi ? 'Thị giác Máy tính' : 'Computer Vision',
          badgeColor: isDark
            ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
            : 'bg-emerald-100 text-emerald-800 border-emerald-300',
          subtitle: (
            <span>
              {isVi ? 'Bộ lọc 3×3:' : 'Kernel:'}{' '}
              <strong className="font-mono text-cyan-600 dark:text-cyan-400">{kernelName}</strong>
              {' • '}
              {isVi ? 'Vị trí quét:' : 'Pos:'}{' '}
              <strong className="font-mono text-amber-600 dark:text-amber-400">[{kernelPos.r}, {kernelPos.c}]</strong>
              {' • '}
              <strong className="font-mono text-slate-600 dark:text-slate-300">6×6 → 4×4 Map</strong>
            </span>
          ),
          impact: isVi
            ? 'Tích chập (Convolution) tận dụng tính bất biến dịch chuyển (translation invariance) và trọng số chia sẻ (weight sharing) để quét ma trận ảnh. Bộ lọc 3×3 trích xuất các đường biên, góc cạnh cục bộ mà không làm bùng nổ số lượng tham số như MLP toàn phần.'
            : 'Convolutions exploit translation invariance and weight sharing across 2D pixels. A compact 3×3 filter slides across the image to extract local visual edges, textures, and patterns without the parameter explosion of fully-connected MLPs.',
        };
      }

      case 'house-price-mlp': {
        const layers = step.state.layers;
        const outputNode = layers && layers.length > 0 ? layers[layers.length - 1].nodes[0] : null;
        const hasComputed = outputNode && typeof outputNode.value === 'number' && outputNode.value > 0;
        const price = hasComputed ? Number(outputNode.value).toFixed(2) : '1.85';
        return {
          icon: <Home className="w-4 h-4 text-amber-500" />,
          title: isVi ? 'Định giá Nhà (Regression MLP)' : 'House Price Regression',
          badge: isVi ? 'Hồi quy Bất động sản' : 'Real-Estate MLP',
          badgeColor: isDark
            ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
            : 'bg-amber-100 text-amber-800 border-amber-300',
          subtitle: (
            <span>
              {isVi ? 'Đặc trưng:' : 'Inputs:'}{' '}
              <strong className="font-mono text-slate-700 dark:text-slate-300">Diện tích, Phòng, Vị trí</strong>
              {' • '}
              {isVi ? 'Giá dự báo:' : 'Est. Price:'}{' '}
              <strong className="font-mono text-emerald-600 dark:text-emerald-400 text-sm font-extrabold">${price}M</strong>
            </span>
          ),
          impact: isVi
            ? 'Mô hình Hồi quy Đa tầng (Regression MLP 3→4→1) kết hợp phi tuyến tính diện tích, số phòng và điểm vị trí để dự báo giá bất động sản. Các nơ-ron ReLU tự động học được giá trị thặng dư phi tuyến tính của vị trí đắc địa.'
            : 'Practical 3→4→1 Regression MLP combining square footage, bedroom count, and location quality to predict property market value. ReLU hidden neurons automatically capture the non-linear premium of prime metro locations.',
        };
      }

      case 'dog-cat-mlp': {
        const layers = step.state.layers;
        const outputNode = layers && layers.length > 0 ? layers[layers.length - 1].nodes[0] : null;
        const hasComputed = outputNode && typeof outputNode.value === 'number' && outputNode.value > 0;
        const dogProb = hasComputed ? Number(outputNode.value) : 0.12;
        const catProb = 1 - dogProb;
        const isCat = catProb >= 0.5;
        return {
          icon: <Sparkles className="w-4 h-4 text-rose-500 dark:text-rose-400" />,
          title: isVi ? 'Phân loại Mèo vs Chó' : 'Cat vs Dog Classifier',
          badge: isVi ? 'Phân loại Nhị phân' : 'Binary Classifier',
          badgeColor: isDark
            ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
            : 'bg-rose-100 text-rose-800 border-rose-300',
          subtitle: (
            <span>
              {isVi ? 'Xác suất P(Mèo):' : 'Prob P(Cat):'}{' '}
              <strong className="font-mono text-emerald-600 dark:text-emerald-400">{(catProb * 100).toFixed(1)}%</strong>
              {' • '}
              {isVi ? 'Dự đoán:' : 'Class:'}{' '}
              <strong className={`font-mono font-bold ${isCat ? 'text-amber-600 dark:text-amber-300' : 'text-sky-600 dark:text-sky-300'}`}>
                {isCat ? (isVi ? '🐱 MÈO (Cat)' : '🐱 CAT') : (isVi ? '🐶 CHÓ (Dog)' : '🐶 DOG')}
              </strong>
            </span>
          ),
          impact: isVi
            ? 'Phân loại hình ảnh bằng mạng sâu 3→4→4→1. Lớp ẩn đa tầng bẻ cong không gian đặc trưng để phân tách hoàn hảo các đặc trưng thị giác phức tạp giữa loài Chó và loài Mèo, đưa ra xác suất qua hàm Sigmoid.'
            : 'Deep 3→4→4→1 binary visual classifier. Multi-layer hidden representations bend and warp the feature space to separate complex feline vs. canine textures, producing a final probability P(Cat) via Sigmoid.',
        };
      }

      case 'mnist-digit-recognition': {
        const mnist = step.state.mnistState;
        const topDigit = mnist?.predictedDigit ?? 7;
        const conf = mnist?.confidence ?? 0.994;

        return {
          icon: <Edit3 className="w-4 h-4 text-cyan-500 dark:text-cyan-400" />,
          title: isVi ? 'Nhận diện Số Viết tay (MNIST)' : 'MNIST Digit Recognition',
          badge: isVi ? 'Mô hình 99.4% CNN' : '99.4% CNN Benchmark',
          badgeColor: isDark
            ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
            : 'bg-cyan-100 text-cyan-800 border-cyan-300',
          subtitle: (
            <span>
              {isVi ? 'Số dự đoán:' : 'Recognized:'}{' '}
              <strong className="font-mono text-emerald-600 dark:text-emerald-400 text-sm font-extrabold">{topDigit}</strong>
              {' • '}
              {isVi ? 'Tin cậy:' : 'Confidence:'}{' '}
              <strong className="font-mono text-cyan-600 dark:text-cyan-400">{(conf * 100).toFixed(1)}%</strong>
              {' • '}
              <span className="text-[10px] opacity-75">
                {isVi ? 'Vẽ tự do trực tiếp lên khung tranh!' : 'Draw freely on canvas!'}
              </span>
            </span>
          ),
          impact: isVi
            ? 'Tiêu chuẩn vàng MNIST (Yann LeCun). Mạng nơ-ron trích xuất từng nét vẽ 28×28 thành 784 điểm ảnh, truyền qua các bộ lọc đặc trưng và chuẩn hóa 10 xác suất Softmax để nhận diện chính xác chữ số viết tay của con người.'
            : 'The iconic MNIST benchmark (Yann LeCun). Features extracted from 28×28 pixel stroke patterns pass through deep filters and are normalized via Softmax into 10 digit probabilities (0–9) for robust handwritten recognition.',
        };
      }

      case 'self-attention': {
        const tokens = step.state.tokens || ['AI', 'is', 'transforming'];
        const numTokens = tokens.length;
        const currentQuery = tokens[tokens.length - 1];

        return {
          icon: <Bot className="w-4 h-4 text-cyan-500 dark:text-cyan-400 animate-pulse" />,
          title: isVi ? 'Cơ chế Causal Self-Attention' : 'Causal Self-Attention',
          badge: isVi ? 'Kiến trúc Transformer' : 'Core Transformer',
          badgeColor: isDark
            ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
            : 'bg-cyan-100 text-cyan-800 border-cyan-300',
          subtitle: (
            <span>
              {isVi ? 'Ngữ cảnh:' : 'Context:'}{' '}
              <strong className="font-mono text-cyan-600 dark:text-cyan-400">{numTokens} tokens</strong>
              {' • '}
              Query:{' '}
              <strong className="font-mono text-amber-600 dark:text-amber-300">"{currentQuery}"</strong>
              {' • '}
              Softmax(Q·K^T / √d_k)·V
            </span>
          ),
          impact: isVi
            ? 'Trái tim của mọi mô hình LLM hiện đại (GPT-4, Claude, LLaMA). Từng từ ngữ tạo ra các vector Query (Truy vấn) và Key (Chìa khóa); tích vô hướng Q·K^T đo lường độ tương đồng ngữ cảnh, cho phép mô hình hiểu được đại từ và mối liên kết ngữ nghĩa đường dài!'
            : 'The foundational engine of all Modern LLMs (GPT-4, Claude, LLaMA). Each token emits Query, Key, and Value vectors; dot products Q·K^T measure semantic affinity across the context window, resolving long-range co-references and syntax dependencies.',
        };
      }

      case 'linear-regression': {
        const w = Number(step.state.weight ?? 0);
        const b = Number(step.state.bias ?? 0);

        return {
          icon: <TrendingUp className="w-4 h-4 text-blue-500 dark:text-cyan-400" />,
          title: isVi ? 'Hồi quy Tuyến tính (MSE)' : 'Linear Regression (MSE)',
          badge: isVi ? 'Tối ưu hóa MSE' : 'MSE Optimization',
          badgeColor: isDark
            ? 'bg-blue-500/20 text-blue-300 border-blue-500/40'
            : 'bg-blue-100 text-blue-800 border-blue-300',
          subtitle: (
            <span>
              y = <strong className="font-mono text-cyan-600 dark:text-cyan-400">{w.toFixed(2)}x {b >= 0 ? '+' : ''}{b.toFixed(2)}</strong>
              {' • '}
              {isVi ? 'Nhấn lên đồ thị để thêm/xóa điểm!' : 'Click graph to add/remove points!'}
            </span>
          ),
          impact: isVi
            ? 'Hồi quy Tuyến tính là nền móng của học có giám sát. Bằng cách cực tiểu hóa MSE Loss qua đạo hàm dL/dw và dL/db, mô hình tìm ra đường thẳng có sai số phần dư (residual) nhỏ nhất để dự báo xu hướng liên tục.'
            : 'Linear regression forms the bedrock of supervised learning. By minimizing MSE Loss via gradients dL/dw and dL/db, the model identifies the best-fit line with minimum residual error for continuous trend forecasting.',
        };
      }

      case 'logistic-regression': {
        const logistic = step.state.logisticState;
        const w = Number(logistic?.weight ?? step.state.weight ?? 0);
        const b = Number(logistic?.bias ?? step.state.bias ?? 0);
        const bdry = isFinite(-b / w) ? -b / w : 0;

        return {
          icon: <Activity className="w-4 h-4 text-emerald-500" />,
          title: isVi ? 'Hồi quy Logistic (Sigmoid)' : 'Logistic Regression (Sigmoid)',
          badge: isVi ? 'Phân loại Nhị phân' : 'Binary Classification',
          badgeColor: isDark
            ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
            : 'bg-emerald-100 text-emerald-800 border-emerald-300',
          subtitle: (
            <span>
              {isVi ? 'Ranh giới x*:' : 'Boundary x*:'}{' '}
              <strong className="font-mono text-cyan-600 dark:text-cyan-400">{bdry.toFixed(2)}</strong>
              {' • '}
              P(y=1) = σ({w.toFixed(2)}x {b >= 0 ? '+' : ''}{b.toFixed(2)})
            </span>
          ),
          impact: isVi
            ? 'Hàm Sigmoid ép trọn điểm số thực vào khoảng xác suất (0, 1). Đặc tính bão hòa hai đầu giúp mô hình kháng ngoại lai vượt trội, tìm ra ranh giới x* phân định dứt khoát giữa hai lớp.'
            : 'The Sigmoid function smoothly bounds real scores into valid probabilities (0, 1). Its saturating tails shield the model from extreme outliers, establishing an optimal boundary x* = -b/w.',
        };
      }

      case 'kmeans-clustering': {
        const km = step.state.kmeansState;
        const iter = km?.iteration ?? 0;
        const kVal = km?.k ?? 3;
        const inertia = km?.inertia ?? 0;
        const phase = km?.phase ?? 'init';
        const phaseLabel = isVi
          ? phase === 'init'
            ? 'Khởi tạo Tâm Cụm (Seeds)'
            : phase === 'assign'
            ? 'Gán Cụm (Voronoi)'
            : phase === 'update'
            ? 'Cập nhật Trọng tâm'
            : 'Đã Hội tụ'
          : phase === 'init'
          ? 'Init Centroids'
          : phase === 'assign'
          ? 'Voronoi Assign'
          : phase === 'update'
          ? 'Mean Update'
          : 'Converged';

        return {
          icon: <Target className="w-4 h-4 text-emerald-500 animate-pulse" />,
          title: isVi ? 'Phân cụm K-Means (Lloyd)' : 'K-Means Clustering (Lloyd)',
          badge: isVi ? 'Học Không Giám Sát' : 'Unsupervised Learning',
          badgeColor: isDark
            ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
            : 'bg-emerald-100 text-emerald-800 border-emerald-300',
          subtitle: (
            <span>
              {isVi ? 'Giai đoạn:' : 'Phase:'}{' '}
              <strong className="font-mono text-cyan-600 dark:text-cyan-400">{phaseLabel}</strong>
              {' • '}
              {isVi ? 'Vòng lặp:' : 'Iter:'}{' '}
              <strong className="font-mono text-indigo-600 dark:text-indigo-400">t = {iter}</strong>
              {' • '}
              <strong className="font-mono text-amber-600 dark:text-amber-400">K = {kVal}</strong>
              {' • '}
              {isVi ? 'Quán tính WCSS:' : 'Inertia J:'}{' '}
              <strong className="font-mono text-emerald-600 dark:text-emerald-400">{inertia.toFixed(2)}</strong>
            </span>
          ),
          impact: isVi
            ? 'K-Means là thuật toán học không giám sát kinh điển (Stuart Lloyd, 1982). Không cần nhãn dán trước, thuật toán tự động gom nhóm N điểm dữ liệu vào k cụm phân biệt bằng cách tối thiểu hóa tổng bình phương khoảng cách tới các tâm cụm (WCSS/Inertia). Ứng dụng rộng rãi trong phân khúc khách hàng thương mại điện tử, nén ảnh (color quantization), và lập chỉ mục vector (IVF-PQ) cho các mô hình AI/LLM!'
            : 'K-Means is the foundational unsupervised clustering algorithm (Stuart Lloyd, 1982). Operating without ground-truth labels, it groups N unlabeled data points into k distinct partitions by iteratively minimizing Within-Cluster Sum of Squares (WCSS / Inertia). Widely deployed in e-commerce customer segmentation, image quantization compression, anomaly detection, and vector search indexing (IVF-PQ) for LLMs!',
        };
      }

      case 'momentum-optimizer': {
        const mom = step.state.momentumState;
        const betaVal = mom?.beta ?? 0.9;
        const lr = mom?.learningRate ?? 0.1;
        const w = mom?.currentW ?? [0, 0];
        const lossVal = mom?.loss ?? 0;
        const vel = mom?.velocity ?? [0, 0];
        const velMag = Math.hypot(vel[0], vel[1]);

        return {
          icon: <Compass className="w-4 h-4 text-cyan-400 animate-pulse" />,
          title: isVi ? 'Gradient Descent với Momentum' : 'Gradient Descent with Momentum',
          badge: isVi ? 'Tối ưu Hóa Gia tốc' : 'Accelerated Optimizer',
          badgeColor: isDark
            ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
            : 'bg-cyan-100 text-cyan-800 border-cyan-300',
          subtitle: (
            <span>
              β = <strong className="font-mono text-cyan-600 dark:text-cyan-400">{betaVal}</strong>
              {' • '}
              α = <strong className="font-mono text-indigo-600 dark:text-indigo-400">{lr}</strong>
              {' • '}
              w = <strong className="font-mono text-slate-700 dark:text-slate-300">({w[0].toFixed(2)}, {w[1].toFixed(2)})</strong>
              {' • '}
              Loss: <strong className="font-mono text-rose-600 dark:text-rose-400">{lossVal.toFixed(3)}</strong>
              {' • '}
              {isVi ? 'Vận tốc ‖v‖:' : 'Speed ‖v‖:'}{' '}
              <strong className="font-mono text-emerald-600 dark:text-emerald-400">{velMag.toFixed(3)}</strong>
            </span>
          ),
          impact: isVi
            ? 'SGD thuần gặp "bệnh nan y" khi đi qua hẻm núi (ravine): dao động dữ dội qua lại giữa hai sườn dốc hẹp mà hầu như không tiến được về phía trước. Thuật toán Heavy-Ball của Polyak (1964) mô phỏng một hòn bi có khối lượng và quán tính vật lý: tích lũy vận tốc v_t theo các hướng dốc ổn định và triệt tiêu các dao động ngược chiều, giúp tăng tốc độ hội tụ gấp 10 lần (1/(1-β))!'
            : 'Vanilla SGD struggles in ravines: oscillating violently across steep narrow walls while making negligible progress along the flat base. Boris Polyak’s Heavy-Ball Momentum (1964) simulates physical inertia: accumulating velocity v_t in consistent directions while averaging out alternating oscillations, accelerating convergence up to 10× (1/(1-β))!',
        };
      }

      default:
        return {
          icon: <Cpu className="w-4 h-4 text-cyan-500 dark:text-cyan-400" />,
          title: preset.title,
          badge: preset.badge,
          badgeColor: isDark
            ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
            : 'bg-cyan-100 text-cyan-800 border-cyan-300',
          subtitle: <span>Step {step.step} / {step.totalSteps}</span>,
          impact: preset.description,
        };
    }
  };

  const meta = getHeaderMeta();

  return (
    <div
      className={`w-full shrink-0 flex flex-col z-20 font-sans select-none rounded-2xl border overflow-hidden shadow-xs mb-2 transition-all ${
        isDark ? 'border-slate-800 bg-[#0d131f]/95' : 'border-slate-200 bg-white/95'
      }`}
    >
      {/* ========================================================= */}
      {/* 1. SCENARIO CONTEXT & INTERACTIVE PARAMETER CONTROLS      */}
      {/* ========================================================= */}
      <div
        className={`px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 transition-colors ${
          isDark ? 'text-slate-200' : 'text-slate-800'
        }`}
      >
        {/* Left: Icon + Title + Badge + Subtitle with Live State */}
        <div className="flex items-center gap-3 shrink-0">
          <div
            className={`p-2 rounded-xl flex items-center justify-center border shrink-0 ${
              isDark ? 'bg-slate-800/80 border-slate-700' : 'bg-slate-100 border-slate-200'
            }`}
          >
            {meta.icon}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span
                className={`text-sm font-bold tracking-tight whitespace-nowrap ${
                  isDark ? 'text-white' : 'text-slate-900'
                }`}
              >
                {meta.title}
              </span>
              <span
                className={`text-[10px] font-mono px-2 py-0.5 rounded-full border font-semibold whitespace-nowrap shrink-0 ${meta.badgeColor}`}
              >
                {meta.badge}
              </span>
            </div>
            <div
              className={`text-[11px] flex items-center gap-2 mt-0.5 whitespace-nowrap ${
                isDark ? 'text-slate-400' : 'text-slate-600'
              }`}
            >
              {meta.subtitle}
            </div>
          </div>
        </div>

        {/* Right: Interactive Live Controls & Variant Buttons */}
        {onChangeInput && variants.length > 0 && (
          <div
            className={`flex items-center gap-1.5 p-1 rounded-xl border shrink-0 max-w-full overflow-x-auto ${
              isDark ? 'bg-[#090d16] border-slate-800' : 'bg-slate-50 border-slate-200'
            }`}
          >
            <div className="flex items-center gap-1 px-1.5 text-[10px] uppercase font-bold text-slate-500 shrink-0">
              <Sliders className="w-3 h-3 text-cyan-500 dark:text-cyan-400" />
              <span className="hidden sm:inline">{isVi ? 'Mẫu:' : 'Presets:'}</span>
            </div>

            <div className="flex items-center gap-1 overflow-x-auto max-w-full">
              {variants.map((v) => {
                const isSelected =
                  inputStr.trim().replace(/\s+/g, '') === v.value.trim().replace(/\s+/g, '');
                const shortLabel = v.label.replace(/\s*\([^)]*\)/, '').trim() || v.label;
                return (
                  <button
                    key={v.label}
                    onClick={() => onChangeInput(v.value)}
                    title={v.label}
                    className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold transition-all whitespace-nowrap border shrink-0 ${
                      isSelected
                        ? isDark
                          ? 'bg-cyan-950 text-cyan-300 border-cyan-700 font-bold shadow-xs'
                          : 'bg-slate-900 text-white border-slate-900 font-bold shadow-xs'
                        : isDark
                        ? 'bg-slate-800/60 border-slate-700/60 text-slate-300 hover:bg-slate-700 hover:text-white'
                        : 'bg-white border-slate-200 text-slate-700 hover:bg-slate-100'
                    }`}
                  >
                    {shortLabel}
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* ========================================================= */}
      {/* 2. REALISTIC CONTEXT BANNER (Matching Image #1)           */}
      {/* ========================================================= */}
      <div
        className={`px-4 py-2 border-t flex items-center gap-2 text-xs transition-colors ${
          isDark
            ? 'border-slate-800/80 bg-[#0f172a]/60 text-slate-300'
            : 'border-slate-200/80 bg-amber-50/60 text-slate-700'
        }`}
      >
        <Info className="w-4 h-4 text-cyan-500 dark:text-cyan-400 shrink-0" />
        <span className="leading-relaxed">
          <strong>{isVi ? 'Ý nghĩa thực tế:' : 'Real-World Impact:'}</strong> {meta.impact}
        </span>
      </div>
    </div>
  );
};
