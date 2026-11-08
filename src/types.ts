export type ThemeMode = 'sketch' | 'studio';

export type Language = 'en' | 'vi';
export type ViewMode = 'desktop-3pane' | 'android-ui' | 'mobile-story';
export type AndroidTab = 'stage' | 'code' | 'theory' | 'curriculum';

export type AppView = 'catalog' | 'visualizer';

export type StudioTab = 'trace' | 'theory';

export type AICategory =
  | 'Classical Machine Learning'
  | 'Deep Learning & Neural Networks'
  | 'Transformers & Modern LLMs';

export type AlgorithmCategory = AICategory;
export type DifficultyLevel = 'Beginner' | 'Intermediate' | 'Advanced';

export type VisualType =
  | 'regression-plot'
  | 'logistic-regression'
  | 'confusion-matrix'
  | 'neural-network'
  | 'single-neuron'
  | 'modern-neuron'
  | 'attention-matrix'
  | 'cnn-convolution'
  | 'mnist-interactive'
  | 'array-bars'
  | 'kmeans-clustering'
  | 'momentum-optimizer'
  | 'nesterov-optimizer'
  | 'batch-gradient';

export type StepAction =
  | 'init'
  | 'forward'
  | 'loss'
  | 'backward'
  | 'update'
  | 'converged'
  | 'compare'
  | 'swap'
  | 'pivot'
  | 'visit'
  | 'found'
  | 'compute'
  | 'done';

export interface DataPoint {
  x: number;
  y: number;
  yPred?: number;
}

export interface NeuralNode {
  label: string;
  value: number;
  activation?: string;
}

export interface NeuralLayer {
  name: string;
  nodes: NeuralNode[];
}

export interface SingleNeuronState {
  inputs: { label: string; val: number }[];
  weights: number[];
  bias: number;
  weightedSum: number;
  activationName: string;
  outputVal: number;
}

export interface BioNeuronState {
  membraneVoltage: number; // in mV, e.g. -70, -55, +40, -80, -70
  phase: 'resting' | 'depolarization' | 'peak' | 'repolarization' | 'refractory';
  naChannels: 'closed' | 'open' | 'inactivated';
  kChannels: 'closed' | 'open';
  dendriteInputs: number[];
  stimulusCurrent: number; // in pA
  history: { t: number; mv: number }[];
}

export interface ModernNeuronState {
  inputs: { label: string; val: number }[];
  weights: number[];
  bias: number;
  zVal: number;
  activationType: 'step' | 'sigmoid' | 'relu' | 'tanh';
  outputVal: number;
  gradientVal: number; // da/dz
}

export interface MnistDigitProb {
  digit: number;
  prob: number;
  logit: number;
}

export interface MnistState {
  grid: number[][];
  flattenedSample: number[];
  hiddenLayer: number[];
  probabilities: MnistDigitProb[];
  predictedDigit: number;
  confidence: number;
  activePhase: 'draw' | 'flatten' | 'hidden' | 'softmax' | 'result';
}

export interface LogisticPoint {
  x: number;
  y: 0 | 1;
  prob?: number;
  loss?: number;
  predLabel?: 0 | 1;
}

export interface LogisticConfusionMatrix {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
}

export interface LogisticState {
  points: LogisticPoint[];
  weight: number;
  bias: number;
  learningRate: number;
  epoch: number;
  totalEpochs: number;
  loss: number;
  threshold: number;
  decisionBoundary: number;
  accuracy: number;
  confusionMatrix: LogisticConfusionMatrix;
  gradW?: number;
  gradB?: number;
}

export type QuadrantType = 'tp' | 'fp' | 'fn' | 'tn';

export interface ConfusionMatrixSample {
  id: number;
  prob: number;
  actual: 0 | 1;
  pred?: 0 | 1;
  status?: QuadrantType;
  label?: string;
}

export interface ConfusionMatrixData {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  total: number;
  threshold: number;
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1Score: number;
  samples: ConfusionMatrixSample[];
  scenario: 'medical' | 'spam' | 'fraud' | 'balanced';
  activeQuadrant?: QuadrantType | null;
  activeSampleIndex?: number | null;
  costNote?: string;
}

export interface KMeansPoint {
  id: number;
  x: number;
  y: number;
  cluster: number; // -1 if unassigned, 0..k-1
  distanceToCentroid?: number;
  distances?: number[];
  label?: string;
}

export interface KMeansCentroid {
  id: number;
  x: number;
  y: number;
  prevX?: number;
  prevY?: number;
  color: string;
  count: number;
}

export interface KMeansState {
  k: number;
  iteration: number;
  maxIterations: number;
  phase: 'init' | 'assign' | 'update' | 'converged';
  points: KMeansPoint[];
  centroids: KMeansCentroid[];
  inertia: number; // Within-Cluster Sum of Squares (WCSS)
  prevInertia?: number;
  centroidShift: number;
  converged: boolean;
  activePointId?: number | null;
  activeClusterId?: number | null;
}

export interface MomentumTrajectoryPoint {
  step: number;
  w1: number;
  w2: number;
  loss: number;
  v1?: number;
  v2?: number;
  grad1?: number;
  grad2?: number;
}

export interface MomentumState {
  beta: number; // Momentum coefficient β ∈ [0, 1)
  learningRate: number; // α
  currentW: [number, number]; // [w1, w2]
  loss: number;
  velocity: [number, number]; // [v1, v2]
  gradient: [number, number]; // [grad1, grad2]
  momentumPath: MomentumTrajectoryPoint[];
  vanillaPath: MomentumTrajectoryPoint[];
  surfaceType: 'ravine' | 'saddle' | 'beale' | 'plateau';
  iteration: number;
  maxIterations: number;
  converged: boolean;
}

export interface NesterovTrajectoryPoint {
  step: number;
  w1: number;
  w2: number;
  loss: number;
  v1?: number;
  v2?: number;
  lookaheadW1?: number;
  lookaheadW2?: number;
  lookaheadLoss?: number;
  grad1?: number; // gradient at lookahead position
  grad2?: number;
}

export interface NesterovState {
  beta: number; // Momentum coefficient β ∈ [0, 1)
  learningRate: number; // α
  currentW: [number, number]; // [w1, w2]
  loss: number;
  lookaheadW: [number, number]; // [w1_ahead, w2_ahead]
  lookaheadLoss: number;
  velocity: [number, number]; // [v1, v2]
  lookaheadGradient: [number, number]; // [grad1, grad2]
  nesterovPath: NesterovTrajectoryPoint[];
  polyakPath: MomentumTrajectoryPoint[]; // Classical Polyak momentum comparison
  vanillaPath: MomentumTrajectoryPoint[]; // Vanilla SGD comparison
  surfaceType: 'ravine' | 'saddle' | 'beale' | 'plateau';
  iteration: number;
  maxIterations: number;
  converged: boolean;
}

export interface BatchDataPoint {
  id: number;
  x: number;
  y: number;
  gradW1: number;
  gradW2: number;
}

export interface BatchTrajectoryPoint {
  step: number;
  w1: number;
  w2: number;
  loss: number;
  grad1: number;
  grad2: number;
}

export interface BatchGradientState {
  batchSize: number; // N (total samples in full batch)
  totalSamples: number; // N
  learningRate: number; // α
  currentW: [number, number]; // [w1, w2]
  loss: number;
  gradient: [number, number]; // exact full-batch gradient [grad1, grad2]
  batchPath: BatchTrajectoryPoint[]; // Smooth full-batch trajectory
  sgdPath: BatchTrajectoryPoint[]; // Stochastic GD trajectory for comparison
  miniBatchPath: BatchTrajectoryPoint[]; // Mini-batch trajectory for comparison
  sampleGradients: BatchDataPoint[]; // Individual sample gradient vectors
  iteration: number;
  maxIterations: number;
  converged: boolean;
}

export type PrimitiveStateValue =
  | string
  | number
  | boolean
  | null
  | undefined
  | number[]
  | string[]
  | number[][]
  | (number | null)[][]
  | { r: number; c: number }
  | DataPoint[]
  | NeuralLayer[]
  | SingleNeuronState
  | ModernNeuronState
  | BioNeuronState
  | { token: string; prob: number }[]
  | MnistState
  | LogisticState
  | LogisticPoint[]
  | ConfusionMatrixData
  | ConfusionMatrixSample[]
  | KMeansState
  | KMeansPoint[]
  | KMeansCentroid[]
  | MomentumState
  | MomentumTrajectoryPoint[]
  | NesterovState
  | NesterovTrajectoryPoint[]
  | BatchGradientState
  | BatchTrajectoryPoint[]
  | BatchDataPoint[];
export interface StepState {
  epoch?: number;
  loss?: number;
  weight?: number;
  bias?: number;
  learningRate?: number;
  gradW?: number;
  gradB?: number;
  points?: DataPoint[];
  layers?: NeuralLayer[];
  tokens?: string[];
  probabilities?: { token: string; prob: number }[];
  arr?: number[];
  i?: number;
  j?: number;
  inputMatrix?: number[][];
  kernel?: number[][];
  featureMap?: (number | null)[][];
  kernelPos?: { r: number; c: number };
  neuronState?: SingleNeuronState;
  modernNeuronState?: ModernNeuronState;
  bioNeuronState?: BioNeuronState;
  mnistState?: MnistState;
  logisticState?: LogisticState;
  confusionMatrixState?: ConfusionMatrixData;
  kmeansState?: KMeansState;
  momentumState?: MomentumState;
  nesterovState?: NesterovState;
  batchGradientState?: BatchGradientState;
  [key: string]: PrimitiveStateValue;
}

export interface VariableDiffItem {
  before: PrimitiveStateValue;
  after: PrimitiveStateValue;
}

export interface TraceStep {
  step: number;
  totalSteps: number;
  line: number; // 1-based active line in code
  action: StepAction;
  targets: number[]; // indices or keys currently highlighted/manipulated
  state: StepState;
  explanation: string;
  invariant?: string;
  variableDiff?: Record<string, VariableDiffItem>;
}

export interface LessonSection {
  title: string;
  content: string;
  formula?: string;
  keyTakeaway?: string;
}

export interface LocalizedPresetInfo {
  title: string;
  category: string;
  badge: string;
  summary: string;
  description: string;
  sections: LessonSection[];
  variants?: { label: string; value: string }[];
}

export interface AlgorithmPreset {
  id: string;
  title: string;
  category: AICategory;
  difficulty: DifficultyLevel;
  badge: string;
  visualType: VisualType;
  formula: string;
  summary: string;
  description: string;
  sections: LessonSection[];
  timeComplexity: string;
  spaceComplexity: string;
  defaultInput: string;
  inputVariants?: { label: string; value: string }[];
  tags: string[];
  pythonCode: string;
  code: string; // for display in code pane
  vi?: LocalizedPresetInfo;
  generateTraces: (inputStr?: string, lang?: Language) => TraceStep[];
}
