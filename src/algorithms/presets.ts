import type {
  AlgorithmPreset,
  TraceStep,
  Language,
  LogisticPoint,
  ConfusionMatrixData,
  ConfusionMatrixSample,
  QuadrantType,
  KMeansPoint,
  KMeansCentroid,
  MomentumTrajectoryPoint,
} from '../types';
import { generateCanonicalDigitGrid, inferMnist } from '../components/MnistStage';
export const ALGORITHM_PRESETS: AlgorithmPreset[] = [
  // ==========================================
  // Category 1: Classical Machine Learning
  // ==========================================
  {
    id: 'linear-regression',
    title: '01. Linear Regression & Gradient Descent',
    category: 'Classical Machine Learning',
    difficulty: 'Beginner',
    badge: 'Optimization',
    visualType: 'regression-plot',
    formula: 'y = w \\cdot x + b, \\quad L = \\frac{1}{N}\\sum_{i=1}^N (\\hat{y}_i - y_i)^2',
    summary: 'Fits an optimal line to 2D continuous data by iteratively minimizing Mean Squared Error (MSE) using gradient descent.',
    description: 'Linear Regression models the relationship between an input feature X and a target variable y. Gradient Descent computes the partial derivatives of the MSE loss with respect to weight w and bias b, updating parameters in the opposite direction of the gradient.',
    sections: [
      {
        title: '1. The Core Objective: Minimizing Residuals',
        content: 'Given data points (X, y), our model predicts y_pred = w * x + b. For each point, the prediction error (residual) is (y_pred - y). We square these errors to heavily penalize large deviations and eliminate negative signs, producing the Mean Squared Error (MSE) objective function.',
        formula: 'MSE(w, b) = \\frac{1}{n} \\sum_{i=1}^n (w \\cdot x_i + b - y_i)^2',
        keyTakeaway: 'The MSE cost function forms a convex parabolic bowl with a single global minimum.',
      },
      {
        title: '2. Gradient Descent Derivation',
        content: 'To find the optimal w and b, we compute the gradient (vector of partial derivatives) of MSE with respect to each parameter. By the chain rule: d/dw = (2/n) * sum(x * (y_pred - y)) and d/db = (2/n) * sum(y_pred - y).',
        formula: 'w \\leftarrow w - \\alpha \\frac{\\partial L}{\\partial w}, \\quad b \\leftarrow b - \\alpha \\frac{\\partial L}{\\partial b}',
        keyTakeaway: 'The learning rate alpha controls step size. Too high causes divergence; too low slows convergence.',
      },
      {
        title: '3. Why It Matters in Modern AI',
        content: 'Linear regression with gradient descent is the conceptual foundation of all deep learning: computing predictions, calculating loss, taking gradients, and backpropagating parameter updates.',
        keyTakeaway: 'Deep neural networks are essentially stacked non-linear regressions optimized by variants of this same gradient rule.',
      },
    ],
    timeComplexity: 'O(epochs · N)',
    spaceComplexity: 'O(1) parameters',
    tags: ['Supervised', 'Regression', 'Gradient Descent', 'MSE Loss', 'NumPy'],
    defaultInput: '(1, 2.2), (2, 2.8), (3, 4.5), (4, 4.9), (5, 6.2)',
    inputVariants: [
      { label: 'Standard', value: '(1, 2.2), (2, 2.8), (3, 4.5), (4, 4.9), (5, 6.2)' },
      { label: 'Noisy', value: '(1, 1.4), (2, 3.8), (3, 3.1), (4, 5.9), (5, 5.2)' },
      { label: 'Outlier', value: '(1, 2.0), (2, 2.5), (3, 3.0), (4, 3.5), (5, 8.5)' },
      { label: 'Steep', value: '(1, 0.5), (2, 2.4), (3, 4.2), (4, 6.1), (5, 8.0)' },
    ],
    vi: {
      title: '01. Hồi quy Tuyến tính & Gradient Descent',
      category: 'Machine Learning Cổ điển',
      badge: 'Gradient Descent',
      summary: 'Khớp đường thẳng tối ưu với dữ liệu 2D bằng cách cực tiểu hóa MSE Loss thông qua thuật toán Gradient Descent.',
      description: 'Hồi quy tuyến tính mô hình hóa tương quan giữa biến đầu vào X và biến mục tiêu y. Thuật toán Gradient Descent tính gradient của hàm MSE Loss theo weight w và bias b, cập nhật tham số ngược hướng dốc để tối ưu mô hình.',
      variants: [
        { label: 'Chuẩn', value: '(1, 2.2), (2, 2.8), (3, 4.5), (4, 4.9), (5, 6.2)' },
        { label: 'Nhiễu', value: '(1, 1.4), (2, 3.8), (3, 3.1), (4, 5.9), (5, 5.2)' },
        { label: 'Ngoại lai', value: '(1, 2.0), (2, 2.5), (3, 3.0), (4, 3.5), (5, 8.5)' },
        { label: 'Dốc cao', value: '(1, 0.5), (2, 2.4), (3, 4.2), (4, 6.1), (5, 8.0)' },
      ],
      sections: [
        {
          title: '1. Mục tiêu Cốt lõi: Cực tiểu hóa Residuals (Sai số dư)',
          content: 'Với mỗi điểm dữ liệu (X, y), mô hình dự đoán y_pred = w * x + b. Phần dư residual là (y_pred - y). Ta bình phương sai số này để phạt nặng sai lệch lớn và loại bỏ dấu âm, tạo nên hàm MSE Loss (Mean Squared Error).',
          formula: 'MSE(w, b) = \\frac{1}{n} \\sum_{i=1}^n (w \\cdot x_i + b - y_i)^2',
          keyTakeaway: 'Hàm MSE Loss có dạng parabol lồi (convex), đảm bảo Gradient Descent sẽ hội tụ về điểm cực tiểu toàn cục duy nhất.',
        },
        {
          title: '2. Đạo hàm và Quy tắc Cập nhật Gradient Descent',
          content: 'Để tìm w và b tối ưu, ta tính gradient của hàm MSE Loss theo từng tham số bằng Chain Rule: d/dw = (2/n) * sum(x * (y_pred - y)) và d/db = (2/n) * sum(y_pred - y).',
          formula: 'w \\leftarrow w - \\alpha \\frac{\\partial L}{\\partial w}, \\quad b \\leftarrow b - \\alpha \\frac{\\partial L}{\\partial b}',
          keyTakeaway: 'Learning rate α quyết định bước nhảy. Nếu quá lớn thuật toán sẽ phân kỳ (diverge); nếu quá nhỏ sẽ hội tụ rất chậm.',
        },
        {
          title: '3. Tầm quan trọng trong Nền tảng Deep Learning',
          content: 'Hồi quy tuyến tính kết hợp Gradient Descent là nền tảng cốt lõi của toàn bộ Deep Learning: Forward pass tính dự đoán, tính Loss, Backpropagation lấy gradient và cập nhật weights.',
          keyTakeaway: 'Mạng nơ-ron sâu bản chất là sự xếp chồng của nhiều layers phi tuyến được tối ưu bằng các biến thể của chính thuật toán Gradient Descent này.',
        },
      ],
    },
    pythonCode: `import numpy as np

def linear_regression(X, y, lr=0.06, epochs=8):
    w, b = 0.0, 0.0  # Initialize weights at zero
    n = len(X)
    
    for epoch in range(epochs):
        # 1. Forward Pass: compute predictions
        y_pred = w * X + b
        
        # 2. Compute Loss: Mean Squared Error
        loss = np.mean((y_pred - y) ** 2)
        
        # 3. Compute Gradients via Chain Rule
        grad_w = (2 / n) * np.sum(X * (y_pred - y))
        grad_b = (2 / n) * np.sum(y_pred - y)
        
        # 4. Parameter Update
        w -= lr * grad_w
        b -= lr * grad_b
        
    return w, b`,
    code: `import numpy as np

def linear_regression(X, y, lr=0.06, epochs=8):
    w, b = 0.0, 0.0  # Initialize weights
    n = len(X)
    for epoch in range(epochs):
        y_pred = w * X + b  # Forward prediction
        loss = np.mean((y_pred - y) ** 2)  # MSE Loss
        grad_w = (2 / n) * np.sum(X * (y_pred - y))
        grad_b = (2 / n) * np.sum(y_pred - y)
        w -= lr * grad_w  # Update weight
        b -= lr * grad_b  # Update bias
    return w, b`,
    generateTraces: (inputStr?: string, lang: Language = 'en'): TraceStep[] => {
      const isVi = lang === 'vi';
      let points = [
        { x: 1, y: 2.2 },
        { x: 2, y: 2.8 },
        { x: 3, y: 4.5 },
        { x: 4, y: 4.9 },
        { x: 5, y: 6.2 },
      ];
      if (inputStr) {
        const matches = [...inputStr.matchAll(/\(?\s*([0-9.]+)\s*[,:]\s*([0-9.]+)\s*\)?/g)];
        if (matches.length >= 2) {
          points = matches.map((m) => ({ x: Number(m[1]), y: Number(m[2]) }));
        }
      }
      const n = points.length;
      const lr = 0.06;
      let w = 0.2;
      let b = 0.5;

      const steps: TraceStep[] = [];

      // Step 0: init
      const initPoints = points.map((p) => ({ ...p, yPred: Number((w * p.x + b).toFixed(2)) }));
      const initLoss = Number(
        (initPoints.reduce((acc, p) => acc + (p.yPred! - p.y) ** 2, 0) / n).toFixed(3)
      );

      steps.push({
        step: 0,
        totalSteps: 0,
        line: 4,
        action: 'init',
        targets: [],
        state: {
          epoch: 0,
          weight: w,
          bias: b,
          learningRate: lr,
          loss: initLoss,
          points: initPoints,
        },
        explanation: isVi
          ? `Khởi tạo weight ban đầu w = ${w}, bias b = ${b}. MSE Loss ban đầu: ${initLoss}.`
          : `Initialized weights at w = ${w}, b = ${b}. Starting Mean Squared Error Loss: ${initLoss}.`,
        invariant: isVi
          ? 'Đường ngẫu nhiên chưa huấn luyện. Bước Gradient Descent ban đầu: epoch = 0.'
          : 'Untrained random line. Gradient step counter epoch = 0.',
      });

      for (let epoch = 1; epoch <= 7; epoch++) {
        const currentPoints = points.map((p) => ({
          ...p,
          yPred: Number((w * p.x + b).toFixed(2)),
        }));
        const lossVal = Number(
          (currentPoints.reduce((acc, p) => acc + (p.yPred! - p.y) ** 2, 0) / n).toFixed(3)
        );

        steps.push({
          step: steps.length,
          totalSteps: 0,
          line: 7,
          action: 'forward',
          targets: [],
          state: {
            epoch,
            weight: Number(w.toFixed(2)),
            bias: Number(b.toFixed(2)),
            loss: lossVal,
            points: currentPoints,
          },
          explanation: isVi
            ? `Epoch #${epoch}: Chạy Forward pass tính y_pred = ${w.toFixed(2)}x + ${b.toFixed(2)}. MSE Loss: ${lossVal}.`
            : `Epoch #${epoch}: Forward pass computed predictions y_pred = ${w.toFixed(2)}x + ${b.toFixed(2)}. Current MSE Loss: ${lossVal}.`,
          invariant: isVi
            ? 'Hàm MSE Loss đánh giá chất lượng khớp đường thẳng hiện tại.'
            : `MSE Loss evaluates current fit quality. Target is local/global minimum.`,
        });

        // Compute Gradients
        const gradW = Number(
          ((2 / n) * currentPoints.reduce((acc, p) => acc + p.x * (p.yPred! - p.y), 0)).toFixed(3)
        );
        const gradB = Number(
          ((2 / n) * currentPoints.reduce((acc, p) => acc + (p.yPred! - p.y), 0)).toFixed(3)
        );

        steps.push({
          step: steps.length,
          totalSteps: 0,
          line: 9,
          action: 'backward',
          targets: [],
          state: {
            epoch,
            weight: Number(w.toFixed(2)),
            bias: Number(b.toFixed(2)),
            gradW,
            gradB,
            loss: lossVal,
            points: currentPoints,
          },
          explanation: isVi
            ? `Tính gradient theo Chain Rule: ∂L/∂w = ${gradW}, ∂L/∂b = ${gradB}. Đi ngược hướng gradient để giảm Loss.`
            : `Calculated gradients via chain rule: ∂L/∂w = ${gradW}, ∂L/∂b = ${gradB}. Negative gradient directs downhill step.`,
          invariant: isVi
            ? 'Vector gradient ∇L chỉ hướng dốc tăng nhanh nhất; ta đi ngược lại để tìm cực tiểu.'
            : 'Gradient vector ∇L points in direction of steepest ascent.',
        });

        // Update weights
        const oldW = Number(w.toFixed(2));
        const oldB = Number(b.toFixed(2));
        w = w - lr * gradW;
        b = b - lr * gradB;

        const updatedPoints = points.map((p) => ({
          ...p,
          yPred: Number((w * p.x + b).toFixed(2)),
        }));
        const newLoss = Number(
          (updatedPoints.reduce((acc, p) => acc + (p.yPred! - p.y) ** 2, 0) / n).toFixed(3)
        );

        steps.push({
          step: steps.length,
          totalSteps: 0,
          line: 11,
          action: 'update',
          targets: [],
          state: {
            epoch,
            weight: Number(w.toFixed(2)),
            bias: Number(b.toFixed(2)),
            loss: newLoss,
            points: updatedPoints,
          },
          explanation: isVi
            ? `Cập nhật tham số: w = ${w.toFixed(2)}, b = ${b.toFixed(2)}. MSE Loss giảm xuống còn ${newLoss}.`
            : `Parameter update: w = ${w.toFixed(2)} (changed by ${(-lr * gradW).toFixed(2)}), b = ${b.toFixed(2)}. Loss dropped to ${newLoss}.`,
          invariant: isVi
            ? 'Hàm MSE Loss đơn điệu giảm dần dưới learning rate phù hợp.'
            : 'Loss strictly non-increasing under appropriate learning rate.',
          variableDiff: {
            w: { before: oldW, after: Number(w.toFixed(2)) },
            b: { before: oldB, after: Number(b.toFixed(2)) },
            loss: { before: lossVal, after: newLoss },
          },
        });
      }

      // Final step: converged
      steps.push({
        step: steps.length,
        totalSteps: 0,
        line: 13,
        action: 'converged',
        targets: [],
        state: {
          epoch: 8,
          weight: Number(w.toFixed(2)),
          bias: Number(b.toFixed(2)),
          loss: Number(
            (
              points.reduce(
                (acc, p) => acc + (Number((w * p.x + b).toFixed(2)) - p.y) ** 2,
                0
              ) / n
            ).toFixed(3)
          ),
          points: points.map((p) => ({
            ...p,
            yPred: Number((w * p.x + b).toFixed(2)),
          })),
        },
        explanation: isVi
          ? `Quá trình Gradient Descent đã hội tụ! Đường hồi quy tối ưu: y = ${w.toFixed(2)}x + ${b.toFixed(2)}. Residuals đạt mức tối thiểu.`
          : `Gradient descent converged! Best fit line: y = ${w.toFixed(2)}x + ${b.toFixed(2)}. Residual variance minimized.`,
        invariant: isVi
          ? '||∇L|| ≈ 0: Vector trọng số đã chạm đáy cực tiểu lồi.'
          : '||∇L|| ≈ 0: Parameter weights reside at local convex minimum.',
      });

      const total = steps.length;
      return steps.map((s, idx) => ({ ...s, step: idx + 1, totalSteps: total }));
    },
  },
  // ==========================================
  // Lesson 02: Logistic Regression & Binary Classification
  // ==========================================
  {
    id: 'logistic-regression',
    title: '02. Logistic Regression & Binary Classification',
    category: 'Classical Machine Learning',
    difficulty: 'Beginner',
    badge: 'Classification',
    visualType: 'logistic-regression',
    formula: 'P(y=1) = \\sigma(wx + b) = \\frac{1}{1 + e^{-(wx + b)}}, \\quad L = -\\frac{1}{N}\\sum_{i=1}^N \\left[y_i \\ln \\hat{y}_i + (1 - y_i) \\ln(1 - \\hat{y}_i)\\right]',
    summary: 'Maps linear features into probabilities via the Sigmoid curve, optimizing Binary Cross-Entropy Loss to separate classes with a decision boundary.',
    description: 'Logistic Regression is the foundational algorithm for binary classification. It passes a linear combination z = w*x + b through the non-linear Sigmoid function σ(z) to bound outputs between 0 and 1. Gradient Descent minimizes Log Loss (Binary Cross-Entropy) to find the optimal decision boundary x* = -b/w.',
    sections: [
      {
        title: '1. The Logistic (Sigmoid) Function & Log-Odds',
        content: 'Linear regression produces outputs from -infinity to +infinity, which makes it unsuitable for predicting probabilities. Logistic regression wraps the linear score z = w*x + b in the Sigmoid activation function σ(z) = 1 / (1 + e^-z), guaranteeing an output bounded strictly between 0 and 1. This originates from modeling the log-odds (logit) ln(p / (1 - p)) as a linear function of x.',
        formula: '\\sigma(z) = \\frac{1}{1 + e^{-z}}, \\quad \\text{logit}(p) = \\ln\\left(\\frac{p}{1 - p}\\right) = w \\cdot x + b',
        keyTakeaway: 'The Sigmoid function smoothly squashes any real number into a valid probability (0, 1).',
      },
      {
        title: '2. Binary Cross-Entropy Loss (Log Loss)',
        content: 'Using Mean Squared Error with Sigmoid causes non-convexity with many flat plateaus and local minima. Instead, we use Binary Cross-Entropy (BCE) derived from Maximum Likelihood Estimation. If y=1, loss is -ln(p); if y=0, loss is -ln(1-p). A confident wrong prediction (e.g. y=1 but p=0.01) incurs near-infinite penalty.',
        formula: 'L_{\\text{BCE}}(w, b) = -\\frac{1}{N}\\sum_{i=1}^N \\Big[y_i \\ln(\\hat{y}_i) + (1 - y_i)\\ln(1 - \\hat{y}_i)\\Big]',
        keyTakeaway: 'BCE is strictly convex for logistic regression, ensuring gradient descent reaches the global minimum.',
      },
      {
        title: '3. Outlier Resistance: Logistic vs. Linear Regression',
        content: 'In linear classification, a point with y=1 far to the right exerts excessive leverage, tilting the regression line and misclassifying borderline samples. In logistic regression, once z is large, σ(z) saturates near 1, causing (p - y) ≈ 0. Consequently, extreme positive outliers generate almost zero gradient and cannot distort the decision boundary.',
        formula: '\\frac{\\partial L}{\\partial w} = \\frac{1}{N} \\sum_{i=1}^N (\\hat{y}_i - y_i) x_i, \\quad \\lim_{z \\to \\infty} (\\hat{y} - 1) = 0',
        keyTakeaway: 'Sigmoid saturation shields logistic regression from being corrupted by distant outliers.',
      },
      {
        title: '4. Decision Boundary & Confusion Matrix',
        content: 'At default threshold τ = 0.5, the decision boundary is the point x* where σ(wx* + b) = 0.5 <=> wx* + b = 0 => x* = -b / w. Predictions are evaluated using the Confusion Matrix: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).',
        formula: 'x^* = -\\frac{b}{w}, \\quad \\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}',
        keyTakeaway: 'Adjusting threshold τ trades off False Positives (precision) against False Negatives (recall).',
      },
    ],
    timeComplexity: 'O(epochs · N)',
    spaceComplexity: 'O(1) parameters',
    tags: ['Supervised', 'Classification', 'Sigmoid', 'BCE Loss', 'Log Loss', 'NumPy'],
    defaultInput: '(1, 0), (2, 0), (3, 0), (4, 0), (6, 1), (7, 1), (8, 1), (9, 1)',
    inputVariants: [
      { label: 'Separable', value: '(1, 0), (2, 0), (3, 0), (4, 0), (6, 1), (7, 1), (8, 1), (9, 1)' },
      { label: 'Noisy Overlap', value: '(1, 0), (2, 0), (3, 0), (4, 1), (5, 0), (6, 1), (7, 0), (8, 1), (9, 1)' },
      { label: 'Outlier Robust', value: '(1, 0), (2, 0), (3, 0), (4, 0), (6, 1), (7, 1), (8, 1), (15, 1)' },
      { label: 'Imbalanced', value: '(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 1), (9, 1)' },
    ],
    vi: {
      title: '02. Hồi quy Logistic & Phân loại Nhị phân',
      category: 'Machine Learning Cổ điển',
      badge: 'Phân loại Nhị phân',
      summary: 'Ánh xạ đặc trưng tuyến tính thành xác suất qua đường cong Sigmoid, tối ưu hóa Binary Cross-Entropy Loss để tìm ranh giới quyết định.',
      description: 'Hồi quy Logistic là thuật toán nền tảng cho các bài toán phân loại nhị phân. Mô hình đưa tổ hợp tuyến tính z = w*x + b qua hàm phi tuyến Sigmoid σ(z) để ép giá trị đầu ra nằm trong khoảng xác suất (0, 1). Thuật toán Gradient Descent cực tiểu hóa Log Loss (Binary Cross-Entropy) để xác định ranh giới quyết định tối ưu x* = -b/w.',
      variants: [
        { label: 'Dễ phân tách', value: '(1, 0), (2, 0), (3, 0), (4, 0), (6, 1), (7, 1), (8, 1), (9, 1)' },
        { label: 'Nhiễu đan xen', value: '(1, 0), (2, 0), (3, 0), (4, 1), (5, 0), (6, 1), (7, 0), (8, 1), (9, 1)' },
        { label: 'Kháng Ngoại lai', value: '(1, 0), (2, 0), (3, 0), (4, 0), (6, 1), (7, 1), (8, 1), (15, 1)' },
        { label: 'Mất cân bằng', value: '(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 1), (9, 1)' },
      ],
      sections: [
        {
          title: '1. Hàm Logistic (Sigmoid) & Tỷ lệ Log-Odds',
          content: 'Hồi quy tuyến tính tạo ra giá trị từ -vô cùng đến +vô cùng, không thể đại diện cho xác suất. Hồi quy Logistic bọc giá trị logit z = w*x + b vào hàm kích hoạt Sigmoid σ(z) = 1 / (1 + e^-z), đảm bảo đầu ra luôn thuộc khoảng (0, 1). Về mặt toán học, điều này tương đương với mô hình hóa log-odds (logit) ln(p / (1 - p)) thành một hàm tuyến tính của x.',
          formula: '\\sigma(z) = \\frac{1}{1 + e^{-z}}, \\quad \\text{logit}(p) = \\ln\\left(\\frac{p}{1 - p}\\right) = w \\cdot x + b',
          keyTakeaway: 'Đường cong chữ S của Sigmoid ánh xạ mượt mà mọi điểm số thực thành một xác suất hợp lệ.',
        },
        {
          title: '2. Hàm Mất mát Binary Cross-Entropy (Log Loss)',
          content: 'Áp dụng hàm MSE cho Sigmoid tạo ra bề mặt phi lồi (non-convex) có nhiều vùng bằng phẳng và cực tiểu cục bộ. Hồi quy Logistic sử dụng hàm Binary Cross-Entropy (BCE) dẫn xuất từ nguyên lý Maximum Likelihood Estimation. Khi nhãn y=1, Loss là -ln(p); khi y=0, Loss là -ln(1-p). Dự đoán sai lệch với độ tự tin cao sẽ bị phạt số điểm mất mát vô cùng lớn.',
          formula: 'L_{\\text{BCE}}(w, b) = -\\frac{1}{N}\\sum_{i=1}^N \\Big[y_i \\ln(\\hat{y}_i) + (1 - y_i)\\ln(1 - \\hat{y}_i)\\Big]',
          keyTakeaway: 'Hàm BCE lồi nghiêm ngặt (strictly convex), đảm bảo Gradient Descent luôn tìm thấy cực tiểu toàn cục duy nhất.',
        },
        {
          title: '3. Nghịch lý Ngoại lai: Hồi quy Logistic vs Tuyến tính',
          content: 'Trong hồi quy tuyến tính, một điểm dữ liệu ngoại lai lệch xa về bên phải sẽ kéo độ dốc đường thẳng xuống, làm sai lệch ngưỡng phân loại của các điểm khác. Trong hồi quy Logistic, khi z lớn, σ(z) bão hòa tiệm cận 1 khiến (p - y) ≈ 0. Do đó, điểm ngoại lai tạo ra gradient gần như bằng 0 và không làm méo mó ranh giới quyết định.',
          formula: '\\frac{\\partial L}{\\partial w} = \\frac{1}{N} \\sum_{i=1}^N (\\hat{y}_i - y_i) x_i, \\quad \\lim_{z \\to \\infty} (\\hat{y} - 1) = 0',
          keyTakeaway: 'Đặc tính bão hòa ở hai đầu của Sigmoid giúp mô hình có khả năng kháng ngoại lai vượt trội so với hồi quy tuyến tính.',
        },
        {
          title: '4. Ranh giới Quyết định & Ma trận Nhầm lẫn',
          content: 'Với ngưỡng mặc định τ = 0.5, ranh giới quyết định là điểm x* mà tại đó σ(wx* + b) = 0.5 <=> wx* + b = 0 => x* = -b / w. Hiệu năng phân loại được đo lường qua Ma trận Nhầm lẫn: Dương tính Thật (TP), Âm tính Thật (TN), Dương tính Giả (FP), và Âm tính Giả (FN).',
          formula: 'x^* = -\\frac{b}{w}, \\quad \\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}',
          keyTakeaway: 'Ranh giới x* phân chia không gian đặc trưng thành hai vùng dự đoán rõ ràng cho Lớp 0 và Lớp 1.',
        },
      ],
    },
    pythonCode: `import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -25, 25)))

def logistic_regression(X, y, lr=0.15, epochs=7):
    x_mean = np.mean(X)
    w, b_c = 0.20, 0.00
    N = len(X)
    
    for epoch in range(epochs):
        # 1. Forward Pass: compute logits and Sigmoid probabilities
        z = w * (X - x_mean) + b_c
        p = sigmoid(z)
        
        # 2. Compute Binary Cross-Entropy Loss
        eps = 1e-12
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        
        # 3. Compute Gradients via Chain Rule
        grad_w = np.mean((p - y) * (X - x_mean))
        grad_b = np.mean(p - y)
        
        # 4. Parameter Update via Gradient Descent
        w -= lr * grad_w
        b_c -= (lr * 2.0) * grad_b
        
    b = b_c - w * x_mean  # Convert to standard form z = w * x + b
    boundary = -b / w     # Decision threshold at P=0.5
    return w, b, boundary`,
    code: `import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, lr=0.15, epochs=7):
    w, b = 0.20, -1.00
    N = len(X)
    for epoch in range(epochs):
        z = w * X + b                   # Compute logits
        p = sigmoid(z)                  # Sigmoid probabilities
        loss = -np.mean(y*np.log(p) + (1-y)*np.log(1-p)) # BCE Loss
        grad_w = np.mean((p - y) * X)   # Weight gradient
        grad_b = np.mean(p - y)         # Bias gradient
        w -= lr * grad_w                # Update weight
        b -= lr * grad_b                # Update bias
    boundary = -b / w                   # Decision boundary x*
    return w, b, boundary`,
    generateTraces: (inputStr?: string, lang: Language = 'en'): TraceStep[] => {
      const isVi = lang === 'vi';
      let points: LogisticPoint[] = [
        { x: 1, y: 0 },
        { x: 2, y: 0 },
        { x: 3, y: 0 },
        { x: 4, y: 0 },
        { x: 6, y: 1 },
        { x: 7, y: 1 },
        { x: 8, y: 1 },
        { x: 9, y: 1 },
      ];
      if (inputStr) {
        const matches = [...inputStr.matchAll(/\(?\s*([0-9.]+)\s*[,:]\s*([0-1])\s*\)?/g)];
        if (matches.length >= 2) {
          points = matches.map((m) => ({ x: Number(m[1]), y: (Number(m[2]) >= 0.5 ? 1 : 0) as 0 | 1 }));
        }
      }

      const N = points.length;
      const x_mean = points.reduce((acc, p) => acc + p.x, 0) / N;
      const lr = 0.15;
      let w = 0.20;
      let b_c = 0.00;
      const sigmoid = (zVal: number) => 1 / (1 + Math.exp(-Math.max(-25, Math.min(25, zVal))));

      const steps: TraceStep[] = [];

      const calcState = (curW: number, curBc: number) => {
        const w_real = Number(curW.toFixed(3));
        const b_real = Number((curBc - curW * x_mean).toFixed(3));
        const bound = curW !== 0 ? -b_real / w_real : 0;
        
        let tp = 0, tn = 0, fp = 0, fn = 0;
        const currentPoints: LogisticPoint[] = points.map((p) => {
          const z = curW * (p.x - x_mean) + curBc;
          const prob = Number(sigmoid(z).toFixed(4));
          const eps = 1e-12;
          const sampleLoss = Number((-(p.y * Math.log(prob + eps) + (1 - p.y) * Math.log(1 - prob + eps))).toFixed(4));
          const predLabel: 0 | 1 = prob >= 0.5 ? 1 : 0;
          if (p.y === 1 && predLabel === 1) tp++;
          else if (p.y === 0 && predLabel === 0) tn++;
          else if (p.y === 0 && predLabel === 1) fp++;
          else if (p.y === 1 && predLabel === 0) fn++;
          return {
            x: p.x,
            y: p.y,
            prob,
            loss: sampleLoss,
            predLabel,
          };
        });

        const totalLoss = Number((currentPoints.reduce((acc, p) => acc + (p.loss ?? 0), 0) / N).toFixed(4));
        const accuracy = Number((((tp + tn) / N) * 100).toFixed(1));

        return {
          w_real,
          b_real,
          bound: Number(bound.toFixed(2)),
          currentPoints,
          totalLoss,
          accuracy,
          confusionMatrix: { tp, tn, fp, fn },
        };
      };

      // Step 0: Init
      const initState = calcState(w, b_c);
      steps.push({
        step: 0,
        totalSteps: 0,
        line: 8,
        action: 'init',
        targets: [],
        state: {
          epoch: 0,
          weight: initState.w_real,
          bias: initState.b_real,
          learningRate: lr,
          loss: initState.totalLoss,
          logisticState: {
            points: initState.currentPoints,
            weight: initState.w_real,
            bias: initState.b_real,
            learningRate: lr,
            epoch: 0,
            totalEpochs: 7,
            loss: initState.totalLoss,
            threshold: 0.5,
            decisionBoundary: initState.bound,
            accuracy: initState.accuracy,
            confusionMatrix: initState.confusionMatrix,
          },
        },
        explanation: isVi
          ? `Khởi tạo trọng số ban đầu: w = ${initState.w_real}, b = ${initState.b_real}. Ranh giới ban đầu x* = ${initState.bound}. Binary Cross-Entropy Loss: ${initState.totalLoss}, Độ chính xác: ${initState.accuracy}%.`
          : `Initialized weights at w = ${initState.w_real}, b = ${initState.b_real}. Initial decision boundary x* = ${initState.bound}. BCE Loss: ${initState.totalLoss}, Accuracy: ${initState.accuracy}%.`,
        invariant: isVi
          ? 'Đường cong Sigmoid chưa huấn luyện. Bước lặp Gradient Descent ban đầu: epoch = 0.'
          : 'Untrained Sigmoid curve. Starting Gradient Descent counter: epoch = 0.',
      });

      // 7 Epochs of Gradient Descent
      for (let epoch = 1; epoch <= 7; epoch++) {
        const fwd = calcState(w, b_c);

        // Step 1: Forward Pass
        steps.push({
          step: steps.length,
          totalSteps: 0,
          line: 12,
          action: 'forward',
          targets: [],
          state: {
            epoch,
            weight: fwd.w_real,
            bias: fwd.b_real,
            loss: fwd.totalLoss,
            logisticState: {
              points: fwd.currentPoints,
              weight: fwd.w_real,
              bias: fwd.b_real,
              learningRate: lr,
              epoch,
              totalEpochs: 7,
              loss: fwd.totalLoss,
              threshold: 0.5,
              decisionBoundary: fwd.bound,
              accuracy: fwd.accuracy,
              confusionMatrix: fwd.confusionMatrix,
            },
          },
          explanation: isVi
            ? `Epoch #${epoch}: Forward pass tính z = ${fwd.w_real}x + (${fwd.b_real}) và xác suất P(y=1) = σ(z). Ranh giới x* = ${fwd.bound}. Độ chính xác: ${fwd.accuracy}%.`
            : `Epoch #${epoch}: Forward pass calculated logits z = ${fwd.w_real}x + (${fwd.b_real}) and Sigmoid probabilities P(y=1) = σ(z). Boundary x* = ${fwd.bound}. Accuracy: ${fwd.accuracy}%.`,
          invariant: isVi
            ? 'Hàm Sigmoid ánh xạ toàn bộ giá trị số thực z vào khoảng xác suất hợp lệ (0, 1).'
            : 'Sigmoid squashes real values into valid probability range (0, 1).',
        });

        // Compute Gradients
        const gradW = Number((fwd.currentPoints.reduce((acc, p) => acc + (p.prob! - p.y) * (p.x - x_mean), 0) / N).toFixed(4));
        const gradB = Number((fwd.currentPoints.reduce((acc, p) => acc + (p.prob! - p.y), 0) / N).toFixed(4));

        // Step 2: Loss & Backward
        steps.push({
          step: steps.length,
          totalSteps: 0,
          line: 20,
          action: 'backward',
          targets: [],
          state: {
            epoch,
            weight: fwd.w_real,
            bias: fwd.b_real,
            gradW,
            gradB,
            loss: fwd.totalLoss,
            logisticState: {
              points: fwd.currentPoints,
              weight: fwd.w_real,
              bias: fwd.b_real,
              learningRate: lr,
              epoch,
              totalEpochs: 7,
              loss: fwd.totalLoss,
              threshold: 0.5,
              decisionBoundary: fwd.bound,
              accuracy: fwd.accuracy,
              confusionMatrix: fwd.confusionMatrix,
              gradW,
              gradB,
            },
          },
          explanation: isVi
            ? `Tính đạo hàm BCE theo Chain Rule: ∂L/∂w = ${gradW}, ∂L/∂b = ${gradB}. Đi ngược hướng gradient để giảm Log Loss.`
            : `Gradients via chain rule: ∂L/∂w = ${gradW}, ∂L/∂b = ${gradB}. Steepest descent direction identified.`,
          invariant: isVi
            ? 'Đạo hàm của hàm Binary Cross-Entropy kết hợp Sigmoid có dạng đối xứng thanh lịch: (p - y) * x.'
            : 'Sigmoid BCE gradient has elegant symmetric form (p - y) * x.',
        });

        // Step 3: Update Weights
        const oldW = fwd.w_real;
        const oldB = fwd.b_real;
        const oldLoss = fwd.totalLoss;

        w -= lr * gradW;
        b_c -= (lr * 2.0) * gradB;

        const updated = calcState(w, b_c);
        steps.push({
          step: steps.length,
          totalSteps: 0,
          line: 24,
          action: 'update',
          targets: [],
          state: {
            epoch,
            weight: updated.w_real,
            bias: updated.b_real,
            loss: updated.totalLoss,
            logisticState: {
              points: updated.currentPoints,
              weight: updated.w_real,
              bias: updated.b_real,
              learningRate: lr,
              epoch,
              totalEpochs: 7,
              loss: updated.totalLoss,
              threshold: 0.5,
              decisionBoundary: updated.bound,
              accuracy: updated.accuracy,
              confusionMatrix: updated.confusionMatrix,
            },
          },
          explanation: isVi
            ? `Cập nhật tham số: w = ${updated.w_real}, b = ${updated.b_real}. BCE Loss giảm xuống còn ${updated.totalLoss}. Ranh giới x* = ${updated.bound}.`
            : `Updated parameters: w = ${updated.w_real}, b = ${updated.b_real}. Loss reduced to ${updated.totalLoss}. Decision boundary shifted to x* = ${updated.bound}.`,
          invariant: isVi
            ? 'Hàm mất mát BCE lồi nghiêm ngặt, đảm bảo mỗi bước cập nhật đều tiến gần đến cực tiểu toàn cục.'
            : 'Strict convexity of BCE guarantees descent towards the unique global minimum.',
          variableDiff: {
            w: { before: oldW, after: updated.w_real },
            b: { before: oldB, after: updated.b_real },
            loss: { before: oldLoss, after: updated.totalLoss },
          },
        });
      }

      // Final Step: Converged
      const finalState = calcState(w, b_c);
      steps.push({
        step: steps.length,
        totalSteps: 0,
        line: 28,
        action: 'converged',
        targets: [],
        state: {
          epoch: 8,
          weight: finalState.w_real,
          bias: finalState.b_real,
          loss: finalState.totalLoss,
          logisticState: {
            points: finalState.currentPoints,
            weight: finalState.w_real,
            bias: finalState.b_real,
            learningRate: lr,
            epoch: 8,
            totalEpochs: 7,
            loss: finalState.totalLoss,
            threshold: 0.5,
            decisionBoundary: finalState.bound,
            accuracy: finalState.accuracy,
            confusionMatrix: finalState.confusionMatrix,
          },
        },
        explanation: isVi
          ? `Mô hình Hồi quy Logistic đã hội tụ! Ranh giới quyết định tối ưu: x* = ${finalState.bound}. Độ chính xác đạt ${finalState.accuracy}%, BCE Loss = ${finalState.totalLoss}.`
          : `Logistic Regression converged! Optimal decision boundary x* = ${finalState.bound}. Accuracy reached ${finalState.accuracy}%, BCE Loss = ${finalState.totalLoss}.`,
        invariant: isVi
          ? '||∇L|| ≈ 0: Ranh giới x* đã phân tách tối ưu không gian đặc trưng giữa hai lớp.'
          : '||∇L|| ≈ 0: Parameter weights have arrived at optimal decision boundary.',
      });

      const total = steps.length;
      return steps.map((s, idx) => ({ ...s, step: idx + 1, totalSteps: total }));
    },
  },
  // ==========================================
  {
    id: 'confusion-matrix',
    title: '03. Confusion Matrix & Classification Metrics',
    category: 'Classical Machine Learning',
    difficulty: 'Beginner',
    badge: 'Evaluation Metrics',
    visualType: 'confusion-matrix',
    formula: '\\text{Accuracy} = \\frac{TP + TN}{N}, \\quad \\text{Precision} = \\frac{TP}{TP + FP}, \\quad \\text{Recall} = \\frac{TP}{TP + FN}, \\quad F_1 = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}',
    summary: 'Deconstructs classification performance into True/False Positives and Negatives, exposing the Accuracy Paradox and navigating the Precision-Recall trade-off across decision thresholds.',
    description: 'In real-world classification (cancer screening, fraud detection, spam filtering), overall accuracy is often dangerously misleading. The Confusion Matrix is the foundational diagnostic tool that tabulates actual outcomes against model predictions into four quadrants: TP, FP, FN, and TN. Adjusting the probability decision threshold τ trades off False Alarms (Precision) against Missed Detections (Recall), harmonized by the F1-Score.',
    sections: [
      {
        title: '1. The Accuracy Paradox & Class Imbalance',
        content: 'In high-stakes applications such as cancer screening or financial fraud detection, positive cases are rare (e.g. 5% positive, 95% negative). A naive "dummy" model predicting all samples as negative achieves 95% Accuracy, yet catches zero sick patients and zero fraud cases (0% Recall)! This dangerous illusion is the Accuracy Paradox. Evaluating imbalanced classification tasks requires breaking down errors into True and False detections.',
        formula: '\\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}, \\quad \\lim_{\\text{Imbalance} \\to \\infty} \\text{Accuracy} \\approx \\text{Majority Class %}',
        keyTakeaway: 'Accuracy is blind to error severity and class imbalance. Never evaluate an imbalanced model on accuracy alone.',
      },
      {
        title: '2. The 2×2 Confusion Matrix & Type I / Type II Errors',
        content: 'The Confusion Matrix cross-tabulates Actual classes (rows) against Predicted classes (columns) into 4 distinct quadrants: True Positive (TP - hit), False Positive (FP - Type I Error, false alarm), False Negative (FN - Type II Error, missed detection), and True Negative (TN - correct rejection). In the real world, the cost of Type I and Type II errors is asymmetric: sending an email to spam (FP) is an inconvenience, but missing malignant cancer (FN) is catastrophic.',
        formula: '\\begin{pmatrix} TP & FN \\\\ FP & TN \\end{pmatrix}, \\quad \\text{Type I Error} = FP, \\quad \\text{Type II Error} = FN',
        keyTakeaway: 'The cost of a False Negative (missed cancer) is often radically higher than a False Positive (extra biopsy).',
      },
      {
        title: '3. Precision, Recall, Specificity & The Harmonic F1-Score',
        content: 'Precision (TP / (TP + FP)) answers: "When the model sounds the alarm, how often is it right?" Recall or Sensitivity (TP / (TP + FN)) answers: "Out of all actual positive cases, how many did the model catch?" Specificity (TN / (TN + FP)) measures true negative clearance. The F1-Score is the harmonic mean of Precision and Recall. Unlike the arithmetic mean, the harmonic mean severely penalizes extreme disparities (e.g. 100% precision with 0% recall collapses F1 to 0).',
        formula: 'F_1 = \\frac{2}{\\frac{1}{\\text{Precision}} + \\frac{1}{\\text{Recall}}} = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}} = \\frac{2TP}{2TP + FP + FN}',
        keyTakeaway: 'The F1-Score delivers a balanced, unified benchmark when you need both high precision and high recall.',
      },
      {
        title: '4. Decision Threshold Tuning & The Precision-Recall Trade-off',
        content: 'Binary classifiers compute a continuous posterior probability P(y=1|x) in [0, 1]. Samples are labeled positive if P >= τ, where τ is the decision threshold. Lowering τ (e.g. τ = 0.30) makes the model aggressive, increasing Recall (fewer missed positives) at the expense of more False Positives. Raising τ (e.g. τ = 0.70) makes the model conservative, maximizing Precision at the cost of missing borderline cases.',
        formula: '\\hat{y}_i = \\begin{cases} 1 & \\text{if } P(y_i=1) \\ge \\tau \\\\ 0 & \\text{if } P(y_i=1) < \\tau \\end{cases}, \\quad \\tau \\in (0, 1)',
        keyTakeaway: 'Threshold τ is an active engineering hyperparameter tuned to match the operational and ethical costs of each specific domain.',
      },
    ],
    timeComplexity: 'O(N)',
    spaceComplexity: 'O(1) matrix',
    tags: ['Supervised', 'Classification', 'Confusion Matrix', 'Precision', 'Recall', 'F1-Score', 'Accuracy Paradox', 'NumPy'],
    defaultInput: 'scenario=medical, threshold=0.50',
    inputVariants: [
      { label: 'Medical (Cancer Screening)', value: 'scenario=medical, threshold=0.50' },
      { label: 'Spam Filter (Urgent Mail)', value: 'scenario=spam, threshold=0.50' },
      { label: 'Credit Fraud (Imbalance)', value: 'scenario=fraud, threshold=0.50' },
      { label: 'Balanced Baseline', value: 'scenario=balanced, threshold=0.50' },
    ],
    vi: {
      title: '03. Ma trận Nhầm lẫn & Chỉ số Đánh giá Phân loại',
      category: 'Machine Learning Cổ điển',
      badge: 'Chỉ số Đánh giá',
      summary: 'Phân rã hiệu năng phân loại thành Dương/Âm tính Thật và Giả, bóc trần Nghịch lý Độ chính xác và tối ưu hóa thỏa hiệp giữa Precision và Recall qua các ngưỡng quyết định.',
      description: 'Trong các bài toán phân loại thực tế (sàng lọc ung thư, phát hiện gian lận ngân hàng, lọc thư rác), tỷ lệ Accuracy tổng thể rất dễ gây hiểu lầm tai hại. Ma trận Nhầm lẫn (Confusion Matrix) là công cụ chẩn đoán cốt lõi lập bảng so sánh giữa nhãn thực tế và nhãn dự đoán thành 4 góc phần tư: TP, FP, FN, TN. Thay đổi ngưỡng quyết định xác suất τ cho phép ta chủ động đánh đổi giữa Báo động nhầm (Precision) và Bỏ sót ca bệnh (Recall), được cân bằng thông qua chỉ số F1-Score.',
      variants: [
        { label: 'Y tế (Sàng lọc Ung thư)', value: 'scenario=medical, threshold=0.50' },
        { label: 'Lọc Thư rác (Spam Filter)', value: 'scenario=spam, threshold=0.50' },
        { label: 'Gian lận Thẻ (Mất cân bằng)', value: 'scenario=fraud, threshold=0.50' },
        { label: 'Dữ liệu Cân bằng', value: 'scenario=balanced, threshold=0.50' },
      ],
      sections: [
        {
          title: '1. Nghịch lý Độ chính xác (Accuracy Paradox) & Dữ liệu Mất cân bằng',
          content: 'Trong các ứng dụng y tế nguy cấp hoặc phát hiện gian lận tài chính, số lượng mẫu dương tính thường rất hiếm (ví dụ 5% bệnh nhân có khối u, 95% khỏe mạnh). Một mô hình ngây thơ luôn luôn đoán Âm tính (khỏe mạnh) sẽ đạt Độ chính xác lên tới 95%, nhưng hoàn toàn bỏ sót 100% bệnh nhân ung thư (Recall = 0%)! Đây chính là Nghịch lý Độ chính xác (Accuracy Paradox). Để đánh giá đúng, ta bắt buộc phải phân tách chi tiết từng loại sai số.',
          formula: '\\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}, \\quad \\lim_{\\text{Imbalance} \\to \\infty} \\text{Accuracy} \\approx \\text{Tỷ lệ Lớp Đa số}',
          keyTakeaway: 'Accuracy không phân biệt được mức độ nghiêm trọng giữa các loại lỗi; không bao giờ dùng Accuracy làm chỉ số duy nhất trên dữ liệu mất cân bằng.',
        },
        {
          title: '2. Ma trận 2×2 & Sai lầm Loại I (FP) vs Sai lầm Loại II (FN)',
          content: 'Ma trận Nhầm lẫn đối chiếu Nhãn thực tế (dòng) với Nhãn dự đoán (cột) thành 4 góc phần tư rõ ràng: Dương tính Thật (TP - bắt trúng), Dương tính Giả (FP - Sai lầm loại I, báo động nhầm), Âm tính Giả (FN - Sai lầm loại II, bỏ sót ca bệnh), và Âm tính Thật (TN - loại trừ đúng). Trong đời thực, cái giá của sai lầm loại I và loại II không bao giờ bằng nhau: phân loại nhầm thư thành spam (FP) chỉ gây bất tiện, nhưng bỏ sót khối u ác tính (FN) có thể cướp đi sinh mạng.',
          formula: '\\begin{pmatrix} TP & FN \\\\ FP & TN \\end{pmatrix}, \\quad \\text{Sai lầm Loại I} = FP, \\quad \\text{Sai lầm Loại II} = FN',
          keyTakeaway: 'Hậu quả của Âm tính Giả (FN - bỏ sót bệnh) nguy hiểm gấp nhiều lần Dương tính Giả (FP - xét nghiệm bổ sung).',
        },
        {
          title: '3. Precision, Recall, Specificity & Điểm F1-Score Điều hòa',
          content: 'Precision (TP / (TP + FP)) trả lời: "Khi mô hình báo động, tỷ lệ thật sự đúng là bao nhiêu?" Recall hay Sensitivity (TP / (TP + FN)) trả lời: "Trong toàn bộ các ca bệnh thật sự, mô hình bắt được bao nhiêu ca?" Specificity (TN / (TN + FP)) đo lường khả năng miễn trừ người khỏe mạnh. Điểm F1-Score là trung bình điều hòa giữa Precision và Recall. Khác với trung bình cộng, trung bình điều hòa phạt cực nặng khi một trong hai chỉ số bị lệch (Precision 100% nhưng Recall 0% khiến F1 rơi thẳng về 0).',
          formula: 'F_1 = \\frac{2}{\\frac{1}{\\text{Precision}} + \\frac{1}{\\text{Recall}}} = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}} = \\frac{2TP}{2TP + FP + FN}',
          keyTakeaway: 'F1-Score là thước đo chuẩn xác nhất khi bài toán đòi hỏi sự cân bằng đồng thời giữa Precision cao và Recall cao.',
        },
        {
          title: '4. Điều chỉnh Ngưỡng Quyết định τ & Đánh đổi Precision - Recall',
          content: 'Mô hình phân loại đưa ra xác suất liên tục P(y=1|x) thuộc [0, 1]. Mẫu được gán nhãn Dương tính nếu P >= τ. Hạ ngưỡng τ (ví dụ τ = 0.30) khiến mô hình nhạy bén hơn, tối đa hóa Recall (hạn chế tối đa bỏ sót ca bệnh) nhưng phải chấp nhận thêm nhiều cảnh báo giả. Tăng ngưỡng τ (ví dụ τ = 0.70) khiến mô hình thận trọng hơn, tối đa hóa Precision (chỉ báo động khi chắc chắn) nhưng dễ bỏ sót ca ở vùng ranh giới.',
          formula: '\\hat{y}_i = \\begin{cases} 1 & \\text{nếu } P(y_i=1) \\ge \\tau \\\\ 0 & \\text{nếu } P(y_i=1) < \\tau \\end{cases}, \\quad \\tau \\in (0, 1)',
          keyTakeaway: 'Ngưỡng τ không bất di bất dịch ở 0.5; kỹ sư AI luôn tinh chỉnh τ dựa trên hàm chi phí và rủi ro thực tế của từng nghiệp vụ.',
        },
      ],
    },
    pythonCode: `import numpy as np

def compute_confusion_matrix(y_true, y_prob, threshold=0.5):
    """
    Calculates Confusion Matrix and diagnostic metrics from scratch.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    # 1. Apply decision threshold tau to obtain binary predictions
    y_pred = (y_prob >= threshold).astype(int)
    
    # 2. Compute the 4 quadrants
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    # 3. Derive key metrics
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "matrix": np.array([[tp, fn], [fp, tn]]),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "specificity": specificity, "f1": f1
    }`,
    code: `import numpy as np

def evaluate_classification(y_true, y_prob, threshold=0.5):
    # 1. Decision thresholding: y_pred in {0, 1}
    y_pred = (y_prob >= threshold).astype(int)
    
    # 2. Compute the 4 quadrants of the Confusion Matrix
    tp = np.sum((y_true == 1) & (y_pred == 1)) # Hit
    fp = np.sum((y_true == 0) & (y_pred == 1)) # Type I Error
    fn = np.sum((y_true == 1) & (y_pred == 0)) # Type II Error
    tn = np.sum((y_true == 0) & (y_pred == 0)) # Correct Rejection
    
    # 3. Derive key diagnostic metrics
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "F1": f1}`,
    generateTraces: (inputStr?: string, lang: Language = 'en'): TraceStep[] => {
      const isVi = lang === 'vi';
      let scenario: 'medical' | 'spam' | 'fraud' | 'balanced' = 'medical';
      let threshold = 0.5;

      if (inputStr) {
        const lower = inputStr.toLowerCase();
        if (lower.includes('spam')) scenario = 'spam';
        else if (lower.includes('fraud')) scenario = 'fraud';
        else if (lower.includes('balanced')) scenario = 'balanced';
        else if (lower.includes('medical')) scenario = 'medical';

        const threshMatch = inputStr.match(/threshold\s*=\s*([0-9.]+)/i);
        if (threshMatch) {
          const val = parseFloat(threshMatch[1]);
          if (!isNaN(val) && val >= 0.05 && val <= 0.95) {
            threshold = val;
          }
        }
      }

      // Generate realistic dataset samples based on scenario
      let baseSamples: { id: number; prob: number; actual: 0 | 1; label: string }[] = [];
      if (scenario === 'medical') {
        baseSamples = [
          { id: 1, prob: 0.95, actual: 1, label: isVi ? 'Khối u ác tính #1' : 'Malignant Tumor #1' },
          { id: 2, prob: 0.89, actual: 1, label: isVi ? 'Khối u ác tính #2' : 'Malignant Tumor #2' },
          { id: 3, prob: 0.84, actual: 1, label: isVi ? 'Khối u ác tính #3' : 'Malignant Tumor #3' },
          { id: 4, prob: 0.78, actual: 1, label: isVi ? 'Khối u ác tính #4' : 'Malignant Tumor #4' },
          { id: 5, prob: 0.72, actual: 1, label: isVi ? 'Khối u ác tính #5' : 'Malignant Tumor #5' },
          { id: 6, prob: 0.64, actual: 1, label: isVi ? 'Khối u ác tính #6' : 'Malignant Tumor #6' },
          { id: 7, prob: 0.55, actual: 1, label: isVi ? 'Khối u ác tính #7' : 'Malignant Tumor #7' },
          { id: 8, prob: 0.42, actual: 1, label: isVi ? 'Khối u ác tính #8 (Vùng mờ)' : 'Malignant Tumor #8 (Borderline)' },
          { id: 9, prob: 0.68, actual: 0, label: isVi ? 'U lành tính #9 (Cảnh báo nhầm)' : 'Benign Cell #9 (Inflamed)' },
          { id: 10, prob: 0.58, actual: 0, label: isVi ? 'U lành tính #10' : 'Benign Cell #10' },
          { id: 11, prob: 0.51, actual: 0, label: isVi ? 'U lành tính #11' : 'Benign Cell #11' },
          { id: 12, prob: 0.46, actual: 0, label: isVi ? 'Mô khỏe mạnh #12' : 'Healthy Tissue #12' },
          { id: 13, prob: 0.39, actual: 0, label: isVi ? 'Mô khỏe mạnh #13' : 'Healthy Tissue #13' },
          { id: 14, prob: 0.33, actual: 0, label: isVi ? 'Mô khỏe mạnh #14' : 'Healthy Tissue #14' },
          { id: 15, prob: 0.27, actual: 0, label: isVi ? 'Mô khỏe mạnh #15' : 'Healthy Tissue #15' },
          { id: 16, prob: 0.21, actual: 0, label: isVi ? 'Mô khỏe mạnh #16' : 'Healthy Tissue #16' },
          { id: 17, prob: 0.16, actual: 0, label: isVi ? 'Mô khỏe mạnh #17' : 'Healthy Tissue #17' },
          { id: 18, prob: 0.11, actual: 0, label: isVi ? 'Mô khỏe mạnh #18' : 'Healthy Tissue #18' },
          { id: 19, prob: 0.07, actual: 0, label: isVi ? 'Mô khỏe mạnh #19' : 'Healthy Tissue #19' },
          { id: 20, prob: 0.03, actual: 0, label: isVi ? 'Mô khỏe mạnh #20' : 'Healthy Tissue #20' },
        ];
      } else if (scenario === 'spam') {
        baseSamples = [
          { id: 1, prob: 0.98, actual: 1, label: isVi ? 'Lừa đảo trúng thưởng #1' : 'Phishing Lottery #1' },
          { id: 2, prob: 0.92, actual: 1, label: isVi ? 'Quảng cáo mua bán #2' : 'Spam Marketing #2' },
          { id: 3, prob: 0.87, actual: 1, label: isVi ? 'Spam bán hàng #3' : 'Spam Promo #3' },
          { id: 4, prob: 0.81, actual: 1, label: isVi ? 'Link độc hại #4' : 'Malicious Link #4' },
          { id: 5, prob: 0.74, actual: 1, label: isVi ? 'Spam tài chính #5' : 'Loan Spam #5' },
          { id: 6, prob: 0.67, actual: 1, label: isVi ? 'Bản tin spam #6' : 'Unsubscribed Blast #6' },
          { id: 7, prob: 0.59, actual: 1, label: isVi ? 'Spam khảo sát #7' : 'Survey Spam #7' },
          { id: 8, prob: 0.44, actual: 1, label: isVi ? 'Spam tinh vi #8' : 'Sneaky Spam #8' },
          { id: 9, prob: 0.76, actual: 0, label: isVi ? 'Hợp đồng đối tác #9 (Bị dính từ khóa)' : 'Client Contract #9 (Keyword Alert)' },
          { id: 10, prob: 0.66, actual: 0, label: isVi ? 'Thư mời phỏng vấn #10' : 'Job Interview Offer #10' },
          { id: 11, prob: 0.54, actual: 0, label: isVi ? 'Hóa đơn ngân hàng #11' : 'Bank Statement #11' },
          { id: 12, prob: 0.45, actual: 0, label: isVi ? 'Email dự án #12' : 'Project Update #12' },
          { id: 13, prob: 0.38, actual: 0, label: isVi ? 'Thư nội bộ #13' : 'Internal Memo #13' },
          { id: 14, prob: 0.30, actual: 0, label: isVi ? 'Báo cáo tuần #14' : 'Weekly Status #14' },
          { id: 15, prob: 0.23, actual: 0, label: isVi ? 'Email khách hàng #15' : 'Customer Query #15' },
          { id: 16, prob: 0.17, actual: 0, label: isVi ? 'Thư cá nhân #16' : 'Personal Mail #16' },
          { id: 17, prob: 0.12, actual: 0, label: isVi ? 'Email trao đổi #17' : 'Casual Thread #17' },
          { id: 18, prob: 0.08, actual: 0, label: isVi ? 'Biên nhận #18' : 'Receipt Confirmation #18' },
          { id: 19, prob: 0.04, actual: 0, label: isVi ? 'Xác thực tài khoản #19' : '2FA Code #19' },
          { id: 20, prob: 0.02, actual: 0, label: isVi ? 'Thông báo hệ thống #20' : 'System Alert #20' },
        ];
      } else if (scenario === 'fraud') {
        baseSamples = [
          { id: 1, prob: 0.96, actual: 1, label: isVi ? 'Rút tiền bất thường lúc 3h #1' : 'Abnormal 3AM Withdrawal #1' },
          { id: 2, prob: 0.88, actual: 1, label: isVi ? 'Thẻ quẹt tại nước ngoài #2' : 'Foreign IP Transaction #2' },
          { id: 3, prob: 0.48, actual: 1, label: isVi ? 'Giao dịch vi mô thử thẻ #3' : 'Micro-probe Charge #3' },
          { id: 4, prob: 0.65, actual: 0, label: isVi ? 'Mua sắm lớn mua TV #4' : 'Large Electronics Buy #4' },
          { id: 5, prob: 0.52, actual: 0, label: isVi ? 'Giao dịch du lịch #5' : 'Travel Hotel Booking #5' },
          { id: 6, prob: 0.43, actual: 0, label: isVi ? 'Thanh toán ăn uống #6' : 'Dining POS #6' },
          { id: 7, prob: 0.37, actual: 0, label: isVi ? 'Mua đồ tạp hóa #7' : 'Grocery Store #7' },
          { id: 8, prob: 0.31, actual: 0, label: isVi ? 'Đổ xăng #8' : 'Gas Station #8' },
          { id: 9, prob: 0.25, actual: 0, label: isVi ? 'Cà phê sáng #9' : 'Coffee Shop #9' },
          { id: 10, prob: 0.20, actual: 0, label: isVi ? 'Tiền điện nước #10' : 'Utility Bill #10' },
          { id: 11, prob: 0.16, actual: 0, label: isVi ? 'Thuê bao xem phim #11' : 'Streaming Sub #11' },
          { id: 12, prob: 0.13, actual: 0, label: isVi ? 'Vé tàu xe #12' : 'Transit Ticket #12' },
          { id: 13, prob: 0.10, actual: 0, label: isVi ? 'Nhà thuốc #13' : 'Pharmacy #13' },
          { id: 14, prob: 0.08, actual: 0, label: isVi ? 'Nạp tiền điện thoại #14' : 'Mobile Topup #14' },
          { id: 15, prob: 0.06, actual: 0, label: isVi ? 'Siêu thị mini #15' : 'Convenience Store #15' },
          { id: 16, prob: 0.05, actual: 0, label: isVi ? 'Mua sách #16' : 'Bookstore #16' },
          { id: 17, prob: 0.04, actual: 0, label: isVi ? 'Tiệm bánh #17' : 'Bakery #17' },
          { id: 18, prob: 0.03, actual: 0, label: isVi ? 'Rửa xe #18' : 'Car Wash #18' },
          { id: 19, prob: 0.02, actual: 0, label: isVi ? 'Tiền gửi xe #19' : 'Parking Fee #19' },
          { id: 20, prob: 0.01, actual: 0, label: isVi ? 'Vé xem phim #20' : 'Cinema Ticket #20' },
        ];
      } else {
        baseSamples = [
          { id: 1, prob: 0.94, actual: 1, label: isVi ? 'Mẫu #1' : 'Sample #1' },
          { id: 2, prob: 0.88, actual: 1, label: isVi ? 'Mẫu #2' : 'Sample #2' },
          { id: 3, prob: 0.81, actual: 1, label: isVi ? 'Mẫu #3' : 'Sample #3' },
          { id: 4, prob: 0.76, actual: 1, label: isVi ? 'Mẫu #4' : 'Sample #4' },
          { id: 5, prob: 0.70, actual: 1, label: isVi ? 'Mẫu #5' : 'Sample #5' },
          { id: 6, prob: 0.65, actual: 1, label: isVi ? 'Mẫu #6' : 'Sample #6' },
          { id: 7, prob: 0.58, actual: 1, label: isVi ? 'Mẫu #7' : 'Sample #7' },
          { id: 8, prob: 0.53, actual: 1, label: isVi ? 'Mẫu #8' : 'Sample #8' },
          { id: 9, prob: 0.44, actual: 1, label: isVi ? 'Mẫu #9' : 'Sample #9' },
          { id: 10, prob: 0.36, actual: 1, label: isVi ? 'Mẫu #10' : 'Sample #10' },
          { id: 11, prob: 0.66, actual: 0, label: isVi ? 'Mẫu #11' : 'Sample #11' },
          { id: 12, prob: 0.59, actual: 0, label: isVi ? 'Mẫu #12' : 'Sample #12' },
          { id: 13, prob: 0.52, actual: 0, label: isVi ? 'Mẫu #13' : 'Sample #13' },
          { id: 14, prob: 0.45, actual: 0, label: isVi ? 'Mẫu #14' : 'Sample #14' },
          { id: 15, prob: 0.38, actual: 0, label: isVi ? 'Mẫu #15' : 'Sample #15' },
          { id: 16, prob: 0.31, actual: 0, label: isVi ? 'Mẫu #16' : 'Sample #16' },
          { id: 17, prob: 0.24, actual: 0, label: isVi ? 'Mẫu #17' : 'Sample #17' },
          { id: 18, prob: 0.17, actual: 0, label: isVi ? 'Mẫu #18' : 'Sample #18' },
          { id: 19, prob: 0.11, actual: 0, label: isVi ? 'Mẫu #19' : 'Sample #19' },
          { id: 20, prob: 0.04, actual: 0, label: isVi ? 'Mẫu #20' : 'Sample #20' },
        ];
      }

      // Helper to compute matrix data at a given threshold
      const evaluateMatrix = (thresh: number, activeQuad: QuadrantType | null = null): ConfusionMatrixData => {
        let tp = 0;
        let fp = 0;
        let fn = 0;
        let tn = 0;

        const evaluated: ConfusionMatrixSample[] = baseSamples.map((s) => {
          const pred: 0 | 1 = s.prob >= thresh ? 1 : 0;
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

        const total = evaluated.length;
        const accuracy = total > 0 ? (tp + tn) / total : 0;
        const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
        const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
        const specificity = tn + fp > 0 ? tn / (tn + fp) : 0;
        const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

        return {
          tp,
          fp,
          fn,
          tn,
          total,
          threshold: thresh,
          accuracy,
          precision,
          recall,
          specificity,
          f1Score,
          samples: evaluated,
          scenario,
          activeQuadrant: activeQuad,
        };
      };

      const initialData = evaluateMatrix(threshold);
      const steps: TraceStep[] = [];

      // STEP 1: INIT & GROUND TRUTH
      steps.push({
        step: 1,
        totalSteps: 8,
        line: 3,
        action: 'init',
        targets: [0],
        state: {
          confusionMatrixState: initialData,
          threshold,
          totalSamples: initialData.total,
        },
        explanation: isVi
          ? `[Khởi tạo Dữ liệu]: Nạp ${initialData.total} mẫu thực tế trong kịch bản "${scenario.toUpperCase()}". Đã có ${baseSamples.filter((s) => s.actual === 1).length} ca Dương tính (y=1) và ${baseSamples.filter((s) => s.actual === 0).length} ca Âm tính (y=0).`
          : `[Data Initialization]: Loaded ${initialData.total} ground truth samples for the "${scenario.toUpperCase()}" scenario (${baseSamples.filter((s) => s.actual === 1).length} positive, ${baseSamples.filter((s) => s.actual === 0).length} negative).`,
        invariant: isVi
          ? 'Tập dữ liệu phản ánh phân phối thực tế của bài toán phân loại.'
          : 'Dataset reflects authentic class distribution of the operational domain.',
      });

      // STEP 2: THRESHOLDING RULE
      steps.push({
        step: 2,
        totalSteps: 8,
        line: 5,
        action: 'forward',
        targets: [1],
        state: {
          confusionMatrixState: initialData,
          threshold,
        },
        explanation: isVi
          ? `[Ngưỡng Quyết định τ = ${threshold.toFixed(2)}]: Mô hình gán nhãn ŷ = 1 cho các mẫu có xác suất P(y=1) ≥ ${threshold.toFixed(2)}, ngược lại gán nhãn ŷ = 0. Đường ranh giới chia không gian thành 2 vùng rõ rệt.`
          : `[Decision Threshold τ = ${threshold.toFixed(2)}]: The model classifies samples with P(y=1) >= ${threshold.toFixed(2)} as positive (ŷ=1), and all others as negative (ŷ=0).`,
        invariant: isVi
          ? `Quy tắc quyết định: ŷ_i = (p_i ≥ ${threshold.toFixed(2)}) ? 1 : 0.`
          : `Decision rule: ŷ_i = (p_i >= ${threshold.toFixed(2)}) ? 1 : 0.`,
      });

      // STEP 3: COUNT TRUE POSITIVES (TP)
      const tpData = evaluateMatrix(threshold, 'tp');
      steps.push({
        step: 3,
        totalSteps: 8,
        line: 8,
        action: 'compute',
        targets: [2],
        state: {
          confusionMatrixState: tpData,
          tp: tpData.tp,
        },
        explanation: isVi
          ? `[Dương tính Thật (TP)]: Phát hiện đúng ${tpData.tp} trường hợp (Thực tế y=1 VÀ Dự đoán ŷ=1). Đây là các ca "bắt trúng đích" mang lại giá trị cao nhất cho hệ thống.`
          : `[True Positives (TP)]: Identified ${tpData.tp} correct positive detections (Actual y=1 AND Predicted ŷ=1). These are the successful "hits".`,
        invariant: isVi
          ? `TP = ${tpData.tp}: Đại diện cho số lượng ca bệnh / ca gian lận đã được phát hiện thành công.`
          : `TP = ${tpData.tp}: True hits where ground truth positive is correctly captured.`,
      });

      // STEP 4: COUNT FALSE POSITIVES (FP - Type I Error)
      const fpData = evaluateMatrix(threshold, 'fp');
      steps.push({
        step: 4,
        totalSteps: 8,
        line: 9,
        action: 'compute',
        targets: [3],
        state: {
          confusionMatrixState: fpData,
          fp: fpData.fp,
        },
        explanation: isVi
          ? `[Dương tính Giả (FP - Sai lầm loại I)]: Có ${fpData.fp} trường hợp báo động nhầm (Thực tế y=0 nhưng Dự đoán ŷ=1). Làm phiền người dùng hoặc tốn chi phí xét nghiệm lại.`
          : `[False Positives (FP - Type I Error)]: Model generated ${fpData.fp} false alarms (Actual y=0 but Predicted ŷ=1), requiring secondary verification.`,
        invariant: isVi
          ? `FP = ${fpData.fp}: Đo lường mức độ báo động giả. Precision giảm khi FP tăng.`
          : `FP = ${fpData.fp}: False alarm rate directly degrades model Precision.`,
      });

      // STEP 5: COUNT FALSE NEGATIVES (FN - Type II Error)
      const fnData = evaluateMatrix(threshold, 'fn');
      steps.push({
        step: 5,
        totalSteps: 8,
        line: 10,
        action: 'compute',
        targets: [4],
        state: {
          confusionMatrixState: fnData,
          fn: fnData.fn,
        },
        explanation: isVi
          ? `[Âm tính Giả (FN - Sai lầm loại II)]: Bỏ sót nguy hiểm ${fnData.fn} trường hợp (Thực tế y=1 nhưng Dự đoán ŷ=0). Trong y tế, đây là bệnh nhân ung thư bị chẩn đoán nhầm là khỏe mạnh!`
          : `[False Negatives (FN - Type II Error)]: Missed ${fnData.fn} critical positive cases (Actual y=1 but Predicted ŷ=0). In medicine, this sends an untreated cancer patient home!`,
        invariant: isVi
          ? `FN = ${fnData.fn}: Sai số nguy hiểm nhất trong chẩn đoán y tế. Recall giảm khi FN tăng.`
          : `FN = ${fnData.fn}: Missed detection risk directly degrades model Recall.`,
      });

      // STEP 6: COUNT TRUE NEGATIVES (TN)
      const tnData = evaluateMatrix(threshold, 'tn');
      steps.push({
        step: 6,
        totalSteps: 8,
        line: 11,
        action: 'compute',
        targets: [5],
        state: {
          confusionMatrixState: tnData,
          tn: tnData.tn,
        },
        explanation: isVi
          ? `[Âm tính Thật (TN)]: Xác nhận loại trừ chính xác ${tnData.tn} trường hợp (Thực tế y=0 VÀ Dự đoán ŷ=0). Người khỏe mạnh được yên tâm, giao dịch bình thường được thông suốt.`
          : `[True Negatives (TN)]: Correctly rejected ${tnData.tn} negative samples (Actual y=0 AND Predicted ŷ=0), allowing routine operations to proceed safely.`,
        invariant: isVi
          ? `TN = ${tnData.tn}: Phản ánh độ đặc hiệu (Specificity = ${(tnData.specificity * 100).toFixed(1)}%).`
          : `TN = ${tnData.tn}: Correct clearance reflecting Specificity = ${(tnData.specificity * 100).toFixed(1)}%.`,
      });

      // STEP 7: THRESHOLD SHIFT & TRADE-OFF SIMULATION
      const tunedThreshold = scenario === 'medical' ? 0.3 : 0.7;
      const tunedData = evaluateMatrix(tunedThreshold, null);
      steps.push({
        step: 7,
        totalSteps: 8,
        line: 5,
        action: 'update',
        targets: [1, 4],
        state: {
          confusionMatrixState: tunedData,
          threshold: tunedThreshold,
        },
        explanation: isVi
          ? scenario === 'medical'
            ? `[Thỏa hiệp Precision - Recall]: Hạ ngưỡng τ xuống 0.30 giúp giảm Âm tính giả FN từ ${initialData.fn} xuống còn ${tunedData.fn}, đẩy Recall tăng vọt lên ${(tunedData.recall * 100).toFixed(1)}%! Bệnh nhân được cứu sống dù phải chấp nhận thêm vài ca cảnh báo nhầm FP.`
            : `[Thỏa hiệp Precision - Recall]: Tăng ngưỡng τ lên 0.70 giúp giảm Dương tính giả FP từ ${initialData.fp} xuống còn ${tunedData.fp}, đẩy Precision tăng lên ${(tunedData.precision * 100).toFixed(1)}%! Hạn chế tối đa việc xóa nhầm thư quan trọng.`
          : scenario === 'medical'
          ? `[Precision-Recall Trade-off]: Lowering threshold to τ = 0.30 slashes False Negatives from ${initialData.fn} down to ${tunedData.fn}, driving Recall up to ${(tunedData.recall * 100).toFixed(1)}%! We prioritize saving lives.`
          : `[Precision-Recall Trade-off]: Raising threshold to τ = 0.70 slashes False Positives from ${initialData.fp} down to ${tunedData.fp}, boosting Precision to ${(tunedData.precision * 100).toFixed(1)}%! Urgent emails are protected.`,
        invariant: isVi
          ? 'Ngưỡng τ điều khiển con lắc giữa Độ chuẩn xác (Precision) và Độ bao phủ (Recall).'
          : 'Threshold τ calibrates the fundamental trade-off between Precision and Recall.',
      });

      // STEP 8: FINAL METRIC SYNTHESIS & HARMONIC F1
      steps.push({
        step: 8,
        totalSteps: 8,
        line: 17,
        action: 'done',
        targets: [0, 1, 2, 3, 4, 5],
        state: {
          confusionMatrixState: initialData,
          accuracy: initialData.accuracy,
          precision: initialData.precision,
          recall: initialData.recall,
          f1Score: initialData.f1Score,
        },
        explanation: isVi
          ? `[Tổng kết Chỉ số Toàn diện]: Accuracy = ${(initialData.accuracy * 100).toFixed(1)}%, Precision = ${(initialData.precision * 100).toFixed(1)}%, Recall = ${(initialData.recall * 100).toFixed(1)}%, Điểm F1 = ${initialData.f1Score.toFixed(3)}. Bạn có thể tự do kéo thanh trượt Ngưỡng τ trên đồ thị để kiểm chứng sự thay đổi tức thì!`
          : `[Full Metric Synthesis]: Accuracy = ${(initialData.accuracy * 100).toFixed(1)}%, Precision = ${(initialData.precision * 100).toFixed(1)}%, Recall = ${(initialData.recall * 100).toFixed(1)}%, F1-Score = ${initialData.f1Score.toFixed(3)}. You can drag the interactive Threshold slider to see real-time recalculations!`,
        invariant: isVi
          ? `F1 = 2·(P·R)/(P+R) = ${initialData.f1Score.toFixed(3)}: Trung bình điều hòa cân bằng hoàn hảo giữa Precision và Recall.`
          : `F1 = 2·(P·R)/(P+R) = ${initialData.f1Score.toFixed(3)}: Harmonic mean harmonizing Precision and Recall.`,
      });

      const total = steps.length;
      return steps.map((s, idx) => ({ ...s, step: idx + 1, totalSteps: total }));
    },
  },

  // ==========================================
  // Lesson 04: K-Means Clustering & Lloyd's Algorithm
  // ==========================================
  {
    id: 'kmeans-clustering',
    title: "04. K-Means Clustering & Lloyd's Algorithm",
    category: 'Classical Machine Learning',
    difficulty: 'Beginner',
    badge: 'Unsupervised Learning',
    visualType: 'kmeans-clustering',
    formula: 'J = \\sum_{k=1}^K \\sum_{x \\in S_k} \\|x - \\mu_k\\|^2, \\quad \\mu_k = \\frac{1}{|S_k|} \\sum_{x \\in S_k} x',
    summary: 'Partitions unlabelled continuous data into K distinct clusters by iteratively assigning points to the closest centroid and recalculating cluster centers.',
    description: 'K-Means is the foundational unsupervised clustering algorithm (Lloyd, 1982). Operating without target labels y, it discovers latent group structures by minimizing Within-Cluster Sum of Squares (WCSS/Inertia). In step 1 (Assignment), points are assigned to their nearest centroid creating Voronoi cells. In step 2 (Update), centroids move to the center of mass of their assigned points. The process repeats until convergence.',
    sections: [
      {
        title: '1. The Core Objective: Minimizing WCSS / Inertia',
        content: 'Unlike supervised learning with target labels y, K-Means groups unlabelled data points into K partitions S = {S_1, S_2, ..., S_K}. The objective is to minimize the Within-Cluster Sum of Squares (WCSS), also known as Inertia J. WCSS measures cluster compactness by summing the squared Euclidean distances from every point to its assigned centroid.',
        formula: 'J(S, \\mu) = \\sum_{k=1}^K \\sum_{x \\in S_k} \\|x - \\mu_k\\|^2',
        keyTakeaway: 'Minimizing WCSS is an NP-hard problem; Lloyd’s algorithm is a greedy coordinate descent that converges to a local minimum.',
      },
      {
        title: "2. Lloyd's 2-Step Alternating Optimization",
        content: 'Stuart Lloyd (1982) formulated the standard K-Means algorithm as an alternating Expectation-Maximization (EM) optimization. Step 1 (Assignment / E-Step): With centroids fixed, each point is assigned to its nearest centroid, partitioning the space into Voronoi polyhedra. Step 2 (Update / M-Step): With cluster assignments fixed, each centroid is recomputed as the arithmetic mean of its assigned points.',
        formula: 'c_i^{(t)} = \\arg\\min_{k} \\|x_i - \\mu_k^{(t)}\\|^2, \\quad \\mu_k^{(t+1)} = \\frac{1}{|S_k^{(t)}|} \\sum_{x \\in S_k^{(t)}} x',
        keyTakeaway: 'Both steps are guaranteed to monotonically decrease or maintain WCSS. Since there are finite partitions, convergence is guaranteed!',
      },
      {
        title: '3. Initialization Sensitivity & K-Means++',
        content: 'Random initialization often gets trapped in poor local optima (e.g., splitting a single dense cluster or stranding empty centroids). David Arthur & Sergei Vassilvitskii (2007) introduced K-Means++: the first centroid is chosen uniformly at random, and each subsequent centroid is sampled with probability proportional to the squared distance D(x)^2 to the nearest existing centroid.',
        formula: 'P(x) = \\frac{D(x)^2}{\\sum_{x\' \\in X} D(x\')^2}',
        keyTakeaway: 'K-Means++ guarantees an expected approximation ratio of O(log K) to the globally optimal WCSS, dramatically speeding up convergence.',
      },
      {
        title: '4. Voronoi Partitions & Vector Database Indexing (IVF)',
        content: 'K-Means assumes isotropic (spherical) clusters with equal variance, producing linear Voronoi decision boundaries. Today, K-Means is essential for modern AI: Inverted File Flat (IVF) indexers in vector databases (FAISS, Milvus, Pinecone) cluster high-dimensional embedding vectors into Voronoi cells, reducing nearest-neighbor search from O(N) to O(N / K).',
        formula: 'V(\\mu_k) = \\left\\{ x \\in \\mathbb{R}^d \\;\\middle|\\; \\|x - \\mu_k\\| \\le \\|x - \\mu_j\\|, \\; \\forall j \\neq k \\right\\}',
        keyTakeaway: 'K-Means serves as the foundational vector quantization (VQ) clustering mechanism in LLM retrieval-augmented generation (RAG).',
      },
    ],
    timeComplexity: 'O(T · K · N · D)',
    spaceComplexity: 'O(N · D + K · D)',
    tags: ['Unsupervised', 'Clustering', 'Lloyd Algorithm', 'WCSS', 'Inertia', 'Voronoi', 'NumPy', 'FAISS'],
    defaultInput: '(2.0, 2.5), (2.5, 3.5), (3.0, 2.0), (7.0, 7.5), (7.5, 8.5), (8.0, 7.0), (3.0, 7.5), (2.5, 8.5), (3.5, 8.0)',
    inputVariants: [
      {
        label: '3 Blobs (Separable)',
        value: '(2.0, 2.5), (2.5, 3.5), (3.0, 2.0), (7.0, 7.5), (7.5, 8.5), (8.0, 7.0), (3.0, 7.5), (2.5, 8.5), (3.5, 8.0)',
      },
      {
        label: '4 Corners (Symmetric)',
        value: '(2.0, 2.0), (2.5, 3.0), (3.0, 2.2), (2.0, 8.0), (2.5, 7.5), (3.0, 8.2), (8.0, 2.0), (7.5, 2.8), (8.2, 3.0), (8.0, 8.0), (7.5, 8.5), (8.5, 7.8)',
      },
      {
        label: 'Unequal Variance',
        value: '(2.0, 2.0), (2.2, 2.3), (2.4, 2.1), (2.1, 2.4), (7.0, 6.0), (8.5, 8.0), (6.5, 8.5), (8.0, 6.5), (4.0, 7.0), (3.0, 8.5), (4.5, 8.0)',
      },
      {
        label: 'Noisy Overlap',
        value: '(3.0, 4.0), (3.5, 4.5), (4.0, 4.2), (4.5, 5.0), (5.0, 4.8), (5.5, 5.5), (6.0, 5.2), (6.5, 6.0), (7.0, 5.8), (7.5, 6.5)',
      },
    ],
    vi: {
      title: '04. Phân cụm K-Means & Thuật toán Lloyd',
      category: 'Machine Learning Cổ điển',
      badge: 'Học Không Giám Sát',
      summary: 'Phân nhóm dữ liệu liên tục không nhãn vào K cụm bằng cách tối thiểu hóa tổng bình phương khoảng cách tới các trọng tâm cụm.',
      description: 'K-Means là thuật toán học không giám sát kinh điển nhất trong Machine Learning (Stuart Lloyd, 1982). Hoạt động hoàn toàn không cần nhãn đích y, thuật toán tự động khám phá cấu trúc ẩn bằng cách tối thiểu hóa tổng bình phương khoảng cách nội cụm (WCSS/Inertia). Bước 1 (Gán cụm): gán từng điểm về tâm cụm gần nhất tạo thành các phân hoạch Voronoi. Bước 2 (Cập nhật): di chuyển tâm cụm về trọng tâm hình học của các điểm vừa gán. Quá trình lặp lại tuần hoàn cho đến khi hội tụ.',
      variants: [
        {
          label: '3 Cụm Phân biệt',
          value: '(2.0, 2.5), (2.5, 3.5), (3.0, 2.0), (7.0, 7.5), (7.5, 8.5), (8.0, 7.0), (3.0, 7.5), (2.5, 8.5), (3.5, 8.0)',
        },
        {
          label: '4 Góc Đối xứng',
          value: '(2.0, 2.0), (2.5, 3.0), (3.0, 2.2), (2.0, 8.0), (2.5, 7.5), (3.0, 8.2), (8.0, 2.0), (7.5, 2.8), (8.2, 3.0), (8.0, 8.0), (7.5, 8.5), (8.5, 7.8)',
        },
        {
          label: 'Độ Biến thiên Khác nhau',
          value: '(2.0, 2.0), (2.2, 2.3), (2.4, 2.1), (2.1, 2.4), (7.0, 6.0), (8.5, 8.0), (6.5, 8.5), (8.0, 6.5), (4.0, 7.0), (3.0, 8.5), (4.5, 8.0)',
        },
        {
          label: 'Đan xen Nhiễu',
          value: '(3.0, 4.0), (3.5, 4.5), (4.0, 4.2), (4.5, 5.0), (5.0, 4.8), (5.5, 5.5), (6.0, 5.2), (6.5, 6.0), (7.0, 5.8), (7.5, 6.5)',
        },
      ],
      sections: [
        {
          title: '1. Mục tiêu Cốt lõi: Cực tiểu hóa WCSS / Quán tính Inertia',
          content: 'Khác với học có giám sát có nhãn mục tiêu y, K-Means gom N điểm dữ liệu không nhãn vào K tập phân hoạch rời rạc S = {S_1, S_2, ..., S_K}. Mục tiêu là cực tiểu hóa hàm mất mát WCSS (Within-Cluster Sum of Squares), hay còn gọi là Quán tính Inertia J. WCSS đo lường độ phân tán nội cụm bằng tổng bình phương khoảng cách Euclid từ mỗi điểm tới tâm cụm tương ứng.',
          formula: 'J(S, \\mu) = \\sum_{k=1}^K \\sum_{x \\in S_k} \\|x - \\mu_k\\|^2',
          keyTakeaway: 'Tìm nghiệm tối ưu toàn cục cho WCSS là bài toán NP-hard; thuật toán Lloyd giải gần đúng bằng phương pháp hạ tọa độ tham lam hội tụ về cực tiểu cục bộ.',
        },
        {
          title: '2. Tối ưu Hóa Xen kẽ 2 Bước của Lloyd (Assign & Update)',
          content: 'Stuart Lloyd (1982) đề xuất thuật toán tối ưu xen kẽ tương tự nguyên lý Expectation-Maximization (EM). Bước 1 (Gán cụm - Assignment): Giữ cố định K tâm cụm, gán mỗi điểm dữ liệu vào tâm cụm gần nó nhất theo khoảng cách Euclid, phân chia không gian thành các ô đa diện Voronoi. Bước 2 (Cập nhật - Update): Giữ cố định phân cụm, tính lại tọa độ mỗi tâm cụm bằng trung bình cộng tọa độ các điểm thuộc cụm đó.',
          formula: 'c_i^{(t)} = \\arg\\min_{k} \\|x_i - \\mu_k^{(t)}\\|^2, \\quad \\mu_k^{(t+1)} = \\frac{1}{|S_k^{(t)}|} \\sum_{x \\in S_k^{(t)}} x',
          keyTakeaway: 'Cả hai bước đều đảm bảo WCSS chỉ giảm hoặc giữ nguyên (đơn điệu). Do số cách phân hoạch là hữu hạn, thuật toán chắc chắn sẽ dừng và hội tụ!',
        },
        {
          title: '3. Độ Nhạy với Khởi tạo & Cải tiến K-Means++',
          content: 'Khởi tạo ngẫu nhiên đơn thuần rất dễ rơi vào bẫy cực tiểu cục bộ kém chất lượng (ví dụ chia cắt một cụm tự nhiên hoặc tạo ra tâm cụm rỗng). David Arthur và Sergei Vassilvitskii (2007) phát minh thuật toán K-Means++: tâm đầu tiên chọn ngẫu nhiên đều, các tâm tiếp theo được chọn với xác suất tỷ lệ thuận với bình phương khoảng cách D(x)^2 tới tâm gần nhất đã có.',
          formula: 'P(x) = \\frac{D(x)^2}{\\sum_{x\' \\in X} D(x\')^2}',
          keyTakeaway: 'K-Means++ chứng minh được kỳ vọng sai số WCSS đạt giới hạn O(log K) so với nghiệm tối ưu lý thuyết, giúp tăng tốc độ hội tụ rõ rệt.',
        },
        {
          title: '4. Phân hoạch Voronoi & Chỉ mục Tìm kiếm Vector (IVF) cho LLM',
          content: 'K-Means ngầm giả định các cụm có dạng hình cầu đối xứng (đẳng hướng) và phương sai bằng nhau, tạo ra các ranh giới quyết định tuyến tính Voronoi. Trong kỷ nguyên Generative AI, K-Means là trái tim của cấu trúc chỉ mục IVF (Inverted File Index) trong cơ sở dữ liệu vector (FAISS, Milvus, Pinecone), phân cụm hàng tỷ vector nhúng thành các ô Voronoi để giảm độ phức tạp tìm kiếm từ O(N) xuống O(N/K).',
          formula: 'V(\\mu_k) = \\left\\{ x \\in \\mathbb{R}^d \\;\\middle|\\; \\|x - \\mu_k\\| \\le \\|x - \\mu_j\\|, \\; \\forall j \\neq k \\right\\}',
          keyTakeaway: 'K-Means là kỹ thuật lượng tử hóa vector (Vector Quantization) cốt lõi hỗ trợ truy xuất ngữ nghĩa siêu tốc trong các hệ thống RAG hiện đại.',
        },
      ],
    },
    pythonCode: `import numpy as np

def kmeans(X, k=3, max_iters=10, tol=1e-4):
    # 1. K-Means++ Centroid Seeding
    n_samples, n_features = X.shape
    centroids = np.empty((k, n_features))
    centroids[0] = X[np.random.randint(n_samples)]
    
    for c_idx in range(1, k):
        dist_sq = np.min([np.sum((X - c)**2, axis=1) for c in centroids[:c_idx]], axis=0)
        probs = dist_sq / np.sum(dist_sq)
        centroids[c_idx] = X[np.random.choice(n_samples, p=probs)]
        
    for iteration in range(max_iters):
        # 2. Assignment Step: assign points to closest centroid
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)
        
        # 3. Update Step: compute center of mass
        new_centroids = np.array([
            X[clusters == j].mean(axis=0) if np.any(clusters == j) else centroids[j]
            for j in range(k)
        ])
        
        # 4. Check Convergence: centroid shift < tol
        shift = np.sum(np.linalg.norm(new_centroids - centroids, axis=1))
        if shift < tol:
            break
        centroids = new_centroids
        
    inertia = np.sum((X - centroids[clusters]) ** 2)
    return centroids, clusters, inertia`,
    code: `import numpy as np

def kmeans(X, k=3, max_iters=10):
    # 1. Initialize k centroids
    centroids = X[np.random.choice(len(X), k, replace=False)].copy()
    for iteration in range(max_iters):
        # 2. Assignment Step: Voronoi nearest centroid
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)
        # 3. Update Step: compute cluster center of mass
        new_centroids = np.array([X[clusters == j].mean(axis=0) for j in range(k)])
        # 4. Check convergence: centroid shift < tol
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids
    inertia = np.sum((X - centroids[clusters]) ** 2)
    return centroids, clusters, inertia`,
    generateTraces: (inputStr?: string, lang: Language = 'en'): TraceStep[] => {
      const isVi = lang === 'vi';
      const CENTROID_COLORS = ['#06b6d4', '#f59e0b', '#f43f5e', '#10b981', '#8b5cf6'];

      let rawPoints: { x: number; y: number }[] = [
        { x: 2.0, y: 2.5 },
        { x: 2.5, y: 3.5 },
        { x: 3.0, y: 2.0 },
        { x: 7.0, y: 7.5 },
        { x: 7.5, y: 8.5 },
        { x: 8.0, y: 7.0 },
        { x: 3.0, y: 7.5 },
        { x: 2.5, y: 8.5 },
        { x: 3.5, y: 8.0 },
      ];

      if (inputStr) {
        const matches = [...inputStr.matchAll(/\(?\s*([0-9.]+)\s*[,:]\s*([0-9.]+)\s*\)?/g)];
        if (matches.length >= 3) {
          rawPoints = matches.map((m) => ({ x: Number(m[1]), y: Number(m[2]) }));
        }
      }

      const numPoints = rawPoints.length;
      let k = 3;
      if (inputStr && (inputStr.includes('4 Corners') || inputStr.includes('k=4') || numPoints >= 12)) {
        k = 4;
      }
      k = Math.min(k, numPoints);

      // Deterministic initial centroid seeds
      let curCentroids: KMeansCentroid[] = [];
      if (k === 3) {
        curCentroids = [
          { id: 0, x: 2.2, y: 2.8, color: CENTROID_COLORS[0], count: 0 },
          { id: 1, x: 7.2, y: 7.2, color: CENTROID_COLORS[1], count: 0 },
          { id: 2, x: 3.2, y: 7.2, color: CENTROID_COLORS[2], count: 0 },
        ];
      } else if (k === 4) {
        curCentroids = [
          { id: 0, x: 2.5, y: 2.5, color: CENTROID_COLORS[0], count: 0 },
          { id: 1, x: 2.5, y: 8.0, color: CENTROID_COLORS[1], count: 0 },
          { id: 2, x: 7.8, y: 2.6, color: CENTROID_COLORS[2], count: 0 },
          { id: 3, x: 7.8, y: 8.0, color: CENTROID_COLORS[3], count: 0 },
        ];
      } else {
        curCentroids = Array.from({ length: k }, (_, idx) => {
          const ptIdx = Math.floor((idx * numPoints) / k);
          return {
            id: idx,
            x: rawPoints[ptIdx].x,
            y: rawPoints[ptIdx].y,
            color: CENTROID_COLORS[idx % CENTROID_COLORS.length],
            count: 0,
          };
        });
      }

      let curPoints: KMeansPoint[] = rawPoints.map((pt, idx) => ({
        id: idx + 1,
        x: pt.x,
        y: pt.y,
        cluster: -1,
      }));

      const steps: TraceStep[] = [];

      // Helper to compute assignment, distances and inertia
      const assignPoints = (pts: KMeansPoint[], cents: KMeansCentroid[]) => {
        let inertia = 0;
        const counts = new Array(cents.length).fill(0);
        const assigned = pts.map((p) => {
          let bestDist = Infinity;
          let bestCluster = 0;
          const dists = cents.map((c, cIdx) => {
            const d = Math.hypot(p.x - c.x, p.y - c.y);
            if (d < bestDist) {
              bestDist = d;
              bestCluster = cIdx;
            }
            return Number(d.toFixed(3));
          });

          inertia += bestDist * bestDist;
          counts[bestCluster]++;
          return {
            ...p,
            cluster: bestCluster,
            distanceToCentroid: Number(bestDist.toFixed(3)),
            distances: dists,
          };
        });

        const updatedCentroids = cents.map((c, i) => ({ ...c, count: counts[i] }));
        return { assigned, updatedCentroids, inertia: Number(inertia.toFixed(2)) };
      };

      // Step 0: Initialization
      const initCalc = assignPoints(curPoints, curCentroids);
      steps.push({
        step: 0,
        totalSteps: 0,
        line: 5,
        action: 'init',
        targets: curCentroids.map((c) => c.id),
        state: {
          loss: initCalc.inertia,
          kmeansState: {
            k,
            iteration: 0,
            maxIterations: 6,
            phase: 'init',
            points: curPoints, // Initially unassigned
            centroids: curCentroids,
            inertia: initCalc.inertia,
            centroidShift: 0,
            converged: false,
          },
        },
        explanation: isVi
          ? `[Khởi tạo K-Means]: Khởi tạo K = ${k} tâm cụm ngẫu nhiên (Seeds) phân bố trong không gian đặc trưng. Toàn bộ ${numPoints} điểm dữ liệu ban đầu đang ở trạng thái chưa phân cụm (unassigned).`
          : `[K-Means Initialization]: Seeded K = ${k} initial centroids across the feature space. All ${numPoints} input data points are currently unassigned.`,
        invariant: isVi
          ? `K = ${k}, \\quad \\mu_k \\in \\mathbb{R}^2: \\text{Khởi tạo K tâm cụm ban đầu}`
          : `K = ${k}, \\quad \\mu_k \\in \\mathbb{R}^2: \\text{K initial cluster centroids seeded}`,
      });

      let maxIters = 4;
      let prevInertia = initCalc.inertia;

      for (let iter = 1; iter <= maxIters; iter++) {
        // Step A: Assignment Step (Voronoi nearest centroid)
        const assignResult = assignPoints(curPoints, curCentroids);
        curPoints = assignResult.assigned;
        curCentroids = assignResult.updatedCentroids;
        const curInertia = assignResult.inertia;

        steps.push({
          step: 0,
          totalSteps: 0,
          line: 8,
          action: 'compute',
          targets: curPoints.map((p) => p.id),
          state: {
            loss: curInertia,
            kmeansState: {
              k,
              iteration: iter,
              maxIterations: maxIters,
              phase: 'assign',
              points: curPoints,
              centroids: curCentroids,
              inertia: curInertia,
              prevInertia,
              centroidShift: 0,
              converged: false,
            },
          },
          explanation: isVi
            ? `[Vòng lặp ${iter} - Gán Cụm Voronoi]: Tính khoảng cách Euclid từ mỗi điểm tới ${k} tâm cụm. Gán mỗi điểm vào tâm cụm gần nhất c_i = argmin ||x_i - μ_k||², hình thành các ranh giới ô Voronoi. Tổng quán tính WCSS J = ${curInertia.toFixed(2)}.`
            : `[Iteration ${iter} - Assignment Step]: Calculated Euclidean distances to all ${k} centroids. Assigned each sample to its nearest centroid c_i = argmin ||x_i - μ_k||², yielding Voronoi cells. Current WCSS Inertia J = ${curInertia.toFixed(2)}.`,
          invariant: `c_i^{(t)} = \\arg\\min_{k} \\|x_i - \\mu_k^{(t)}\\|^2, \\quad J = ${curInertia.toFixed(2)}`,
        });

        // Step B: Centroid Update Step (Center of mass)
        let totalShift = 0;
        const nextCentroids: KMeansCentroid[] = curCentroids.map((c, cIdx) => {
          const clusterPoints = curPoints.filter((p) => p.cluster === cIdx);
          if (clusterPoints.length === 0) return c;
          const avgX = Number((clusterPoints.reduce((acc, p) => acc + p.x, 0) / clusterPoints.length).toFixed(2));
          const avgY = Number((clusterPoints.reduce((acc, p) => acc + p.y, 0) / clusterPoints.length).toFixed(2));
          const shift = Math.hypot(avgX - c.x, avgY - c.y);
          totalShift += shift;
          return {
            ...c,
            prevX: c.x,
            prevY: c.y,
            x: avgX,
            y: avgY,
            count: clusterPoints.length,
          };
        });

        totalShift = Number(totalShift.toFixed(3));
        const updatedAssign = assignPoints(curPoints, nextCentroids);
        curCentroids = nextCentroids;
        prevInertia = curInertia;
        const newInertia = updatedAssign.inertia;

        steps.push({
          step: 0,
          totalSteps: 0,
          line: 10,
          action: 'update',
          targets: curCentroids.map((c) => c.id),
          state: {
            loss: newInertia,
            kmeansState: {
              k,
              iteration: iter,
              maxIterations: maxIters,
              phase: 'update',
              points: curPoints,
              centroids: curCentroids,
              inertia: newInertia,
              prevInertia: curInertia,
              centroidShift: totalShift,
              converged: false,
            },
          },
          explanation: isVi
            ? `[Vòng lặp ${iter} - Cập nhật Trọng tâm]: Tái tính tọa độ ${k} tâm cụm bằng trung bình cộng tọa độ các điểm thuộc cụm (μ_k = 1/|S_k| Σ x). Tổng độ dịch chuyển tâm Δμ = ${totalShift.toFixed(2)}. Quán tính WCSS giảm xuống ${newInertia.toFixed(2)}.`
            : `[Iteration ${iter} - Centroid Update]: Shifted centroids to the geometric center of mass (μ_k = 1/|S_k| Σ x). Total centroid displacement Δμ = ${totalShift.toFixed(2)}. Inertia lowered to ${newInertia.toFixed(2)}.`,
          invariant: `\\mu_k^{(t+1)} = \\frac{1}{|S_k|} \\sum_{x \\in S_k} x, \\quad \\Delta \\mu = ${totalShift.toFixed(2)}`,
        });

        if (totalShift < 0.05 || iter === maxIters) {
          break;
        }
      }

      // Step Final: Convergence
      const finalResult = assignPoints(curPoints, curCentroids);
      steps.push({
        step: 0,
        totalSteps: 0,
        line: 14,
        action: 'done',
        targets: curCentroids.map((c) => c.id),
        state: {
          loss: finalResult.inertia,
          kmeansState: {
            k,
            iteration: steps.length / 2,
            maxIterations: maxIters,
            phase: 'converged',
            points: finalResult.assigned,
            centroids: finalResult.updatedCentroids,
            inertia: finalResult.inertia,
            centroidShift: 0,
            converged: true,
          },
        },
        explanation: isVi
          ? `[Hội tụ Thành công!]: Thuật toán K-Means đã hội tụ hoàn toàn. Vị trí các tâm cụm đã ổn định (Δμ < 10⁻⁴), phân cụm không còn thay đổi. Cấu trúc phân hoạch Voronoi đạt trạng thái cân bằng với Quán tính WCSS tối ưu J = ${finalResult.inertia.toFixed(2)}!`
          : `[Convergence Achieved!]: K-Means has fully converged. Centroids are stabilized (Δμ < 10⁻⁴) and cluster assignments remain invariant. Voronoi partitioning is at equilibrium with optimal WCSS J = ${finalResult.inertia.toFixed(2)}!`,
        invariant: `\\Delta \\mu \\to 0, \\quad J^* = ${finalResult.inertia.toFixed(2)} \\quad (\\text{Converged})`,
      });

      const total = steps.length;
      return steps.map((s, idx) => ({ ...s, step: idx + 1, totalSteps: total }));
    },
  },

  // ==========================================
  // Lesson 05: Gradient Descent with Momentum
  // ==========================================
  {
    id: 'momentum-optimizer',
    title: '05. Gradient Descent with Momentum (Accelerated Optimization)',
    category: 'Classical Machine Learning',
    difficulty: 'Intermediate',
    badge: 'Optimizer',
    visualType: 'momentum-optimizer',
    formula: 'v_t = \\beta v_{t-1} + \\alpha \\nabla L(\\theta_t), \\quad \\theta_{t+1} = \\theta_t - v_t',
    summary: 'Accelerates gradient descent and suppresses ravine oscillations by modeling physical inertia via an exponential moving average of velocity.',
    description: 'Vanilla Gradient Descent struggles severely in ravines—surfaces where curvature is far steeper in one direction than another—causing wild, inefficient oscillations. Boris Polyak (1964) introduced the Heavy-Ball Method (Momentum), which simulates a heavy physical particle with mass and inertia. Past gradients compound with friction decay β into a velocity vector v_t, maintaining velocity down shallow valley floors while canceling perpendicular vibrations.',
    sections: [
      {
        title: '1. The Ravine Problem & Condition Number',
        content: 'In non-spherical loss landscapes (ravines, canyons), the Hessian matrix has vastly unequal eigenvalues (condition number κ = λ_max / λ_min >> 1). The gradient points almost perpendicularly to the optimal direction, forcing standard SGD to oscillate wildly across the steep ravine walls while making painfully slow progress along the flat base.',
        formula: 'L(w_1, w_2) = \\frac{1}{2}(w_1^2 + 10 w_2^2), \\quad \\kappa = \\frac{\\lambda_{\\max}}{\\lambda_{\\min}} = 10',
        keyTakeaway: 'Vanilla SGD requires an impractically small learning rate to avoid divergence across steep walls, freezing progress along shallow slopes.',
      },
      {
        title: "2. Polyak's Heavy-Ball Momentum Formulation",
        content: 'Polyak (1964) modeled optimization as a heavy rolling ball. At step t, the update velocity v_t accumulates the current gradient scaled by learning rate α plus a decayed fraction β of the previous velocity. Along consistent valley directions, velocity compounds to α / (1 - β) (a 10× speedup at β = 0.9). Along oscillating directions, alternating gradients cancel out by destructive interference.',
        formula: 'v_t = \\beta v_{t-1} + \\alpha \\nabla L(\\theta_t), \\quad \\theta_{t+1} = \\theta_t - v_t, \\quad v_{\\text{terminal}} = \\frac{\\alpha}{1 - \\beta} \\nabla L',
        keyTakeaway: 'Momentum provides low-pass filtering: high-frequency oscillations are smoothed out, while low-frequency consistent gradients are amplified up to 1/(1-β)×.',
      },
      {
        title: '3. Classical Momentum vs. Nesterov Accelerated Gradient (NAG)',
        content: 'Standard Polyak momentum computes the gradient at the current position θ_t before applying momentum β*v_{t-1}. Yurii Nesterov (1983) introduced a brilliant look-ahead enhancement: compute the gradient at the predicted future point (θ_t - β*v_{t-1}). If the momentum is carrying the ball too fast up the opposite slope, the look-ahead gradient applies an immediate braking force.',
        formula: 'v_t^{\\text{NAG}} = \\beta v_{t-1} + \\alpha \\nabla L(\\theta_t - \\beta v_{t-1}), \\quad \\theta_{t+1} = \\theta_t - v_t^{\\text{NAG}}',
        keyTakeaway: 'Nesterov look-ahead gradient prevents momentum overshooting, improving theoretical convergence from O(1/t) to the optimal O(1/t²).',
      },
      {
        title: '4. The Backbone of Modern AI Optimizers (Adam)',
        content: 'Momentum forms the first-order directional memory (1st moment m_t = E[g]) in modern AI training. Combining Momentum with second-moment adaptive learning rates (RMSprop v_t = E[g²]) created Adam (Kingma & Ba, 2014)—the standard optimizer powering all modern Transformers, LLMs (GPT-4, Claude, LLaMA), and deep vision architectures.',
        formula: 'm_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t, \\quad v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2, \\quad \\theta_{t+1} = \\theta_t - \\frac{\\alpha}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t',
        keyTakeaway: 'Without momentum, training deep multi-layer neural networks and LLMs would suffer from catastrophic vanishing gradients and stagnation in saddle points.',
      },
    ],
    timeComplexity: 'O(T · D)',
    spaceComplexity: 'O(D) velocity state',
    tags: ['Optimization', 'Gradient Descent', 'Momentum', 'Polyak', 'Nesterov', 'Adam', 'Ravine', 'NumPy'],
    defaultInput: 'start=(4.5, 1.8), beta=0.9, lr=0.1',
    inputVariants: [
      {
        label: 'Elongated Ravine (β=0.9)',
        value: 'start=(4.5, 1.8), beta=0.9, lr=0.1',
      },
      {
        label: 'Vanilla SGD (No Momentum β=0.0)',
        value: 'start=(4.5, 1.8), beta=0.0, lr=0.1',
      },
      {
        label: 'High Inertia (β=0.95)',
        value: 'start=(4.5, 1.8), beta=0.95, lr=0.1',
      },
      {
        label: 'Moderate Damping (β=0.6)',
        value: 'start=(4.5, 1.8), beta=0.6, lr=0.1',
      },
    ],
    vi: {
      title: '05. Gradient Descent với Momentum (Tối ưu Hóa Gia tốc)',
      category: 'Machine Learning Cổ điển',
      badge: 'Thuật toán Tối ưu',
      summary: 'Gia tốc hóa Gradient Descent và triệt tiêu dao động nghẽn hẻm núi bằng mô hình quán tính vật lý qua trung bình trượt hàm mũ.',
      description: 'Gradient Descent thuần (Vanilla SGD) gặp trở ngại nghiêm trọng trên các bề mặt hàm mất mát dạng "hẻm núi" (ravine)—nơi độ cong dốc gấp nhiều lần theo một chiều so với chiều còn lại—dẫn đến hiện tượng dao động dội qua lại liên tục và tiến triển cực kỳ chậm. Boris Polyak (1964) phát minh thuật toán Hòn bi Nặng (Heavy-Ball Momentum), mô phỏng một hạt vật lý có khối lượng và quán tính: các gradient trong quá khứ được tích lũy theo hệ số suy giảm ma sát β vào vector vận tốc v_t. Vận tốc duy trì gia tốc cao dọc theo đáy thung lũng phẳng và triệt tiêu dao động ngang, giúp tăng tốc độ hội tụ lên đến 10 lần.',
      variants: [
        {
          label: 'Hẻm Dốc Hẹp (β=0.9)',
          value: 'start=(4.5, 1.8), beta=0.9, lr=0.1',
        },
        {
          label: 'SGD Thuần (Không Momentum β=0.0)',
          value: 'start=(4.5, 1.8), beta=0.0, lr=0.1',
        },
        {
          label: 'Quán tính Cao (β=0.95)',
          value: 'start=(4.5, 1.8), beta=0.95, lr=0.1',
        },
        {
          label: 'Hãm Vừa phải (β=0.6)',
          value: 'start=(4.5, 1.8), beta=0.6, lr=0.1',
        },
      ],
      sections: [
        {
          title: '1. Bệnh Nan y "Hẻm Núi" & Hiện tượng Dao động của SGD Thuần',
          content: 'Trong không gian mất mát bất đẳng hướng, ma trận Hessian có các trị riêng chênh lệch lớn (chỉ số điều kiện κ = λ_max / λ_min >> 1). Vector gradient chỉ gần như vuông góc với hướng cực tiểu toàn cục, khiến thuật toán SGD thuần bật nảy dữ dội qua lại giữa hai vách dốc hẹp mà hầu như không di chuyển được dọc theo đáy thung lũng.',
          formula: 'L(w_1, w_2) = \\frac{1}{2}(w_1^2 + 10 w_2^2), \\quad \\kappa = \\frac{\\lambda_{\\max}}{\\lambda_{\\min}} = 10',
          keyTakeaway: 'SGD thuần buộc phải chọn learning rate cực nhỏ để tránh phân kỳ qua hai sườn dốc hẹp, làm đóng băng tốc độ tiến về cực tiểu.',
        },
        {
          title: '2. Công thức Hòn bi Nặng (Heavy-Ball) của Polyak',
          content: 'Polyak (1964) mô hình hóa quá trình tối ưu như một hòn bi nặng lăn xuống dốc có ma sát. Tại bước t, vận tốc v_t tích lũy gradient hiện tại cộng với quán tính từ vận tốc bước trước suy giảm theo tỷ lệ β. Dọc theo hướng dốc đồng nhất, vận tốc tích lũy đạt giá trị giới hạn α / (1 - β) (tăng tốc gấp 10 lần khi β = 0.9). Dọc theo hướng dao động đối xứng, các gradient ngược dấu triệt tiêu nhau hoàn toàn.',
          formula: 'v_t = \\beta v_{t-1} + \\alpha \\nabla L(\\theta_t), \\quad \\theta_{t+1} = \\theta_t - v_t, \\quad v_{\\text{terminal}} = \\frac{\\alpha}{1 - \\ beta} \\nabla L',
          keyTakeaway: 'Momentum hoạt động như bộ lọc thông thấp (low-pass filter): lọc sạch các dao động tần số cao và khuếch đại gia tốc hướng dốc nhất quán.',
        },
        {
          title: '3. So sánh Momentum Cổ điển & Gia tốc Nesterov (NAG)',
          content: 'Polyak Momentum tính gradient tại vị trí hiện tại θ_t trước khi cộng quán tính β*v_{t-1}. Yurii Nesterov (1983) cải tiến bằng cơ chế "nhìn trước": tính gradient tại điểm dự đoán tiếp theo (θ_t - β*v_{t-1}). Nếu quán tính đang đẩy hòn bi lăn quá nhanh lên sườn dốc bên kia, gradient nhìn trước sẽ đóng vai trò như một lực phanh chủ động.',
          formula: 'v_t^{\\text{NAG}} = \\beta v_{t-1} + \\alpha \\nabla L(\\theta_t - \\beta v_{t-1}), \\quad \\theta_{t+1} = \\theta_t - v_t^{\\text{NAG}}',
          keyTakeaway: 'Gia tốc Nesterov hạn chế tối đa hiện tượng vượt ngưỡng (overshooting), cải thiện tốc độ hội tụ lý thuyết từ O(1/t) lên mức tối ưu O(1/t²).',
        },
        {
          title: '4. Nền tảng của các Trình Tối ưu AI Hiện đại (RMSprop & Adam)',
          content: 'Momentum là thành phần ký ức định hướng bậc nhất (1st moment m_t = E[g]) của các thuật toán tối ưu hiện đại. Khi kết hợp Momentum với tốc độ học thích ứng theo phương sai bậc hai (RMSprop v_t = E[g²]), ta có thuật toán Adam (Kingma & Ba, 2014)—trình tối ưu mặc định huấn luyện hầu hết các mô hình Transformer, LLM (GPT-4, Claude, LLaMA) và mạng nơ-ron sâu ngày nay.',
          formula: 'm_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t, \\quad v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2, \\quad \\theta_{t+1} = \\theta_t - \\frac{\\alpha}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t',
          keyTakeaway: 'Không có Momentum, việc huấn luyện các mô hình AI sâu hàng trăm tầng sẽ bị đình trệ nghiêm trọng tại các điểm yên ngựa (saddle points) và vùng gradient triệt tiêu.',
        },
      ],
    },
    pythonCode: `import numpy as np

def momentum_gradient_descent(w_init, lr=0.1, beta=0.9, max_iters=15):
    # Loss: L(w1, w2) = 0.5 * (w1^2 + 10 * w2^2)
    # Gradient: [w1, 10 * w2]
    w = np.array(w_init, dtype=float)
    v = np.zeros_like(w)
    path = []

    for step in range(max_iters):
        loss = 0.5 * (w[0]**2 + 10 * w[1]**2)
        grad = np.array([w[0], 10 * w[1]])
        
        # 1. Update velocity with momentum decay
        v = beta * v + lr * grad
        
        # 2. Update parameters using velocity vector
        w = w - v
        path.append((w.copy(), loss, v.copy(), grad.copy()))
        
        if np.linalg.norm(grad) < 1e-3:
            break

    return w, path`,
    code: `import numpy as np

def momentum_sgd(w_init, lr=0.1, beta=0.9, max_iters=15):
    w = np.array(w_init, dtype=float)
    v = np.zeros_like(w)
    for step in range(max_iters):
        grad = np.array([w[0], 10 * w[1]])     # 1. Compute gradient
        loss = 0.5 * (w[0]**2 + 10 * w[1]**2)   # 2. Compute loss
        v = beta * v + lr * grad                # 3. Update velocity
        w = w - v                               # 4. Parameter step
        if np.linalg.norm(grad) < 1e-3:
            break
    return w, loss`,
    generateTraces: (inputStr?: string, lang: Language = 'en'): TraceStep[] => {
      const isVi = lang === 'vi';

      let startW: [number, number] = [4.5, 1.8];
      let beta = 0.9;
      let lr = 0.1;

      if (inputStr) {
        const startMatch = inputStr.match(/start\s*=\s*\(?\s*([-0-9.]+)\s*[,:]\s*([-0-9.]+)\s*\)?/);
        if (startMatch) {
          startW = [Number(startMatch[1]), Number(startMatch[2])];
        }
        const betaMatch = inputStr.match(/beta\s*=\s*([0-9.]+)/);
        if (betaMatch) {
          beta = Number(betaMatch[1]);
        }
        const lrMatch = inputStr.match(/lr\s*=\s*([0-9.]+)/);
        if (lrMatch) {
          lr = Number(lrMatch[1]);
        }
      }


      // Precompute Vanilla SGD trajectory for side-by-side comparison
      const maxSteps = 12;
      const fullVanillaPath: MomentumTrajectoryPoint[] = [];
      let vanW1 = startW[0];
      let vanW2 = startW[1];
      const vanLr = 0.16; // Moderate learning rate to demonstrate realistic ravine oscillation

      for (let s = 0; s <= maxSteps; s++) {
        const l = 0.5 * (vanW1 * vanW1 + 10 * vanW2 * vanW2);
        const g1 = vanW1;
        const g2 = 10 * vanW2;
        fullVanillaPath.push({
          step: s,
          w1: Number(vanW1.toFixed(3)),
          w2: Number(vanW2.toFixed(3)),
          loss: Number(l.toFixed(4)),
          grad1: Number(g1.toFixed(3)),
          grad2: Number(g2.toFixed(3)),
        });
        vanW1 -= vanLr * g1;
        vanW2 -= vanLr * g2;
      }

      // Precompute Momentum trajectory
      const fullMomentumPath: MomentumTrajectoryPoint[] = [];
      let momW1 = startW[0];
      let momW2 = startW[1];
      let v1 = 0;
      let v2 = 0;

      for (let s = 0; s <= maxSteps; s++) {
        const l = 0.5 * (momW1 * momW1 + 10 * momW2 * momW2);
        const g1 = momW1;
        const g2 = 10 * momW2;
        fullMomentumPath.push({
          step: s,
          w1: Number(momW1.toFixed(3)),
          w2: Number(momW2.toFixed(3)),
          loss: Number(l.toFixed(4)),
          v1: Number(v1.toFixed(4)),
          v2: Number(v2.toFixed(4)),
          grad1: Number(g1.toFixed(3)),
          grad2: Number(g2.toFixed(3)),
        });
        v1 = beta * v1 + lr * g1;
        v2 = beta * v2 + lr * g2;
        momW1 -= v1;
        momW2 -= v2;
      }

      const steps: TraceStep[] = [];

      // Step 0: Initialization
      const initPoint = fullMomentumPath[0];
      steps.push({
        step: 0,
        totalSteps: 0,
        line: 5,
        action: 'init',
        targets: [],
        state: {
          loss: initPoint.loss,
          momentumState: {
            beta,
            learningRate: lr,
            currentW: [initPoint.w1, initPoint.w2],
            loss: initPoint.loss,
            velocity: [0, 0],
            gradient: [initPoint.grad1 || 0, initPoint.grad2 || 0],
            momentumPath: [initPoint],
            vanillaPath: [fullVanillaPath[0]],
            surfaceType: 'ravine',
            iteration: 0,
            maxIterations: maxSteps,
            converged: false,
          },
        },
        explanation: isVi
          ? `[Khởi tạo Momentum]: Điểm xuất phát θ_0 = (${initPoint.w1}, ${initPoint.w2}) trên sườn hẻm dốc. Vận tốc ban đầu v_0 = (0, 0). Hệ số quán tính β = ${beta}, tốc độ học α = ${lr}.`
          : `[Momentum Init]: Initial position θ_0 = (${initPoint.w1}, ${initPoint.w2}) on the steep ravine wall. Initial velocity v_0 = (0, 0). Momentum factor β = ${beta}, learning rate α = ${lr}.`,
        invariant: `v_0 = (0, 0), \\quad \\theta_0 = (${initPoint.w1}, ${initPoint.w2})`,
      });

      for (let s = 1; s <= maxSteps; s++) {
        const curPt = fullMomentumPath[s];
        const vMag = Math.hypot(curPt.v1 || 0, curPt.v2 || 0);
        const isLast = s === maxSteps;

        steps.push({
          step: 0,
          totalSteps: 0,
          line: isLast ? 11 : 9,
          action: isLast ? 'done' : 'update',
          targets: [],
          state: {
            loss: curPt.loss,
            momentumState: {
              beta,
              learningRate: lr,
              currentW: [curPt.w1, curPt.w2],
              loss: curPt.loss,
              velocity: [curPt.v1 || 0, curPt.v2 || 0],
              gradient: [curPt.grad1 || 0, curPt.grad2 || 0],
              momentumPath: fullMomentumPath.slice(0, s + 1),
              vanillaPath: fullVanillaPath.slice(0, s + 1),
              surfaceType: 'ravine',
              iteration: s,
              maxIterations: maxSteps,
              converged: isLast || curPt.loss < 0.05,
            },
          },
          explanation: isLast
            ? isVi
              ? `[Hội tụ Hoàn tất]: Sau ${s} bước, Momentum đã dẫn dắt tham số tiệm cận đáy hẻm θ* = (0, 0) với Loss = ${curPt.loss.toFixed(4)}. So sánh với SGD thuần (đường đỏ), quán tính đã triệt tiêu hoàn toàn dao động ngang!`
              : `[Convergence Complete]: After ${s} steps, Momentum guided parameters near global optimum θ* = (0, 0) with Loss = ${curPt.loss.toFixed(4)}. Compared to oscillating Vanilla SGD (red), lateral vibrations are completely extinguished!`
            : isVi
            ? `[Bước ${s} - Tích lũy Quán tính]: Vector vận tốc v_${s} = β·v_${s - 1} + α·∇L tích lũy động lượng dọc theo trục w₁, triệt tiêu dao động trên trục w₂. Vận tốc ‖v‖ = ${vMag.toFixed(3)}, Loss giảm còn ${curPt.loss.toFixed(4)}.`
            : `[Step ${s} - Velocity Accumulation]: Velocity vector v_${s} = β·v_${s - 1} + α·∇L accelerates along the w₁ floor while canceling transverse w₂ oscillations. Speed ‖v‖ = ${vMag.toFixed(3)}, Loss reduced to ${curPt.loss.toFixed(4)}.`,
          invariant: `v_{${s}} = ${beta} v_{${s - 1}} + ${lr} \\nabla L, \\quad L = ${curPt.loss.toFixed(4)}`,
        });

        if (curPt.loss < 0.02) break;
      }

      const total = steps.length;
      return steps.map((s, idx) => ({ ...s, step: idx + 1, totalSteps: total }));
    },
  },

  // ==========================================
  // Category 2: Deep Learning & Neural Networks
  // ==========================================
  {
    id: 'single-neuron',
    title: '01. The Artificial Neuron (Perceptron)',
    category: 'Deep Learning & Neural Networks',
    difficulty: 'Beginner',
    badge: 'Neural Building Block',
    visualType: 'single-neuron',
    formula: 'z = \\sum_{i=1}^n w_i x_i + b = \\mathbf{w}^T \\mathbf{x} + b, \\quad a = \\sigma(z) = \\frac{1}{1 + e^{-z}}',
    summary: 'The foundational atomic unit of deep learning: aggregates weighted inputs, adds a threshold bias, and fires via an activation function.',
    description: 'Inspired by biological neurons (dendrites, soma, axon), Frank Rosenblatt’s Perceptron computes a linear combination z = w^T x + b followed by an activation function a = f(z). Traces how inputs scale through synaptic weights to produce binary or probability decisions.',
    inputVariants: [
      { label: 'AND Gate (x=[1, 1])', value: 'x1: 1.0, x2: 1.0' },
      { label: 'OR Gate (x=[0, 1])', value: 'x1: 0.0, x2: 1.0' },
      { label: 'Below Threshold (x=[0, 0])', value: 'x1: 0.0, x2: 0.0' },
      { label: 'Continuous (x=[0.8, -0.5])', value: 'x1: 0.8, x2: -0.5' },
    ],
    vi: {
      title: '01. Nơ-ron Nhân tạo (Perceptron)',
      category: 'Deep Learning & Mạng Nơ-ron',
      badge: 'Viên gạch Nơ-ron',
      summary: 'Đơn vị nguyên tử nền tảng của học sâu: tổng hợp đầu vào có trọng số, cộng bias ngưỡng và kích hoạt qua hàm phi tuyến để ra quyết định.',
      description: 'Lấy cảm hứng từ nơ-ron sinh học (sợi nhánh, thân tế bào, sợi trục), Perceptron của Frank Rosenblatt tính tổ hợp tuyến tính z = w^T x + b và kích hoạt phi tuyến a = f(z). Mô phỏng từng luồng tín hiệu từ đầu vào qua trọng số đến quyết định kích hoạt.',
      variants: [
        { label: 'Cổng AND (x=[1, 1])', value: 'x1: 1.0, x2: 1.0' },
        { label: 'Cổng OR (x=[0, 1])', value: 'x1: 0.0, x2: 1.0' },
        { label: 'Dưới ngưỡng (x=[0, 0])', value: 'x1: 0.0, x2: 0.0' },
        { label: 'Liên tục (x=[0.8, -0.5])', value: 'x1: 0.8, x2: -0.5' },
      ],
      sections: [
        {
          title: '1. Cấu trúc Nơ-ron Nhân tạo (Từ Sinh học đến Toán học)',
          content: 'Lấy cảm hứng từ nơ-ron sinh học: Sợi nhánh (Dendrites) nhận tín hiệu đầu vào x_i, Khớp thần kinh (Synapses) khuếch đại hoặc làm suy giảm tín hiệu qua trọng số w_i, Thân tế bào (Soma) cộng dồn toàn bộ z = ∑ w_i x_i + b, và Sợi trục (Axon) phát xung đầu ra a = f(z).',
          formula: 'z = \\mathbf{w}^T \\mathbf{x} + b = w_1 x_1 + w_2 x_2 + \\dots + w_n x_n + b',
          keyTakeaway: 'Weights (w) quyết định tầm quan trọng của từng đặc trưng; Bias (b) điều chỉnh ngưỡng kích hoạt nơ-ron dễ hay khó phát xung.',
        },
        {
          title: '2. Các Hàm Kích hoạt: Step Function, Sigmoid và ReLU',
          content: 'Perceptron ban đầu của Rosenblatt sử dụng hàm bước nhảy (Step function) nhị phân 0 hoặc 1. Mạng nơ-ron hiện đại chuyển sang các hàm kích hoạt mượt mà có đạo hàm như Sigmoid (1 / (1 + e^-z)) hoặc ReLU (max(0, z)) để phục vụ việc tính gradient khi huấn luyện.',
          formula: '\\sigma(z) = \\frac{1}{1 + e^{-z}}, \\quad \\text{ReLU}(z) = \\max(0, z)',
          keyTakeaway: 'Tính khả vi (differentiable) của các hàm kích hoạt mượt cho phép thuật toán Gradient Descent tính đạo hàm và tinh chỉnh trọng số.',
        },
        {
          title: '3. Giới hạn Ranh giới Quyết định Tuyến tính & Bài toán XOR',
          content: 'Một nơ-ron đơn lẻ chỉ có thể tạo ra một siêu phẳng tuyến tính w1*x1 + w2*x2 + b = 0 chia không gian thành hai nửa. Nó có thể học được cổng logic AND và OR, nhưng thất bại hoàn toàn trước cổng XOR (Minsky & Papert, 1969).',
          keyTakeaway: 'Hạn chế này thúc đẩy sự ra đời của Mạng nơ-ron Đa tầng (MLP) với các tầng ẩn để giải quyết các bài toán phi tuyến phức tạp.',
        },
      ],
    },
    sections: [
      {
        title: '1. Anatomy of an Artificial Neuron (Biology to Math)',
        content: 'Inspired by biological neurons: Dendrites receive input signals x_i, Synapses apply signal strength via weights w_i, the Cell Body (Soma) computes the weighted sum z = ∑ w_i x_i + b, and the Axon fires an activation signal a = f(z).',
        formula: 'z = \\mathbf{w}^T \\mathbf{x} + b = w_1 x_1 + w_2 x_2 + \\dots + w_n x_n + b',
        keyTakeaway: 'Weights (w) scale feature importance; Bias (b) shifts the threshold for how easily the neuron fires.',
      },
      {
        title: '2. Activation Functions: From Step Function to Sigmoid & ReLU',
        content: 'Rosenblatt’s historical perceptron used a discrete step function (0 or 1). Modern deep learning requires smooth, differentiable activations like Sigmoid or ReLU so gradients can be propagated backward during gradient descent training.',
        formula: '\\sigma(z) = \\frac{1}{1 + e^{-z}}, \\quad \\text{ReLU}(z) = \\max(0, z)',
        keyTakeaway: 'Smooth differentiable activations allow infinitesimal weight nudges to produce measurable loss changes.',
      },
      {
        title: '3. The Linear Decision Boundary & The XOR Limit',
        content: 'A single neuron defines a linear decision boundary w1*x1 + w2*x2 + b = 0 dividing space into two half-planes. It easily solves AND and OR gates, but is mathematically incapable of separating the non-linear XOR function (Minsky & Papert, 1969).',
        keyTakeaway: 'Overcoming the single neuron’s linear boundary limit directly necessitated Multi-Layer Perceptrons (MLPs).',
      },
    ],
    timeComplexity: 'O(n) inputs dot-product',
    spaceComplexity: 'O(n) parameters',
    tags: ['Perceptron', 'Neural Network', 'Activation', 'Sigmoid', 'PyTorch'],
    defaultInput: 'x1: 1.0, x2: 1.0',
    pythonCode: `import numpy as np

def single_neuron(x, w, b, activation="sigmoid"):
    # 1. Linear Weighted Sum + Bias
    z = np.dot(x, w) + b
    
    # 2. Non-linear Activation
    if activation == "sigmoid":
        a = 1 / (1 + np.exp(-z))
    elif activation == "relu":
        a = np.maximum(0, z)
    elif activation == "step":
        a = 1.0 if z >= 0 else 0.0
    return z, a

# Logic AND gate weights configuration
w = np.array([1.5, 1.5])
b = -2.0
x = np.array([1.0, 1.0])
z, output = single_neuron(x, w, b, activation="sigmoid")
print(f"z = {z:.2f}, Output probability = {output:.4f}")`,
    code: `import numpy as np

def neuron(x, w, b):
    # 1. Weighted sum
    z = np.dot(x, w) + b
    # 2. Sigmoid activation
    a = 1 / (1 + np.exp(-z))
    return z, a`,
    generateTraces: (inputStr?: string, lang: Language = 'en'): TraceStep[] => {
      const isVi = lang === 'vi';
      let x1 = 1.0;
      let x2 = 1.0;

      if (inputStr) {
        const m1 = inputStr.match(/x1:\s*(-?[0-9.]+)/i);
        const m2 = inputStr.match(/x2:\s*(-?[0-9.]+)/i);
        if (m1) x1 = Number(m1[1]);
        if (m2) x2 = Number(m2[1]);
      }

      const w1 = 1.5;
      const w2 = 1.5;
      const b = -2.0;
      const weightedSum = Number((w1 * x1 + w2 * x2 + b).toFixed(2));
      const outputProb = Number((1 / (1 + Math.exp(-weightedSum))).toFixed(3));

      const steps: TraceStep[] = [];

      // Step 0: Input Signal Reception
      steps.push({
        step: 0,
        totalSteps: 0,
        line: 4,
        action: 'init',
        targets: [0, 1],
        state: {
          neuronState: {
            inputs: [
              { label: 'x₁', val: x1 },
              { label: 'x₂', val: x2 },
            ],
            weights: [w1, w2],
            bias: b,
            weightedSum: 0,
            activationName: 'Pending',
            outputVal: 0,
          },
        },
        explanation: isVi
          ? `Đầu vào x = [${x1}, ${x2}] được tiếp nhận bởi sợi nhánh (dendrites). Trọng số khớp thần kinh: w₁ = ${w1}, w₂ = ${w2}, bias ngưỡng b = ${b}.`
          : `Inputs x = [${x1}, ${x2}] received at dendrites. Synaptic weights: w₁ = ${w1}, w₂ = ${w2}, threshold bias b = ${b}.`,
        invariant: isVi
          ? 'Tín hiệu đầu vào sẵn sàng truyền qua các kết nối synap.'
          : 'Input signals ready for synaptic scaling.',
      });

      // Step 1: Synaptic Weighting (w1*x1 and w2*x2)
      steps.push({
        step: 1,
        totalSteps: 0,
        line: 5,
        action: 'compute',
        targets: [0, 1],
        state: {
          neuronState: {
            inputs: [
              { label: 'x₁', val: x1 },
              { label: 'x₂', val: x2 },
            ],
            weights: [w1, w2],
            bias: b,
            weightedSum: Number((w1 * x1 + w2 * x2).toFixed(2)),
            activationName: 'Scaling',
            outputVal: 0,
          },
        },
        explanation: isVi
          ? `Tín hiệu khuếch đại qua synap: w₁·x₁ = ${w1}·${x1} = ${(w1 * x1).toFixed(2)}, w₂·x₂ = ${w2}·${x2} = ${(w2 * x2).toFixed(2)}.`
          : `Synaptic transmission: w₁·x₁ = ${w1}·${x1} = ${(w1 * x1).toFixed(2)}, w₂·x₂ = ${w2}·${x2} = ${(w2 * x2).toFixed(2)}.`,
        invariant: isVi
          ? 'Phép nhân trọng số điều chỉnh cường độ đóng góp của từng đặc trưng.'
          : 'Synaptic scaling weights feature importance.',
      });

      // Step 2: Summation and Bias in Soma (z = w^T x + b)
      steps.push({
        step: 2,
        totalSteps: 0,
        line: 5,
        action: 'forward',
        targets: [2],
        state: {
          neuronState: {
            inputs: [
              { label: 'x₁', val: x1 },
              { label: 'x₂', val: x2 },
            ],
            weights: [w1, w2],
            bias: b,
            weightedSum,
            activationName: 'z = ∑ + b',
            outputVal: 0,
          },
        },
        explanation: isVi
          ? `Thân nơ-ron (Soma) tính tổng và cộng bias: z = (${(w1 * x1).toFixed(2)} + ${(w2 * x2).toFixed(2)}) + (${b}) = ${weightedSum}. ${weightedSum >= 0 ? 'Giá trị vượt ngưỡng kích hoạt (z ≥ 0)!' : 'Giá trị dưới ngưỡng kích hoạt (z < 0).'}`
          : `Soma aggregates inputs and adds bias: z = (${(w1 * x1).toFixed(2)} + ${(w2 * x2).toFixed(2)}) + (${b}) = ${weightedSum}. ${weightedSum >= 0 ? 'Above activation threshold (z ≥ 0)!' : 'Below threshold (z < 0).'}`,
        invariant: isVi
          ? 'Tổ hợp tuyến tính z đo khoảng cách đại số tới ranh giới quyết định.'
          : 'Affine combination z measures signed distance to decision boundary.',
      });

      // Step 3: Activation Function Firing (Sigmoid)
      steps.push({
        step: 3,
        totalSteps: 0,
        line: 7,
        action: 'done',
        targets: [3],
        state: {
          neuronState: {
            inputs: [
              { label: 'x₁', val: x1 },
              { label: 'x₂', val: x2 },
            ],
            weights: [w1, w2],
            bias: b,
            weightedSum,
            activationName: 'Sigmoid',
            outputVal: outputProb,
          },
        },
        explanation: isVi
          ? `Hàm kích hoạt Sigmoid phát xung đầu ra: a = σ(${weightedSum}) = ${outputProb} (xác suất ${(outputProb * 100).toFixed(1)}%). Nơ-ron đưa ra quyết định ${outputProb >= 0.5 ? 'KÍCH HOẠT (Output = 1)' : 'KHÔNG KÍCH HOẠT (Output = 0)'}!`
          : `Sigmoid activation fires: a = σ(${weightedSum}) = ${outputProb} (${(outputProb * 100).toFixed(1)}% confidence). Neuron decides: ${outputProb >= 0.5 ? 'FIRE (Class 1)' : 'REST (Class 0)'}!`,
        invariant: isVi
          ? 'Đầu ra được nén về đoạn [0, 1] biểu thị xác suất tin cậy của nơ-ron.'
          : 'Output compressed to [0, 1] interval representing class firing probability.',
      });

      const total = steps.length;
      return steps.map((s, idx) => ({ ...s, step: idx + 1, totalSteps: total }));
    },
  },
  {
    id: 'modern-neuron',
    title: '02. The Modern Neuron vs. Perceptron',
    category: 'Deep Learning & Neural Networks',
    difficulty: 'Beginner',
    badge: 'Activation & Calculus',
    visualType: 'modern-neuron',
    formula: '\\text{Perceptron: } a = \\text{step}(z) \\quad \\longleftrightarrow \\quad \\text{Modern Neuron: } a = \\sigma(z), \\quad \\frac{da}{dz} = a(1 - a)',
    summary: 'Reveals the revolution that unlocked deep learning: replacing the Perceptron’s rigid step function with smooth, differentiable activations (Sigmoid & ReLU) to enable Gradient Descent.',
    description: 'Frank Rosenblatt’s 1957 Perceptron computed a linear weighted sum z = w^T x + b, but passed it through a discontinuous step function where da/dz = 0 everywhere. Modern neurons substitute continuous non-linearities (Sigmoid, Tanh, ReLU) with non-zero gradients, allowing Backpropagation to guide parameter updates.',
    inputVariants: [
      { label: 'Modern Sigmoid (Smooth Gradient)', value: 'activation: sigmoid, z: 1.2' },
      { label: 'Modern ReLU (Deep Learning Standard)', value: 'activation: relu, z: 1.5' },
      { label: 'Classic Perceptron (Zero Gradient)', value: 'activation: step, z: 1.2' },
      { label: 'Vanishing Gradient Saturation', value: 'activation: sigmoid, z: 4.8' },
    ],
    vi: {
      title: '02. Nơ-ron AI Hiện đại vs. Perceptron',
      category: 'Deep Learning & Mạng Nơ-ron',
      badge: 'Hàm Kích hoạt & Đạo hàm',
      summary: 'Cú hích lịch sử khai sinh học sâu: thay thế hàm bước nhảy cụt ngủn của Perceptron bằng các hàm kích hoạt khả vi (Sigmoid & ReLU) để thuật toán Gradient Descent có thể tối ưu trọng số.',
      description: 'Perceptron (1957) tính tổng trọng số z = w^T x + b nhưng đi qua hàm bước nhảy có đạo hàm da/dz = 0 ở mọi nơi, khiến máy tính không thể học bằng đạo hàm. Nơ-ron hiện đại thay thế bằng các hàm phi tuyến liên tục (Sigmoid, Tanh, ReLU) có gradient sống, mở đường cho thuật toán Lan truyền ngược (Backpropagation).',
      variants: [
        { label: 'Modern Sigmoid (Gradient Mượt)', value: 'activation: sigmoid, z: 1.2' },
        { label: 'Modern ReLU (Chuẩn Deep Learning)', value: 'activation: relu, z: 1.5' },
        { label: 'Classic Perceptron (Gradient Bằng 0)', value: 'activation: step, z: 1.2' },
        { label: 'Triệt tiêu Gradient (Saturation)', value: 'activation: sigmoid, z: 4.8' },
      ],
      sections: [
        {
          title: '1. Lỗ hổng Chí mạng của Perceptron: Bài toán Gradient Bằng 0',
          content: 'Hàm bước nhảy Heaviside (a = 1 nếu z ≥ 0, ngược lại 0) hoàn toàn nằm ngang ở mọi điểm trừ z = 0. Đạo hàm da/dz = 0 ở khắp nơi! Khi tính gradient để tối ưu trọng số theo Chain Rule: ∂L/∂w = (∂L/∂a) * (da/dz) * (∂z/∂w) = (∂L/∂a) * 0 * x = 0. Dòng gradient hoàn toàn bị chặn đứng! Máy tính không thể dùng vi tích phân hay Gradient Descent để huấn luyện mạng Perceptron.',
          formula: '\\frac{da}{dz} = 0 \\quad (\\forall z \\ne 0) \\implies \\frac{\\partial L}{\\partial w} = 0 \\quad (\\text{Không thể học bằng Gradient Descent!})',
          keyTakeaway: 'Đạo hàm bằng 0 làm đóng băng quá trình học: thay đổi trọng số một chút không tạo ra bất kỳ phản hồi độ dốc nào.',
        },
        {
          title: '2. Giải pháp Đột phá: Hàm Kích hoạt Khả vi (Sigmoid & Tanh)',
          content: 'Năm 1986, Rumelhart, Hinton và Williams thay thế hàm bước nhảy bằng hàm Logistic Sigmoid liên tục: a = σ(z) = 1 / (1 + e^-z). Đạo hàm của Sigmoid có dạng đối xứng hình chuông tuyệt đẹp: da/dz = a * (1 - a). Nhờ đó, gradient luôn lớn hơn 0 trên miền hoạt động, cho phép thuật toán Lan truyền ngược (Backpropagation) hoạt động trơn tru.',
          formula: '\\sigma(z) = \\frac{1}{1 + e^{-z}}, \\quad \\frac{d\\sigma}{dz} = \\sigma(z)(1 - \\sigma(z)) = a(1 - a)',
          keyTakeaway: 'Hàm kích hoạt mượt mà cung cấp độ dốc (gradient) dẫn đường cho Gradient Descent trượt dốc xuống đáy mất mát tối thiểu.',
        },
        {
          title: '3. Kỷ nguyên Hiện đại: Từ Sigmoid sang ReLU (Trị dứt điểm Triệt tiêu Gradient)',
          content: 'Dù giải quyết được bài toán đạo hàm bằng 0, đạo hàm Sigmoid chỉ đạt tối đa 0.25 tại z=0 và tiến về 0 khi |z| > 4 (bão hòa). Khi nhân dồn chuỗi số thập phân này qua 10 lớp, gradient biến mất (0.25^10 ≈ 10^-6). Mạng sâu hiện đại chuyển sang ReLU (Rectified Linear Unit): a = max(0, z). Đạo hàm của ReLU là hằng số 1.0 cho mọi z > 0, cho phép mạng nơ-ron sâu hàng trăm tầng huấn luyện ổn định mà không bị suy hao tín hiệu.',
          formula: '\\text{ReLU}(z) = \\max(0, z), \\quad \\frac{d}{dz}\\text{ReLU} = 1.0 \\quad (z > 0)',
          keyTakeaway: 'Đạo hàm hằng số 1.0 của ReLU chính là động cơ cốt lõi vận hành toàn bộ kỷ nguyên Deep Learning và các mô hình Transformer hiện đại.',
        },
      ],
    },
    sections: [
      {
        title: '1. The Perceptron’s Fatal Flaw: The Zero-Gradient Problem',
        content: 'The Heaviside step function (a = 1 if z ≥ 0 else 0) is flat everywhere. Its derivative da/dz = 0 for all z ≠ 0. When computing parameter gradients via the Chain Rule: ∂L/∂w = (∂L/∂a) * (da/dz) * (∂z/∂w) = (∂L/∂a) * 0 * x = 0. The gradient signal is entirely dead! Calculus and Gradient Descent cannot train a network of step-function perceptrons.',
        formula: '\\frac{da}{dz} = 0 \\quad (\\forall z \\ne 0) \\implies \\frac{\\partial L}{\\partial w} = 0 \\quad (\\text{Dead Gradient Flow})',
        keyTakeaway: 'A zero derivative freezes the learning process: small weight tweaks produce zero feedback.',
      },
      {
        title: '2. The Modern Solution: Smooth Differentiable Activations (Sigmoid & Tanh)',
        content: 'In 1986, Rumelhart, Hinton, and Williams replaced the step function with the smooth logistic Sigmoid: a = σ(z) = 1 / (1 + e^-z). Its derivative is mathematically elegant: da/dz = a * (1 - a). Now, the gradient is non-zero, allowing infinitesimal weight changes to continuously steer the model toward lower loss.',
        formula: '\\sigma(z) = \\frac{1}{1 + e^{-z}}, \\quad \\frac{d\\sigma}{dz} = \\sigma(z)(1 - \\sigma(z)) = a(1 - a)',
        keyTakeaway: 'Smooth differentiable curves provide the continuous slope (gradient) that guides Gradient Descent downhill toward zero loss.',
      },
      {
        title: '3. From Sigmoid to ReLU: Overcoming Vanishing Gradients in Deep Nets',
        content: 'While Sigmoid revived neural networks, its derivative peaks at only 0.25 and approaches 0 when |z| > 4 (saturation). Multiplying these fractions across 10 layers causes gradients to vanish (0.25^10 ≈ 10^-6). Modern deep learning adopted ReLU: a = max(0, z). Its derivative is a clean constant 1.0 for all positive inputs, allowing networks with hundreds of layers to train without signal degradation.',
        formula: '\\text{ReLU}(z) = \\max(0, z), \\quad \\frac{d}{dz}\\text{ReLU} = 1.0 \\quad (z > 0)',
        keyTakeaway: 'ReLU’s constant gradient of 1.0 is the engine powering modern Deep Learning and Transformer architectures.',
      },
    ],
    timeComplexity: 'O(1) forward & gradient step',
    spaceComplexity: 'O(1) activations',
    tags: ['Activation Function', 'Gradient Flow', 'Sigmoid', 'ReLU', 'Perceptron vs Neuron'],
    defaultInput: 'activation: sigmoid, z: 1.2',
    pythonCode: `import numpy as np

# Comparing Perceptron vs Modern Neuron Activations and Gradients
def activation_and_gradient(z, kind="sigmoid"):
    if kind == "step":
        # Classic Perceptron: flat derivative, gradient = 0
        a = 1.0 if z >= 0 else 0.0
        grad = 0.0  # Zero gradient! Breaks backprop
    elif kind == "sigmoid":
        # Modern Sigmoid: smooth differentiable curve
        a = 1.0 / (1.0 + np.exp(-z))
        grad = a * (1.0 - a)  # Continuous gradient!
    elif kind == "relu":
        # Modern ReLU: avoids vanishing gradient
        a = max(0.0, z)
        grad = 1.0 if z > 0 else 0.0
    elif kind == "tanh":
        a = np.tanh(z)
        grad = 1.0 - a**2
    return a, grad

# Example comparison at z = 1.2
z = 1.2
for mode in ["step", "sigmoid", "relu"]:
    out, grad = activation_and_gradient(z, mode)
    print(f"[{mode.upper():7s}] Output a = {out:.4f} | Gradient da/dz = {grad:.4f}")`,
    code: `import numpy as np

def modern_neuron(z, activation="sigmoid"):
    # 1. Activation computation
    if activation == "sigmoid":
        a = 1 / (1 + np.exp(-z))
        grad = a * (1 - a)     # Smooth gradient!
    elif activation == "relu":
        a = np.maximum(0, z)
        grad = 1.0 if z > 0 else 0.0
    elif activation == "step": # Classic Perceptron
        a = 1.0 if z >= 0 else 0.0
        grad = 0.0             # Zero gradient!
    return a, grad`,
    generateTraces: (inputStr?: string, lang: Language = 'en'): TraceStep[] => {
      const isVi = lang === 'vi';
      let z = 1.2;
      let actType: 'step' | 'sigmoid' | 'relu' | 'tanh' = 'sigmoid';

      if (inputStr) {
        if (inputStr.includes('relu')) actType = 'relu';
        else if (inputStr.includes('step')) actType = 'step';
        else if (inputStr.includes('tanh')) actType = 'tanh';

        const m = inputStr.match(/z:\s*(-?[0-9.]+)/i);
        if (m) z = Number(m[1]);
      }

      const stepA = z >= 0 ? 1.0 : 0.0;
      const stepGrad = 0.0;

      const sigA = Number((1 / (1 + Math.exp(-z))).toFixed(3));
      const sigGrad = Number((sigA * (1 - sigA)).toFixed(3));

      const reluA = Number((Math.max(0, z)).toFixed(3));
      const reluGrad = z > 0 ? 1.0 : 0.0;

      const steps: TraceStep[] = [];

      // Step 0: Linear Combination z = w^T x + b
      steps.push({
        step: 0,
        totalSteps: 0,
        line: 4,
        action: 'init',
        targets: [0],
        state: {
          modernNeuronState: {
            inputs: [
              { label: 'x₁', val: 0.8 },
              { label: 'x₂', val: 0.4 },
            ],
            weights: [1.2, 0.5],
            bias: 0.04,
            zVal: z,
            activationType: 'step',
            outputVal: 0,
            gradientVal: 0,
          },
        },
        explanation: isVi
          ? `Tổng hợp tuyến tính: z = w₁·x₁ + w₂·x₂ + b = (1.2·0.8 + 0.5·0.4) + 0.04 = ${z.toFixed(2)}. Giá trị z sẵn sàng đưa qua các hàm kích hoạt để so sánh.`
          : `Linear combination computed: z = w₁·x₁ + w₂·x₂ + b = (1.2·0.8 + 0.5·0.4) + 0.04 = ${z.toFixed(2)}. Value z ready for activation comparison.`,
        invariant: isVi
          ? 'Cả Perceptron và Nơ-ron hiện đại đều dùng chung phép biến đổi tuyến tính z = w^T x + b.'
          : 'Both Perceptrons and Modern Neurons share the identical affine linear sum z = w^T x + b.',
      });

      // Step 1: The Classic Perceptron (Step Function) Evaluation
      steps.push({
        step: 1,
        totalSteps: 0,
        line: 10,
        action: 'compare',
        targets: [1],
        state: {
          modernNeuronState: {
            inputs: [
              { label: 'x₁', val: 0.8 },
              { label: 'x₂', val: 0.4 },
            ],
            weights: [1.2, 0.5],
            bias: 0.04,
            zVal: z,
            activationType: 'step',
            outputVal: stepA,
            gradientVal: stepGrad,
          },
        },
        explanation: isVi
          ? `Mô hình Perceptron Cổ điển (1957): Sử dụng hàm bước nhảy Heaviside. Với z = ${z.toFixed(2)} ≥ 0, Output a = ${stepA}. Đạo hàm da/dz = 0.000! Gradient bị triệt tiêu hoàn toàn (Zero-Gradient Problem), khiến thuật toán Gradient Descent bất lực.`
          : `Classic Perceptron (1957): Employs the Heaviside step function. For z = ${z.toFixed(2)} ≥ 0, Output a = ${stepA}. Derivative da/dz = 0.000! Dead gradient freezes Backpropagation.`,
        invariant: isVi
          ? 'Perceptron: Đạo hàm da/dz = 0 ở mọi nơi; không thể tối ưu hóa bằng vi tích phân.'
          : 'Classic Perceptron: da/dz = 0 everywhere; calculus-based learning is impossible.',
      });

      // Step 2: The Modern Neuron (Sigmoid Activation)
      steps.push({
        step: 2,
        totalSteps: 0,
        line: 6,
        action: 'forward',
        targets: [2],
        state: {
          modernNeuronState: {
            inputs: [
              { label: 'x₁', val: 0.8 },
              { label: 'x₂', val: 0.4 },
            ],
            weights: [1.2, 0.5],
            bias: 0.04,
            zVal: z,
            activationType: 'sigmoid',
            outputVal: sigA,
            gradientVal: sigGrad,
          },
        },
        explanation: isVi
          ? `Nơ-ron Hiện đại (Sigmoid): Đường cong mượt biến đổi z = ${z.toFixed(2)} thành xác suất liên tục a = ${sigA}. Đạo hàm da/dz = a(1 - a) = ${sigGrad} > 0! Gradient sống cho phép Backpropagation tính toán cập nhật trọng số.`
          : `Modern Neuron (Sigmoid): Smooth S-curve transforms z = ${z.toFixed(2)} into continuous probability a = ${sigA}. Derivative da/dz = a(1 - a) = ${sigGrad} > 0! Non-zero slope enables Gradient Descent.`,
        invariant: isVi
          ? 'Sigmoid: Tính khả vi (Differentiable) mở đường cho thuật toán Lan truyền ngược (Backpropagation).'
          : 'Sigmoid: Differentiable continuity unlocks Backpropagation and weight updates.',
      });

      // Step 3: Modern Deep Learning Standard (ReLU)
      steps.push({
        step: 3,
        totalSteps: 0,
        line: 8,
        action: 'done',
        targets: [3],
        state: {
          modernNeuronState: {
            inputs: [
              { label: 'x₁', val: 0.8 },
              { label: 'x₂', val: 0.4 },
            ],
            weights: [1.2, 0.5],
            bias: 0.04,
            zVal: z,
            activationType: actType === 'relu' ? 'relu' : 'sigmoid',
            outputVal: actType === 'relu' ? reluA : sigA,
            gradientVal: actType === 'relu' ? reluGrad : sigGrad,
          },
        },
        explanation: isVi
          ? `Chuẩn Học sâu Hiện đại (ReLU): a = max(0, ${z.toFixed(2)}) = ${reluA}. Đạo hàm da/dz = ${reluGrad}.0 (Hằng số cực đại)! Không bị bão hòa, giải quyết triệt để vấn đề Triệt tiêu Gradient (Vanishing Gradient) trong các mạng sâu hàng trăm lớp.`
          : `Modern Deep Learning Engine (ReLU): a = max(0, ${z.toFixed(2)}) = ${reluA}. Constant gradient da/dz = ${reluGrad}.0! Completely eliminates Vanishing Gradients, allowing hundreds of deep layers to learn.`,
        invariant: isVi
          ? 'Khai sinh Deep Learning: Thay thế hàm bước nhảy bằng hàm kích hoạt khả vi đã biến Perceptron thành Nơ-ron AI hiện đại.'
          : 'Foundation of Deep Learning: Swapping step functions for differentiable activations transformed Perceptrons into Modern Neurons.',
      });

      const total = steps.length;
      return steps.map((s, idx) => ({ ...s, step: idx + 1, totalSteps: total }));
    },
  },


  {
    id: 'neural-network-mlp',
    title: '03. Multi-Layer Perceptron (MLP) Forward Pass',
    category: 'Deep Learning & Neural Networks',
    difficulty: 'Intermediate',
    badge: 'Feedforward',
    visualType: 'neural-network',
    formula: 'h = \\text{ReLU}(W_1 x + b_1), \\quad \\hat{y} = \\sigma(W_2 h + b_2)',
    summary: 'Propagates input activations through hidden layers with ReLU non-linearities to calculate final network predictions.',
    description: 'A Multi-Layer Perceptron (MLP) maps non-linear input features into high-dimensional representations. Traces how weighted sums and activation functions fire across artificial neurons.',
    inputVariants: [
      { label: 'Standard', value: 'x1: 0.8, x2: 0.4' },
      { label: 'High Drive', value: 'x1: 1.5, x2: 1.2' },
      { label: 'Negative In', value: 'x1: -0.6, x2: 0.8' },
      { label: 'Zero In', value: 'x1: 0.0, x2: 0.0' },
    ],
    vi: {
      title: '03. Mạng Nơ-ron Đa tầng (MLP) Forward Pass',
      category: 'Deep Learning & Mạng Nơ-ron',
      badge: 'Forward Pass',
      summary: 'Lan truyền activations qua các hidden layers với hàm phi tuyến ReLU để tính toán output dự đoán.',
      description: 'Mạng Multi-Layer Perceptron (MLP) ánh xạ các features đầu vào sang không gian biểu diễn nhiều chiều. Quan sát tổ hợp tuyến tính và các hàm activation lan truyền qua từng neuron.',
      variants: [
        { label: 'Chuẩn', value: 'x1: 0.8, x2: 0.4' },
        { label: 'Biên độ cao', value: 'x1: 1.5, x2: 1.2' },
        { label: 'Đầu vào âm', value: 'x1: -0.6, x2: 0.8' },
        { label: 'Điểm gốc 0', value: 'x1: 0.0, x2: 0.0' },
      ],
      sections: [
        {
          title: '1. Tại sao Hidden Layers tạo nên Tính Phi tuyến?',
          content: 'Một linear layer đơn lẻ chỉ có thể học được ranh giới quyết định tuyến tính. Bằng cách xếp chồng các layers với hàm kích hoạt phi tuyến như ReLU (max(0, z)), mạng nơ-ron có thể xấp xỉ bất kỳ hàm số liên tục nào (Universal Approximation Theorem).',
          formula: '\\text{ReLU}(z) = \\max(0, z)',
          keyTakeaway: 'Nếu không có hàm phi tuyến, việc xếp chồng 100 linear layers sẽ sụp đổ thành đúng một linear layer duy nhất.',
        },
        {
          title: '2. Phép Nhân Ma trận và Kích hoạt Neurons',
          content: 'Với mỗi layer l, ta tính z^[l] = W^[l] * a^[l-1] + b^[l], sau đó áp dụng hàm activation a^[l] = g(z^[l]). Forward pass xâu chuỗi các phép tính này từ input layer đến output layer.',
          formula: 'z^{[1]} = W^{[1]} x + b^{[1]}, \\quad a^{[1]} = \\text{ReLU}(z^{[1]})',
          keyTakeaway: 'Phép nhân ma trận chạy song song cực nhanh trên GPU với các Tensor Cores.',
        },
        {
          title: '3. Kết nối Forward Pass với Backpropagation',
          content: 'Trong giai đoạn inference, mô hình chỉ chạy Forward pass. Trong giai đoạn training, sai số output tạo ra gradient truyền ngược (Backpropagation) theo Chain Rule để cập nhật weights.',
          keyTakeaway: 'Activations từ Forward pass được lưu trong memory (cache) để tái sử dụng khi tính gradient ở pha Backpropagation.',
        },
      ],
    },
    sections: [
      {
        title: '1. Why Hidden Layers Enable Non-Linearity',
        content: 'A single linear layer can only learn linear decision boundaries (hyperplanes). By stacking layers with non-linear activation functions like ReLU (max(0, z)), deep neural networks can approximate any continuous mathematical function (Universal Approximation Theorem).',
        formula: '\\text{ReLU}(z) = \\max(0, z)',
        keyTakeaway: 'Without non-linear activations, stacking 100 linear layers collapses mathematically into just one linear layer.',
      },
      {
        title: '2. The Matrix Multiplication Step',
        content: 'For layer l, we compute z^[l] = W^[l] * a^[l-1] + b^[l], followed by element-wise activation a^[l] = g(z^[l]). Forward propagation chains these operations from input to output.',
        formula: 'z^{[1]} = W^{[1]} x + b^{[1]}, \\quad a^{[1]} = \\text{ReLU}(z^{[1]})',
        keyTakeaway: 'Matrix multiplications run in parallel on GPUs, which is why AI hardware excels at matrix tensor cores.',
      },
      {
        title: '3. Connecting Forward to Backprop',
        content: 'During inference, only the forward pass runs. During training, the prediction error generates gradients that travel backwards via the Chain Rule to update each weight matrix.',
        keyTakeaway: 'The forward activations computed here are cached in memory for use during the backward gradient pass.',
      },
    ],
    timeComplexity: 'O(∑ d_{in} · d_{out})',
    spaceComplexity: 'O(∑ d_l) activations',
    tags: ['Neural Network', 'MLP', 'ReLU', 'Forward Pass', 'PyTorch'],
    defaultInput: 'x1: 0.8, x2: 0.4',
    pythonCode: `import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 2 features, Hidden: 3 neurons, Output: 1 score
        self.fc1 = nn.Linear(2, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Hidden Layer Linear Projection
        z1 = self.fc1(x)
        # 2. ReLU Activation
        h1 = self.relu(z1)
        # 3. Output Layer Linear Projection
        z2 = self.fc2(h1)
        # 4. Sigmoid Probability Output
        y_hat = self.sigmoid(z2)
        return y_hat`,
    code: `import torch
import torch.nn as nn

def forward_mlp(x, W1, b1, W2, b2):
    # 1. Hidden Layer Weighted Sum
    z1 = np.dot(x, W1) + b1
    # 2. Non-linear Activation (ReLU)
    h1 = np.maximum(0, z1)
    # 3. Output Layer Weighted Sum
    z2 = np.dot(h1, W2) + b2
    # 4. Sigmoid Probability Activation
    y_hat = 1 / (1 + np.exp(-z2))
    return y_hat`,
    generateTraces: (_inputStr?: string, lang: Language = 'en'): TraceStep[] => {
      const isVi = lang === 'vi';
      const steps: TraceStep[] = [];

      // Step 1: Input Layer
      steps.push({
        step: 0,
        totalSteps: 0,
        line: 2,
        action: 'init',
        targets: [0, 1],
        state: {
          layers: [
            {
              name: isVi ? 'Tầng Đầu vào (Input)' : 'Input Layer',
              nodes: [
                { label: 'x₁', value: 0.8 },
                { label: 'x₂', value: 0.4 },
              ],
            },
            {
              name: isVi ? 'Tầng Ẩn (Hidden)' : 'Hidden Layer',
              nodes: [
                { label: 'h₁', value: 0 },
                { label: 'h₂', value: 0 },
                { label: 'h₃', value: 0 },
              ],
            },
            {
              name: isVi ? 'Tầng Đầu ra (Output)' : 'Output Layer',
              nodes: [{ label: 'ŷ', value: 0 }],
            },
          ],
        },
        explanation: isVi
          ? 'Tầng đầu vào khởi tạo với vector đặc trưng x = [0.8, 0.4]. Sẵn sàng tính toán lan truyền tiến.'
          : 'Input Layer initialized with feature vector x = [0.8, 0.4]. Ready to compute forward activation.',
        invariant: isVi
          ? 'Đặc trưng đầu vào được chuẩn hóa trong khoảng [0, 1].'
          : 'Input features are normalized in range [0, 1].',
      });

      // Step 2: Linear transformation into Hidden Layer (z1)
      steps.push({
        step: 1,
        totalSteps: 0,
        line: 4,
        action: 'compute',
        targets: [0, 1, 2],
        state: {
          layers: [
            {
              name: isVi ? 'Tầng Đầu vào' : 'Input Layer',
              nodes: [
                { label: 'x₁', value: 0.8 },
                { label: 'x₂', value: 0.4 },
              ],
            },
            {
              name: isVi ? 'Tầng Ẩn (Giá trị z₁)' : 'Hidden Layer (Pre-activation z₁)',
              nodes: [
                { label: 'h₁', value: 1.12 },
                { label: 'h₂', value: -0.45 },
                { label: 'h₃', value: 0.84 },
              ],
            },
            {
              name: isVi ? 'Tầng Đầu ra' : 'Output Layer',
              nodes: [{ label: 'ŷ', value: 0 }],
            },
          ],
        },
        explanation: isVi
          ? 'Tính tổ hợp tuyến tính z₁ = W₁ · x + b₁. Lưu ý nơ-ron h₂ = -0.45 mang giá trị âm.'
          : 'Computed linear combination z₁ = W₁ · x + b₁. Notice node h₂ = -0.45 is negative.',
        invariant: isVi
          ? 'Phép chiếu affine z = Wx + b bảo toàn tính tuyến tính không gian.'
          : 'Affine projection z = Wx + b preserves linear dimensionality.',
      });

      // Step 3: ReLU Activation
      steps.push({
        step: 2,
        totalSteps: 0,
        line: 6,
        action: 'forward',
        targets: [0, 1, 2],
        state: {
          layers: [
            {
              name: isVi ? 'Tầng Đầu vào' : 'Input Layer',
              nodes: [
                { label: 'x₁', value: 0.8 },
                { label: 'x₂', value: 0.4 },
              ],
            },
            {
              name: isVi ? 'Tầng Ẩn (Đã kích hoạt ReLU)' : 'Hidden Layer (Activated h₁)',
              nodes: [
                { label: 'h₁', value: 1.12, activation: 'ReLU' },
                { label: 'h₂', value: 0.0, activation: 'ReLU' }, // clamped by ReLU!
                { label: 'h₃', value: 0.84, activation: 'ReLU' },
              ],
            },
            {
              name: isVi ? 'Tầng Đầu ra' : 'Output Layer',
              nodes: [{ label: 'ŷ', value: 0 }],
            },
          ],
        },
        explanation: isVi
          ? 'Áp dụng hàm kích hoạt ReLU: max(0, z). Nơ-ron h₂ (-0.45) bị triệt tiêu về 0.0 (tính thưa thớt của ReLU).'
          : 'Applied ReLU activation: max(0, z). Node h₂ (-0.45) clamped to 0.0 (inactive neuron sparsity).',
        invariant: isVi
          ? 'ReLU(z) ép giá trị không âm: a ∈ [0, ∞).'
          : 'ReLU(z) enforces non-negativity: a ∈ [0, ∞).',
      });

      // Step 4: Output Layer projection & Sigmoid
      steps.push({
        step: 3,
        totalSteps: 0,
        line: 8,
        action: 'done',
        targets: [0],
        state: {
          loss: 0.042,
          layers: [
            {
              name: isVi ? 'Tầng Đầu vào' : 'Input Layer',
              nodes: [
                { label: 'x₁', value: 0.8 },
                { label: 'x₂', value: 0.4 },
              ],
            },
            {
              name: isVi ? 'Tầng Ẩn (Hoàn tất)' : 'Hidden Layer (Activated)',
              nodes: [
                { label: 'h₁', value: 1.12 },
                { label: 'h₂', value: 0.0 },
                { label: 'h₃', value: 0.84 },
              ],
            },
            {
              name: isVi ? 'Tầng Đầu ra (Sigmoid)' : 'Output Layer',
              nodes: [{ label: 'ŷ', value: 0.87, activation: 'Sigmoid' }],
            },
          ],
        },
        explanation: isVi
          ? 'Lan truyền xuôi hoàn tất! Nơ-ron đầu ra ŷ = 0.87 (87% xác suất nhãn dương). Mất mát Binary Cross-Entropy: 0.042.'
          : 'Forward pass complete! Final output neuron ŷ = 0.87 (87% positive class probability). Binary Cross-Entropy Loss: 0.042.',
        invariant: isVi
          ? 'Đầu ra được chuẩn hóa thành phân phối xác suất hợp lệ qua hàm Sigmoid.'
          : 'Output normalized into valid class probability distribution via Sigmoid.',
      });

      const total = steps.length;
      return steps.map((s, idx) => ({ ...s, step: idx + 1, totalSteps: total }));
    },
  },

  {
    id: 'cnn-feature-map',
    title: '04. Convolutional Neural Network (CNN) 2D Feature Map',
    category: 'Deep Learning & Neural Networks',
    difficulty: 'Intermediate',
    badge: 'Computer Vision',
    visualType: 'cnn-convolution',
    formula: 'S(i, j) = (I * K)(i, j) = \\sum_{m} \\sum_{n} I(i + m, j + n) \\cdot K(m, n)',
    summary: 'Slides a learnable 3x3 filter kernel over an input image matrix to detect local visual features like edges and textures.',
    description: 'Convolutional layers use shared weights and local receptive fields to achieve translation invariance. Traces step-by-step 2D kernel sliding, element-wise dot products, bias addition, and ReLU activation.',
    inputVariants: [
      { label: 'Vertical Edge (Sobel-X)', value: 'vertical-edge' },
      { label: 'Horizontal Edge (Sobel-Y)', value: 'horizontal-edge' },
      { label: 'Ridge Filter', value: 'ridge' },
      { label: 'Identity Filter', value: 'identity' },
    ],
    vi: {
      title: '04. Mạng Nơ-ron Tích chập (CNN) & Bản đồ Đặc trưng (Feature Map)',
      category: 'Deep Learning & Mạng Nơ-ron',
      badge: 'Computer Vision',
      summary: 'Trượt kernel lọc 3x3 qua ma trận ảnh đầu vào để trích xuất các đặc trưng thị giác cục bộ như đường biên và họa tiết.',
      description: 'Tầng tích chập (Conv2d) sử dụng chia sẻ trọng số và vùng tiếp nhận cục bộ để đạt tính bất biến tịnh tiến. Trực quan hóa từng bước trượt kernel 2D, nhân tích chập vô hướng và kích hoạt ReLU.',
      variants: [
        { label: 'Đường biên dọc (Sobel-X)', value: 'vertical-edge' },
        { label: 'Đường biên ngang (Sobel-Y)', value: 'horizontal-edge' },
        { label: 'Bộ lọc Ridge', value: 'ridge' },
        { label: 'Bộ lọc Identity', value: 'identity' },
      ],
      sections: [
        {
          title: '1. Tại sao Tích chập (Convolution) tối ưu cho Thị giác Máy tính?',
          content: 'Nếu làm phẳng một bức ảnh 2D thành một vector dài để đưa vào MLP, số lượng trọng số sẽ bùng nổ lên hàng triệu tham số và làm mất hoàn toàn cấu trúc không gian 2D. Tầng tích chập dùng một kernel nhỏ (ví dụ 3x3) trượt khắp bức ảnh, chia sẻ chung trọng số cho toàn bộ vị trí.',
          formula: '(I * K)(i, j) = \\sum_{m=0}^{k-1} \\sum_{n=0}^{k-1} I(i+m, j+n) K(m, n)',
          keyTakeaway: 'Tính bất biến tịnh tiến (Translation Invariance): Một đường biên hay chi tiết được nhận diện chính xác bất kể nó nằm ở góc nào của bức ảnh.',
        },
        {
          title: '2. Bản đồ Đặc trưng (Feature Map) hình thành như thế nào?',
          content: 'Tại mỗi vị trí trượt, kernel tính tích vô hướng phần tử (element-wise multiplication) với vùng ảnh tương ứng (receptive field) rồi cộng dồn lại. Giá trị tổng này đại diện cho cường độ đặc trưng tại vị trí đó.',
          keyTakeaway: 'Kích thước đầu ra: Output = (Input - Kernel + 2*Padding) / Stride + 1. Ví dụ ảnh 5x5 qua kernel 3x3 tạo ra Feature Map 3x3.',
        },
        {
          title: '3. Kích hoạt ReLU và Tầng Gom cụm (Max Pooling)',
          content: 'Sau khi tích chập, hàm ReLU (max(0, z)) loại bỏ các giá trị âm để tăng tính phi tuyến. Tiếp đó, tầng Max Pooling lấy giá trị cực đại trong từng cửa sổ 2x2 để giảm một nửa kích thước không gian và giảm chi phí tính toán.',
          keyTakeaway: 'Các tầng tích chập đầu tiên học các đường biên/góc cơ bản; các tầng sâu hơn tổng hợp thành hình dạng, bộ phận và vật thể hoàn chỉnh.',
        },
      ],
    },
    sections: [
      {
        title: '1. Why Convolutions for Computer Vision?',
        content: 'Flattening a 2D image into a 1D vector for a standard MLP destroys 2D spatial relationships and causes an explosion of millions of redundant parameters. CNNs solve this by sliding a compact 3x3 kernel across the entire image with shared weights.',
        formula: '(I * K)(i, j) = \\sum_{m=0}^{k-1} \\sum_{n=0}^{k-1} I(i+m, j+n) K(m, n)',
        keyTakeaway: 'Translation Invariance: Features like edges, textures, or shapes are detected identically regardless of their spatial location in the frame.',
      },
      {
        title: '2. Generating the 2D Feature Map',
        content: 'At each stride position, the kernel performs an element-wise dot product with the underlying receptive field patch and sums the results into a single scalar in the output feature map.',
        keyTakeaway: 'Output Spatial Dimension Formula: Out = (Input - Kernel + 2*Padding) / Stride + 1. A 5x5 image convolved with a 3x3 kernel yields a 3x3 feature map.',
      },
      {
        title: '3. Non-Linear ReLU and Downsampling via Max Pooling',
        content: 'Following convolution, ReLU activations clamp negative responses to introduce sparsity. Downstream Max Pooling layers extract the maximum activation in 2x2 windows, reducing dimensions while retaining prominent visual features.',
        keyTakeaway: 'Early CNN layers capture low-level edges; deeper layers assemble these edges into eyes, wheels, and complex objects.',
      },
    ],
    timeComplexity: 'O(H · W · kH · kW)',
    spaceComplexity: 'O(C_{out} · H_{out} · W_{out})',
    tags: ['CNN', 'Computer Vision', 'PyTorch', 'Feature Map', 'Sobel Filter'],
    defaultInput: 'vertical-edge',
    pythonCode: `import torch
import torch.nn as nn
import numpy as np

# 1. PyTorch Convolutional Layer
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input channel, 1 output feature map, 3x3 kernel
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)   # 2D Convolution
        x = self.relu(x)   # Non-linear activation
        x = self.pool(x)   # Downsampling
        return x

# 2. Pure NumPy 2D Convolution Operation
def conv2d_forward(image, kernel, bias=0.0):
    H, W = image.shape
    kH, kW = kernel.shape
    out_H, out_W = H - kH + 1, W - kW + 1
    feature_map = np.zeros((out_H, out_W))
    
    for r in range(out_H):
        for c in range(out_W):
            receptive_field = image[r:r+kH, c:c+kW]
            feature_map[r, c] = np.sum(receptive_field * kernel) + bias
    return feature_map`,
    code: `import torch
import torch.nn as nn

def conv2d(image, kernel, bias=0.0):
    H, W = image.shape
    kH, kW = kernel.shape
    feature_map = np.zeros((H - kH + 1, W - kW + 1))
    for r in range(H - kH + 1):
        for c in range(W - kW + 1):
            patch = image[r:r+kH, c:c+kW]
            feature_map[r, c] = np.sum(patch * kernel) + bias
    return np.maximum(0, feature_map)  # ReLU`,
    generateTraces: (_inputStr?: string, lang: Language = 'en'): TraceStep[] => {
      const isVi = lang === 'vi';
      // 5x5 input image containing a vertical boundary between high intensity (10) and low (0)
      const inputMatrix = [
        [10, 10, 10,  0,  0],
        [10, 10, 10,  0,  0],
        [10, 10, 10,  0,  0],
        [10, 10, 10,  0,  0],
        [10, 10, 10,  0,  0],
      ];
      // 3x3 Sobel Vertical Edge Detection Filter
      const kernel = [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
      ];

      const steps: TraceStep[] = [];

      // Step 0: Init at (0, 0)
      steps.push({
        step: 0,
        totalSteps: 0,
        line: 4,
        action: 'init',
        targets: [0],
        state: {
          inputMatrix,
          kernel,
          kernelPos: { r: 0, c: 0 },
          featureMap: [
            [0, null, null],
            [null, null, null],
            [null, null, null],
          ],
        },
        explanation: isVi
          ? 'Khởi tạo ma trận ảnh 5x5 và Kernel lọc biên dọc Sobel 3x3 tại vị trí (r=0, c=0). Receptive field nằm trên vùng đồng nhất nên tổng tích chập = 0.'
          : 'Initialized 5x5 image matrix and 3x3 Sobel vertical edge kernel at (r=0, c=0). Uniform receptive field yields dot product = 0.',
        invariant: isVi
          ? 'Kích thước Feature Map đầu ra: (5 - 3 + 1) x (5 - 3 + 1) = 3x3.'
          : 'Output Feature Map dimensions: (H - kH + 1) x (W - kW + 1) = 3x3.',
      });

      // Step 1: Slide to (0, 1) across the vertical edge!
      steps.push({
        step: 1,
        totalSteps: 0,
        line: 8,
        action: 'forward',
        targets: [1],
        state: {
          inputMatrix,
          kernel,
          kernelPos: { r: 0, c: 1 },
          featureMap: [
            [0, 40, null],
            [null, null, null],
            [null, null, null],
          ],
        },
        explanation: isVi
          ? 'Trượt Kernel sang vị trí (r=0, c=1). Cửa sổ bao trùm ranh giới sáng-tối! Tích vô hướng đạt giá trị cao = 40.0. Phát hiện đường biên dọc!'
          : 'Slid Kernel horizontally to (r=0, c=1). Window spans the edge transition! Dot product peaks at 40.0. Strong vertical edge detected!',
        invariant: isVi
          ? 'Tích chập nhân từng phần tử: ∑ (ReceptiveField ⊙ Kernel) = 1(10) + 2(10) + 1(10) = 40.'
          : 'Element-wise multiplication: ∑ (Patch ⊙ Kernel) = 1(10) + 2(10) + 1(10) = 40.',
      });

      // Step 2: Slide to (0, 2)
      steps.push({
        step: 2,
        totalSteps: 0,
        line: 8,
        action: 'forward',
        targets: [2],
        state: {
          inputMatrix,
          kernel,
          kernelPos: { r: 0, c: 2 },
          featureMap: [
            [0, 40, 30],
            [null, null, null],
            [null, null, null],
          ],
        },
        explanation: isVi
          ? 'Trượt Kernel sang vị trí (r=0, c=2). Hoàn thành hàng đầu tiên của Feature Map. Giá trị tích chập = 30.0.'
          : 'Slid Kernel to (r=0, c=2). Completed row 0 of Feature Map with value 30.0.',
        invariant: isVi
          ? 'Kernel trượt ngang với Stride = 1 đến khi chạm biên phải.'
          : 'Horizontal stride = 1 maintained until row boundary.',
      });

      // Step 3: Stride down to row 1, col 1 (center)
      steps.push({
        step: 3,
        totalSteps: 0,
        line: 8,
        action: 'forward',
        targets: [4],
        state: {
          inputMatrix,
          kernel,
          kernelPos: { r: 1, c: 1 },
          featureMap: [
            [0, 40, 30],
            [0, 40, 30],
            [null, null, null],
          ],
        },
        explanation: isVi
          ? 'Xuống hàng tiếp theo (Stride dọc = 1), trượt tới trung tâm (r=1, c=1). Tiếp tục phát hiện đường biên dọc với giá trị cực đại 40.0.'
          : 'Strided down to row 1, centered at (r=1, c=1). Senses vertical edge continuously with peak response 40.0.',
        invariant: isVi
          ? 'Chia sẻ tham số (Parameter sharing): Sử dụng cùng một bộ lọc 3x3 cho toàn bộ các vị trí ảnh.'
          : 'Parameter sharing: Identical 3x3 weights applied across all spatial receptive fields.',
      });

      // Step 4: Done - Full Feature Map and ReLU
      steps.push({
        step: 4,
        totalSteps: 0,
        line: 10,
        action: 'done',
        targets: [8],
        state: {
          inputMatrix,
          kernel,
          kernelPos: { r: 2, c: 2 },
          featureMap: [
            [0, 40, 30],
            [0, 40, 30],
            [0, 40, 30],
          ],
        },
        explanation: isVi
          ? 'Quá trình tích chập hoàn tất! Tạo thành Feature Map 3x3 với dải biên dọc kích hoạt mạnh ở cột giữa (40.0). Áp dụng ReLU giữ nguyên các đặc trưng tích cực.'
          : '2D Convolution complete! Generated full 3x3 Feature Map exhibiting a bright vertical activation band (40.0) where the edge exists.',
        invariant: isVi
          ? 'Bản đồ đặc trưng trích xuất thành công cạnh dọc từ ảnh gốc 5x5.'
          : 'Feature map successfully extracts structural vertical edge from raw 5x5 pixels.',
      });

      const total = steps.length;
      return steps.map((s, idx) => ({ ...s, step: idx + 1, totalSteps: total }));
    },
  },

  {
    id: 'house-price-mlp',
    title: '05. House Price Prediction with Neural Network',
    category: 'Deep Learning & Neural Networks',
    difficulty: 'Intermediate',
    badge: 'Regression MLP',
    visualType: 'neural-network',
    formula: '\\hat{price} = W_2 \\cdot \\text{ReLU}(W_1 \\mathbf{x} + b_1) + b_2',
    summary: 'Predicts house prices from area, bedrooms, and location score using a 3→4→1 ReLU MLP regression network.',
    description: 'A practical regression MLP maps three normalized house features (area m², bedrooms, location quality score) through a hidden ReLU layer of 4 neurons to produce a continuous price prediction in millions USD. Demonstrates how MLPs replace linear regression for non-linear real-estate patterns.',
    inputVariants: [
      { label: 'Suburban (80m², 2BR, 0.7)', value: 'area: 0.80, rooms: 0.40, loc: 0.70' },
      { label: 'City Center (120m², 3BR, 0.95)', value: 'area: 1.00, rooms: 0.75, loc: 0.95' },
      { label: 'Rural (60m², 2BR, 0.30)', value: 'area: 0.60, rooms: 0.40, loc: 0.30' },
      { label: 'Luxury (200m², 5BR, 1.0)', value: 'area: 1.00, rooms: 1.00, loc: 1.00' },
    ],
    vi: {
      title: '05. Dự đoán Giá Nhà bằng Neural Network',
      category: 'Deep Learning & Mạng Nơ-ron',
      badge: 'Regression MLP',
      summary: 'Dự đoán giá nhà từ diện tích, số phòng ngủ và điểm vị trí bằng mạng MLP hồi quy 3→4→1 với ReLU.',
      description: 'MLP hồi quy thực tế ánh xạ 3 đặc trưng nhà (diện tích m², số phòng ngủ, điểm chất lượng vị trí) qua một hidden layer ReLU 4 nơ-ron để dự đoán giá liên tục tính bằng tỷ đồng. Minh chứng cách MLP vượt qua hồi quy tuyến tính trong các bài toán bất động sản phi tuyến.',
      variants: [
        { label: 'Ngoại ô (80m², 2PN, 0.7)', value: 'area: 0.80, rooms: 0.40, loc: 0.70' },
        { label: 'Trung tâm (120m², 3PN, 0.95)', value: 'area: 1.00, rooms: 0.75, loc: 0.95' },
        { label: 'Vùng quê (60m², 2PN, 0.30)', value: 'area: 0.60, rooms: 0.40, loc: 0.30' },
        { label: 'Cao cấp (200m², 5PN, 1.0)', value: 'area: 1.00, rooms: 1.00, loc: 1.00' },
      ],
      sections: [
        {
          title: '1. Tại sao MLP tốt hơn Hồi quy Tuyến tính cho Bất động sản?',
          content: 'Giá nhà phụ thuộc vào các tương tác phi tuyến phức tạp: vị trí đắt đỏ x diện tích lớn x tầng cao → giá tăng theo hàm mũ, không tuyến tính. Hồi quy tuyến tính y = wx + b buộc phải mô hình hóa tất cả bằng một siêu phẳng phẳng. Hidden ReLU layers của MLP giúp học các ngưỡng phi tuyến như "vị trí > 0.8 AND diện tích > 0.9 → giá tăng đột biến".',
          formula: '\\text{ReLU}(z) = \\max(0, z) \\quad \\Rightarrow \\quad \\text{học ngưỡng phi tuyến}',
          keyTakeaway: 'MLP với hidden layers là "Universal Approximator": có thể xấp xỉ bất kỳ hàm số liên tục nào, kể cả đường cong giá bất động sản.',
        },
        {
          title: '2. Chuẩn hóa đặc trưng (Feature Normalization) trước khi đưa vào MLP',
          content: 'Diện tích tính bằng m² (60–300), số phòng (1–6), và điểm vị trí (0–1) có thang đo hoàn toàn khác nhau. Nếu không chuẩn hóa, gradient descent sẽ bị thiên vị nặng về đặc trưng có giá trị lớn (diện tích). Min-Max normalization đưa tất cả về [0, 1].',
          formula: 'x_{\\text{norm}} = \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}}',
          keyTakeaway: 'Batch Normalization trong deep learning hiện đại thực hiện bước chuẩn hóa này ngay bên trong từng layer.',
        },
        {
          title: '3. Hàm Mất mát MSE cho Bài toán Hồi quy',
          content: 'Khác với bài toán phân loại dùng Binary Cross-Entropy, hồi quy giá nhà dùng Mean Squared Error (MSE): tính trung bình bình phương khoảng cách giữa giá dự đoán và giá thực tế. Backpropagation cực tiểu hóa MSE theo toàn bộ tham số W₁, b₁, W₂, b₂.',
          formula: 'L_{MSE} = \\frac{1}{N} \\sum_{i=1}^N (\\hat{y}_i - y_i)^2',
          keyTakeaway: 'MSE phạt nặng sai số lớn do bình phương, thúc đẩy mô hình học cẩn thận với các nhà có giá đặc biệt.',
        },
      ],
    },
    sections: [
      {
        title: '1. Why MLP Beats Linear Regression for Real Estate',
        content: 'House prices depend on complex non-linear interactions: premium location × large area × high floor → price increases exponentially, not linearly. Linear regression y = wx + b forces all of this into a flat hyperplane. MLP hidden layers learn non-linear thresholds like "location > 0.8 AND area > 0.9 → price spikes".',
        formula: '\\text{ReLU}(z) = \\max(0, z) \\quad \\Rightarrow \\quad \\text{non-linear threshold learning}',
        keyTakeaway: 'MLP is the Universal Approximator: it can approximate any continuous function — including the complex curves of real estate prices.',
      },
      {
        title: '2. Feature Normalization Before Feeding into MLP',
        content: 'Area in m² (60–300), bedrooms (1–6), and location score (0–1) are on completely different scales. Without normalization, gradient descent is heavily biased toward the highest-magnitude feature. Min-Max normalization maps everything to [0, 1].',
        formula: 'x_{\\text{norm}} = \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}}',
        keyTakeaway: 'Modern deep learning handles this with Batch Normalization layers inside the network itself.',
      },
      {
        title: '3. MSE Loss Function for Regression',
        content: 'Unlike classification (Binary Cross-Entropy), house price regression uses Mean Squared Error (MSE): the average squared distance between predicted and actual price. Backpropagation minimizes MSE over all parameters W₁, b₁, W₂, b₂.',
        formula: 'L_{MSE} = \\frac{1}{N} \\sum_{i=1}^N (\\hat{y}_i - y_i)^2',
        keyTakeaway: 'MSE heavily penalizes large errors via squaring, driving the model to be precise on outlier-priced houses.',
      },
    ],
    timeComplexity: 'O(d_in · d_h + d_h · d_out)',
    spaceComplexity: 'O(d_h) hidden activations',
    tags: ['Neural Network', 'MLP', 'Regression', 'ReLU', 'House Price', 'PyTorch'],
    defaultInput: 'area: 0.80, rooms: 0.40, loc: 0.70',
    pythonCode: `import torch
import torch.nn as nn

class HousePriceNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 features → 4 ReLU hidden → 1 price output
        self.net = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        return self.net(x)

# Inputs: [area_norm, rooms_norm, location_score]
x = torch.tensor([[0.80, 0.40, 0.70]])
model = HousePriceNet()
price = model(x)
print(f"Predicted price: \${price.item():.2f}M")`,
    code: `import numpy as np

def house_price_mlp(x, W1, b1, W2, b2):
    # 1. Hidden: Linear + ReLU
    z1 = np.dot(x, W1) + b1
    h1 = np.maximum(0, z1)    # ReLU
    # 2. Output: Linear (regression)
    z2 = np.dot(h1, W2) + b2
    return z2                  # price in $M`,
    generateTraces: (inputStr?: string, lang: Language = 'en'): TraceStep[] => {
      const isVi = lang === 'vi';
      let area = 0.80, rooms = 0.40, loc = 0.70;
      if (inputStr) {
        const ma = inputStr.match(/area:\s*(-?[0-9.]+)/i);
        const mr = inputStr.match(/rooms:\s*(-?[0-9.]+)/i);
        const ml = inputStr.match(/loc:\s*(-?[0-9.]+)/i);
        if (ma) area = Number(ma[1]);
        if (mr) rooms = Number(mr[1]);
        if (ml) loc = Number(ml[1]);
      }
      // Fixed weights for repeatable educational demo
      const W1 = [[0.6, 0.3, -0.2, 0.8], [0.4, 0.7, 0.5, -0.3], [0.9, 0.2, 0.6, 0.4]];
      const b1 = [-0.1, 0.2, -0.3, 0.1];
      const W2 = [0.5, 0.4, 0.3, 0.6];
      const b2 = 0.15;
      const x = [area, rooms, loc];
      const z1 = b1.map((bi, j) => Number((x[0]*W1[0][j] + x[1]*W1[1][j] + x[2]*W1[2][j] + bi).toFixed(3)));
      const h1 = z1.map(z => Number(Math.max(0, z).toFixed(3)));
      const price = Number((h1.reduce((s, h, j) => s + h * W2[j], b2)).toFixed(3));
      const inNodes = [
        { label: 'x₁', value: area, activation: isVi ? 'Diện tích' : 'Area' },
        { label: 'x₂', value: rooms, activation: isVi ? 'Phòng ngủ' : 'Rooms' },
        { label: 'x₃', value: loc, activation: isVi ? 'Vị trí' : 'Location' },
      ];
      const steps: TraceStep[] = [];

      steps.push({
        step: 0, totalSteps: 0, line: 1, action: 'init', targets: [],
        state: {
          layers: [
            { name: isVi ? 'Đặc trưng Nhà (x)' : 'House Features (x)', nodes: inNodes },
            { name: isVi ? 'Tầng ẩn ReLU' : 'Hidden Layer (ReLU)',
              nodes: [{ label: 'h₁', value: 0 }, { label: 'h₂', value: 0 }, { label: 'h₃', value: 0 }, { label: 'h₄', value: 0 }] },
            { name: isVi ? 'Giá dự đoán ($M)' : 'Predicted Price ($M)',
              nodes: [{ label: 'ŷ', value: 0, activation: isVi ? 'Đang tính...' : 'Pending' }] },
          ],
        },
        explanation: isVi
          ? `Đầu vào: Diện tích = ${area.toFixed(2)}, Phòng ngủ = ${rooms.toFixed(2)}, Vị trí = ${loc.toFixed(2)}. Sẵn sàng lan truyền tiến qua mạng hồi quy.`
          : `Input: Area = ${area.toFixed(2)}, Rooms = ${rooms.toFixed(2)}, Location = ${loc.toFixed(2)}. Ready to forward propagate through regression network.`,
        invariant: isVi ? 'Đặc trưng đầu vào đã chuẩn hóa trong [0, 1] bằng Min-Max Normalization.' : 'Input features normalized to [0, 1] via Min-Max Normalization.',
      });

      steps.push({
        step: 0, totalSteps: 0, line: 4, action: 'compute', targets: [],
        state: {
          layers: [
            { name: isVi ? 'Đặc trưng Nhà (x)' : 'House Features (x)', nodes: inNodes },
            { name: isVi ? 'Tầng ẩn — z₁ = W₁·x + b₁' : 'Hidden Layer — z₁ = W₁·x + b₁',
              nodes: [{ label: 'z₁', value: z1[0] }, { label: 'z₂', value: z1[1] }, { label: 'z₃', value: z1[2] }, { label: 'z₄', value: z1[3] }] },
            { name: isVi ? 'Giá dự đoán ($M)' : 'Predicted Price ($M)',
              nodes: [{ label: 'ŷ', value: 0, activation: isVi ? 'Đang tính...' : 'Pending' }] },
          ],
        },
        explanation: isVi
          ? `Phép chiếu affine: z₁ = W₁·x + b₁ → [${z1.map(v => v.toFixed(2)).join(', ')}]. Một số giá trị âm sẽ bị ReLU triệt tiêu.`
          : `Affine projection: z₁ = W₁·x + b₁ → [${z1.map(v => v.toFixed(2)).join(', ')}]. Negative values will be killed by ReLU.`,
        invariant: isVi ? 'Tổ hợp tuyến tính z = Wx + b bảo toàn tính tuyến tính.' : 'Linear combination z = Wx + b preserves linearity.',
      });

      steps.push({
        step: 0, totalSteps: 0, line: 6, action: 'forward', targets: [],
        state: {
          layers: [
            { name: isVi ? 'Đặc trưng Nhà (x)' : 'House Features (x)', nodes: inNodes },
            { name: isVi ? 'Tầng ẩn — h₁ = ReLU(z₁)' : 'Hidden Layer — h₁ = ReLU(z₁)',
              nodes: [{ label: 'h₁', value: h1[0], activation: 'ReLU' }, { label: 'h₂', value: h1[1], activation: 'ReLU' }, { label: 'h₃', value: h1[2], activation: 'ReLU' }, { label: 'h₄', value: h1[3], activation: 'ReLU' }] },
            { name: isVi ? 'Giá dự đoán ($M)' : 'Predicted Price ($M)',
              nodes: [{ label: 'ŷ', value: 0, activation: isVi ? 'Đang tính...' : 'Pending' }] },
          ],
        },
        explanation: isVi
          ? `ReLU triệt tiêu giá trị âm: h₁ = max(0, z₁) → [${h1.map(v => v.toFixed(2)).join(', ')}]. Tính thưa thớt giúp mạng tập trung vào tín hiệu quan trọng nhất.`
          : `ReLU kills negatives: h₁ = max(0, z₁) → [${h1.map(v => v.toFixed(2)).join(', ')}]. Sparsity focuses the network on the strongest feature signals.`,
        invariant: isVi ? 'ReLU đảm bảo h₁ ∈ [0, ∞) — gradient luôn dương với nơ-ron đang kích hoạt.' : 'ReLU ensures h₁ ∈ [0, ∞) — gradient stays positive for active neurons.',
      });

      steps.push({
        step: 0, totalSteps: 0, line: 8, action: 'done', targets: [],
        state: {
          layers: [
            { name: isVi ? 'Đặc trưng Nhà (x)' : 'House Features (x)', nodes: inNodes },
            { name: isVi ? 'Tầng ẩn (Hoàn tất)' : 'Hidden Layer (Done)',
              nodes: [{ label: 'h₁', value: h1[0] }, { label: 'h₂', value: h1[1] }, { label: 'h₃', value: h1[2] }, { label: 'h₄', value: h1[3] }] },
            { name: isVi ? 'Giá dự đoán ($M)' : 'Predicted Price ($M)',
              nodes: [{ label: 'ŷ', value: price, activation: isVi ? `$${price.toFixed(2)}M` : `$${price.toFixed(2)}M` }] },
          ],
        },
        explanation: isVi
          ? `Hoàn tất! Tầng đầu ra tuyến tính: giá = W₂·h₁ + b₂ = $${price.toFixed(2)}M. Không dùng Sigmoid vì đây là bài toán hồi quy giá trị liên tục.`
          : `Done! Output linear projection: price = W₂·h₁ + b₂ = $${price.toFixed(2)}M. No Sigmoid — this is continuous-value regression, not classification.`,
        invariant: isVi ? 'Tầng đầu ra hồi quy không có hàm kích hoạt (identity) để giá trị không bị giới hạn trong [0,1].' : 'Regression output has no activation (identity) — price is unbounded, unlike probability.',
      });

      const total = steps.length;
      return steps.map((s, idx) => ({ ...s, step: idx + 1, totalSteps: total }));
    },
  },

  {
    id: 'dog-cat-mlp',
    title: '06. Dog vs. Cat Classification (MLP Classifier)',
    category: 'Deep Learning & Neural Networks',
    difficulty: 'Intermediate',
    badge: 'Binary Classification',
    visualType: 'neural-network',
    formula: 'P(\\text{Cat}) = \\sigma(W_2 \\cdot \\text{ReLU}(W_1 \\mathbf{x} + b_1) + b_2)',
    summary: 'Classifies images as Dog or Cat from 3 visual features (texture, brightness, edge density) using a 3→4→4→1 deep MLP.',
    description: 'A two-hidden-layer MLP binary classifier that takes normalized visual features extracted from animal images and outputs a class probability via Sigmoid. Demonstrates how stacking non-linear layers enables complex decision boundaries needed to separate cats from dogs.',
    inputVariants: [
      { label: 'Likely Cat (smooth, bright)', value: 'texture: 0.2, bright: 0.8, edges: 0.3' },
      { label: 'Likely Dog (rough, dark)', value: 'texture: 0.8, bright: 0.4, edges: 0.9' },
      { label: 'Ambiguous', value: 'texture: 0.5, bright: 0.5, edges: 0.5' },
      { label: 'High Contrast Dog', value: 'texture: 0.9, bright: 0.3, edges: 1.0' },
    ],
    vi: {
      title: '06. Phân loại Chó vs. Mèo bằng MLP',
      category: 'Deep Learning & Mạng Nơ-ron',
      badge: 'Phân loại Nhị phân',
      summary: 'Phân loại ảnh là Chó hoặc Mèo từ 3 đặc trưng thị giác (kết cấu, độ sáng, mật độ cạnh) dùng MLP sâu 3→4→4→1.',
      description: 'MLP phân loại nhị phân hai hidden layers nhận đặc trưng thị giác chuẩn hóa từ ảnh động vật và xuất xác suất lớp qua Sigmoid. Minh họa cách xếp chồng các lớp phi tuyến giúp học ranh giới quyết định phức tạp để phân biệt mèo và chó.',
      variants: [
        { label: 'Khả năng là Mèo (mịn, sáng)', value: 'texture: 0.2, bright: 0.8, edges: 0.3' },
        { label: 'Khả năng là Chó (thô, tối)', value: 'texture: 0.8, bright: 0.4, edges: 0.9' },
        { label: 'Mơ hồ (50/50)', value: 'texture: 0.5, bright: 0.5, edges: 0.5' },
        { label: 'Chó tương phản cao', value: 'texture: 0.9, bright: 0.3, edges: 1.0' },
      ],
      sections: [
        {
          title: '1. Tại sao cần 2 Hidden Layers để Phân loại Chó/Mèo?',
          content: 'Chó và mèo có hình dạng lông (texture), tư thế và nét mặt khác nhau, nhưng ranh giới phân loại là phi tuyến và phức tạp. Layer 1 học các đặc trưng nguyên thủy như "có mịn không?". Layer 2 học các tổ hợp cao hơn như "mịn + mắt tròn → mèo". Chỉ một layer ẩn không đủ sức học được sự kết hợp phức tạp này.',
          formula: 'a^{[2]} = \\text{ReLU}(W^{[2]} \\cdot \\text{ReLU}(W^{[1]} x + b^{[1]}) + b^{[2]})',
          keyTakeaway: 'Mỗi layer học đặc trưng ở độ trừu tượng ngày càng cao hơn — đây là bản chất của Deep Learning.',
        },
        {
          title: '2. Hàm Sigmoid và Ý nghĩa Xác suất Lớp',
          content: 'Tầng đầu ra không dùng ReLU mà dùng Sigmoid để ép đầu ra vào [0, 1]. Giá trị này được diễn giải là P(Chó) — xác suất hình ảnh thuộc lớp Chó. Quy tắc quyết định: nếu P(Chó) > 0.5 → "Chó", ngược lại → "Mèo".',
          formula: '\\hat{y} = \\sigma(z) = \\frac{1}{1+e^{-z}} \\in (0, 1)',
          keyTakeaway: 'Binary Cross-Entropy Loss: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)] — phạt nặng khi mô hình tự tin sai.',
        },
        {
          title: '3. Từ MLP đến Mạng CNN hiện đại (Bước tiếp theo)',
          content: 'MLP này nhận đặc trưng thủ công (handcrafted). Mạng CNN hiện đại (ResNet, EfficientNet) tự động trích xuất đặc trưng từ pixel gốc qua hàng chục Conv layers. MLP vẫn là "đầu phân loại" (classification head) ở cuối mọi CNN hiện đại.',
          keyTakeaway: 'MLP hiểu nguyên lý phân loại nhị phân. CNN chỉ thêm bước tự học đặc trưng từ pixel — cấu trúc quyết định vẫn là MLP Sigmoid này.',
        },
      ],
    },
    sections: [
      {
        title: '1. Why 2 Hidden Layers for Dog/Cat Classification?',
        content: 'Dogs and cats differ in fur texture, posture, and facial structure — but the decision boundary is highly non-linear. Layer 1 learns primitive features like "is it smooth?". Layer 2 combines them: "smooth + round eyes → cat". A single hidden layer cannot capture such high-order combinations.',
        formula: 'a^{[2]} = \\text{ReLU}(W^{[2]} \\cdot \\text{ReLU}(W^{[1]} x + b^{[1]}) + b^{[2]})',
        keyTakeaway: 'Each layer learns features at an increasingly abstract level — this hierarchy is the essence of Deep Learning.',
      },
      {
        title: '2. Sigmoid Output and Class Probability Interpretation',
        content: 'The output layer uses Sigmoid (not ReLU) to squash the final logit into [0, 1]. This is interpreted as P(Dog). Decision rule: if P(Dog) > 0.5 → "Dog", else → "Cat".',
        formula: '\\hat{y} = \\sigma(z) = \\frac{1}{1+e^{-z}} \\in (0, 1)',
        keyTakeaway: 'Binary Cross-Entropy Loss penalizes confident wrong predictions the most — driving the model to be accurately calibrated.',
      },
      {
        title: '3. From MLP to Modern CNN (The Next Step)',
        content: 'This MLP takes handcrafted features. Modern CNNs (ResNet, EfficientNet) automatically extract features from raw pixels through dozens of Conv layers. The MLP Sigmoid head at the end remains the same — CNNs just add automatic feature extraction before it.',
        keyTakeaway: 'Understanding this MLP is understanding the classification head of every modern vision model.',
      },
    ],
    timeComplexity: 'O(d₁·d₂ + d₂·d₃ + d₃·1)',
    spaceComplexity: 'O(d₂ + d₃) hidden activations',
    tags: ['MLP', 'Binary Classification', 'Sigmoid', 'ReLU', 'Computer Vision', 'PyTorch'],
    defaultInput: 'texture: 0.2, bright: 0.8, edges: 0.3',
    pythonCode: `import torch
import torch.nn as nn

class DogCatMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 visual features → 4 → 4 → 1 (Sigmoid prob)
        self.net = nn.Sequential(
            nn.Linear(3, 4), nn.ReLU(),
            nn.Linear(4, 4), nn.ReLU(),
            nn.Linear(4, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

# Features: [texture_score, brightness, edge_density]
x = torch.tensor([[0.2, 0.8, 0.3]])  # likely cat
model = DogCatMLP()
prob = model(x).item()
label = "Dog" if prob > 0.5 else "Cat"
print(f"P(Dog) = {prob:.3f} → Predicted: {label}")`,
    code: `import numpy as np

def dog_cat_mlp(x, W1,b1, W2,b2, W3,b3):
    h1 = np.maximum(0, np.dot(x,W1) + b1)  # ReLU
    h2 = np.maximum(0, np.dot(h1,W2) + b2) # ReLU
    z3 = np.dot(h2, W3) + b3
    y  = 1 / (1 + np.exp(-z3))             # Sigmoid
    return y  # P(Dog)`,
    generateTraces: (inputStr?: string, lang: Language = 'en'): TraceStep[] => {
      const isVi = lang === 'vi';
      let tex = 0.2, bright = 0.8, edges = 0.3;
      if (inputStr) {
        const mt = inputStr.match(/texture:\s*(-?[0-9.]+)/i);
        const mb = inputStr.match(/bright:\s*(-?[0-9.]+)/i);
        const me = inputStr.match(/edges:\s*(-?[0-9.]+)/i);
        if (mt) tex = Number(mt[1]);
        if (mb) bright = Number(mb[1]);
        if (me) edges = Number(me[1]);
      }
      const x = [tex, bright, edges];
      const W1 = [[-0.08, 0.06, 0.57, 0.87], [1.2, 1.31, -1.34, 1.22], [-0.1, 1.27, 0.5, 0.25]];
      const b1 = [-0.02, -0.32, 0.57, 0.11];
      const W2 = [[0.35, -0.81, 0.73, -0.36], [-0.2, 1.13, -0.04, -0.92], [-0.89, 1.33, 0.21, -0.3], [0.9, -0.22, 0.79, 1.41]];
      const b2 = [-0.19, 0.18, -0.29, 0.36];
      const W3 = [-1.29, 1.24, -1.03, 1.0];
      const b3 = -0.48;
      const z1 = b1.map((bi,j)=>Number((x[0]*W1[0][j]+x[1]*W1[1][j]+x[2]*W1[2][j]+bi).toFixed(3)));
      const h1 = z1.map(z=>Number(Math.max(0,z).toFixed(3)));
      const z2 = b2.map((bi,j)=>Number((h1[0]*W2[0][j]+h1[1]*W2[1][j]+h1[2]*W2[2][j]+h1[3]*W2[3][j]+bi).toFixed(3)));
      const h2 = z2.map(z=>Number(Math.max(0,z).toFixed(3)));
      const z3 = Number((h2.reduce((s,h,j)=>s+h*W3[j],b3)).toFixed(3));
      const pDog = Number((1/(1+Math.exp(-z3))).toFixed(3));
      const label = pDog > 0.5 ? (isVi ? 'CHÓ 🐶' : 'DOG 🐶') : (isVi ? 'MÈO 🐱' : 'CAT 🐱');
      const inNodes = [
        { label: 'x₁', value: tex, activation: isVi ? 'Kết cấu' : 'Texture' },
        { label: 'x₂', value: bright, activation: isVi ? 'Độ sáng' : 'Brightness' },
        { label: 'x₃', value: edges, activation: isVi ? 'Đường cạnh' : 'Edges' },
      ];
      const steps: TraceStep[] = [];

      steps.push({
        step: 0, totalSteps: 0, line: 1, action: 'init', targets: [],
        state: {
          layers: [
            { name: isVi ? 'Đặc trưng Ảnh (x)' : 'Image Features (x)', nodes: inNodes },
            { name: isVi ? 'Tầng ẩn 1 (ReLU)' : 'Hidden Layer 1 (ReLU)',
              nodes: [{ label:'h1₁',value:0},{label:'h1₂',value:0},{label:'h1₃',value:0},{label:'h1₄',value:0}] },
            { name: isVi ? 'Tầng ẩn 2 (ReLU)' : 'Hidden Layer 2 (ReLU)',
              nodes: [{ label:'h2₁',value:0},{label:'h2₂',value:0},{label:'h2₃',value:0},{label:'h2₄',value:0}] },
            { name: isVi ? 'Xác suất P(Chó)' : 'P(Dog) Output',
              nodes: [{ label: 'ŷ', value: 0, activation: isVi ? 'Đang tính...' : 'Pending' }] },
          ],
        },
        explanation: isVi
          ? `Đặc trưng ảnh: Kết cấu=${tex.toFixed(2)}, Độ sáng=${bright.toFixed(2)}, Cạnh=${edges.toFixed(2)}. Chuẩn bị lan truyền tiến qua mạng phân loại 2 hidden layers.`
          : `Image features: Texture=${tex.toFixed(2)}, Brightness=${bright.toFixed(2)}, Edges=${edges.toFixed(2)}. Forwarding through 2-hidden-layer classifier.`,
        invariant: isVi ? 'Đặc trưng chuẩn hóa ∈ [0,1]. Mạng có cấu trúc 3→4→4→1.' : 'Features normalized ∈ [0,1]. Network architecture: 3→4→4→1.',
      });

      steps.push({
        step: 0, totalSteps: 0, line: 4, action: 'compute', targets: [],
        state: {
          layers: [
            { name: isVi ? 'Đặc trưng Ảnh (x)' : 'Image Features (x)', nodes: inNodes },
            { name: isVi ? 'Tầng ẩn 1 — z₁' : 'Hidden Layer 1 — z₁',
              nodes: z1.map((v,i)=>({label:`z₁${i+1}`,value:v})) },
            { name: isVi ? 'Tầng ẩn 2 (ReLU)' : 'Hidden Layer 2 (ReLU)',
              nodes: [{label:'h2₁',value:0},{label:'h2₂',value:0},{label:'h2₃',value:0},{label:'h2₄',value:0}] },
            { name: isVi ? 'Xác suất P(Chó)' : 'P(Dog) Output',
              nodes: [{ label: 'ŷ', value: 0, activation: isVi ? 'Đang tính...' : 'Pending' }] },
          ],
        },
        explanation: isVi
          ? `Tầng ẩn 1 tổ hợp tuyến tính: z₁ = W₁·x + b₁ → [${z1.map(v=>v.toFixed(2)).join(', ')}]. Phát hiện các đặc trưng nguyên thủy.`
          : `Hidden Layer 1 linear combination: z₁ = W₁·x + b₁ → [${z1.map(v=>v.toFixed(2)).join(', ')}]. Detecting primitive features.`,
        invariant: isVi ? 'Layer 1 học đặc trưng cấp thấp: cạnh, màu sắc, kết cấu.' : 'Layer 1 learns low-level features: edges, color, texture.',
      });

      steps.push({
        step: 0, totalSteps: 0, line: 5, action: 'forward', targets: [],
        state: {
          layers: [
            { name: isVi ? 'Đặc trưng Ảnh (x)' : 'Image Features (x)', nodes: inNodes },
            { name: isVi ? 'Tầng ẩn 1 — h₁ = ReLU(z₁)' : 'Hidden Layer 1 — h₁ = ReLU(z₁)',
              nodes: h1.map((v,i)=>({label:`h1${i+1}`,value:v,activation:'ReLU'})) },
            { name: isVi ? 'Tầng ẩn 2 — z₂' : 'Hidden Layer 2 — z₂',
              nodes: z2.map((v,i)=>({label:`z₂${i+1}`,value:v})) },
            { name: isVi ? 'Xác suất P(Chó)' : 'P(Dog) Output',
              nodes: [{ label: 'ŷ', value: 0, activation: isVi ? 'Đang tính...' : 'Pending' }] },
          ],
        },
        explanation: isVi
          ? `ReLU(z₁) → h₁=[${h1.map(v=>v.toFixed(2)).join(', ')}]. Tầng ẩn 2 phóng chiếu: z₂=[${z2.map(v=>v.toFixed(2)).join(', ')}]. Học tổ hợp đặc trưng bậc cao.`
          : `ReLU(z₁) → h₁=[${h1.map(v=>v.toFixed(2)).join(', ')}]. Layer 2 projection: z₂=[${z2.map(v=>v.toFixed(2)).join(', ')}]. Learning high-level feature combos.`,
        invariant: isVi ? 'Layer 2 học đặc trưng bậc cao: "mịn + sáng → mèo", "thô + nhiều cạnh → chó".' : 'Layer 2 learns high-level: "smooth + bright → cat", "rough + many edges → dog".',
      });

      steps.push({
        step: 0, totalSteps: 0, line: 7, action: 'compute', targets: [],
        state: {
          layers: [
            { name: isVi ? 'Đặc trưng Ảnh (x)' : 'Image Features (x)', nodes: inNodes },
            { name: isVi ? 'Tầng ẩn 1 (Hoàn tất)' : 'Hidden Layer 1 (Done)',
              nodes: h1.map((v,i)=>({label:`h1${i+1}`,value:v})) },
            { name: isVi ? 'Tầng ẩn 2 — h₂ = ReLU(z₂)' : 'Hidden Layer 2 — h₂ = ReLU(z₂)',
              nodes: h2.map((v,i)=>({label:`h2${i+1}`,value:v,activation:'ReLU'})) },
            { name: isVi ? 'Xác suất P(Chó)' : 'P(Dog) Output',
              nodes: [{ label: 'ŷ', value: 0, activation: isVi ? 'Đang tính...' : 'Pending' }] },
          ],
        },
        explanation: isVi
          ? `ReLU tầng 2: h₂=[${h2.map(v=>v.toFixed(2)).join(', ')}]. Đặc trưng bậc cao đã trích xuất, sẵn sàng cho tầng đầu ra Sigmoid.`
          : `Layer 2 ReLU: h₂=[${h2.map(v=>v.toFixed(2)).join(', ')}]. High-level features extracted, ready for Sigmoid output.`,
        invariant: isVi ? 'Mỗi hidden layer học một tầng trừu tượng mới.' : 'Each hidden layer learns a new level of abstraction.',
      });

      steps.push({
        step: 0, totalSteps: 0, line: 9, action: 'done', targets: [],
        state: {
          layers: [
            { name: isVi ? 'Đặc trưng Ảnh (x)' : 'Image Features (x)', nodes: inNodes },
            { name: isVi ? 'Tầng ẩn 1' : 'Hidden Layer 1',
              nodes: h1.map((v,i)=>({label:`h1${i+1}`,value:v})) },
            { name: isVi ? 'Tầng ẩn 2' : 'Hidden Layer 2',
              nodes: h2.map((v,i)=>({label:`h2${i+1}`,value:v})) },
            { name: isVi ? 'Kết quả Phân loại' : 'Classification Result',
              nodes: [{ label: pDog > 0.5 ? '🐶' : '🐱', value: pDog, activation: label }] },
          ],
        },
        explanation: isVi
          ? `Tầng đầu ra: z₃=${z3.toFixed(3)}, P(Chó) = σ(${z3.toFixed(3)}) = ${pDog.toFixed(3)}. Quyết định: ${label}${pDog > 0.5 ? ` (${(pDog*100).toFixed(1)}% tự tin)` : ` (${((1-pDog)*100).toFixed(1)}% là Mèo)`}.`
          : `Output: z₃=${z3.toFixed(3)}, P(Dog) = σ(${z3.toFixed(3)}) = ${pDog.toFixed(3)}. Decision: ${label}${pDog > 0.5 ? ` (${(pDog*100).toFixed(1)}% confident)` : ` (${((1-pDog)*100).toFixed(1)}% Cat)`}.`,
        invariant: isVi ? 'Sigmoid đảm bảo P(Chó) ∈ (0,1). Quy tắc: P > 0.5 → Chó, P ≤ 0.5 → Mèo.' : 'Sigmoid ensures P(Dog) ∈ (0,1). Decision rule: P > 0.5 → Dog, ≤ 0.5 → Cat.',
      });

      const total = steps.length;
      return steps.map((s, idx) => ({ ...s, step: idx + 1, totalSteps: total }));
    },
  },

  {
    id: 'mnist-digit-recognition',
    title: '07. MNIST Handwritten Digit Recognition (Interactive MLP)',
    category: 'Deep Learning & Neural Networks',
    difficulty: 'Intermediate',
    badge: 'Multiclass Softmax',
    visualType: 'mnist-interactive',
    formula: 'P(y = k \\mid \\mathbf{x}) = \\frac{e^{z_k}}{\\sum_{j=0}^{9} e^{z_j}}, \\quad \\mathbf{z} = W_2 \\cdot \\text{ReLU}(W_1 \\mathbf{x} + b_1) + b_2',
    summary: 'Recognizes handwritten digits (0–9) from 28×28 grayscale images into 10 Softmax probabilities with an interactive drawing canvas.',
    description: 'The iconic MNIST benchmark: unrolls 28×28 pixel matrices into 784-dimensional vectors, processes through hidden ReLU feature detectors, and produces 10 output class logits normalized by Softmax. Includes an interactive canvas where you can draw digits freely.',
    inputVariants: [
      { label: 'Bảng vẽ trực tiếp (Trống)', value: 'digit: draw' },
    ],
    vi: {
      title: '07. Nhận diện Chữ số Viết tay MNIST (Tương tác Vẽ số)',
      category: 'Deep Learning & Mạng Nơ-ron',
      badge: 'Phân loại 10 Lớp Softmax',
      summary: 'Nhận diện chữ số viết tay (0–9) từ ảnh 28×28 thành 10 xác suất Softmax kèm bảng vẽ tương tác trực tiếp.',
      description: 'Bài toán kinh điển MNIST: chuyển bức vẽ 28×28 điểm ảnh thành vector 784 chiều, lan truyền qua các nơ-ron ẩn ReLU để nhận diện nét vẽ và tính 10 xác suất Softmax. Tích hợp bảng vẽ trực tiếp để người dùng tự do vẽ số bằng chuột hoặc cảm ứng.',
      variants: [
        { label: 'Bảng vẽ trực tiếp (Trống)', value: 'digit: draw' },
      ],
      sections: [
        {
          title: '1. Bài toán Chuẩn MNIST & Vector hóa Ảnh 2D thành 784 Chiều',
          content: 'Bộ dữ liệu MNIST là "Hello World" của thị giác máy tính. Mỗi bức ảnh xám kích thước 28×28 chứa 784 điểm ảnh với cường độ từ 0 (nền đen) đến 1 (nét vẽ trắng). Để đưa vào mạng nơ-ron truyền thẳng (MLP), ta "duỗi phẳng" (flatten) ma trận 2D thành một vector 1D 784 chiều x ∈ ℝ^784.',
          formula: '\\mathbf{x} = \\text{vec}(I_{28 \\times 28}) = [x_1, x_2, \\dots, x_{784}]^T \\in [0, 1]^{784}',
          keyTakeaway: 'Mỗi phần tử trong vector 784 chiều tương ứng với độ sáng của đúng một điểm ảnh trên bức vẽ 28×28.',
        },
        {
          title: '2. Tầng ẩn ReLU: Nhận diện Nét Hình học Nguyên thủy',
          content: '784 đặc trưng đầu vào được phóng chiếu qua ma trận trọng số W₁ và kích hoạt bằng hàm ReLU. Mỗi nơ-ron ẩn học cách nhận diện một đặc trưng hình học cục bộ: nơ-ron phát hiện nét cong tròn khép kín (số 0, 8, 9), nơ-ron phát hiện nét thẳng đứng (số 1, 4), hoặc nét gạch ngang (số 2, 7, 5).',
          formula: '\\mathbf{h} = \\text{ReLU}(W_1 \\mathbf{x} + b_1) = \\max(0, W_1 \\mathbf{x} + b_1)',
          keyTakeaway: 'ReLU triệt tiêu các nơ-ron không phát hiện được nét tương ứng về 0, tạo tính thưa thớt (sparsity) giúp mạng học hiệu quả.',
        },
        {
          title: '3. Hàm Softmax: Phân phối Xác suất Đa lớp cho 10 Chữ số',
          content: 'Khác với phân loại nhị phân (chó vs mèo) dùng hàm Sigmoid cho 1 nơ-ron, phân loại 10 lớp sinh ra 10 điểm số thô (logits) z_k. Hàm Softmax lấy số mũ e^(z_k) để đảm bảo giá trị dương, rồi chia cho tổng lũy thừa của cả 10 số, tạo thành phân phối xác suất có tổng bằng 1.0 (100%).',
          formula: 'P(y = k \\mid \\mathbf{x}) = \\frac{e^{z_k}}{\\sum_{j=0}^{9} e^{z_j}}, \\quad \\sum_{k=0}^{9} P(y=k) = 1.0',
          keyTakeaway: 'Softmax khuếch đại nơ-ron có điểm cao nhất, tạo ra một người chiến thắng rõ ràng với độ tin cậy vượt trội.',
        },
        {
          title: '4. Hàm Mất mát Categorical Cross-Entropy',
          content: 'Khi huấn luyện với nhãn đúng one-hot y_true, mô hình tối ưu hàm mất mát Categorical Cross-Entropy: L = -log(P(y_true)). Nếu nhãn thật là số 7, hàm mất mát sẽ tiệm cận 0 khi P(7) đạt 100%. Mọi sai sót đều sinh gradient truyền ngược qua 784 trọng số để tinh chỉnh mô hình.',
          formula: 'L_{\\text{CE}} = -\\sum_{k=0}^{9} y_k \\log(\\hat{y}_k) = -\\log(P(y_{\\text{true}}))',
          keyTakeaway: 'Cross-Entropy phạt rất nặng khi mô hình tự tin sai nhãn, thúc đẩy độ chính xác cao trên toàn bộ 10 chữ số.',
        },
      ],
    },
    sections: [
      {
        title: '1. The MNIST Benchmark & 2D Image Vectorization (784 Pixels)',
        content: 'MNIST is the quintessential "Hello World" of computer vision. A 28×28 grayscale image contains 784 pixels with brightness from 0 (background) to 1 (stroke). To feed this into a Dense Neural Network (MLP), we unroll the 2D grid row-by-row into a 784-dimensional vector x ∈ ℝ^784.',
        formula: '\\mathbf{x} = \\text{vec}(I_{28 \\times 28}) = [x_1, x_2, \\dots, x_{784}]^T \\in [0, 1]^{784}',
        keyTakeaway: 'Each entry in the 784 vector represents the grayscale intensity of exactly one pixel coordinate.',
      },
      {
        title: '2. Hidden Layer Feature Extraction with ReLU',
        content: 'The 784 inputs are projected into hidden neurons via z_1 = W_1 * x + b_1 and activated by ReLU. Each hidden neuron acts as a learned stroke detector: some activate on closed loops (digits 0, 8, 9), some on vertical straight stems (digit 1), and others on horizontal cross-bars (digits 2, 7).',
        formula: '\\mathbf{h} = \\text{ReLU}(W_1 \\mathbf{x} + b_1) = \\max(0, W_1 \\mathbf{x} + b_1)',
        keyTakeaway: 'ReLU clamps inactive stroke detectors to 0, promoting sparsity and robust feature isolation.',
      },
      {
        title: '3. Softmax Function: Converting Raw Logits into 10-Class Probabilities',
        content: 'Unlike binary classification (which uses Sigmoid for a single output), 10-class MNIST produces 10 raw scores (logits) z_k. Softmax exponentiates each score (ensuring positive values) and divides by the total sum, producing a normalized probability distribution where all 10 probabilities sum to exactly 1.0 (100%).',
        formula: 'P(y = k \\mid \\mathbf{x}) = \\frac{e^{z_k}}{\\sum_{j=0}^{9} e^{z_j}}, \\quad \\sum_{k=0}^{9} P(y=k) = 1.0',
        keyTakeaway: 'Softmax exponentiation magnifies the highest scoring digit, creating a distinct, confident prediction winner.',
      },
      {
        title: '4. Categorical Cross-Entropy Loss for Multiclass Classification',
        content: 'During supervised training with one-hot targets, the model minimizes Categorical Cross-Entropy: L = -log(P(y_true)). If the ground truth is digit 7, loss approaches zero when P(7) approaches 100%. Gradients propagate back through all 784 input weights simultaneously.',
        formula: 'L_{\\text{CE}} = -\\sum_{k=0}^{9} y_k \\log(\\hat{y}_k) = -\\log(P(y_{\\text{true}}))',
        keyTakeaway: 'Categorical Cross-Entropy heavily penalizes confident incorrect classifications, driving high accuracy.',
      },
    ],
    timeComplexity: 'O(784 · d_h + d_h · 10)',
    spaceComplexity: 'O(784 + d_h + 10) activations',
    tags: ['MNIST', 'Computer Vision', 'Softmax', 'Multiclass', 'Interactive Drawing', 'PyTorch'],
    defaultInput: 'digit: draw',
    pythonCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 784 inputs (28x28) -> 64 hidden -> 10 class logits
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # 1. Flatten 2D image (B, 1, 28, 28) -> (B, 784)
        x = x.view(-1, 784)
        # 2. Hidden Layer + ReLU
        h = F.relu(self.fc1(x))
        # 3. Output Logits (10 classes: digits 0-9)
        logits = self.fc2(h)
        # 4. Softmax Probabilities
        probs = F.softmax(logits, dim=-1)
        return probs

model = MNISTNet()
x_img = torch.randn(1, 1, 28, 28)
output_probs = model(x_img)
predicted = torch.argmax(output_probs, dim=-1).item()
print(f"Predicted digit: {predicted}")`,
    code: `import numpy as np

def mnist_forward(image_28x28, W1, b1, W2, b2):
    # 1. Flatten 2D image into 784D vector
    x = image_28x28.flatten()
    # 2. Hidden Layer linear projection + ReLU
    h = np.maximum(0, np.dot(x, W1) + b1)
    # 3. Output logits (10 classes)
    z = np.dot(h, W2) + b2
    # 4. Softmax probability distribution
    exp_z = np.exp(z - np.max(z))
    probs = exp_z / np.sum(exp_z)
    pred_digit = np.argmax(probs)
    return pred_digit, probs`,
    generateTraces: (inputStr?: string, lang: Language = 'en'): TraceStep[] => {
      const isVi = lang === 'vi';
      let digit: number | null = null;
      if (inputStr) {
        const match = inputStr.match(/digit:\s*(\d)/i);
        if (match) {
          digit = parseInt(match[1], 10);
        }
      }
      const grid = digit !== null
        ? generateCanonicalDigitGrid(digit)
        : Array.from({ length: 28 }, () => Array(28).fill(0));
      const inference = inferMnist(grid);
      const flattenedSample = grid.flat().slice(100, 220);
      const hiddenLayer = Array.from({ length: 16 }, (_, idx) =>
        (digit !== null && ((idx * 7 + digit * 3) % 5 !== 0))
          ? Number((0.2 + ((idx * 31 + digit * 17) % 80) / 100).toFixed(2))
          : 0
      );
      const steps: TraceStep[] = [];

      // Step 1: Input Drawing Matrix (28x28)
      steps.push({
        step: 0,
        totalSteps: 0,
        line: 3,
        action: 'init',
        targets: [],
        state: {
          mnistState: {
            grid,
            flattenedSample,
            hiddenLayer,
            probabilities: inference.probabilities,
            predictedDigit: inference.predictedDigit,
            confidence: inference.confidence,
            activePhase: 'draw',
          },
        },
        explanation: isVi
          ? (digit !== null
              ? `Bức vẽ chữ số mẫu ${digit} trên ma trận 28×28. Bạn có thể bấm Xóa để vẽ số của riêng mình.`
              : 'Bảng vẽ trực tiếp 28×28 đã sẵn sàng. Hãy dùng chuột hoặc ngón tay vẽ thẳng chữ số (0 - 9) lên bảng đen!')
          : (digit !== null
              ? `Canonical sample digit ${digit} loaded on 28×28 matrix. Click Clear to draw your own custom digit.`
              : 'Interactive 28×28 canvas ready. Draw any digit (0 - 9) directly onto the black input board!'),
        invariant: isVi
          ? (digit !== null
              ? 'Mỗi pixel chuẩn hóa trong đoạn [0, 1]. Kích thước ảnh chuẩn MNIST: 28×28 = 784 pixels.'
              : 'Bảng vẽ khởi tạo trống, sẵn sàng nhận nét vẽ tay trực tiếp.')
          : (digit !== null
              ? 'Each pixel intensity normalized in [0, 1]. Canonical MNIST size: 28×28 = 784 pixels.'
              : 'Canvas initialized blank, ready for direct freehand drawing.'),
      });

      // Step 2: Vector Flattening
      steps.push({
        step: 0,
        totalSteps: 0,
        line: 5,
        action: 'compute',
        targets: [],
        state: {
          mnistState: {
            grid,
            flattenedSample,
            hiddenLayer,
            probabilities: inference.probabilities,
            predictedDigit: inference.predictedDigit,
            confidence: inference.confidence,
            activePhase: 'flatten',
          },
        },
        explanation: isVi
          ? 'Duỗi phẳng ma trận 2D (28×28) thành vector 1D 784 chiều: x = vec(I) ∈ ℝ⁷⁸⁴. Chuẩn bị cho phép nhân ma trận trọng số.'
          : 'Flattening 2D spatial image (28×28) into a 1D vector of 784 features: x = vec(I) ∈ ℝ⁷⁸⁴. Ready for matrix multiplication.',
        invariant: isVi
          ? 'Phép duỗi phẳng bảo toàn toàn bộ 784 giá trị điểm ảnh theo thứ tự hàng.'
          : 'Vector unrolling preserves all 784 pixel intensities in row-major order.',
      });

      // Step 3: Hidden Layer ReLU Features
      steps.push({
        step: 0,
        totalSteps: 0,
        line: 7,
        action: 'forward',
        targets: [],
        state: {
          mnistState: {
            grid,
            flattenedSample,
            hiddenLayer,
            probabilities: inference.probabilities,
            predictedDigit: inference.predictedDigit,
            confidence: inference.confidence,
            activePhase: 'hidden',
          },
        },
        explanation: isVi
          ? 'Lan truyền tiến qua tầng ẩn W₁·x + b₁ và kích hoạt bằng ReLU. Các nơ-ron phát hiện nét cong, nét thẳng và vòng khép kín kích hoạt mạnh mẽ.'
          : 'Forward pass through hidden layer W₁·x + b₁ activated by ReLU. Feature detectors for curves, loops, and straight stems fire.',
        invariant: isVi
          ? 'ReLU(z) = max(0, z) loại bỏ các giá trị âm, kích hoạt các nét đặc trưng hình học tương ứng.'
          : 'ReLU(z) = max(0, z) clamps negative activations to zero, isolating active stroke features.',
      });

      // Step 4: Output Layer Logits
      steps.push({
        step: 0,
        totalSteps: 0,
        line: 9,
        action: 'compute',
        targets: [],
        state: {
          mnistState: {
            grid,
            flattenedSample,
            hiddenLayer,
            probabilities: inference.probabilities,
            predictedDigit: inference.predictedDigit,
            confidence: inference.confidence,
            activePhase: 'softmax',
          },
        },
        explanation: isVi
          ? `Tầng đầu ra tính 10 điểm số thô (logits) z₀...z₉. Chữ số ${inference.predictedDigit} đạt điểm số logit cao nhất.`
          : `Output layer computes 10 raw logits z₀...z₉. Digit ${inference.predictedDigit} receives the highest logit score.`,
        invariant: isVi
          ? '10 logits biểu thị tương quan tuyến tính chưa chuẩn hóa của từng chữ số.'
          : '10 logits represent unnormalized linear compatibility scores for each digit class.',
      });

      // Step 5: Softmax Probability Distribution & Result
      steps.push({
        step: 0,
        totalSteps: 0,
        line: 12,
        action: 'done',
        targets: [],
        state: {
          mnistState: {
            grid,
            flattenedSample,
            hiddenLayer,
            probabilities: inference.probabilities,
            predictedDigit: inference.predictedDigit,
            confidence: inference.confidence,
            activePhase: 'result',
          },
        },
        explanation: isVi
          ? `Hàm Softmax hoàn tất! Nhận diện chính xác: Chữ số ${inference.predictedDigit} với độ tin cậy ${(inference.confidence * 100).toFixed(1)}%. Tổng 10 xác suất đúng bằng 100%.`
          : `Softmax normalization complete! Predicted: Digit ${inference.predictedDigit} with ${(inference.confidence * 100).toFixed(1)}% confidence. All 10 probabilities sum to 100%.`,
        invariant: isVi
          ? 'Hàm Softmax đảm bảo ∑ P(y=k) = 1.0 và P ∈ (0, 1). Chữ số có xác suất cao nhất được chọn làm kết quả.'
          : 'Softmax guarantees ∑ P(y=k) = 1.0 and P ∈ (0, 1). The argmax digit is chosen as the final prediction.',
      });

      const total = steps.length;
      return steps.map((s, idx) => ({ ...s, step: idx + 1, totalSteps: total }));
    },
  },


  // ==========================================
  // Category 3: Transformers & Modern LLMs
  // ==========================================
  {
    id: 'self-attention',
    title: '01. Self-Attention Mechanism (Q, K, V)',
    category: 'Transformers & Modern LLMs',
    difficulty: 'Advanced',
    badge: 'Core Transformer',
    visualType: 'attention-matrix',
    formula: '\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V',
    summary: 'Calculates dynamic relevance scores between all token pairs in an input prompt to capture contextual meaning.',
    description: 'The foundation of all Modern LLMs (GPT-4, Claude, LLaMA). Each token generates Query (Q), Key (K), and Value (V) embedding vectors. The dot product of Query and Keys measures semantic compatibility.',
    inputVariants: [
      { label: 'Standard', value: 'The, quick, brown, fox, jumps' },
      { label: 'Short Prompt', value: 'AI, models, learn, patterns' },
      { label: 'Code Tokens', value: 'def, forward, layer, loss' },
    ],
    vi: {
      title: '01. Cơ chế Self-Attention (Q, K, V)',
      category: 'Transformer & Mô hình LLM',
      badge: 'Transformer Core',
      summary: 'Tính toán điểm attention liên quan ngữ cảnh động giữa tất cả các cặp tokens trong câu.',
      description: 'Nền tảng của các mô hình LLM hiện đại. Mỗi token tạo ra các vector nhúng Query (Q), Key (K) và Value (V). Tích vô hướng giữa Query và Key đo lường mức độ tương quan ngữ cảnh trong câu.',
      variants: [
        { label: 'Chuẩn', value: 'The, quick, brown, fox, jumps' },
        { label: 'Câu ngắn', value: 'AI, models, learn, patterns' },
        { label: 'Mã nguồn', value: 'def, forward, layer, loss' },
      ],
      sections: [
        {
          title: '1. Tại sao Self-Attention tạo nên Cuộc Cách mạng NLP?',
          content: 'Các kiến trúc tuần hoàn trước đây (RNN/LSTM) xử lý từng từ nối tiếp nhau, gây nghẽn memory khi gặp câu dài. Cơ chế Self-Attention cho phép mỗi từ tương tác trực tiếp với mọi từ khác trong văn bản với đường dẫn O(1), đồng thời huấn luyện song song cực nhanh trên GPU.',
          formula: 'S_{ij} = \\frac{q_i \\cdot k_j^T}{\\sqrt{d_k}}',
          keyTakeaway: 'Self-Attention giải quyết triệt để vấn đề phụ thuộc tầm xa (long-range dependency) bằng ma trận dot-product.',
        },
        {
          title: '2. Vai trò của Query, Key và Value',
          content: 'Query là câu truy vấn tìm kiếm, Keys là các thẻ chỉ mục trên mỗi token trong ngữ cảnh, và Values là thông tin ngữ nghĩa thực tế được trích xuất khi Query và Key match nhau.',
          keyTakeaway: 'Hàm Softmax chuẩn hóa dot-products thành phân phối xác suất có tổng bằng 1.0, điều hướng mô hình tập trung sự chú ý.',
        },
        {
          title: '3. Causal Masking trong Mô hình Generative LLMs',
          content: 'Trong các mô hình autoregressive (như GPT), token ở hiện tại không được phép nhìn trước tương lai. Causal mask gán các vị trí tương lai bằng -inf, khiến Softmax triệt tiêu xác suất về 0.',
          formula: '\\text{Masked}(S_{ij}) = -\\infty \\quad \\text{if } j > i',
          keyTakeaway: 'Causal mask đảm bảo nguyên tắc autoregressive một chiều trong quá trình training và generation.',
        },
      ],
    },
    sections: [
      {
        title: '1. Why Self-Attention Revolutionized NLP',
        content: 'Earlier recurrent models (RNNs/LSTMs) processed words sequentially, causing memory bottlenecks over long paragraphs. Self-Attention allows every word to interact directly with every other word in constant O(1) path length, running massively in parallel on GPUs.',
        formula: 'S_{ij} = \\frac{q_i \\cdot k_j^T}{\\sqrt{d_k}}',
        keyTakeaway: 'Attention solves the long-range dependency problem by replacing sequential recurrence with matrix dot products.',
      },
      {
        title: '2. The Role of Query, Key, and Value',
        content: 'Think of Query as a search query ("What is the subject of this sentence?"), Keys as indexed tags on words in the context, and Values as the actual semantic content retrieved when a Query and Key match.',
        keyTakeaway: 'Softmax turns dot products into a probability distribution summing to 1.0, directing the model where to focus.',
      },
      {
        title: '3. Causal Masking in Generative LLMs',
        content: 'In autoregressive generation (GPT models), tokens must not peek into the future. A lower-triangular causal mask fills future positions with -inf, so softmax zeroes them out.',
        formula: '\\text{Masked}(S_{ij}) = -\\infty \\quad \\text{if } j > i',
        keyTakeaway: 'Causal masking enforces strict temporal directionality during training and generation.',
      },
    ],
    timeComplexity: 'O(N² · d)',
    spaceComplexity: 'O(N · d) KV-Cache',
    tags: ['Transformer', 'Attention', 'Softmax', 'LLMs', 'PyTorch'],
    defaultInput: 'The, quick, brown, fox, jumps',
    pythonCode: `import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, d_k):
    # 1. Compute Raw Attention Scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # 2. Apply Causal Mask (prevent looking ahead)
    seq_len = Q.size(-2)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 3. Softmax Normalization
    weights = F.softmax(scores, dim=-1)
    
    # 4. Weighted Sum of Values
    output = torch.matmul(weights, V)
    return output, weights`,
    code: `import torch
import torch.nn.functional as F

def forward_attention(Q, K, V, d_k):
    # 1. Scaled Dot-Product of Query & Key
    scores = torch.matmul(Q, K.T) / (d_k ** 0.5)
    # 2. Causal Masking (Lower Triangular)
    scores = apply_causal_mask(scores)
    # 3. Softmax Probabilities
    attn_weights = F.softmax(scores, dim=-1)
    # 4. Contextual Value Aggregation
    context = torch.matmul(attn_weights, V)
    return context`,
    generateTraces: (inputStr?: string, lang: Language = 'en'): TraceStep[] => {
      const isVi = lang === 'vi';
      const raw = inputStr || 'The, quick, brown, fox, jumps';
      const tokens = raw
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean);
      const activeTokens = tokens.length > 0 ? tokens : ['The', 'quick', 'brown', 'fox', 'jumps'];
      const lastToken = activeTokens[activeTokens.length - 1];

      const steps: TraceStep[] = [];

      steps.push({
        step: 0,
        totalSteps: 0,
        line: 2,
        action: 'init',
        targets: [activeTokens.length - 1],
        state: {
          tokens: activeTokens,
          probabilities: [
            { token: 'over', prob: 0.12 },
            { token: 'high', prob: 0.10 },
            { token: 'away', prob: 0.08 },
            { token: 'through', prob: 0.05 },
          ],
        },
        explanation: isVi
          ? `Cửa sổ ngữ cảnh khởi tạo với ${activeTokens.length} tokens. Vector truy vấn Query được chiếu từ token cuối cùng "${lastToken}".`
          : `Context window initialized with ${activeTokens.length} tokens. Active Query vector projected from last token "${lastToken}".`,
        invariant: isVi
          ? 'Chuỗi tự hồi quy: các token trước đó được lưu trữ trong KV-Cache.'
          : 'Autoregressive sequence: prior tokens stored in KV-Cache.',
      });

      steps.push({
        step: 1,
        totalSteps: 0,
        line: 5,
        action: 'compute',
        targets: [activeTokens.length - 1],
        state: {
          tokens: activeTokens,
          probabilities: [
            { token: 'over', prob: 0.28 },
            { token: 'high', prob: 0.18 },
            { token: 'away', prob: 0.12 },
            { token: 'through', prob: 0.08 },
          ],
        },
        explanation: isVi
          ? `Tích vô hướng tỉ lệ: Q · K^T / √d_k. Đo lường mức độ tương quan ngữ cảnh giữa từ "${lastToken}" với các từ trước đó.`
          : `Scaled dot-product: Q · K^T / √d_k. Computes contextual correlation scores between "${lastToken}" and all prior tokens.`,
        invariant: isVi
          ? 'Chia tỉ lệ cho 1/√d_k ngăn giá trị tích vô hướng quá lớn rơi vào vùng gradient biến mất của Softmax.'
          : 'Scaling by 1/√d_k prevents dot-product values from growing too large into vanishing gradient softmax regions.',
      });

      steps.push({
        step: 2,
        totalSteps: 0,
        line: 7,
        action: 'forward',
        targets: [1, 3, activeTokens.length - 1],
        state: {
          tokens: activeTokens,
          probabilities: [
            { token: 'over', prob: 0.52 },
            { token: 'high', prob: 0.20 },
            { token: 'away', prob: 0.14 },
            { token: 'onto', prob: 0.08 },
          ],
        },
        explanation: isVi
          ? `Áp dụng Softmax: Điểm chú ý cao nhất tập trung vào chủ ngữ "fox" (44%) và động từ "${lastToken}" (38%).`
          : `Softmax applied across row: Highest attention score placed on subject token "fox" (44%) and verb "${lastToken}" (38%).`,
        invariant: isVi
          ? '∑_j Attention[i, j] = 1.0 trên toàn bộ chiều của chuỗi.'
          : '∑_j Attention[i, j] = 1.0 across the sequence dimension.',
      });

      steps.push({
        step: 3,
        totalSteps: 0,
        line: 9,
        action: 'done',
        targets: [activeTokens.length - 1],
        state: {
          tokens: activeTokens,
          probabilities: [
            { token: 'over', prob: 0.76 },
            { token: 'high', prob: 0.13 },
            { token: 'away', prob: 0.05 },
            { token: 'onto', prob: 0.04 },
            { token: 'into', prob: 0.02 },
          ],
        },
        explanation: isVi
          ? `Tổng hợp giá trị Value hoàn tất! Vector ngữ cảnh được chuyển qua Language Model Head. Token tiếp theo có xác suất cao nhất: "over" (76%).`
          : `Value aggregation complete! Context vector passed to Language Model Head. Top-1 predicted next token: "over" (76% probability).`,
        invariant: isVi
          ? 'Bước sinh tự hồi quy hoàn tất; sẵn sàng chọn mẫu cho token tiếp theo.'
          : 'Autoregressive generation step complete; ready to sample next token.',
      });

      const total = steps.length;
      return steps.map((s, idx) => ({ ...s, step: idx + 1, totalSteps: total }));
    },
  },
];

// Helper getters for localized preset content
export function getPresetTitle(preset: AlgorithmPreset, lang: Language): string {
  return lang === 'vi' && preset.vi ? preset.vi.title : preset.title;
}

export function getPresetCategory(preset: AlgorithmPreset, lang: Language): string {
  return lang === 'vi' && preset.vi ? preset.vi.category : preset.category;
}

export function getPresetSummary(preset: AlgorithmPreset, lang: Language): string {
  return lang === 'vi' && preset.vi ? preset.vi.summary : preset.summary;
}

export function getPresetDescription(preset: AlgorithmPreset, lang: Language): string {
  return lang === 'vi' && preset.vi ? preset.vi.description : preset.description;
}

export function getPresetBadge(preset: AlgorithmPreset, lang: Language): string {
  return lang === 'vi' && preset.vi ? preset.vi.badge : preset.badge;
}

export function getPresetSections(preset: AlgorithmPreset, lang: Language) {
  return lang === 'vi' && preset.vi ? preset.vi.sections : preset.sections;
}

export function getPresetVariants(preset: AlgorithmPreset, lang: Language) {
  return lang === 'vi' && preset.vi && preset.vi.variants ? preset.vi.variants : preset.inputVariants;
}
