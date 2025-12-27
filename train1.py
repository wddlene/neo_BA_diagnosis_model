import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, classification_report, accuracy_score, roc_auc_score, confusion_matrix, brier_score_loss, matthews_corrcoef
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, cross_val_predict, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
import math
import warnings
import joblib 
import itertools
from scipy.stats import gaussian_kde
import os

# ====================
# 专业可视化设置（从BATL_GBMMP3neo.ipynb导入）
# ====================
# 设置全局样式
plt.style.use('seaborn-v0_8-darkgrid')

# 定义专业色彩方案
CUSTOM_COLORS = {
    'primary': '#2E86AB',    # 深蓝色 - 主色
    'secondary': '#A23B72',  # 紫色 - 次要色
    'tertiary': '#F18F01',   # 橙色 - 第三色
    'accent': '#C73E1D',     # 红色 - 强调色
    'success': '#2A9D8F',    # 绿色 - 成功
    'warning': '#E9C46A',    # 黄色 - 警告
    'dark': '#264653',       # 深色 - 背景
    'light': '#E9ECEF',      # 浅色 - 背景
    'gray1': '#6C757D',      # 灰色1
    'gray2': '#ADB5BD',      # 灰色2
    'blue_gradient': ['#1A2980', '#26D0CE'],  # 蓝色渐变
    'red_gradient': ['#FF416C', '#FF4B2B'],   # 红色渐变
    'green_gradient': ['#11998E', '#38EF7D'], # 绿色渐变
}

# 设置matplotlib全局参数
plt.rcParams.update({
    'figure.figsize': (14, 10),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
})

# 设置seaborn样式
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.1)
sns.set_style("whitegrid", {
    'grid.linestyle': ':',
    'grid.alpha': 0.2,
    'axes.edgecolor': '0.3',
    'axes.linewidth': 1.1,
})

# 创建自定义颜色映射
blue_cmap = sns.light_palette(CUSTOM_COLORS['primary'], as_cmap=True)
red_cmap = sns.light_palette(CUSTOM_COLORS['accent'], as_cmap=True)
diverging_cmap = sns.diverging_palette(240, 10, as_cmap=True)

print("✅ Visualization settings completed")

# 加载数据（使用与notebook相同的特征）
print("📊 Loading neonatal biliary atresia data...")
data = pd.read_csv("BAGBMMP1209_nonscaled.csv")

# 提取目标变量和特征 - 使用相同的5个特征
data_target = data['BA']
data_features = data[['GB_length', 'Abnormal_GEI', 'GGT', 'DBIL', 'MMP7']]

# 数据标准化
print("🔧 Standardizing features...")
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = pd.DataFrame(scaler.fit_transform(data_features), columns=data_features.columns)

# 分割训练和测试集（与notebook相同的参数）
print("✂️ Splitting training and test sets...")
class_x_train, class_x_test, class_y_train, class_y_test = train_test_split(
    data_scaled, data_target, test_size=0.3, random_state=42, stratify=data_target
)

# 确保标签是numpy数组格式
class_y_train = class_y_train.values.ravel()
class_y_test = class_y_test.values.ravel()

# 显示数据集信息
print(f"📈 Dataset information:")
print(f"  Total samples: {data.shape[0]}")
print(f"  Number of features: {data_scaled.shape[1]}")
print(f"  Training set size: {class_x_train.shape[0]} samples")
print(f"  Test set size: {class_x_test.shape[0]} samples")
print(f"  Training class distribution: {pd.Series(class_y_train).value_counts().to_dict()}")
print(f"  Test class distribution: {pd.Series(class_y_test).value_counts().to_dict()}")

# 计算AUC置信区间函数
def calculate_auc_ci(y_true, y_pred, n_bootstraps=2000, alpha=0.95):
    """Calculate AUC confidence intervals using bootstrap method"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(np.unique(y_true)) < 2:
        return 0.5, (0.0, 1.0)
    
    n = len(y_true)
    bootstrapped_auc = []
    original_auc = roc_auc_score(y_true, y_pred)
    
    # Bootstrap sampling
    for _ in range(n_bootstraps):
        indices = np.random.choice(range(n), n, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        auc_val = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_auc.append(auc_val)
    
    if len(bootstrapped_auc) == 0:
        return original_auc, (0.0, 1.0)
    
    # Calculate percentile confidence intervals
    sorted_auc = np.sort(bootstrapped_auc)
    lower_idx = int(n_bootstraps * (1 - alpha) / 2)
    upper_idx = int(n_bootstraps * (1 + alpha) / 2)
    
    ci_lower = sorted_auc[lower_idx] if lower_idx < len(sorted_auc) else sorted_auc[0]
    ci_upper = sorted_auc[upper_idx] if upper_idx < len(sorted_auc) else sorted_auc[-1]
    
    return original_auc, (ci_lower, ci_upper)

# ====================
# 从BATL_GBMMP3neo.ipynb导入的增强可视化函数
# ====================
def create_modern_roc_curve(y_true, y_pred_proba, set_name="", ax=None):
    """
    Create modern ROC curve
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_val, (ci_lower, ci_upper) = calculate_auc_ci(y_true, y_pred_proba)
    
    # Main ROC curve
    ax.plot(fpr, tpr, color=CUSTOM_COLORS['primary'], 
            linewidth=3, alpha=0.9,
            label=f'AUC = {auc_val:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})')
    
    # Fill confidence interval
    ax.fill_between(fpr, tpr, alpha=0.2, color=CUSTOM_COLORS['primary'])
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label='Random classifier')
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Mark optimal point
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], 
               s=150, zorder=5, color=CUSTOM_COLORS['accent'],
               edgecolors='white', linewidth=2,
               label=f'Optimal point (J={j_scores[optimal_idx]:.3f})')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='medium')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='medium')
    
    title = f'ROC Curve - {set_name}' if set_name else 'ROC Curve'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add performance summary
    summary_text = (f'Optimal threshold: {optimal_threshold:.3f}\n'
                    f'Sensitivity: {tpr[optimal_idx]:.3f}\n'
                    f'Specificity: {1-fpr[optimal_idx]:.3f}\n'
                    f'Youden index: {j_scores[optimal_idx]:.3f}')
    
    ax.text(0.6, 0.2, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    return auc_val, optimal_threshold

def create_enhanced_confusion_matrix(y_true, y_pred, set_name="", ax=None):
    """
    Create enhanced confusion matrix
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=blue_cmap, vmin=0, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Proportion', rotation=270, labelpad=20)
    
    # Set labels
    classes = ['Non-BA', 'BA']
    tick_marks = np.arange(len(classes))
    
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='medium')
    ax.set_ylabel('True Label', fontsize=12, fontweight='medium')
    
    title = f'Confusion Matrix - {set_name}' if set_name else 'Confusion Matrix'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i-0.1, f'{cm[i, j]}',
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
                fontsize=12, fontweight='bold')
        
        ax.text(j, i+0.1, f'({cm_normalized[i, j]:.1%})',
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
                fontsize=10)
    
    # Calculate performance metrics
    accuracy = np.trace(cm) / np.sum(cm)
    sensitivity = cm[1,1] / cm[1,:].sum() if cm[1,:].sum() > 0 else 0
    specificity = cm[0,0] / cm[0,:].sum() if cm[0,:].sum() > 0 else 0
    ppv = cm[1,1] / cm[:,1].sum() if cm[:,1].sum() > 0 else 0
    npv = cm[0,0] / cm[:,0].sum() if cm[:,0].sum() > 0 else 0
    
    # Add metrics box
    metrics_text = (f'Accuracy: {accuracy:.3f}\n'
                    f'Sensitivity: {sensitivity:.3f}\n'
                    f'Specificity: {specificity:.3f}\n'
                    f'PPV: {ppv:.3f}\n'
                    f'NPV: {npv:.3f}')
    
    ax.text(1.7, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    return cm

def create_probability_distribution(y_true, y_proba, set_name="", ax=None):
    """
    Create probability distribution histogram
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate BA and non-BA probabilities
    ba_probs = y_proba[y_true == 1]
    non_ba_probs = y_proba[y_true == 0]
    
    # Create histogram
    bins = np.linspace(0, 1, 31)
    
    ax.hist(non_ba_probs, bins=bins, alpha=0.7, label='Non-BA',
            color=CUSTOM_COLORS['primary'], edgecolor='black', density=True)
    ax.hist(ba_probs, bins=bins, alpha=0.7, label='BA',
            color=CUSTOM_COLORS['accent'], edgecolor='black', density=True)
    
    # Add decision threshold line
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Threshold (0.5)')
    
    # Add probability density curves
    if len(ba_probs) > 1:
        kde_ba = gaussian_kde(ba_probs)
        x_ba = np.linspace(0, 1, 100)
        ax.plot(x_ba, kde_ba(x_ba), color='darkred', linewidth=2, alpha=0.8)
    
    if len(non_ba_probs) > 1:
        kde_non_ba = gaussian_kde(non_ba_probs)
        x_non_ba = np.linspace(0, 1, 100)
        ax.plot(x_non_ba, kde_non_ba(x_non_ba), color='darkblue', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='medium')
    ax.set_ylabel('Density', fontsize=12, fontweight='medium')
    
    title = f'Probability Distribution - {set_name}' if set_name else 'Probability Distribution'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics
    stats_text = (f'Non-BA (n={len(non_ba_probs)})\n'
                  f'  Mean: {non_ba_probs.mean():.3f}\n'
                  f'  SD: {non_ba_probs.std():.3f}\n\n'
                  f'BA (n={len(ba_probs)})\n'
                  f'  Mean: {ba_probs.mean():.3f}\n'
                  f'  SD: {ba_probs.std():.3f}')
    
    ax.text(0.72, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    return ax

def create_calibration_curve(y_true, y_proba, set_name="", ax=None):
    """
    Create calibration curve
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy='quantile')
    
    # Create scatter plot with lines
    ax.plot(prob_pred, prob_true, 'o-', linewidth=2.5, markersize=8,
            color=CUSTOM_COLORS['primary'], label='Model calibration')
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Perfect calibration')
    
    # Fill areas
    ax.fill_between(prob_pred, prob_pred, prob_true, where=(prob_true >= prob_pred),
                    alpha=0.2, color=CUSTOM_COLORS['accent'], label='Overconfident')
    ax.fill_between(prob_pred, prob_pred, prob_true, where=(prob_true < prob_pred),
                    alpha=0.2, color=CUSTOM_COLORS['success'], label='Underconfident')
    
    # Add histogram showing sample distribution
    ax2 = ax.twinx()
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(y_proba, bins) - 1
    bin_counts = [np.sum(bin_indices == i) for i in range(10)]
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    ax2.bar(bin_centers, bin_counts, width=0.08, alpha=0.3,
            color=CUSTOM_COLORS['gray1'], label='Sample count')
    ax2.set_ylabel('Sample count', fontsize=10)
    ax2.set_ylim(0, max(bin_counts) * 1.3)
    
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='medium')
    ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='medium')
    
    title = f'Calibration Curve - {set_name}' if set_name else 'Calibration Curve'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Merge legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Calculate Brier score
    brier = brier_score_loss(y_true, y_proba)
    ax.text(0.05, 0.95, f'Brier score: {brier:.4f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    return brier

print("✅ Enhanced visualization functions loaded")

# ====================
# 训练随机森林模型（防止过拟合）
# ====================
def train_random_forest_model():
    print("\n" + "="*60)
    print("🌲 Training Random Forest Model for Neonatal BA Diagnosis")
    print("="*60)
    
    # 使用与notebook相同的RF参数（防止过拟合）
    rf_model = RandomForestClassifier(
        n_estimators=100,        # 树的数量
        max_depth=5,            # 限制树深度
        min_samples_split=10,   # 分割所需最小样本数
        min_samples_leaf=5,     # 叶节点所需最小样本数
        max_features=0.5,       # 每棵树使用的特征比例
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print("\n🚀 Training model...")
    rf_model.fit(class_x_train, class_y_train)
    
    return rf_model

# ====================
# 模型评估函数
# ====================
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Random Forest"):
    """
    评估模型并生成可视化
    """
    print(f"\n{'='*60}")
    print(f"📈 {model_name} - Performance Evaluation")
    print('='*60)
    
    # 训练集预测
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    # 测试集预测
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算训练集指标
    train_metrics = {
        'train_Accuracy': accuracy_score(y_train, y_train_pred),
        'train_Precision': precision_score(y_train, y_train_pred),
        'train_Recall': recall_score(y_train, y_train_pred),
        'train_F1': f1_score(y_train, y_train_pred),
        'train_AUC': roc_auc_score(y_train, y_train_proba),
        'train_MCC': matthews_corrcoef(y_train, y_train_pred)
    }
    
    # 计算特异性
    cm_train = confusion_matrix(y_train, y_train_pred)
    if cm_train.shape == (2, 2):
        train_metrics['train_Specificity'] = cm_train[0,0] / cm_train[0,:].sum() if cm_train[0,:].sum() > 0 else 0
    
    # 计算测试集指标
    test_metrics = {
        'Accuracy': accuracy_score(y_test, y_test_pred),
        'Precision': precision_score(y_test, y_test_pred),
        'Recall': recall_score(y_test, y_test_pred),
        'F1': f1_score(y_test, y_test_pred),
        'AUC': roc_auc_score(y_test, y_test_proba),
        'MCC': matthews_corrcoef(y_test, y_test_pred)
    }
    
    # 计算特异性
    cm_test = confusion_matrix(y_test, y_test_pred)
    if cm_test.shape == (2, 2):
        test_metrics['Specificity'] = cm_test[0,0] / cm_test[0,:].sum() if cm_test[0,:].sum() > 0 else 0
    
    # AUC置信区间
    train_auc, train_auc_ci = calculate_auc_ci(y_train, y_train_proba)
    test_auc, test_auc_ci = calculate_auc_ci(y_test, y_test_proba)
    
    train_metrics['train_AUC_CI'] = train_auc_ci
    test_metrics['AUC_CI'] = test_auc_ci
    
    # 组合指标
    combined_metrics = {**train_metrics, **test_metrics}
    
    # 创建性能比较DataFrame
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Specificity', 'MCC'],
        'Training': [
            f"{train_metrics['train_Accuracy']:.4f}",
            f"{train_metrics['train_Precision']:.4f}",
            f"{train_metrics['train_Recall']:.4f}",
            f"{train_metrics['train_F1']:.4f}",
            f"{train_metrics['train_AUC']:.4f}",
            f"{train_metrics.get('train_Specificity', 0):.4f}",
            f"{train_metrics['train_MCC']:.4f}"
        ],
        'Test': [
            f"{test_metrics['Accuracy']:.4f}",
            f"{test_metrics['Precision']:.4f}",
            f"{test_metrics['Recall']:.4f}",
            f"{test_metrics['F1']:.4f}",
            f"{test_metrics['AUC']:.4f}",
            f"{test_metrics.get('Specificity', 0):.4f}",
            f"{test_metrics['MCC']:.4f}"
        ]
    })
    
    print("\n📊 Performance Metrics Comparison (Training vs Test):")
    print(metrics_df.to_string(index=False))
    
    # 分类报告
    print("\n📋 Detailed Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['Non-BA', 'BA']))
    
    # 创建可视化
    print("\n🎨 Generating comprehensive visualizations...")
    visualize_comprehensive_results(y_train, y_train_proba, y_train_pred, 
                                   y_test, y_test_proba, y_test_pred, model_name)
    
    # 找到最优阈值
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\n🎯 ROC-based Optimal Threshold: {optimal_threshold:.3f}")
    print(f"  Sensitivity at optimal threshold: {tpr[optimal_idx]:.3f}")
    print(f"  Specificity at optimal threshold: {1-fpr[optimal_idx]:.3f}")
    
    combined_metrics['optimal_threshold'] = optimal_threshold
    
    return combined_metrics

def visualize_comprehensive_results(y_train, y_train_proba, y_train_pred,
                                  y_test, y_test_proba, y_test_pred, model_name):
    """
    创建综合可视化图表
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. ROC曲线（训练集）
    create_modern_roc_curve(y_train, y_train_proba, "Training Set", ax=axes[0, 0])
    
    # 2. ROC曲线（测试集）
    create_modern_roc_curve(y_test, y_test_proba, "Test Set", ax=axes[0, 1])
    
    # 3. 混淆矩阵（训练集）
    create_enhanced_confusion_matrix(y_train, y_train_pred, "Training Set", ax=axes[1, 0])
    
    # 4. 混淆矩阵（测试集）
    create_enhanced_confusion_matrix(y_test, y_test_pred, "Test Set", ax=axes[1, 1])
    
    # 5. 概率分布
    create_probability_distribution(y_test, y_test_proba, "Test Set", ax=axes[2, 0])
    
    # 6. 校准曲线
    create_calibration_curve(y_test, y_test_proba, "Test Set", ax=axes[2, 1])
    
    plt.suptitle(f'{model_name} - Comprehensive Analysis', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存为PDF
    pdf_path = f'random_forest_comprehensive_analysis.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"  📄 PDF saved: {pdf_path}")
    
    plt.show()

# ====================
# 特征重要性分析
# ====================
def analyze_feature_importance(model, feature_names):
    """
    分析特征重要性
    """
    print("\n📊 Feature Importance Analysis")
    print(f"{'='*60}")
    
    # 获取特征重要性
    rf_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_importance
    }).sort_values('Importance', ascending=False)
    
    print("Random Forest Feature Importance:")
    print(feature_importance_df.to_string(index=False))
    
    # 绘制特征重要性图
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    bars = plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors)
    plt.xlabel('Importance (Gini Index)', fontsize=12, fontweight='medium')
    plt.ylabel('Feature', fontsize=12, fontweight='medium')
    plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # 添加数值标签
    for i, (v, bar) in enumerate(zip(feature_importance_df['Importance'], bars)):
        plt.text(v + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{v:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('feature_importance.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
    return feature_importance_df

# ====================
# 交叉验证
# ====================
def perform_cross_validation(model, X, y, n_splits=5):
    """
    执行交叉验证
    """
    print("\n🔬 Cross-Validation Validation")
    print(f"{'='*60}")
    
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 计算AUC交叉验证分数
    cv_auc = cross_val_score(
        model, 
        X, 
        y,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    print(f"Cross-validation AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(f"Individual fold AUCs: {cv_auc}")
    
    # 可视化交叉验证结果
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, n_splits+1), cv_auc, color=CUSTOM_COLORS['primary'], alpha=0.7)
    ax.axhline(y=cv_auc.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean AUC = {cv_auc.mean():.3f}')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(f'{n_splits}-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, n_splits+1))
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cross_validation_results.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
    return cv_auc

# ====================
# 主训练函数
# ====================
def main():
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    
    # 忽略警告
    warnings.filterwarnings('ignore')
    
    print("\n" + "="*60)
    print("🚀 Neonatal Biliary Atresia Random Forest Model")
    print("="*60)
    
    # 训练随机森林模型
    rf_model = train_random_forest_model()
    
    # 评估模型
    feature_names = class_x_train.columns.tolist()
    metrics = evaluate_model(rf_model, class_x_train, class_y_train, 
                            class_x_test, class_y_test, "Random Forest")
    
    # 分析特征重要性
    feature_importance_df = analyze_feature_importance(rf_model, feature_names)
    
    # 执行交叉验证
    cv_scores = perform_cross_validation(rf_model, data_scaled, data_target)
    
    # 检查过拟合
    print("\n🔍 Overfitting Check:")
    train_acc = metrics['train_Accuracy']
    test_acc = metrics['Accuracy']
    if train_acc > 0.95 and test_acc < 0.85:
        print("⚠️ Warning: Model may be overfitting!")
        print(f"  Training accuracy: {train_acc:.3f}")
        print(f"  Test accuracy: {test_acc:.3f}")
        print(f"  Difference: {train_acc - test_acc:.3f}")
    else:
        print("✅ Model generalization performance is good")
    
    # 保存模型和标准化器
    joblib.dump(rf_model, 'neonatal_ba_rf_model.pkl')
    joblib.dump(scaler, 'neonatal_ba_scaler.pkl')
    print("\n✅ Model and scaler saved successfully")
    
    # 保存特征信息
    feature_info = {
        'features': feature_names,
        'feature_count': len(feature_names),
        'feature_descriptions': {
            'GB_length': 'Gallbladder Length (mm)',
            'Abnormal_GEI': 'Abnormal Gallbladder Emptying Index (Binary)',
            'GGT': 'Gamma-Glutamyl Transferase (U/L)',
            'DBIL': 'Direct Bilirubin (μmol/L)',
            'MMP7': 'Matrix Metalloproteinase-7 (ng/mL)'
        }
    }
    joblib.dump(feature_info, 'feature_info.pkl')
    print("✅ Feature information saved")
    
    # 保存性能指标
    performance_data = {
        'model': 'RandomForestClassifier',
        'metrics': metrics,
        'feature_importance': feature_importance_df.to_dict('records'),
        'cv_scores': cv_scores.tolist(),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'dataset_size': data.shape[0],
        'train_size': class_x_train.shape[0],
        'test_size': class_x_test.shape[0],
        'feature_names': feature_names
    }
    joblib.dump(performance_data, 'performance_metrics.pkl')
    
    # 创建性能总结CSV文件
    summary_df = pd.DataFrame({
        'Metric': ['AUC', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'MCC'],
        'Training': [
            metrics['train_AUC'],
            metrics['train_Accuracy'],
            metrics['train_Precision'],
            metrics['train_Recall'],
            metrics.get('train_Specificity', 0),
            metrics['train_F1'],
            metrics['train_MCC']
        ],
        'Test': [
            metrics['AUC'],
            metrics['Accuracy'],
            metrics['Precision'],
            metrics['Recall'],
            metrics.get('Specificity', 0),
            metrics['F1'],
            metrics['MCC']
        ]
    })
    summary_df.to_csv('model_performance_summary.csv', index=False)
    
    print("✅ Performance metrics saved")
    
    # 临床建议
    print("\n" + "="*60)
    print("🏥 Clinical Application Recommendations")
    print("="*60)
    
    if feature_importance_df is not None:
        print("\n📋 Top 3 Predictive Features:")
        for i, (feature, importance) in enumerate(feature_importance_df.head(3).values):
            print(f"   {i+1}. {feature}: Importance={importance:.3f}")
    
    # 找到最优阈值
    y_test_proba = rf_model.predict_proba(class_x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(class_y_test, y_test_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\n🎯 Recommended Decision Threshold: {optimal_threshold:.3f}")
    print(f"  Sensitivity at this threshold: {tpr[optimal_idx]:.3f}")
    print(f"  Specificity at this threshold: {1-fpr[optimal_idx]:.3f}")
    
    # 保存临床建议
    clinical_recommendations = {
        'optimal_threshold': optimal_threshold,
        'sensitivity': tpr[optimal_idx],
        'specificity': 1-fpr[optimal_idx],
        'top_feature_1': feature_importance_df.iloc[0]['Feature'] if feature_importance_df is not None else '',
        'top_feature_1_importance': feature_importance_df.iloc[0]['Importance'] if feature_importance_df is not None else 0,
        'top_feature_2': feature_importance_df.iloc[1]['Feature'] if feature_importance_df is not None else '',
        'top_feature_2_importance': feature_importance_df.iloc[1]['Importance'] if feature_importance_df is not None else 0,
        'top_feature_3': feature_importance_df.iloc[2]['Feature'] if feature_importance_df is not None else '',
        'top_feature_3_importance': feature_importance_df.iloc[2]['Importance'] if feature_importance_df is not None else 0
    }
    
    clinical_df = pd.DataFrame([clinical_recommendations])
    clinical_df.to_csv('clinical_recommendations.csv', index=False)
    print("✅ Clinical recommendations saved")
    
    # 最终输出
    print("\n" + "="*60)
    print("🎉 Training completed successfully!")
    print("="*60)
    print("\n📁 Generated files:")
    print("  - neonatal_ba_rf_model.pkl (Random Forest model)")
    print("  - neonatal_ba_scaler.pkl (Feature scaler)")
    print("  - feature_info.pkl (Feature information)")
    print("  - performance_metrics.pkl (Performance metrics)")
    print("  - model_performance_summary.csv (Performance summary)")
    print("  - clinical_recommendations.csv (Clinical recommendations)")
    print("  - random_forest_comprehensive_analysis.pdf (Comprehensive visualization)")
    print("  - feature_importance.pdf (Feature importance plot)")
    print("  - cross_validation_results.pdf (Cross-validation results)")
    
    return rf_model, scaler, metrics, feature_names

if __name__ == "__main__":
    try:
        model, scaler, metrics, feature_names = main()
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()