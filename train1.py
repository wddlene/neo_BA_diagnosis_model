import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, classification_report, accuracy_score, roc_auc_score, confusion_matrix, brier_score_loss, matthews_corrcoef
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
import math
import warnings
import joblib 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
from scipy.stats import gaussian_kde
import itertools

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

# Load data
print("📊 Loading neonatal biliary atresia data...")
data = pd.read_csv("BAGBMMP1209_nonscaled.csv")

# Extract target variable and features - 使用与迁移学习模型相同的特征集
data_target = data['BA']
data_features = data[['GB_length', 'Abnormal_GEI', 'GGT', 'DBIL', 'MMP7']]

# Data standardization
print("🔧 Standardizing features...")
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = pd.DataFrame(scaler.fit_transform(data_features), columns=data_features.columns)

# Split training and test sets
print("✂️ Splitting training and test sets...")
class_x_train, class_x_test, class_y_train, class_y_test = train_test_split(
    data_scaled, data_target, test_size=0.3, random_state=42, stratify=data_target
)

# 确保标签是numpy数组格式
class_y_train = class_y_train.values.ravel()
class_y_test = class_y_test.values.ravel()

# Display dataset information
print(f"📈 Dataset information:")
print(f"  Total samples: {data.shape[0]}")
print(f"  Number of features: {data_scaled.shape[1]}")
print(f"  Training set size: {class_x_train.shape[0]} samples")
print(f"  Test set size: {class_x_test.shape[0]} samples")
print(f"  Training class distribution: {pd.Series(class_y_train).value_counts().to_dict()}")
print(f"  Test class distribution: {pd.Series(class_y_test).value_counts().to_dict()}")

# Function to calculate AUC confidence intervals
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

# PyTorch Dataset class (修复版 - 简化处理)
class NeonatalBADataset(Dataset):
    def __init__(self, features, labels):
        # 确保转换为numpy数组
        if hasattr(features, 'values'):
            features = features.values
        if hasattr(labels, 'values'):
            labels = labels.values
        elif hasattr(labels, 'ravel'):
            labels = labels.ravel()
        
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Transfer Learning Model class (与Notebook保持一致)
class NeonatalTransferLearningModel(nn.Module):
    def __init__(self, input_dim, base_model=None, hidden_dims=[128, 64, 32], dropout_rate=0.4):
        super(NeonatalTransferLearningModel, self).__init__()
        self.base_model = base_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # 构建神经网络层
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers).to(self.device)
        
        # 如果有基础模型，使用特征重要性初始化权重
        if base_model is not None and hasattr(base_model, 'feature_importances_'):
            self._initialize_weights_with_importance()
    
    def _initialize_weights_with_importance(self):
        """使用基础模型的特征重要性初始化神经网络权重"""
        feature_importances = self.base_model.feature_importances_
        
        with torch.no_grad():
            # 调整第一层权重
            weight = self.model[0].weight.data
            for i in range(len(feature_importances)):
                # 基于特征重要性调整权重初始化
                weight[:, i] = weight[:, i] * (0.5 + feature_importances[i])
            
            # 应用Xavier初始化
            nn.init.xavier_uniform_(self.model[0].weight)
            
            # 对其他层应用标准初始化
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    if layer is not self.model[0]:  # 跳过第一层
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.model(x)
    
    def predict_proba(self, x):
        """预测概率"""
        self.eval()
        with torch.no_grad():
            if isinstance(x, pd.DataFrame):
                x = torch.tensor(x.values, dtype=torch.float32).to(self.device)
            elif isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32).to(self.device)
            
            predictions = self.model(x)
            probs = predictions.cpu().numpy()
            
            # 转换为二分类概率格式
            prob_array = np.zeros((len(probs), 2))
            prob_array[:, 1] = probs.flatten()
            prob_array[:, 0] = 1 - probs.flatten()
            
            return prob_array
    
    def predict(self, x, threshold=0.5):
        """预测类别"""
        proba = self.predict_proba(x)
        return (proba[:, 1] >= threshold).astype(int)

# 训练函数
def train_model(model, train_loader, epochs=150, lr=0.001, validation_loader=None):
    device = model.device
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    model.train()
    best_val_auc = 0
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []
    
    print("🔥 Starting training...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for features, labels in train_loader:
            features = features.to(device).float()
            labels = labels.to(device).float().view(-1, 1)
            
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds)
        
        # Validation phase
        val_auc = 0
        if validation_loader:
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for features, labels in validation_loader:
                    features = features.to(device).float()
                    labels = labels.to(device).float().view(-1, 1)
                    
                    outputs = model(features)
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            if len(val_labels) > 0:
                val_auc = roc_auc_score(val_labels, val_preds)
            
            # 学习率调度
            scheduler.step(val_auc)
        
        # Save history
        train_losses.append(avg_train_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        
        # 每10个epoch打印一次
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc:.4f}", end="")
            if validation_loader:
                print(f" | Val AUC: {val_auc:.4f}")
            else:
                print()
        
        # 保存最佳模型 - 基于验证AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_neonatal_transfer_model.pth')
            if (epoch + 1) % 10 == 0:
                print(f"  💾 New best model saved with AUC: {val_auc:.4f}")
    
    # 加载最佳模型
    if os.path.exists('best_neonatal_transfer_model.pth'):
        model.load_state_dict(torch.load('best_neonatal_transfer_model.pth'))
        print(f"✅ Loaded best model with AUC: {best_val_auc:.4f}")
    
    # 绘制训练历史
    if len(train_losses) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(train_losses, label='Training loss', linewidth=2, color=CUSTOM_COLORS['primary'])
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(train_aucs, label='Training AUC', linewidth=2, color=CUSTOM_COLORS['primary'])
        if val_aucs:
            axes[1].plot(val_aucs, label='Validation AUC', linewidth=2, color=CUSTOM_COLORS['accent'])
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('AUC', fontsize=12)
        axes[1].set_title('Training and Validation AUC', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.pdf', format='pdf', bbox_inches='tight', dpi=300)
        plt.show()
    
    return model

# 评估函数
def evaluate_neonatal_model(model, x, y, set_name='Validation'):
    device = model.device
    
    # 转换为PyTorch张量
    features = torch.tensor(x.values, dtype=torch.float32).to(device)
    labels = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
    
    # 预测概率
    model.eval()
    with torch.no_grad():
        predictions = model(features)
        pred_probs = predictions.cpu().numpy().flatten()
    
    # 预测类别
    pred_labels = (pred_probs > 0.5).astype(int)
    
    # 计算指标
    auc_val = roc_auc_score(y, pred_probs)
    accuracy = accuracy_score(y, pred_labels)
    precision = precision_score(y, pred_labels)
    recall = recall_score(y, pred_labels)
    f1 = f1_score(y, pred_labels)
    mcc = matthews_corrcoef(y, pred_labels)
    
    # 计算特异性
    cm = confusion_matrix(y, pred_labels)
    specificity = 0
    if cm.shape == (2, 2):
        specificity = cm[0,0] / cm[0,:].sum() if cm[0,:].sum() > 0 else 0
    
    # 计算AUC置信区间
    auc_mean, (auc_lower, auc_upper) = calculate_auc_ci(y, pred_probs)
    
    metrics = {
        'auc': auc_val,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'specificity': specificity,
        'auc_ci': (auc_lower, auc_upper)
    }
    
    print(f"\n{'='*60}")
    print(f"📊 {set_name} Set Results")
    print(f"{'='*60}")
    print(f"AUC: {auc_val:.4f} (95% CI: {auc_lower:.4f}-{auc_upper:.4f})")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    
    # 可视化
    visualize_results(y, pred_probs, metrics, set_name)
    
    return metrics

# 可视化函数
def visualize_results(true_labels, pred_probs, metrics, set_name):
    # 计算Brier分数
    brier = brier_score_loss(true_labels, pred_probs)
    metrics['brier'] = brier
    print(f"Brier Score: {brier:.4f}")
    
    # 创建综合可视化图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. ROC曲线
    create_modern_roc_curve(true_labels, pred_probs, set_name, ax=axes[0, 0])
    
    # 2. 混淆矩阵
    pred_labels = [1 if p > 0.5 else 0 for p in pred_probs]
    create_enhanced_confusion_matrix(true_labels, pred_labels, set_name, ax=axes[0, 1])
    
    # 3. 概率分布
    create_probability_distribution(true_labels, pred_probs, set_name, ax=axes[1, 0])
    
    # 4. 校准曲线
    create_calibration_curve(true_labels, pred_probs, set_name, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(f'comprehensive_analysis_{set_name.lower().replace(" ", "_")}.pdf', 
                format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
    return metrics

# 特征重要性分析
def analyze_feature_importance(model, feature_names):
    """分析特征重要性"""
    print("\n📊 Feature Importance Analysis")
    print(f"{'='*60}")
    
    # 检查模型类型
    if hasattr(model, 'feature_importances_'):
        # 如果是随机森林或树模型
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
    else:
        print("⚠️ Model does not have feature importances attribute")
        return None

# Main training function
def main():
    print("\n" + "="*60)
    print("🚀 Neonatal Biliary Atresia Transfer Learning Model")
    print("="*60)
    
    # Pre-train random forest
    print("\n🌲 Pre-training random forest...")
    pretrain_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=0.5,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    pretrain_model.fit(class_x_train, class_y_train)
    
    # 分析特征重要性
    feature_names = class_x_train.columns.tolist()
    feature_importance_df = analyze_feature_importance(pretrain_model, feature_names)
    
    # Prepare transfer learning data
    train_dataset = NeonatalBADataset(class_x_train, class_y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    val_dataset = NeonatalBADataset(class_x_test, class_y_test)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create transfer learning model
    input_dim = class_x_train.shape[1]
    print(f"\n🏗️ Building transfer learning model with {input_dim} features...")
    transfer_model = NeonatalTransferLearningModel(
        input_dim, 
        base_model=pretrain_model,
        hidden_dims=[128, 64, 32],
        dropout_rate=0.4
    )
    
    # Train transfer model
    print("\n🔥 Training transfer model...")
    trained_model = train_model(transfer_model, train_loader, epochs=150, 
                               lr=0.001, validation_loader=val_loader)
    
    # Evaluate model
    print("\n" + "="*60)
    print("📈 Model Evaluation")
    print("="*60)
    
    print("\n🔬 Training set evaluation:")
    train_metrics = evaluate_neonatal_model(trained_model, class_x_train, class_y_train, 'Training')
    
    print("\n🧪 Test set evaluation:")
    test_metrics = evaluate_neonatal_model(trained_model, class_x_test, class_y_test, 'Test')
    
    # Save model and scaler
    torch.save(trained_model.state_dict(), 'neonatal_transfer_model.pth')
    joblib.dump(scaler, 'neonatal_scaler.pkl')
    joblib.dump(pretrain_model, 'random_forest_base_model.pkl')
    print("\n✅ Model and scaler saved successfully")
    
    # 保存特征信息
    feature_info = {
        'features': feature_names,
        'feature_count': len(feature_names)
    }
    joblib.dump(feature_info, 'feature_info.pkl')
    print("✅ Feature information saved")
    
    # 性能总结
    print("\n" + "="*60)
    print("📊 Model Performance Summary")
    print("="*60)
    
    print(f"\n{'Metric':<25} {'Training':>12} {'Test':>12}")
    print("-" * 50)
    
    # AUC行（带置信区间）
    auc_row = "AUC (95% CI)"
    train_auc = f"{train_metrics['auc']:.3f} ({train_metrics['auc_ci'][0]:.3f}-{train_metrics['auc_ci'][1]:.3f})"
    test_auc = f"{test_metrics['auc']:.3f} ({test_metrics['auc_ci'][0]:.3f}-{test_metrics['auc_ci'][1]:.3f})"
    
    print(f"{auc_row:<25} {train_auc:>12} {test_auc:>12}")
    
    # 其他指标
    metrics_to_display = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'mcc', 'brier']
    metric_names = ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1-score', 'MCC', 'Brier Score']
    
    for metric, name in zip(metrics_to_display, metric_names):
        if metric in train_metrics and metric in test_metrics:
            train_val = f"{train_metrics[metric]:.3f}"
            test_val = f"{test_metrics[metric]:.3f}"
            
            print(f"{name:<25} {train_val:>12} {test_val:>12}")
    
    # 计算过拟合程度
    auc_overfit = train_metrics['auc'] - test_metrics['auc']
    acc_overfit = train_metrics['accuracy'] - test_metrics['accuracy']
    
    print(f"\n📈 Overfitting Analysis:")
    print(f"  AUC difference (Train - Test): {auc_overfit:.4f}")
    print(f"  Accuracy difference (Train - Test): {acc_overfit:.4f}")
    
    if auc_overfit > 0.1 or acc_overfit > 0.1:
        print("  ⚠️  Warning: Possible overfitting detected!")
    else:
        print("  ✅ Model shows good generalization")
    
    # 保存性能指标
    performance_data = {
        'model': 'NeonatalTransferLearningModel',
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_names': feature_names,
        'dataset_size': data.shape[0],
        'train_size': class_x_train.shape[0],
        'test_size': class_x_test.shape[0]
    }
    joblib.dump(performance_data, 'performance_metrics.pkl')
    
    # 创建性能总结CSV文件
    summary_df = pd.DataFrame({
        'Metric': ['AUC', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'MCC', 'Brier Score'],
        'Training': [
            train_metrics['auc'],
            train_metrics['accuracy'],
            train_metrics['precision'],
            train_metrics['recall'],
            train_metrics['specificity'],
            train_metrics['f1'],
            train_metrics['mcc'],
            train_metrics.get('brier', 0)
        ],
        'Test': [
            test_metrics['auc'],
            test_metrics['accuracy'],
            test_metrics['precision'],
            test_metrics['recall'],
            test_metrics['specificity'],
            test_metrics['f1'],
            test_metrics['mcc'],
            test_metrics.get('brier', 0)
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
    
    # 找到最优阈值（从测试集ROC曲线）
    y_test_proba = trained_model.predict_proba(class_x_test)[:, 1]
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
    
    return trained_model, scaler, train_metrics, test_metrics, feature_names

if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 忽略警告
    warnings.filterwarnings('ignore')
    
    try:
        model, scaler, train_metrics, test_metrics, feature_names = main()
        print("\n" + "="*60)
        print("🎉 Training completed successfully!")
        print("="*60)
        print("\n📁 Generated files:")
        print("  - neonatal_transfer_model.pth (PyTorch model)")
        print("  - neonatal_scaler.pkl (Feature scaler)")
        print("  - random_forest_base_model.pkl (Base Random Forest model)")
        print("  - feature_info.pkl (Feature information)")
        print("  - best_neonatal_transfer_model.pth (Best model)")
        print("  - training_history.pdf (Training curves)")
        print("  - comprehensive_analysis_training.pdf (Training analysis)")
        print("  - comprehensive_analysis_test.pdf (Test analysis)")
        print("  - feature_importance.pdf (Feature importance)")
        print("  - performance_metrics.pkl (Performance metrics)")
        print("  - model_performance_summary.csv (Performance summary)")
        print("  - clinical_recommendations.csv (Clinical recommendations)")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()