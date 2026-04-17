import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, accuracy_score,
                             precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('data.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print("Dataset shape:", df.shape)
print("Target distribution:\n", df['diagnosis'].value_counts())

# ── 2. Prepare Features & Target ─────────────────────────────────────────────
X = df.drop(columns=['id', 'diagnosis'])
y = (df['diagnosis'] == 'M').astype(int)   # Malignant=1, Benign=0

# ── 3. Train / Test Split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Feature Scaling ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 5. Hyperparameter Tuning (Grid Search) ───────────────────────────────────
param_grid = {
    'C':      [0.1, 1, 10, 100],
    'gamma':  ['scale', 'auto', 0.001, 0.01],
    'kernel': ['rbf', 'linear']
}
grid = GridSearchCV(SVC(probability=True), param_grid,
                    cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_s, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best CV Accuracy:", round(grid.best_score_, 4))

best_svm = grid.best_estimator_

# ── 6. Evaluate on Test Set ──────────────────────────────────────────────────
y_pred = best_svm.predict(X_test_s)
y_prob = best_svm.predict_proba(X_test_s)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

cm       = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc  = auc(fpr, tpr)
print("ROC AUC:", round(roc_auc, 4))

# ── 7. Cross-Validation Scores ───────────────────────────────────────────────
cv_scores = cross_val_score(best_svm, X_train_s, y_train, cv=5, scoring='accuracy')
print("5-Fold CV Scores:", cv_scores.round(4), "| Mean:", round(cv_scores.mean(), 4))

# ── 8. Feature Importance (Linear SVM) ──────────────────────────────────────
lin_svm = SVC(kernel='linear', C=grid.best_params_.get('C', 1), probability=True)
lin_svm.fit(X_train_s, y_train)
coef     = np.abs(lin_svm.coef_[0])
feat_imp = pd.Series(coef, index=X.columns).sort_values(ascending=False)

# ── 9. Plotting ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0f172a')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_facecolor('#1e293b')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

colors = ['#38bdf8', '#f472b6', '#4ade80', '#facc15', '#c084fc']

# — Confusion Matrix —
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'],
            linewidths=0.5, linecolor='#0f172a',
            annot_kws={'size': 14, 'color': 'white'})
ax1.set_title('Confusion Matrix', color='white', fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('Predicted', color='#94a3b8')
ax1.set_ylabel('Actual',    color='#94a3b8')
ax1.tick_params(colors='#94a3b8')

# — ROC Curve —
ax2.plot(fpr, tpr, color='#38bdf8', lw=2, label=f'AUC = {roc_auc:.4f}')
ax2.plot([0, 1], [0, 1], '--', color='#475569', lw=1)
ax2.fill_between(fpr, tpr, alpha=0.15, color='#38bdf8')
ax2.set_title('ROC Curve',          color='white', fontsize=13, fontweight='bold')
ax2.set_xlabel('False Positive Rate', color='#94a3b8')
ax2.set_ylabel('True Positive Rate',  color='#94a3b8')
ax2.tick_params(colors='#94a3b8')
ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white')

# — 5-Fold CV Accuracy —
bars = ax3.bar(range(1, 6), cv_scores, color=colors,
               edgecolor='#0f172a', linewidth=1.5)
ax3.axhline(cv_scores.mean(), color='white', linestyle='--',
            lw=1.5, label=f'Mean: {cv_scores.mean():.4f}')
ax3.set_title('5-Fold CV Accuracy', color='white', fontsize=13, fontweight='bold')
ax3.set_xlabel('Fold',     color='#94a3b8')
ax3.set_ylabel('Accuracy', color='#94a3b8')
ax3.tick_params(colors='#94a3b8')
ax3.set_ylim(0.9, 1.0)
ax3.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
for bar, val in zip(bars, cv_scores):
    ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
             f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=9)

# — Top 15 Feature Importances —
top15 = feat_imp.head(15)
ax4.barh(range(len(top15)), top15.values[::-1],
         color='#4ade80', edgecolor='#0f172a', linewidth=0.5)
ax4.set_yticks(range(len(top15)))
ax4.set_yticklabels(top15.index[::-1], color='#94a3b8', fontsize=8)
ax4.set_title('Top 15 Feature Importances\n(Linear SVM |coef|)',
              color='white', fontsize=12, fontweight='bold')
ax4.set_xlabel('|Coefficient|', color='#94a3b8')
ax4.tick_params(colors='#94a3b8')

# — Predicted Probability Distribution —
ax5.hist(y_prob[y_test == 0], bins=20, alpha=0.7, color='#38bdf8',
         label='Benign',    edgecolor='#0f172a')
ax5.hist(y_prob[y_test == 1], bins=20, alpha=0.7, color='#f472b6',
         label='Malignant', edgecolor='#0f172a')
ax5.axvline(0.5, color='white', linestyle='--', lw=1.5, label='Threshold=0.5')
ax5.set_title('Predicted Probability Distribution',
              color='white', fontsize=12, fontweight='bold')
ax5.set_xlabel('P(Malignant)', color='#94a3b8')
ax5.set_ylabel('Count',        color='#94a3b8')
ax5.tick_params(colors='#94a3b8')
ax5.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white')

# — Performance Metrics Bar Chart —
metrics_names = ['Accuracy', 'Precision\n(Malig.)', 'Recall\n(Malig.)',
                 'F1\n(Malig.)', 'ROC AUC']
metrics_vals  = [
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred),
    roc_auc
]
bars3 = ax6.bar(metrics_names, metrics_vals, color=colors,
                edgecolor='#0f172a', linewidth=1.5)
ax6.set_ylim(0.85, 1.02)
ax6.set_title('Performance Metrics', color='white', fontsize=13, fontweight='bold')
ax6.set_ylabel('Score', color='#94a3b8')
ax6.tick_params(colors='#94a3b8', axis='y')
ax6.tick_params(colors='white',   axis='x', labelsize=8)
for bar, val in zip(bars3, metrics_vals):
    ax6.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
             f'{val:.4f}', ha='center', va='bottom',
             color='white', fontsize=9, fontweight='bold')

fig.suptitle('SVM Classification — Breast Cancer Dataset',
             color='white', fontsize=16, fontweight='bold', y=0.98)

plt.savefig('svm_results.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.show()
print("\nDone! Plot saved as svm_results.png")