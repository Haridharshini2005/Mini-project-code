# ================================================
# COMPLETE THYROID DISEASE PREDICTION PIPELINE
# ================================================
import pandas as pd
import numpy as np
import warnings, os, joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (accuracy_score, classification_report,
                              ConfusionMatrixDisplay, roc_auc_score, roc_curve)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ── STEP 1: Download Dataset ──────────────────────────────────
print("⬇️ Downloading dataset...")

urls = [
    "https://raw.githubusercontent.com/Raghav-Saboo/Thyroid-Disease-Detection/main/thyroidDF.csv",
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/thyroid_disease.csv",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/thyroid0387.csv",
]

df_raw = None
for url in urls:
    try:
        df_raw = pd.read_csv(url)
        print(f"✅ Loaded from: {url}")
        print(f"   Shape: {df_raw.shape}")
        break
    except Exception as e:
        print(f"❌ Failed: {url} → {e}")

if df_raw is None:
    print("\n⬇️ Trying UCI direct download...")
    try:
        import urllib.request
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhypo.data",
            "allhypo.data"
        )
        cols = ['age','sex','on_thyroxine','query_on_thyroxine','on_antithyroid_meds',
                'sick','pregnant','thyroid_surgery','I131_treatment','query_hypothyroid',
                'query_hyperthyroid','lithium','goitre','tumor','hypopituitary','psych',
                'TSH_measured','TSH','T3_measured','T3','TT4_measured','TT4',
                'T4U_measured','T4U','FTI_measured','FTI','TBG_measured','TBG',
                'referral_source','target']
        df_raw = pd.read_csv("allhypo.data", names=cols, na_values='?')
        print("✅ Loaded from UCI. Shape:", df_raw.shape)
    except Exception as e:
        print(f"❌ UCI failed: {e}")

if df_raw is None:
    print("\n⚠️ All downloads failed — using synthetic data for demo...")
    from sklearn.datasets import make_classification
    X_syn, y_syn = make_classification(
        n_samples=3000, n_features=20, n_classes=3,
        n_informative=10, n_redundant=5, random_state=42,
        weights=[0.7, 0.2, 0.1]
    )
    syn_cols = ['age','sex','on_thyroxine','TSH','T3','TT4','T4U','FTI','TBG',
                'goitre','tumor','pregnant','sick','lithium','psych',
                'thyroid_surgery','query_hypothyroid','query_hyperthyroid',
                'on_antithyroid_meds','hypopituitary']
    df_raw = pd.DataFrame(X_syn, columns=syn_cols)
    df_raw['target'] = y_syn

df_raw.to_csv("thyroidDF.csv", index=False)
print("\n💾 Saved as thyroidDF.csv")
print("Columns:", df_raw.columns.tolist())
print("\nTarget value counts:\n", df_raw.iloc[:,-1].value_counts().head(10))

# ── STEP 2: Find Target Column ────────────────────────────────
full_name_map = {0:'Negative', 1:'Hypothyroid', 2:'Hyperthyroid'}

possible = ['target','Target','class','Class','diagnosis','Diagnosis',
            'ThryroidClass','binaryClass','classes','label','Label']
target_col = None
for col in possible:
    if col in df_raw.columns:
        target_col = col
        break
if target_col is None:
    target_col = df_raw.columns[-1]
print(f"\n✅ Target column: '{target_col}'")
print("Unique values:\n", df_raw[target_col].value_counts())

# ── STEP 3: Map Labels ────────────────────────────────────────
raw_vals  = df_raw[target_col].unique()
label_map = {}
for val in raw_vals:
    v      = str(val).strip().lower().replace(' ','_')
    v_orig = str(val).strip()
    if   any(x in v for x in ['neg','normal','euthyroid','none','-','false']):
        label_map[val] = 0
    elif any(x in v for x in ['hypo','compensated','primary_h',
                               'secondary_h','subnormal']):
        label_map[val] = 1
    elif any(x in v for x in ['hyper','toxic','goitre','graves','t3_toxic']):
        label_map[val] = 2
    elif v_orig in ['A','B','C','D','K','AK','BK','CK','DK','GK','FK','MK','KJ']:
        label_map[val] = 1
    elif v_orig in ['E','F','G','H','I','J','L','M','N','O','P','Q','R','S',
                    'GI','OI','LJ','GKJ']:
        label_map[val] = 2
    elif v_orig == '-':
        label_map[val] = 0
    else:
        try:    label_map[val] = int(float(v_orig)) % 3
        except: label_map[val] = 0

print("\n📋 Label mapping:")
for k, v in label_map.items():
    print(f"   '{k}' → {v} ({full_name_map[v]})")

df_raw['label'] = df_raw[target_col].map(label_map)
print("\n✅ Class distribution:")
print(df_raw['label'].map(full_name_map).value_counts())

# ── STEP 4: Drop Unwanted Columns ────────────────────────────
drop_cols = [c for c in [target_col, 'patient_id', 'referral_source']
             if c in df_raw.columns and c != 'label']
df_raw.drop(columns=drop_cols, inplace=True)

# ── STEP 5: Encode Object Columns ────────────────────────────
for col in df_raw.select_dtypes(include='object').columns:
    if col == 'label': continue
    df_raw[col] = df_raw[col].map(
        {'t':1,'f':0,'T':1,'F':0,'M':1,'m':1,
         'y':1,'n':0,'Y':1,'N':0}
    ).fillna(df_raw[col])
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

# ── STEP 6: Separate X and y ─────────────────────────────────
X = df_raw.drop('label', axis=1)
y = df_raw['label'].values
print("\n✅ Features shape:", X.shape)
print("Classes in y:", np.unique(y, return_counts=True))

# ── STEP 7: Drop All-NaN Columns then Impute ─────────────────
all_nan_cols = X.columns[X.isnull().all()].tolist()
if all_nan_cols:
    print(f"\n⚠️ Dropping fully-NaN columns: {all_nan_cols}")
    X = X.drop(columns=all_nan_cols)

imputer = SimpleImputer(strategy='median')
X_imp   = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
print("✅ Imputation done. Shape:", X_imp.shape)
print("Missing values remaining:", X_imp.isnull().sum().sum())

# ── STEP 8: Train/Test Split ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y, test_size=0.2, random_state=42, stratify=y)
print("\n✅ Train:", X_train.shape, "| Test:", X_test.shape)

# ── STEP 9: Scale ─────────────────────────────────────────────
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("✅ Scaling done.")

# ── STEP 10: SMOTE ───────────────────────────────────────────
unique_cls, cls_counts = np.unique(y_train, return_counts=True)
k_neighbors = max(1, min(5, int(cls_counts.min()) - 1))
print(f"\n✅ SMOTE k_neighbors={k_neighbors}")

smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_bal, y_train_bal = smote.fit_resample(X_train_sc, y_train)

print("After SMOTE:")
for u, c in zip(*np.unique(y_train_bal, return_counts=True)):
    print(f"  {full_name_map[u]}: {c}")

# ── STEP 11: Feature Selection ────────────────────────────────
k_feat      = min(15, X_train_bal.shape[1])
selector    = SelectKBest(mutual_info_classif, k=k_feat)
X_train_sel = selector.fit_transform(X_train_bal, y_train_bal)
X_test_sel  = selector.transform(X_test_sc)
selected    = X_imp.columns[selector.get_support()]
print(f"\n✅ Selected {k_feat} features:")
print(list(selected))

# ── STEP 12: Base Models ──────────────────────────────────────
print("\n📊 Individual model results:")
print("-" * 45)
results = {}
base_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM':                 SVC(probability=True, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':             XGBClassifier(random_state=42, eval_metric='mlogloss'),
}
for name, m in base_models.items():
    m.fit(X_train_sel, y_train_bal)
    acc = accuracy_score(y_test, m.predict(X_test_sel))
    results[name] = acc
    print(f"  {name:22s}: {acc*100:.2f}%")

# ── STEP 13: Auto-detect Actual Classes ──────────────────────
actual_classes = sorted(np.unique(y_test))
target_names   = [full_name_map[c] for c in actual_classes]
print(f"\n✅ Actual classes : {actual_classes}")
print(f"✅ Target names   : {target_names}")

# ── STEP 14: Stacking Model ───────────────────────────────────
print("\n🔧 Training Stacking Model...")
stacking_model = StackingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
        ('xgb', XGBClassifier(random_state=42, eval_metric='mlogloss')),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    stack_method='predict_proba'
)
stacking_model.fit(X_train_sel, y_train_bal)
y_pred = stacking_model.predict(X_test_sel)

stacking_acc = accuracy_score(y_test, y_pred)
results['Stacking Model'] = stacking_acc
print(f"\n✅ Stacking Accuracy: {stacking_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      labels=actual_classes,
      target_names=target_names))

# ── STEP 15: Confusion Matrix ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7,5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=target_names,
    cmap='Blues', ax=ax
)
plt.title('Confusion Matrix — Stacking Model', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("✅ Saved confusion_matrix.png")

# ── STEP 16: Correlation Matrix ───────────────────────────────
plt.figure(figsize=(12,10))
corr = X_imp[list(selected)].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5)
plt.title('Correlation Matrix of Selected Features', fontsize=13)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150)
plt.show()
print("✅ Saved correlation_matrix.png")

# ── STEP 17: ROC-AUC Curve (handles 2 or 3 classes) ──────────
y_prob = stacking_model.predict_proba(X_test_sel)
colors = ['#185FA5','#854F0B','#3B6D11']

plt.figure(figsize=(8,6))
if len(actual_classes) == 2:
    fpr, tpr, _ = roc_curve(y_test, y_prob[:,1],
                             pos_label=actual_classes[1])
    auc_val   = roc_auc_score(y_test, y_prob[:,1])
    macro_auc = auc_val
    plt.plot(fpr, tpr, color=colors[0], lw=2,
             label=f'{target_names[1]} (AUC = {auc_val:.3f})')
else:
    y_test_bin = label_binarize(y_test, classes=actual_classes)
    for i, (cls, col) in enumerate(zip(target_names, colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:,i], y_prob[:,i])
        auc_i = roc_auc_score(y_test_bin[:,i], y_prob[:,i])
        plt.plot(fpr, tpr, color=col, lw=2,
                 label=f'{cls} (AUC = {auc_i:.3f})')
    macro_auc = roc_auc_score(
        label_binarize(y_test, classes=actual_classes),
        y_prob, multi_class='ovr', average='macro')

plt.plot([0,1],[0,1],'k--', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve — Stacking Model', fontsize=13)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()
print(f"✅ Macro AUC-ROC: {macro_auc:.4f}")

# ── STEP 18: Model Comparison Bar Chart ──────────────────────
plt.figure(figsize=(9,5))
names = list(results.keys())
accs  = [v*100 for v in results.values()]
bars  = plt.bar(names, accs,
                color=['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2'])
plt.ylim(85, 101)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Model-wise Accuracy Comparison', fontsize=13)
plt.xticks(rotation=15, ha='right')
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.2,
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()
print("✅ Saved model_comparison.png")

# ── STEP 19: Save Artifacts ───────────────────────────────────
joblib.dump(stacking_model, 'thyroid_model.pkl')
joblib.dump(scaler,         'thyroid_scaler.pkl')
joblib.dump(selector,       'thyroid_selector.pkl')
joblib.dump(imputer,        'thyroid_imputer.pkl')
print("\n✅ Saved: thyroid_model.pkl")
print("✅ Saved: thyroid_scaler.pkl")
print("✅ Saved: thyroid_selector.pkl")
print("✅ Saved: thyroid_imputer.pkl")

# ── STEP 20: Test Prediction ──────────────────────────────────
sample_raw = X_test.iloc[[0]]
sample_imp = pd.DataFrame(imputer.transform(sample_raw), columns=X.columns)
sample_sc  = scaler.transform(sample_imp)
sample_sel = selector.transform(sample_sc)

pred = stacking_model.predict(sample_sel)[0]
prob = stacking_model.predict_proba(sample_sel)[0]

print(f"\n🔬 Sample Prediction : {full_name_map[pred]}")
print(f"   Confidence        : {max(prob)*100:.1f}%")
for cls_idx, cls_name, cls_prob in zip(actual_classes, target_names, prob):
    print(f"   {cls_name:15s}: {cls_prob:.3f}")

print("\n🎉 Full pipeline complete!")
