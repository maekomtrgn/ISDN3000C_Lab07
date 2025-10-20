# Lab 07 — White Box AI (Toaster)

This README explains how the main models used in the notebook work (Logistic Regression, Decision Tree, SVM, Random Forest, XGBoost). The provided dataset (Siyan and Professor) should be used for all evaluation and submission.

---

## Contents
- `ISDN3000C_Lab07.ipynb` — Primary notebook (Steps 1–10). Run cells top-to-bottom.

---

## Quick start (Colab)
1. Open Google Colab: https://colab.research.google.com
2. Open notebook from GitHub: use the **GitHub** tab and paste your forked repo URL, then open `ISDN3000C_Lab07.ipynb`.
3. (Optional) Install missing libs at the top of the notebook:
```python
!pip install -q plotly scikit-learn xgboost
```
4. Load the premade dataset by  Professor in Colab. 

5. Verify column names: `ToastingTime`, `BreadThickness`, `AmbientTemp`, `IsFrozen`, `ToastState`.
6. Create train/test splits: 80/20
```python
from sklearn.model_selection import train_test_split
X = df[['ToastingTime','BreadThickness','AmbientTemp','IsFrozen']]
y = df['ToastState']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
7. Run notebook cells (top → bottom). Save a copy to Drive and then to your fork: File > Save a copy in Drive, then File > Save a copy in GitHub.

---

## Submission
- Fork the template repository: https://github.com/siyanhu/ISDN3000C_Lab07_template
- Work in your fork, run and complete the notebook, and push the final `ISDN3000C_Lab07.ipynb` to your fork.
- Submit the fork link to Canvas (one submission per group).

---

## How the notebook evaluates models (overview)
The notebook trains and compares several classifiers. For each candidate model we recommend:
- A 5-fold cross-validation score (mean ± std).
- A held-out test set score (accuracy on `X_test`).
- Timings: measure training and inference times (use `time.perf_counter()` for higher precision).
- For robustness, run timing N times and report the median.

Recommended evaluation snippet (Colab / notebook cell):
```python
import time
from sklearn.model_selection import cross_val_score

# CV
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print('CV mean ± std:', cv_scores.mean(), cv_scores.std())

# Timed fit & predict
t0 = time.perf_counter()
model.fit(X_train, y_train)
t1 = time.perf_counter()
preds = model.predict(X_test)
t2 = time.perf_counter()
train_time = t1 - t0
infer_time = t2 - t1

from sklearn.metrics import accuracy_score
print('Test accuracy:', accuracy_score(y_test, preds))
print('Train time:', train_time, 'Infer time:', infer_time)
```

---

## Model notes & Colab-ready examples
Below are short explanations and code snippets extracted/adapted from the notebook to help you understand and run each model.

### 1) Logistic Regression
- Use for linear separable-ish problems; fast and interpretable (coef_ shows feature weights).
- Key args: `solver` (optimization algorithm), `max_iter` (iterations), `penalty` (L1/L2), `C` (inverse regularization strength).
- Multiclass: `liblinear` uses one-vs-rest; `lbfgs` and `newton-cg` support multinomial.

Example (Colab cell):
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

pipe = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', max_iter=500))
pipe.fit(X_train, y_train)
print('Test acc:', accuracy_score(y_test, pipe.predict(X_test)))

# Check iterations (if you want the underlying estimator)
est = pipe.named_steps['logisticregression']
print('n_iter_:', getattr(est, 'n_iter_', None))
```

Convergence tips:
- If you see ConvergenceWarning, increase `max_iter` or change solver.
- For `sag`/`saga`, scale features first. These solvers are faster on large data.

### 2) Decision Tree
- Interpretable; controls like `max_depth` manage overfitting.
- Visualize with `plot_tree`.

Example:
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
print('Test acc:', accuracy_score(y_test, clf.predict(X_test)))

import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=['Under','Perfect','Burnt'], max_depth=3)
plt.show()
```

Overfitting check: train trees with `max_depth` from 1..20, plot train vs test accuracy to find where test accuracy stops improving.

### 3) Support Vector Machine (SVC)
- Good for complex boundaries. Use scaling (StandardScaler) before SVC.
- Key params: `kernel` (rbf, linear, poly), `C` (regularization), `gamma` (kernel coefficient).
- Kernel = 'rbf' often works well; tune `C` and `gamma` via GridSearch.

Example:
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=100, gamma=0.1))
pipe.fit(X_train, y_train)
print('Test acc:', accuracy_score(y_test, pipe.predict(X_test)))
```

GridSearch example (Colab-friendly):
```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])
param_grid = {'svm__kernel':['rbf'],'svm__C':[0.1,1,10,100],'svm__gamma':[0.001,0.01,0.1,'scale']}
gs = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)
```

### 4) Random Forest
- Ensemble of decision trees; robust and gives `feature_importances_`.
- Key param: `n_estimators` (number of trees), `max_depth`, `max_features`.

Example:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print('Test acc:', accuracy_score(y_test, rf.predict(X_test)))

import pandas as pd
pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
```

### 5) XGBoost
- Gradient boosting that often outperforms RandomForest on structured data. Use `eval_metric='logloss'` to silence label-encoder warnings.
- Key params: `n_estimators`, `learning_rate`, `max_depth`, `subsample`.

Example:
```python
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb = XGBClassifier(n_estimators=300, learning_rate=0.3, max_depth=3, subsample=0.8, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
print('Test acc:', accuracy_score(y_test, xgb.predict(X_test)))

pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False)
```

GridSearch for XGBoost (small grid recommended for Colab):
```python
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators':[100,300,500],'learning_rate':[0.3,0.1,0.05],'max_depth':[3,5],'subsample':[0.8,1.0]}
gs = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=42), param_grid, cv=3, n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)
```

---

## Feature interpretation
- Tree-based models provide `feature_importances_` (impurity-based) — useful but biased.
- For more reliable interpretation, compute permutation importance (`sklearn.inspection.permutation_importance`) and partial dependence plots (`sklearn.inspection.partial_dependence` / `PartialDependenceDisplay`).

---



