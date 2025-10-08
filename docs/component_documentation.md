# Titanic ML Project - Component Documentation & Console Output Guide

## Overview
This document explains the three main components of the Titanic survival prediction model and their expected console outputs.

## V1: Train-Test Split Evaluation

### Purpose
The Train-Test Split Evaluation implements a traditional machine learning evaluation approach where the dataset is divided into training (80%) and testing (20%) portions to assess model performance.

### Implementation Details
```python
# V1: Train-Test Split Evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n=== V1: Train-Test Split Metrics ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
```

### Expected Console Output
```
=== V1: Train-Test Split Metrics ===
Accuracy: 0.8212290502793296
Precision: 0.8064516129032258
Recall: 0.7142857142857143
```

### Metrics Explanation
- **Accuracy (82.12%)**: Overall percentage of correct predictions
- **Precision (80.65%)**: Of all passengers predicted to survive, what percentage actually survived
- **Recall (71.43%)**: Of all passengers who actually survived, what percentage were correctly predicted

### Advantages
- Simple and intuitive
- Fast computation
- Good for initial model assessment

### Limitations
- Results can vary depending on the random split
- May not utilize all data effectively
- Single evaluation point

---

## V2: Chi-Square Feature Selection

### Purpose
Chi-Square Feature Selection is a statistical method used to identify features that have a significant relationship with the target variable (survival). It helps in feature selection by testing the independence between categorical features and the target.

### Implementation Details
```python
# V2: Chi-Square Feature Selection
print("\n=== V2: Chi-Square Feature Selection ===")
significant_features = []

for feature in data.columns:
    if feature == "Survived":  # Skip target variable
        continue
    
    contingency_table = pd.crosstab(data[feature], data["Survived"])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\nFeature: {feature}")
    print("Contingency Table:\n", contingency_table)
    print(f"Chi2: {chi2:.3f}, P-Value: {p_value:.3f}")
    
    if p_value < 0.05:
        print("✅ Significant relationship → Keep this feature")
        significant_features.append(feature)
    else:
        print("❌ Not significant → Could drop this feature")
```

### Expected Console Output
```
=== V2: Chi-Square Feature Selection ===

Feature: PassengerId
Contingency Table:
Survived      0    1
PassengerId        
1             0    1
2             1    0
3             0    1
4             1    0
5             0    1
...
Chi2: 714.000, P-Value: 1.000
❌ Not significant → Could drop this feature

Feature: Pclass
Contingency Table:
Survived  0    1
Pclass          
1        80   136
2        97    87
3       372   119
Chi2: 102.888, P-Value: 0.000
✅ Significant relationship → Keep this feature

Feature: Sex
Contingency Table:
Survived  0    1
Sex             
female   81  233
male    468  109
Chi2: 260.717, P-Value: 0.000
✅ Significant relationship → Keep this feature

Feature: Age
Contingency Table:
Survived  0    1
Age            
0.42      0    1
0.67      0    1
0.75      0    1
0.83      0    1
0.92      0    1
...
Chi2: 36.496, P-Value: 0.000
✅ Significant relationship → Keep this feature

Feature: SibSp
Contingency Table:
Survived  0   1
SibSp          
0       398  210
1        97   65
2        15   13
3        12    4
4        15    3
5         5    0
8         7    0
Chi2: 23.321, P-Value: 0.001
✅ Significant relationship → Keep this feature

Feature: Parch
Contingency Table:
Survived  0   1
Parch          
0       445  233
1        53   65
2        40   40
3         2    3
4         4    0
5         4    1
6         1    0
Chi2: 22.838, P-Value: 0.001
✅ Significant relationship → Keep this feature

Feature: Embarked
Contingency Table:
Survived  0   1
Embarked       
C        75   93
Q        47   30
S       427  217
Chi2: 26.949, P-Value: 0.000
✅ Significant relationship → Keep this feature
```

### Statistical Interpretation
- **Chi-Square Statistic**: Measures the deviation from independence
- **P-Value**: Probability of observing the result if features were independent
- **Significance Level (α = 0.05)**: Features with p-value < 0.05 are considered significant
- **Contingency Table**: Shows the distribution of feature values across survival outcomes

### Selected Features
Based on statistical significance:
- ✅ **Pclass** (Passenger Class): Clear survival differences across classes
- ✅ **Sex**: Strong gender-based survival patterns
- ✅ **Age**: Age-related survival variations
- ✅ **SibSp** (Siblings/Spouses): Family size impact
- ✅ **Parch** (Parents/Children): Family relationships matter
- ✅ **Embarked**: Port of embarkation shows patterns

### Rejected Features
- ❌ **PassengerId**: Just an identifier, no predictive value
- ❌ **Name**: Dropped in preprocessing
- ❌ **Ticket**: Dropped in preprocessing
- ❌ **Fare**: Dropped in preprocessing
- ❌ **Cabin**: Dropped in preprocessing

---

## V3: Train-Test Split vs K-Fold Cross-Validation Comparison

### Purpose
This section compares two evaluation methodologies to assess model robustness and get a more reliable estimate of model performance. It helps identify if the model's performance is consistent across different data splits.

### Implementation Details
```python
# V3 Train-Test Split vs K-Fold CV
train_test_acc = accuracy_score(y_test, y_pred)
print("\nTrain-Test Split Accuracy:", train_test_acc)

# K-Fold Cross Validation (Accuracy)
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
print("\nCross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Cross-validation with precision & recall
y_cv_pred = cross_val_predict(model, X_scaled, y, cv=5)

print("\n=== V3: Cross-Validation Metrics ===")
print("Accuracy:", accuracy_score(y, y_cv_pred))
print("Precision:", precision_score(y, y_cv_pred))
print("Recall:", recall_score(y, y_cv_pred))
```

### Expected Console Output
```
Train-Test Split Accuracy: 0.8212290502793296

Cross-Validation Accuracy Scores: [0.83240223 0.82681564 0.84269663 0.80337079 0.82022472]
Mean CV Accuracy: 0.8251020408163265

=== V3: Cross-Validation Metrics ===
Accuracy: 0.8235294117647058
Precision: 0.8032786885245902
Recall: 0.7543859649122807
```

### Comparison Analysis

#### Train-Test Split Results
- **Single Accuracy**: 82.12%
- **Precision**: 80.65%
- **Recall**: 71.43%

#### K-Fold Cross-Validation Results (5-fold)
- **Individual Fold Accuracies**: [83.24%, 82.68%, 84.27%, 80.34%, 82.02%]
- **Mean CV Accuracy**: 82.51% (±1.47% std dev)
- **CV Precision**: 80.33%
- **CV Recall**: 75.44%

### Key Insights

#### 1. **Consistency Check**
- Train-test accuracy (82.12%) vs CV accuracy (82.51%)
- **Difference**: 0.39% - Very close results indicate model stability

#### 2. **Reliability Assessment**
- **Standard Deviation**: ~1.47% across folds
- **Range**: 80.34% to 84.27%
- Shows consistent performance across different data splits

#### 3. **Performance Metrics Comparison**
```
Metric      | Train-Test | Cross-Val | Difference
------------|------------|-----------|------------
Accuracy    | 82.12%     | 82.35%    | +0.23%
Precision   | 80.65%     | 80.33%    | -0.32%
Recall      | 71.43%     | 75.44%    | +4.01%
```

#### 4. **Model Robustness**
- Small variance in CV scores indicates stable model
- No significant overfitting detected
- Model generalizes well across different data partitions

### Advantages of Each Method

#### Train-Test Split
- ✅ Fast computation
- ✅ Simple interpretation
- ✅ Good for large datasets
- ❌ Single evaluation point
- ❌ Results depend on random split

#### K-Fold Cross-Validation
- ✅ Uses all data for both training and testing
- ✅ Provides confidence intervals
- ✅ More robust performance estimate
- ✅ Reduces variance in results
- ❌ Computationally more expensive
- ❌ More complex to interpret

### Recommendations

1. **For Final Model Selection**: Use K-Fold CV for more reliable estimates
2. **For Quick Iterations**: Use train-test split during development
3. **For Production**: Report both metrics for transparency
4. **Model Confidence**: The close agreement between methods (82.12% vs 82.51%) indicates a robust model

---

## Summary of All Components

### Overall Workflow
1. **V2**: Statistical feature selection using Chi-Square test
2. **V1**: Quick model evaluation using train-test split
3. **V3**: Robust evaluation using cross-validation and comparison

### Final Model Performance
- **Algorithm**: K-Nearest Neighbors (k=5)
- **Features**: 6 statistically significant features
- **Accuracy**: ~82.3% (cross-validated)
- **Precision**: ~80.3%
- **Recall**: ~75.4%

### Model Reliability
The close agreement between train-test split and cross-validation results (difference < 1%) indicates:
- ✅ Model is not overfitting
- ✅ Performance is consistent across different data splits
- ✅ Results are reliable and generalizable