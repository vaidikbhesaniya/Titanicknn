# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from scipy.stats import chi2_contingency

# data=pd.read_csv("Titanic.csv")

# data=data.drop('Name',axis=1)
# data=data.drop('Ticket',axis=1)
# data=data.drop('Fare',axis=1)
# data=data.drop('Cabin',axis=1)

# # data['Family']=data['SibSp']+data['Parch']

# # data=data.drop('SibSp',axis=1)
# # data=data.drop('Parch',axis=1)


# for feature in data.columns:
#     contingency_table = pd.crosstab(data[feature], data["Survived"])
#     chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
#     print(f"\nFeature: {feature}")
#     print("Contingency Table:\n", contingency_table)
#     print(f"Chi2: {chi2:.3f}, P-Value: {p_value:.3f}")
    
#     if p_value < 0.05:
#         print("✅ Significant relationship → Keep this feature")
#     else:
#         print("❌ Not significant → Could drop this feature")


import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.stats import chi2_contingency

# Load dataset
data = pd.read_csv("Titanic.csv")

# Drop unnecessary columns
data = data.drop(['Name', 'Ticket', 'Fare', 'Cabin'], axis=1)


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

# Keep only significant features + target
data = data[significant_features + ["Survived"]]

# Drop rows with missing values
data = data.dropna()

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Split features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


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


# Save Model and Preprocessing Objects

print("\n=== Saving Model and Preprocessing Objects ===")

# Save the trained model
pickle.dump(model, open('./models/titanic_knn_model.pkl', 'wb'))
print("✅ Model saved as 'titanic_knn_model.pkl'")

# Save the scaler for consistent preprocessing
pickle.dump(scaler, open('./models/scaler.pkl', 'wb'))
print("✅ Scaler saved as 'scaler.pkl'")

# Save feature names for future use
feature_names = X.columns.tolist()
pickle.dump(feature_names, open('./models/feature_names.pkl', 'wb'))
print("✅ Feature names saved as 'feature_names.pkl'")

# Save significant features list
pickle.dump(significant_features, open('./models/significant_features.pkl', 'wb'))
print("✅ Significant features saved as 'significant_features.pkl'")

print("\n🎉 All pickle files generated successfully!")


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
