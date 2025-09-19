import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.stats import chi2_contingency

data=pd.read_csv("Titanic.csv")

data=data.drop('Name',axis=1)
data=data.drop('Ticket',axis=1)
data=data.drop('Fare',axis=1)
data=data.drop('Cabin',axis=1)

# data['Family']=data['SibSp']+data['Parch']

# data=data.drop('SibSp',axis=1)
# data=data.drop('Parch',axis=1)


for feature in data.columns:
    contingency_table = pd.crosstab(data[feature], data["Survived"])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\nFeature: {feature}")
    print("Contingency Table:\n", contingency_table)
    print(f"Chi2: {chi2:.3f}, P-Value: {p_value:.3f}")
    
    if p_value < 0.05:
        print("✅ Significant relationship → Keep this feature")
    else:
        print("❌ Not significant → Could drop this feature")


