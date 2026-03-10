from ucimlrepo import fetch_ucirepo 
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# fetch dataset 
heart_failure_clinical_records = fetch_ucirepo(id=519) 
  
# data (as pandas dataframes) 
a = heart_failure_clinical_records.data.features 
b = heart_failure_clinical_records.data.targets 
  
# metadata 
print(heart_failure_clinical_records.metadata) 
  
# variable information 
print(heart_failure_clinical_records.variables) 


# combiner X et y
data = pd.concat([a, b], axis=1)
data.columns = [col.lower() for col in data.columns]
X = data.drop("death_event", axis=1)
y = data["death_event"]
smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)


sns.countplot(x=y_resampled)

plt.title("Balanced Class Distribution after SMOTE")
plt.xlabel("DEATH_EVENT")
plt.ylabel("Number of Patients")

plt.show()