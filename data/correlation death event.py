from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# fetch dataset
heart_failure_clinical_records = fetch_ucirepo(id=519)

# features et target
X = heart_failure_clinical_records.data.features
y = heart_failure_clinical_records.data.targets

# combiner
data = pd.concat([X, y], axis=1)

# matrice de corrélation
corr_matrix = data.corr()

# corrélation avec la variable cible
corr_target = corr_matrix["death_event"].sort_values(ascending=False)

print("Correlation with death_event:")
print(corr_target)

# graphique
corr_target.drop("death_event").plot(kind="bar", figsize=(10,5))

plt.title("Correlation with Death Event")
plt.ylabel("Correlation coefficient")
plt.show()