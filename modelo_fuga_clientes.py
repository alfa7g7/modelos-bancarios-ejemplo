import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Datos sintéticos: Modelo de Fuga (Churn)
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'antiguedad': np.random.randint(1, 20, size=n),
    'productos': np.random.randint(1, 5, size=n),
    'transacciones_mes': np.random.randint(0, 30, size=n),
    'satisfaccion': np.random.uniform(1, 5, size=n),
    'churn': np.random.binomial(1, 0.1, size=n) # 10% fuga
})

X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Reporte de clasificación - Modelo de Fuga (Churn)")
print(classification_report(y_test, y_pred))

# Visualización: Matriz de confusión
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Matriz de confusión - Fuga de clientes")
plt.show()

# Visualización: Importancia de variables
importances = model.feature_importances_
plt.figure()
plt.bar(X.columns, importances)
plt.title("Importancia de variables - Fuga de clientes")
plt.ylabel("Importancia")
plt.xticks(rotation=45)
plt.show()