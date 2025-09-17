import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 3. Datos sintéticos: Score de Riesgo Bancario (Credit Scoring)
np.random.seed(17)
n = 1000
df = pd.DataFrame({
    'historial_credito': np.random.randint(0, 10, size=n),
    'ingresos': np.random.uniform(500, 10000, size=n),
    'edad': np.random.randint(18, 70, size=n),
    'relacion_banco': np.random.randint(1, 10, size=n),
    'riesgo': np.random.binomial(1, 0.2, size=n) # 20% alto riesgo
})

X = df.drop('riesgo', axis=1)
y = df['riesgo']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=17)
model = GradientBoostingClassifier(n_estimators=100, random_state=17)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Reporte de clasificación - Score de Riesgo Bancario")
print(classification_report(y_test, y_pred))

# Visualización: Matriz de confusión
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Matriz de confusión - Score de Riesgo Bancario")
plt.show()

# Visualización: Importancia de variables
importances = model.feature_importances_
plt.figure()
plt.bar(X.columns, importances)
plt.title("Importancia de variables - Score de Riesgo Bancario")
plt.ylabel("Importancia")
plt.xticks(rotation=45)
plt.show()