import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 2. Datos sintéticos: Modelo de Cobranza (Probabilidad de pago)
np.random.seed(7)
n = 1000
df = pd.DataFrame({
    'dias_atraso': np.random.randint(0, 90, size=n),
    'monto_deuda': np.random.uniform(100, 10000, size=n),
    'pagos_previos': np.random.randint(0, 12, size=n),
    'contactos': np.random.randint(0, 5, size=n),
    'pagara': np.random.binomial(1, 0.7, size=n) # 70% paga
})

X = df.drop('pagara', axis=1)
y = df['pagara']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=7)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Reporte de clasificación - Modelo de Cobranza")
print(classification_report(y_test, y_pred))

# Visualización: Matriz de confusión
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Matriz de confusión - Cobranza")
plt.show()

# Visualización: Importancia de variables
importances = np.abs(model.coef_[0])
plt.figure()
plt.bar(X.columns, importances)
plt.title("Importancia de variables - Cobranza")
plt.ylabel("Importancia relativa (absoluta)")
plt.xticks(rotation=45)
plt.show()