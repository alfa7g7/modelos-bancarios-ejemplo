import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import psutil

# Verificar hardware disponible
print(f"🖥️ Hardware disponible:")
print(f"CPU Cores: {psutil.cpu_count(logical=False)} físicos, {psutil.cpu_count(logical=True)} lógicos")
print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

# 1. Datos sintéticos: Modelo de Fuga (Churn) - Optimizado para hardware potente
np.random.seed(42)
n = 10000  # 10x más datos
print(f"📊 Generando dataset de fuga con {n:,} registros...")

df = pd.DataFrame({
    'antiguedad': np.random.randint(1, 20, size=n),
    'productos': np.random.randint(1, 8, size=n),  # Más productos
    'transacciones_mes': np.random.poisson(25, size=n),  # Distribución más realista
    'satisfaccion': np.random.uniform(1, 5, size=n),
    'edad': np.random.randint(18, 80, size=n),
    'ingresos_mensuales': np.random.lognormal(8, 0.5, size=n),
    'deuda_total': np.random.lognormal(6, 1, size=n),
    'score_credito': np.random.normal(650, 100, size=n),
})

# Generar churn más realista
churn_prob = (
    0.2 * (df['antiguedad'] < 2) +           # Clientes nuevos
    0.3 * (df['satisfaccion'] < 2.5) +       # Baja satisfacción
    0.2 * (df['productos'] == 1) +           # Un solo producto
    0.1 * (df['deuda_total'] > df['ingresos_mensuales'] * 0.5) +  # Alto endeudamiento
    0.1 * (df['score_credito'] < 600) +      # Mal score crediticio
    np.random.rand(n) * 0.1                  # Ruido aleatorio
)

df['churn'] = (churn_prob > 0.4).astype(int)
print(f"✅ Dataset generado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f"📈 Tasa de churn: {df['churn'].mean():.2%}")

X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Función objetivo para optimización bayesiana
def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42,
        'n_jobs': -1  # Usar todos los cores
    }
    
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# Optimización bayesiana
print("🔍 Optimizando Random Forest...")
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective_rf, n_trials=100, n_jobs=-1)

print(f"Mejor score: {study.best_value:.4f}")
print(f"Mejores parámetros: {study.best_params}")

# Entrenar modelo optimizado
model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n📊 Reporte de clasificación - Modelo de Fuga (Churn) Optimizado")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

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