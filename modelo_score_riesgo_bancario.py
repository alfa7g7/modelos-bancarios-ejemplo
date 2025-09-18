import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import psutil

# Verificar hardware disponible
print(f"🖥️ Hardware disponible:")
print(f"CPU Cores: {psutil.cpu_count(logical=False)} físicos, {psutil.cpu_count(logical=True)} lógicos")
print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

# 3. Datos sintéticos: Score de Riesgo Bancario (Credit Scoring) - Optimizado
np.random.seed(17)
n = 10000  # 10x más datos
print(f"📊 Generando dataset de score de riesgo con {n:,} registros...")

df = pd.DataFrame({
    'historial_credito': np.random.randint(0, 10, size=n),  # Más rango
    'ingresos': np.random.lognormal(8, 0.5, size=n),       # Distribución log-normal
    'edad': np.random.randint(18, 80, size=n),             # Más rango de edad
    'relacion_banco': np.random.randint(1, 15, size=n),    # Más años de relación
    'deuda_total': np.random.lognormal(6, 1, size=n),      # Deuda total
    'productos_activos': np.random.randint(1, 8, size=n),  # Productos bancarios
    'empleo_estable': np.random.binomial(1, 0.7, size=n),  # Empleo estable
    'educacion': np.random.choice([1, 2, 3, 4], size=n, p=[0.2, 0.4, 0.3, 0.1]),  # Nivel educativo
    'vivienda': np.random.choice([1, 2, 3], size=n, p=[0.5, 0.3, 0.2]),  # Tipo de vivienda
})

# Generar riesgo más realista
riesgo_prob = (
    0.25 * (df['historial_credito'] < 3) +              # Mal historial
    0.2 * (df['ingresos'] < 2000) +                     # Bajos ingresos
    0.15 * (df['edad'] < 25) +                          # Jóvenes
    0.1 * (df['deuda_total'] > df['ingresos'] * 0.5) + # Alto endeudamiento
    0.1 * (df['empleo_estable'] == 0) +                 # Empleo inestable
    0.1 * (df['productos_activos'] == 1) +              # Pocos productos
    0.05 * (df['relacion_banco'] < 2) +                 # Poca relación bancaria
    np.random.rand(n) * 0.05                            # Ruido aleatorio
)

df['riesgo'] = (riesgo_prob > 0.3).astype(int)
print(f"✅ Dataset generado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f"📈 Tasa de riesgo: {df['riesgo'].mean():.2%}")

X = df.drop('riesgo', axis=1)
y = df['riesgo']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=17)

# Función objetivo para optimización bayesiana
def objective_gb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 17
    }
    
    model = GradientBoostingClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# Optimización bayesiana
print("🔍 Optimizando Gradient Boosting...")
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=17))
study.optimize(objective_gb, n_trials=100, n_jobs=-1)

print(f"Mejor score: {study.best_value:.4f}")
print(f"Mejores parámetros: {study.best_params}")

# Entrenar modelo optimizado
model = GradientBoostingClassifier(**study.best_params, random_state=17)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n📊 Reporte de clasificación - Score de Riesgo Bancario Optimizado")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

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