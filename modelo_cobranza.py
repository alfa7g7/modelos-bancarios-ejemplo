import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import psutil

# Verificar hardware disponible
print(f"🖥️ Hardware disponible:")
print(f"CPU Cores: {psutil.cpu_count(logical=False)} físicos, {psutil.cpu_count(logical=True)} lógicos")
print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

# 2. Datos sintéticos: Modelo de Cobranza (Probabilidad de pago) - Optimizado
np.random.seed(7)
n = 10000  # 10x más datos
print(f"📊 Generando dataset de cobranza con {n:,} registros...")

df = pd.DataFrame({
    'dias_atraso': np.random.randint(0, 180, size=n),  # Más rango
    'monto_deuda': np.random.lognormal(6, 1, size=n),  # Distribución más realista
    'pagos_previos': np.random.poisson(2, size=n),     # Distribución de Poisson
    'contactos': np.random.poisson(3, size=n),         # Más contactos
    'edad_cliente': np.random.randint(18, 80, size=n),
    'ingresos_mensuales': np.random.lognormal(8, 0.5, size=n),
    'score_credito': np.random.normal(650, 100, size=n),
    'productos_activos': np.random.randint(1, 6, size=n),
})

# Generar pago más realista
pago_prob = (
    0.3 * (df['dias_atraso'] < 30) +                    # Pago temprano
    0.2 * (df['pagos_previos'] > 2) +                   # Historial de pagos
    0.15 * (df['contactos'] > 3) +                      # Muchos contactos
    0.1 * (df['monto_deuda'] < df['ingresos_mensuales'] * 0.3) +  # Deuda baja vs ingresos
    0.1 * (df['score_credito'] > 700) +                 # Buen score crediticio
    0.1 * (df['productos_activos'] > 2) +               # Múltiples productos
    np.random.rand(n) * 0.05                            # Ruido aleatorio
)

df['pagara'] = (pago_prob > 0.4).astype(int)
print(f"✅ Dataset generado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f"📈 Tasa de pago: {df['pagara'].mean():.2%}")

X = df.drop('pagara', axis=1)
y = df['pagara']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=7)

# Función objetivo para optimización bayesiana
def objective_lr(trial):
    C = trial.suggest_float('C', 0.01, 100.0, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    max_iter = trial.suggest_int('max_iter', 100, 1000)
    
    # Seleccionar solver compatible
    if penalty == 'elasticnet':
        solver = 'saga'
    elif penalty == 'l1':
        solver = trial.suggest_categorical('solver_l1', ['liblinear', 'saga'])
    else:
        solver = trial.suggest_categorical('solver_l2', ['liblinear', 'lbfgs', 'saga'])
    
    model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter, random_state=7)
    
    try:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    except:
        return 0.0

# Optimización bayesiana
print("🔍 Optimizando Regresión Logística...")
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=7))
study.optimize(objective_lr, n_trials=100, n_jobs=-1)

print(f"Mejor score: {study.best_value:.4f}")
print(f"Mejores parámetros: {study.best_params}")

# Crear modelo optimizado
best_params = study.best_params.copy()
if best_params['penalty'] == 'elasticnet':
    best_params['solver'] = 'saga'
elif best_params['penalty'] == 'l1':
    best_params['solver'] = best_params.get('solver_l1', 'liblinear')
else:
    best_params['solver'] = best_params.get('solver_l2', 'liblinear')

# Remover parámetros auxiliares
best_params = {k: v for k, v in best_params.items() if not k.startswith('solver_')}
best_params['random_state'] = 7

model = LogisticRegression(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n📊 Reporte de clasificación - Modelo de Cobranza Optimizado")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

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