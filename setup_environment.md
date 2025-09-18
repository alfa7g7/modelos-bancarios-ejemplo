# 🚀 Configuración del Entorno - Modelos Bancarios

## 📋 Requisitos del Sistema
- **Python**: 3.8+ (recomendado 3.10)
- **RAM**: 8GB+ (recomendado 16GB+)
- **GPU**: NVIDIA RTX (opcional, para XGBoost GPU)
- **CUDA**: 11.0+ (solo si usas GPU)

## 🔧 Instalación Paso a Paso

### 1. Crear Entorno Virtual
```bash
# Crear entorno virtual
python -m venv venv_bancario

# Activar entorno (Windows)
venv_bancario\Scripts\activate

# Activar entorno (Linux/Mac)
source venv_bancario/bin/activate
```

### 2. Instalar Dependencias
```bash
# Actualizar pip
python -m pip install --upgrade pip

# Instalar dependencias básicas
pip install -r requirements.txt

# Si tienes GPU NVIDIA, descomenta en requirements.txt:
# xgboost[gpu]==1.7.6
```

### 3. Verificar Instalación
```bash
# Verificar Python
python --version

# Verificar librerías
python -c "import numpy, pandas, sklearn, xgboost, optuna; print('✅ Todas las librerías instaladas correctamente')"

# Verificar GPU (opcional)
nvidia-smi
```

## 🎮 Configuración GPU (Opcional)

### Para XGBoost con GPU:
1. Instalar CUDA Toolkit 11.0+
2. Descomentar `xgboost[gpu]==1.7.6` en requirements.txt
3. Reinstalar: `pip install xgboost[gpu]==1.7.6`

### Verificar GPU:
```python
import xgboost as xgb
print("XGBoost version:", xgb.__version__)
# Debería mostrar soporte GPU si está configurado
```

## 📊 Estructura del Proyecto
```
modelos-bancarios-ejemplo/
├── venv_bancario/          # Entorno virtual
├── requirements.txt        # Dependencias
├── setup_environment.md    # Esta guía
├── notebook_churn_Version3.ipynb
├── notebook_cobranza_Version2.ipynb
├── notebook_score_riesgo_Version2.ipynb
├── modelo_fuga_clientes.py
├── modelo_cobranza.py
├── modelo_score_riesgo_bancario.py
└── README.md
```

## 🚨 Troubleshooting

### Error: "No module named pip"
```bash
python -m ensurepip --upgrade
```

### Error: "CUDA not found"
- Verificar instalación de CUDA
- Usar versión CPU: `pip install xgboost==1.7.6`

### Error: "Permission denied"
- Ejecutar como administrador (Windows)
- Usar `sudo` (Linux/Mac)

## ✅ Verificación Final
```python
# Ejecutar este script para verificar todo
import sys
print(f"Python: {sys.version}")

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import optuna
import psutil

print("✅ Todas las librerías funcionan correctamente")
print(f"CPU Cores: {psutil.cpu_count()}")
print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
```

## 🎯 Próximos Pasos
1. Activar entorno: `venv_bancario\Scripts\activate`
2. Ejecutar notebooks optimizados
3. Probar modelos Python mejorados
