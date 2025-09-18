# 🏦 Modelos Bancarios Avanzados con Optimización Bayesiana

Este repositorio contiene **modelos bancarios optimizados** con técnicas avanzadas de machine learning, incluyendo **optimización bayesiana** con Optuna, análisis por segmentos y aprovechamiento de hardware de alto rendimiento.

## 🚀 Características Principales

- **Optimización Bayesiana** con Optuna (200 trials por modelo)
- **Análisis por Segmentos** de cliente detallado
- **Soporte GPU** para XGBoost (RTX 4070)
- **Paralelización** completa (32 cores)
- **Dataset Sintético** realista de 10,000 registros
- **Comparación Rigurosa** entre modelos optimizados

## 📁 Estructura del Proyecto

### **📊 Notebooks Optimizados:**
- **`notebook_churn_Version3.ipynb`**: Predicción de fuga de clientes con análisis por segmentos
- **`notebook_cobranza_Version2.ipynb`**: Predicción de recuperación de cartera optimizada
- **`notebook_score_riesgo_Version2.ipynb`**: Credit scoring con optimización bayesiana

### **🐍 Scripts Python:**
- **`modelo_fuga_clientes.py`**: Script standalone para churn prediction
- **`modelo_cobranza.py`**: Script standalone para cobranza
- **`modelo_score_riesgo_bancario.py`**: Script standalone para credit scoring

### **📋 Documentación:**
- **`requirements.txt`**: Dependencias optimizadas y compatibles
- **`setup_environment.md`**: Guía de configuración del entorno
- **`test_models.py`**: Script de pruebas automatizadas

## ⚡ Instalación Rápida

### **1. Clonar el repositorio:**
```bash
git clone https://github.com/tu-usuario/modelos-bancarios-ejemplo.git
cd modelos-bancarios-ejemplo
```

### **2. Crear entorno virtual (Recomendado):**
```bash
# Windows
python -m venv venv_bancario
venv_bancario\Scripts\activate

# Linux/Mac
python -m venv venv_bancario
source venv_bancario/bin/activate
```

### **3. Instalar dependencias:**
```bash
pip install -r requirements.txt
```

### **4. Ejecutar notebooks:**
```bash
jupyter notebook
```

## 🎯 Resultados Destacados

### **📊 Rendimiento por Segmento (XGBoost Optimizado):**
- **Retail**: 99.1% accuracy, 96.7% precision
- **Premium**: 99.2% accuracy, 98.9% precision  
- **Pyme**: 99.7% accuracy, 98.6% precision
- **Empresarial**: 99.6% accuracy, 100% precision

### **⚡ Optimizaciones Implementadas:**
- **200 trials** de optimización bayesiana por modelo
- **Validación cruzada** 5-fold para evaluación robusta
- **Paralelización** completa aprovechando 32 cores
- **Soporte GPU** para XGBoost con RTX 4070
- **Dataset ampliado** a 10,000 registros sintéticos

## 🔧 Configuración de Hardware

### **💻 Hardware Optimizado:**
- **CPU**: Intel Core i9 13900HX (32 cores)
- **RAM**: 64GB DDR5 4800MHz
- **GPU**: NVIDIA RTX 4070 Laptop
- **Almacenamiento**: SSD NVMe

### **⚙️ Configuración GPU (Opcional):**
```bash
# Para habilitar GPU en XGBoost, descomenta en requirements.txt:
# xgboost[gpu]==1.7.6
```

## 📈 Uso de Datos

### **🎲 Datos Sintéticos (Incluidos):**
Los notebooks generan automáticamente datasets sintéticos realistas de 10,000 registros con:
- Variables demográficas y financieras
- Patrones de comportamiento bancario
- Distribuciones realistas de churn/cobranza/riesgo

### **📁 Datos Externos (Opcional):**
```python
# Cargar tus propios datos CSV
data = pd.read_csv('tu_archivo.csv')
```

## 🧪 Pruebas Automatizadas

```bash
# Ejecutar pruebas de todos los modelos
python test_models.py
```

## 📚 Recursos y Referencias

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [XGBoost GPU Support](https://xgboost.readthedocs.io/en/latest/gpu/index.html)
- [Scikit-learn Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Pandas Performance Tips](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👨‍💻 Autor

Desarrollado con optimización para hardware de alto rendimiento y técnicas avanzadas de machine learning.

---

**⭐ Si este proyecto te fue útil, ¡dale una estrella al repositorio!**