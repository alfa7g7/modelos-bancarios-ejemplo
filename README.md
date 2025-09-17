# Modelos Bancarios de Ejemplo

Este repositorio contiene **tres ejemplos prácticos** de modelos bancarios implementados en Python utilizando datos sintéticos:

- Modelo de **fuga de clientes (churn)**
- Modelo de **cobranza (probabilidad de pago)**
- **Score de riesgo bancario (credit scoring)**

Cada ejemplo incluye generación de datos, entrenamiento, evaluación, visualización y explicación.

---

## 1. Modelo de Fuga de Clientes (`modelo_fuga_clientes.py`)

Predice si un cliente está en riesgo de abandonar el banco.  
Variables consideradas: antigüedad, cantidad de productos, transacciones mensuales y satisfacción.

- Algoritmo: `RandomForestClassifier`
- Métricas: Reporte de clasificación y matriz de confusión
- Visualización: Importancia de variables

---

## 2. Modelo de Cobranza (`modelo_cobranza.py`)

Estima la probabilidad de que una deuda sea pagada.  
Variables consideradas: días de atraso, monto de deuda, pagos previos y contactos realizados con el cliente.

- Algoritmo: `LogisticRegression`
- Métricas: Reporte de clasificación y matriz de confusión
- Visualización: Importancia de variables

---

## 3. Score de Riesgo Bancario (`score_riesgo_bancario.py`)

Calcula el riesgo crediticio de un cliente.  
Variables consideradas: historial de crédito, ingresos, edad y relación con el banco.

- Algoritmo: `GradientBoostingClassifier`
- Métricas: Reporte de clasificación y matriz de confusión
- Visualización: Importancia de variables

---

## Requisitos

- Python >= 3.7
- pandas
- numpy
- scikit-learn
- matplotlib

Instala dependencias con:
```bash
pip install -r requirements.txt
```

## Ejecución

Cada script se ejecuta por separado y genera un reporte de clasificación y visualizaciones:

```bash
python modelo_fuga_clientes.py
python modelo_cobranza.py
python score_riesgo_bancario.py
```

---

## Visualizaciones

Todos los modelos generan una **matriz de confusión** y un gráfico de **importancia de variables** para interpretar el desempeño y relevancia de los atributos.

---

## Uso y adaptación

Estos ejemplos pueden servir como base para:
- Prototipos de modelos bancarios
- Pruebas de concepto de analítica financiera
- Enseñanza de ciencia de datos aplicada a banca

Puedes modificar los scripts para ajustarlos a tus propios datos o necesidades.

---

¿Dudas o sugerencias? Abre un issue o contacta al autor.
