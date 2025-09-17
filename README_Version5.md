# Modelos Bancarios Ejemplo

Este portafolio incluye ejemplos prácticos de modelos bancarios para pruebas técnicas y aprendizaje. Cada caso está documentado en su propio notebook y puede ejecutarse usando datos sintéticos o archivos CSV externos propios.

## Estructura

- **notebook_churn.ipynb:** Predicción de fuga de clientes (churn prediction).
- **notebook_cobranza.ipynb:** Predicción de recuperación de cartera (cobranza).
- **notebook_score_riesgo.ipynb:** Credit scoring y score de riesgo bancario.

Cada notebook incluye:
- Opción para generar datos sintéticos y cargar CSV externo.
- Análisis exploratorio de datos (EDA) y visualizaciones gráficas.
- Preprocesamiento, encoding y escalado.
- Modelos con regresión logística y XGBoost.
- Evaluación con métricas, matriz de confusión, curva ROC e importancia de variables.
- Explicaciones en markdown para entender cada paso.

## Instalación y uso

1. Clona el repositorio.
2. Instala dependencias:

```bash
pip install -r requirements.txt
```

3. Abre y ejecuta los notebooks con Jupyter Notebook, JupyterLab o VSCode.

## Ejemplo de carga de datos

Puedes usar los datos sintéticos incluidos o cargar tus propios archivos CSV, por ejemplo:

```python
data = pd.read_csv('clientes_churn.csv')
```

## Recursos recomendados

- [zygmuntz/awesome-credit-scoring](https://github.com/zygmuntz/awesome-credit-scoring)
- [mikekang/credit-scoring](https://github.com/mikekang/credit-scoring)
- [IBM churn-prediction](https://github.com/IBM/customer-churn-prediction)
- [Credit Collections ML](https://github.com/llSourcell/Credit_Collection_ML)
- [XGBoost Official Examples](https://github.com/dmlc/xgboost/tree/master/demo)
- [Kaggle churn modeling notebook](https://www.kaggle.com/code/shubhendra7/customer-churn-modelling)

## Preguntas técnicas sugeridas

- ¿Cómo elegir las variables más relevantes en un modelo bancario?
- ¿Qué métricas usar para modelos de churn/cobranza/score?
- ¿Cómo manejar desbalance de clases?
- ¿Qué ventajas tiene XGBoost frente a regresión logística para estos casos?
- ¿Cómo interpretar la importancia de variables?
- ¿Qué técnicas de feature engineering son útiles en banca?

## Contribuciones

¡Ideas, mejoras y ejemplos extra son bienvenidos!