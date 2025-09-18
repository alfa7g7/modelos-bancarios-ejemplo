# Preguntas técnicas sugeridas y respuestas modelo para evaluación en banca

---

**1. ¿Cómo elegir las variables más relevantes en un modelo bancario?**

La selección de variables relevantes implica análisis estadístico, conocimiento del negocio y técnicas automáticas como feature importance, pruebas de correlación, selección recursiva (RFE), LASSO, o métodos de árboles de decisión. Es importante evitar variables redundantes, irrelevantes o que puedan introducir sesgos, y considerar el impacto regulatorio (por ejemplo, evitar variables prohibidas). Se debe usar tanto EDA como algoritmos para justificar la selección y validarla con métricas de desempeño.

_Respuesta del candidato:_  
"En mi proceso, comienzo con un análisis exploratorio de datos para identificar correlaciones y patrones. Utilizo métodos como la importancia de variables en modelos de árbol (por ejemplo, XGBoost), selección recursiva de características y regularización LASSO para automatizar la selección. También valido la relevancia de cada variable con expertos del negocio y verifico que ninguna variable esté prohibida por regulación, asegurando que solo queden las más predictivas y explicables en el modelo."

---

**2. ¿Qué métricas usar para modelos de churn/cobranza/score?**

No basta con la precisión: en banca es fundamental usar métricas como AUC-ROC, precisión, recall, F1-score, matriz de confusión, Gini, KS statistic y lift, según el objetivo del modelo. Se debe distinguir entre métricas para clasificación binaria (churn/cobranza) y para score de riesgo (probabilidad de incumplimiento), y argumentar la elección de la métrica según el contexto y el costo de errores.

_Respuesta del candidato:_  
"Para la predicción de churn y cobranza, prefiero el uso de AUC-ROC para comparar modelos y F1-score/recall para medir el desempeño en clases minoritarias. En scoring de riesgo, utilizo el estadístico KS y el coeficiente Gini para evaluar la capacidad de discriminación. Siempre elijo la métrica que mejor se alinea con el objetivo de negocio y el impacto económico de los errores, por ejemplo, minimizando falsos negativos en cobranza."

---

**3. ¿Cómo manejar desbalance de clases?**

Se deben aplicar técnicas como oversampling (SMOTE), undersampling, generación de datos sintéticos, ajuste de pesos en la función de pérdida, y uso de algoritmos robustos al desbalance (como XGBoost con parámetro scale_pos_weight). Es importante evaluar con métricas sensibles al desbalance y considerar el impacto en la interpretación del modelo.

_Respuesta del candidato:_  
"Cuando enfrento desbalance de clases, suelo emplear SMOTE para aumentar la cantidad de casos minoritarios y undersampling para equilibrar las clases. Ajusto los pesos en la función de pérdida si el modelo lo permite (por ejemplo, scale_pos_weight en XGBoost). Complemento esto con el uso de métricas como F1-score y AUC-ROC, que reflejan mejor el desempeño bajo desbalance. Además, reviso que el proceso no afecte la interpretabilidad ni el valor de negocio."

---

**4. ¿Qué ventajas tiene XGBoost frente a regresión logística para estos casos?**

XGBoost es más potente para capturar relaciones no lineales y manejar interacciones de variables automáticamente, suele tener mejor desempeño en problemas complejos y desbalanceados, y ofrece interpretabilidad vía importancia de variables. Sin embargo, regresión logística es más simple, rápida, interpretada fácilmente por negocio y reguladores, y útil como benchmark. Se debe saber cuándo preferir uno sobre otro según el caso bancario.

_Respuesta del candidato:_  
"XGBoost destaca por su capacidad para detectar relaciones no lineales y gestionar variables de forma automática, además de ser muy eficiente en conjuntos de datos desbalanceados. Lo uso cuando el objetivo es maximizar la precisión y la robustez del modelo. En cambio, la regresión logística la prefiero cuando la interpretabilidad y la trazabilidad son más relevantes, como en modelos que deben ser explicados a reguladores o para benchmarking inicial."

---

**5. ¿Cómo interpretar la importancia de variables?**

Se deben utilizar métodos como gain, split count, SHAP values para modelos de árbol (XGBoost), y coeficientes para regresión logística. La importancia ayuda a entender el modelo y justificar decisiones ante reguladores, pero no siempre implica causalidad. Es clave comunicar estos resultados al negocio.

_Respuesta del candidato:_  
"En XGBoost, analizo la importancia de variables con métricas como gain y frecuencia de splits, y complemento con SHAP values para interpretabilidad local y global. En regresión logística interpreto los coeficientes, su magnitud y significancia estadística. Comunico estos hallazgos al negocio destacando que la importancia no implica causalidad, y que se deben monitorear las variables más influyentes en el tiempo."

---

**6. ¿Qué técnicas de feature engineering son útiles en banca?**

Es recomendable aplicar transformación de variables (log, box-cox), creación de variables agregadas (saldos promedio, ratios), codificación para variables categóricas, binning de score, tratamiento de valores nulos, detección y manejo de outliers, y generación de variables temporales (histórico de pagos, antigüedad). El conocimiento del negocio es esencial para crear variables relevantes.

_Respuesta del candidato:_  
"Desarrollo variables agregadas como el promedio de saldo, la antigüedad de la cuenta y ratios de pago, empleo transformaciones logarítmicas para variables sesgadas y realizo binning en scores para mejorar la segmentación. Codifico variables categóricas con one-hot o target encoding y trato los nulos según su contexto. Además, creo variables temporales que reflejan el comportamiento histórico del cliente, siempre en colaboración con expertos del negocio para maximizar el valor predictivo."

---