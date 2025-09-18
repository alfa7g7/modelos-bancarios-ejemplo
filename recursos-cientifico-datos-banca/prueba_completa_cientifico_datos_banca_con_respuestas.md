# Prueba completa para selección de Científico de Datos en Banca

## Parte 1: Preguntas técnicas
(Ver archivo preguntas_respuestas_banca.md)

---

## Parte 2: Ejercicio práctico

**Caso:**  
Supongamos que una entidad bancaria quiere predecir la probabilidad de fuga de clientes ("churn") usando datos de comportamiento financiero, demográficos y de interacción con el banco.

**Respuesta modelo:**  
"Desarrollaría el modelo siguiendo estos pasos:

1. **Análisis exploratorio:** Exploro la distribución de variables, identifico correlaciones y outliers. Visualizo la proporción de clientes que han abandonado el banco y reviso las variables con mayor impacto potencial.
2. **Feature engineering:** Transformo variables numéricas sesgadas usando logaritmos, creo agregados como promedio de saldo y frecuencia de interacción, y codifico variables categóricas relevantes.
3. **Manejo de desbalance:** Aplico SMOTE para oversampling de la clase minoritaria y ajusto el parámetro `scale_pos_weight` en XGBoost.
4. **Entrenamiento:** Entreno modelos de regresión logística y XGBoost, comparando desempeño con AUC-ROC, precisión, recall y F1-score.
5. **Evaluación:** Presento matriz de confusión, curva ROC y gain/lift para negocio. Interpreto la importancia de variables con coeficientes y SHAP values.
6. **Recomendación:** Entrego un reporte con los pasos, resultados y recomendaciones para el banco, incluyendo sugerencias para monitoreo del modelo en producción y próximos pasos de mejora."

---

## Parte 3: Preguntas sobre ética y regulación

1. **¿Qué variables pueden estar prohibidas por regulación en modelos bancarios?**  
   "Variables como género, raza, religión, orientación sexual, y en muchos casos edad, pueden estar prohibidas por regulación como la Ley de Igualdad de Trato o normativas locales de protección de datos. Además, cualquier variable que permita perfilamiento discriminatorio debe ser excluida, siguiendo las recomendaciones de compliance y auditoría."

2. **¿Cómo garantizar que tu modelo no discrimine por factores como género, raza o edad?**  
   "Excluyo explícitamente variables sensibles y realizo análisis de equidad, por ejemplo, revisando métricas de desempeño por subgrupo (ejemplo: tasa de rechazo por género). Si detecto sesgo, ajusto el modelo y reporto los hallazgos. También valido regularmente el comportamiento del modelo en producción para prevenir discriminación inadvertida."

3. **¿Qué harías si tu modelo tiene un desempeño desigual entre diferentes segmentos de clientes?**  
   "Analizo las causas del sesgo, ajusto el preprocesamiento, eligiendo técnicas de balanceo específicas o aplicando fairness constraints. Si es necesario, rediseño el modelo para mejorar la equidad y comunico de forma transparente los resultados, involucrando al área de compliance."

---

## Parte 4: Preguntas sobre comunicación y negocio

1. **Explica en términos sencillos cómo funciona tu modelo a un gerente de negocio que no sabe de ciencia de datos.**  
   "Mi modelo analiza variables como el uso de productos, transacciones y comportamiento del cliente para estimar la probabilidad de que abandone el banco. Utiliza patrones aprendidos de clientes anteriores para anticipar riesgos y ayudar al banco a tomar acciones preventivas."

2. **¿Cómo interpretarías y comunicarías el riesgo de un cliente según el score generado?**  
   "El score indica la probabilidad de que un cliente abandone el banco: un valor alto significa mayor riesgo, y bajo menor riesgo. Recomiendo segmentar clientes según el score para priorizar campañas de retención, enfocando recursos en quienes más lo necesitan."

3. **¿Cómo justificarías ante un regulador por qué tu modelo es confiable y ético?**  
   "Presento evidencia de exclusión de variables sensibles, documentación del proceso de selección de variables y validación de equidad. Explico las métricas usadas, la metodología de testeo y muestro reportes de fairness. Además, garantizo la trazabilidad del modelo y su monitoreo continuo."

---

## Parte 5: Manejo de datos reales y validación

1. **¿Cómo tratarías los datos faltantes, valores atípicos y errores en los datos?**  
   "Para datos faltantes, utilizo imputación basada en la media, mediana o modelos específicos según el patrón de ausencia. Los valores atípicos los identifico y decido si eliminarlos o transformarlos según su impacto en el modelo. Para errores, implemento validaciones automáticas y reglas de negocio en la limpieza de datos."

2. **¿Qué harías si tienes que integrar información de distintas fuentes (por ejemplo, bases internas y datos externos)?**  
   "Realizo mapeo de variables, estandarizo formatos y valido la calidad de cada fuente. Utilizo claves únicas para la integración y realizo pruebas de consistencia para asegurar que no haya duplicados ni inconsistencias. Documento el proceso y hago pruebas piloto antes de usar los datos en producción."

3. **¿Cómo monitorearías el desempeño del modelo una vez implementado? ¿Qué indicadores revisarías y cómo actuarías si detectas una degradación?**  
   "Configuro dashboards de monitoreo con métricas como precisión, AUC-ROC y distribución de scores. Analizo el drift de variables y resultados por segmento. Si detecto degradación, reviso el origen (cambio en datos, comportamiento de clientes) y decido si reentrenar el modelo, ajustar parámetros o rediseñar variables. Reporto hallazgos al equipo y negocio."

---

## Parte 6: Bonus (opcional)

**Elige uno:**

- **Propón una mejora al proceso de onboarding de modelos en la banca (desde prototipo hasta producción).**  
  "Implementaría una pipeline automatizada con validaciones de datos, pruebas de fairness y documentación obligatoria antes de pasar a producción. Añadiría un proceso de aprobación por parte de compliance y auditoría. También establecería métricas de monitoreo en tiempo real y alertas automáticas para detectar drift o anomalías."

- **Explica cómo usarías explainable AI (XAI) en modelos bancarios y qué beneficios aporta.**  
  "Utilizaría XAI, por ejemplo con SHAP values, para explicar por qué el modelo toma cada decisión, tanto a usuarios internos como ante reguladores. Esto aumenta la confianza y facilita la interpretación, permitiendo detectar sesgos y errores antes de afectar a los clientes."

- **Diseña una estrategia para enriquecer la información de clientes usando fuentes externas, garantizando privacidad y cumplimiento regulatorio.**  
  "Integro fuentes externas como burós de crédito o redes sociales solo con consentimiento explícito del cliente, cumpliendo GDPR y leyes locales. Encripto los datos sensibles y aplico anonimización cuando corresponda. Realizo auditorías regulares y mantengo la trazabilidad del uso de la información."

---

**Recomendaciones para el evaluador:**  
- Evalúa tanto el conocimiento técnico como la capacidad de comunicación y ética profesional.
- Da preferencia a candidatos que demuestran pensamiento crítico, rigor analítico y comprensión del negocio bancario.

---