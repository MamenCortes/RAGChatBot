# Evaluación de Contenido Generado por RAG
## Definición de Métricas
### Completeness

Completeness evalúa si la respuesta generada incluye toda la información importante que debería incluir según el contexto recuperado. Una respuesta completa proporciona una cobertura adecuada de los puntos clave y detalles relevantes, mientras que una respuesta incompleta puede omitir información crítica o relevante.

| Valor | Interpretación                                   |
| ----: | ------------------------------------------------ |    
|  1.00 | Contiene todos los puntos necesarios             |
|  0.75 | Contiene la mayoría; faltan detalles secundarios |
|  0.50 | Cubre aproximadamente la mitad                   |
|  0.25 | Solo incluye una parte pequeña                   |
|  0.00 | No proporciona la información solicitada         |
### Faithfulness

Faithfulness evalúa si todo lo que dice la respuesta está respaldado por el contexto recuperado. Una respuesta fiel refleja con precisión la información contenida en los documentos de referencia, mientras que una respuesta infiel puede incluir afirmaciones no respaldadas o contradictorias con la evidencia disponible.

| Valor | Interpretación                                                |
| ----: | ------------------------------------------------------------- |   
|  1.00 | Todas las afirmaciones están respaldadas                      |
|  0.75 | Contenido principalmente respaldado, con alguna adición menor |
|  0.50 | Mezcla similar de afirmaciones respaldadas y no respaldadas   |
|  0.25 | La mayoría no está respaldada o contradice las evidencias     |
|  0.00 | La respuesta es completamente incompatible con las evidencias |

## Número Total de Preguntas Evaluadas: 31
---
## Resumen de Métricas  

| Métrica      | Media  | Mediana | Desviación Estándar | Mínimo | Máximo |
| ------------ | ------ | ------- | ------------------ | ------ | ------ |
| Completeness | 0.61 | 0.75 | 0.36 | 0.00 | 1.00 |
| Faithfulness | 0.85 | 1.00 | 0.21 | 0.25 | 1.00 |    

## Observaciones de la Evaluación de Contenido
Durante la evaluación se observó que había varios casos en los que la respuesta generada decía que no existía contexto que respondiera a dicha pregunta (Cuando la evidencia si responde a dicha pregunta). En estos casos la completitud se definió como 0 y la fiabilidad como 1 ya que el sistema reconoció de manera adecuada que no sabía responder a la preguntas.

También hay varias ocasiones en las que se observó que el ground_truth no recogía toda la información relevante que sí estaba presente en los contextos recuperados. En estos casos, se tuvo en cuenta también la información recuperada por el sistema de retrieval para la evaluación. 
Se reconoce que esto es una limitación de la selección manual de las respuestas más que una limitación del sistema de RAG. 

También se pudo observar que aquellas respuestas donde la completitud era baja, se debía en muchos casos a que el sistema de retrieval no había recuperado toda la información relevante para responder a la pregunta. Esto indica que el sistema de retrieval es un componente crítico para la generación de respuestas completas y precisas.
Aún así, en la mayoría de casos, la respuesta se mantiene fiel a la evidencia recuperada, lo que indica que el sistema de generación de respuestas es capaz de utilizar la información disponible de manera efectiva.


