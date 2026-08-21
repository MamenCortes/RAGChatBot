# Validación de los documentos finales

Fecha UTC: 2026-08-14T15:25:43.757Z

Este informe valida el corpus final de un único revisor. No equivale a adjudicación externa.

- PASS: Todos los question_id de claims existen
- PASS: Todos los question_id de evidencias existen
- PASS: Todos los claim_id enlazados están activos
- PASS: Toda evidencia final tiene texto
- PASS: Toda evidencia final tiene grado 2 o 3
- PASS: Toda evidencia final cubre al menos un claim activo
- PASS: No hay evidencias duplicadas por pregunta
- PASS: Toda evidencia tiene al menos un enlace
- PASS: No se incluyeron decisiones humanas reject/uncertain
- PASS: No se incluyeron claims retirados
- PASS: No hay columnas de embeddings o secretos
- PASS: Los claims reescritos tienen texto final
- PASS: No quedan notas de segmentación sin interpretar

## Recuentos de control

- Claims originales: 155
- Claims activos: 152
- Claims retirados: 3
- Claims reescritos: 13
- Evidencias finales revisadas por una persona: 41
- Evidencias finales machine-only no revisadas: 138
- Evidencias provisionales excluidas por revisión humana o falta de claim activo: 45
- Evidencias con fuente no localizada por nombre en docs/: 3
- Claims totalmente cubiertos por cualquier evidencia final: 150/152
- Claims totalmente cubiertos por evidencia revisada por una persona: 34/152

## Limitaciones

- Las evidencias machine_only_unreviewed conservan la anotación automática provisional y requieren revisión humana futura.
- No se realizó adjudicación independiente.
- Una fuente no localizada en docs/ puede seguir siendo un chunk verificable de base de datos; este proceso no volvió a consultar la base de datos.
- Los claims reescritos enlazados solo con evidencia automática deben volver a revisarse contra el nuevo texto.
