# Evidence set final — revisión de una sola persona

## Qué contiene

Este directorio contiene una versión final operativa construida sin modificar los archivos anteriores. Los claims marcados con `rewrite to:` se sustituyeron conservando su ID; los marcados con `reject` se retiraron y permanecen trazables en `claim_revision_map.csv`.

El evidence set combina dos niveles de revisión:

- `human_reviewed`: evidencia aceptada manualmente con grado 2 o 3.
- `machine_only_unreviewed`: evidencia provisional automática que no apareció en la cola revisada.

Las decisiones humanas `reject` o `uncertain` no forman parte del evidence set. Ninguna fila se etiqueta como adjudicada.

## Archivo que debe usarse como evidence set

Utiliza `evidence_set_final_single_reviewer.csv`. Para evaluación por claims, acompáñalo de `claims_final_single_reviewer.csv` y `evidence_claim_links_final_single_reviewer.csv`.

## Recuentos

- Preguntas: 31
- Claims activos: 152
- Claims reescritos: 13
- Claims retirados: 3
- Evidencias finales: 179
- Revisadas y aceptadas por una persona: 41
- Machine-only no revisadas: 138
- Claims totalmente cubiertos: 150/152
- Casos pendientes de revisión o evidencia adicional: 118

## Resumen por pregunta

| question_id | claims | evidencias | revisadas | machine-only | claims cubiertos | pendientes |
|---|---:|---:|---:|---:|---:|---:|
| Q001 | 1 | 1 | 0 | 1 | 1 | 1 |
| Q002 | 5 | 7 | 4 | 3 | 5 | 0 |
| Q003 | 1 | 1 | 0 | 1 | 1 | 1 |
| Q004 | 1 | 1 | 0 | 1 | 1 | 1 |
| Q005 | 1 | 3 | 1 | 2 | 1 | 0 |
| Q006 | 1 | 2 | 1 | 1 | 1 | 0 |
| Q007 | 2 | 2 | 0 | 2 | 2 | 2 |
| Q008 | 3 | 4 | 2 | 2 | 3 | 1 |
| Q009 | 2 | 1 | 0 | 1 | 2 | 2 |
| Q010 | 5 | 6 | 0 | 6 | 5 | 5 |
| Q011 | 2 | 2 | 0 | 2 | 2 | 2 |
| Q012 | 1 | 3 | 1 | 2 | 1 | 0 |
| Q013 | 11 | 8 | 1 | 7 | 11 | 10 |
| Q014 | 2 | 3 | 2 | 1 | 2 | 1 |
| Q015 | 10 | 20 | 1 | 19 | 10 | 8 |
| Q016 | 6 | 6 | 5 | 1 | 6 | 3 |
| Q017 | 1 | 1 | 0 | 1 | 1 | 1 |
| Q018 | 6 | 6 | 1 | 5 | 6 | 5 |
| Q019 | 5 | 8 | 2 | 6 | 5 | 3 |
| Q020 | 14 | 23 | 6 | 17 | 14 | 11 |
| Q021 | 7 | 5 | 1 | 4 | 7 | 6 |
| Q022 | 2 | 1 | 0 | 1 | 2 | 2 |
| Q023 | 3 | 3 | 2 | 1 | 3 | 1 |
| Q024 | 1 | 1 | 0 | 1 | 1 | 1 |
| Q025 | 8 | 8 | 3 | 5 | 8 | 5 |
| Q026 | 30 | 27 | 0 | 27 | 28 | 30 |
| Q027 | 4 | 1 | 0 | 1 | 4 | 4 |
| Q028 | 11 | 16 | 6 | 10 | 11 | 8 |
| Q029 | 1 | 1 | 0 | 1 | 1 | 1 |
| Q030 | 3 | 6 | 1 | 5 | 3 | 2 |
| Q031 | 2 | 2 | 1 | 1 | 2 | 1 |

## Limitación principal

Este resultado es un corpus de evidencia provisional generado y verificado automáticamente, complementado con revisión de una sola persona y pendiente de adjudicación humana independiente. Las filas `machine_only_unreviewed` no deben presentarse como revisadas manualmente.
