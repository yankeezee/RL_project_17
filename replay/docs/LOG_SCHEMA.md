# Replay diagnostics schema v1

Код и машинные поля на английском. `synthetic` обязателен при создании logger/export:
значение true сопровождает все подготовленные демонстрационные данные.

## Файлы на run

- `<run_id>.metadata.json`: seed, regime, synthetic, произвольные реальные сведения
  о config hash, git commit, learner checkpoint, replay/environment RNG seeds.
- `<run_id>.batches.jsonl`: одна строка на выполненное replay/optimizer обновление.
- `<run_id>.samples.jsonl`: опционально одна строка на каждый draw, включая дубликаты.

Время sampling измеряется внутри buffer.sample; excludes logging, learner, priority update.
Priority-update и JSONL overhead измеряются отдельно в benchmark.
Память `buffer.metadata_bytes` точно суммирует nbytes ndarray, включая padding деревьев;
не включает payloads, список ссылок, Python objects, временные batch arrays и process RSS.

## Batch JSONL

| Поле | Определение |
|---|---|
| `run_id, seed, regime, synthetic` | Контекст, не смешивать разные runs без явной агрегации |
| `method, decisions, optimizer_step, beta` | Метод и две разные временные шкалы |
| `buffer_size, group_sizes` | N и [N0,N1] в момент sampling, не capacity |
| `batch_size` | B, по протоколу 64 |
| `buffer_outage_fraction` | f=N1/N |
| `sample_outage_fraction` | m1/B |
| `allocation_difference` | m1/B-f |
| `allocation_ratio` | (m1/B)/f; JSON null при f=0 |
| `minimum_probability` | min по всему текущему буферу для выбранного sampler |
| `sampler_seconds` | Только время выборки, probabilities, weights, сборки batch |
| `weighted_huber_numerator` | sum по B: w_i Huber(delta_i) |
| `loss_denominator` | B; не sum(w) |
| `weighted_huber_batch_mean` | numerator/B, loss для backprop |

`groups["0"]` и `groups["1"]` содержат:

| Поле | Определение |
|---|---|
| `sample_count` | m_c, число draws группы с учётом повторов |
| `expected_count_from_buffer_share` | B*Nc/N, baseline share-preserving allocation |
| `td_abs_sum, td_abs_sq_sum, td_abs_mean` | Moments фактических sampled TD residuals |
| `replay_count_sum` | Сумма lifetime counts после batch по draws, повторы не удаляются |
| `weighted_huber_numerator` | sum_{draws in group} w_i Huber(delta_i) |
| `within_group_denominator` | m_c |
| `within_group_weighted_huber_mean` | group numerator/m_c; null при m_c=0 |
| `batch_denominator` | B |
| `contribution_to_batch_mean` | group numerator/B; 0 при m_c=0 |

Сумма двух `contribution_to_batch_mean` равна общей weighted loss.
Within-group mean отвечает на другой вопрос и не складывается в общую loss.
Дубликат участвует в числителе и знаменателе каждый раз, как в реальном loss.
TD error берётся от того же forward pass, по которому делался optimizer step.
Эти величины описывают loss, но не непосредственно норму или направление градиента.

## Draw JSONL

`transition_id` уникален за время жизни экземпляра buffer, `slot` переиспользуется FIFO.
`draw` различает дубликаты внутри optimizer_step. Поля: tag, actual probability,
globally normalized weight, td_abs, huber, weighted_huber и
`replay_count_after_batch`. Последний включает **все** draws этого batch:
два появления одного transition_id имеют одинаковый итоговый count, не последовательные counts.
Перезаписанный слот начинает count с нуля и новым ID.

График TD vs count использует deterministic reservoir, seed=7919, максимум 4000
draw rows. Это draw-weighted scatter: часто сэмплируемые переходы встречаются чаще.
Для переходов, находящихся в frozen snapshot, есть отдельный replay_count/td_abs
в CSV: одна строка на переход. Его TD относится к последнему refresh, не обязательно
к последнему состоянию сети. Для новых priorities td_abs пустой, а не вымышленный.

## Динамический буфер и агрегация

На окне обновлений T сравнивайте:

```
observed = sum_t m1_t
expected_from_occupancy = sum_t B_t * N1_t/N_t
excess_fraction = (observed - expected_from_occupancy) / sum_t B_t
allocation_ratio = observed / expected_from_occupancy  # null если expected=0
```

Это baseline по доле буфера, а не expected PER mass.
Последняя доля буфера не заменяет эту сумму. Реализованные plots и
`*.allocation.json` используют именно такое сопоставление.
Для loss агрегируйте числители и знаменатели; нельзя усреднять within-group means
невзвешенно, когда counts различаются. Не сравнивайте абсолютные replay counts
старых и новых переходов как равный opportunity without age exposure adjustment.

## Frozen analysis

`frozen.csv`: slot, transition_id, tag, raw_priority, td_abs, replay_count.
`frozen.metadata.json`: N, group sizes, alpha=.6, epsilon=1e-6, synthetic,
metadata и SHA-256 CSV. `analyze_frozen` проверяет hash/конвенцию до расчётов.

`probabilities.csv` сохраняет исходные колонки и p_uniform/p_per/p_mixture/p_grouped,
а также w для заданного beta. `summary.json` содержит expected group mass,
excess, ratio, min/max/quantiles P, effective probability support 1/sum(P^2).
Последняя величина описывает концентрацию вероятностей, не ESS фактических IS weights.
Сам frozen export не содержит observations и не позволяет восстановить learner/RNG.

На реальных логах `synthetic=false`, но отсутствие пометки synthetic само по себе
не удостоверяет происхождение: команда должна сохранять настоящие run/checkpoint metadata.

## Совместимое расширение после cycle 3

К batch schema v1 добавлены поля, исходные определения не менялись:

- `sampler_outage_probability`: точная sum_{i:c_i=1} P(i) в момент sampling.
- `sampler_expected_allocation_difference`: эта вероятность минус N1/N.
- `groups[c].sampler_group_probability`: marginal probability данной группы.
- `groups[c].expected_count_from_sampler`: B * marginal probability.

Для uniform/grouped ожидаемая probability равна Nc/N; для PER используется
sum priority mass группы / total mass; для mixture — полная смесь обеих долей.
Это ожидаемые значения прежней allocation diagnostics, не новая performance metric.
Графики читают и старые, и расширенные batch logs; expected curve появляется
только если поля доступны. Не заменяйте observed counts этими expectations.

Отдельный replay `state_dict` теперь содержит payloads/RNG/cursor и позволяет
продолжение sampler. Он не меняет спецификацию frozen CSV, описанную выше,
и не заменяет полный checkpoint learner/environment.
