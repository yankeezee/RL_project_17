# Контракт для Areg и Leon

## Подключение без изменения learner

Подключите `outage_replay/` как пакет или перенесите его в репозиторий.
`ReplayBuffer(capacity=50_000, seed=replay_seed)` хранит opaque transition payload.
Публичный API:

```python
slot, transition_id = buffer.add(
    transition,
    current_delivered=current_injected_delivery,
    next_delivered=next_injected_delivery,  # None допустим только при task end
    terminated=terminated,
    timeout=scenario_timeout,
)
batch = buffer.sample(64, method="grouped", beta=beta_schedule(decisions, 300_000))
# batch.transitions, indices, transition_ids, tags, probabilities, weights
# optimizer.zero_grad(); weighted_loss.backward(); optimizer.step()
buffer.update_priorities(batch.indices, detached_td_errors,
                         transition_ids=batch.transition_ids)
logger.log(batch, td_errors=detached_td_errors,
           unreduced_huber=detached_unreduced_huber,
           decisions=decisions, optimizer_step=optimizer_step)
```

Остальные `method`: `uniform`, `per`, `mixture`. Все используют одну и ту же
реализацию хранения и интерфейс. Существующий PER команды можно сохранить, если
на общих фиксированных данных совпадают probabilities, weights, новые priorities,
sampling-with-replacement и обработка дубликатов. Иначе согласовать разницу до main runs.

Каждая выборка возвращает вероятности и веса **на момент sampling**.
Нельзя пересчитывать их после вставки переходов или refresh приоритетов.
`transition_ids` защищают от изменения приоритета уже перезаписанного слота;
при stale update весь запрос отклоняется до мутации. Буфер не потокобезопасен.

## Leon: флаги и границы эпизода

Передавайте именно injected delivery flags текущего decision observation
и истинного next observation этого перехода, до autoreset.
Не используйте target detection, наличие координат, заполненные нулями пиксели
или маску любого старого элемента four-step history вместо delivery flag.
Доставленный кадр без цели имеет `current_delivered=True`.

`z = terminated or scenario_timeout`. Для z=True next flag игнорируется:
можно передать None. Timeout в proposal завершает конечную задачу, bootstrap=0.
Произвольное внешнее truncation/остановку эксперимента не объявляйте scenario timeout
без решения команды. Не смешивайте next state перехода с reset state следующего эпизода.
Храните observation/history как независимые снимки: не передавайте переиспользуемый
мутируемый массив среды. Payload не копируется буфером.

Пример payload-контракта и корректного target masking есть в
`examples/learner_integration.py`. Конечные переходы там вообще не проходят next-state
forward: None/NaN terminal observation не загрязняет target.

Среда остаётся общей: predict_position, четыре tics на действие, fixed history,
time remaining, default reward/episode limit. Базовый dropout Leon остаётся общим
для всех методов; здесь outage masks не генерируются.

## Areg: TD error, loss, расписание

```python
# y = rewards + gamma * (1-z) * Q_target(next, argmax Q_online(next))
# Не вычисляйте Q(next) для отсутствующих terminal observations.
delta = y - q_selected
per_item = torch.nn.functional.smooth_l1_loss(
    q_selected, y, reduction="none", beta=1.0)
w = torch.as_tensor(batch.weights, device=per_item.device, dtype=per_item.dtype)
assert per_item.shape == w.shape == (64,)
loss = (w * per_item).mean()
```

Не нормируйте веса ещё раз по batch; не делите loss на сумму весов.
Не применяйте alpha к TD errors в learner: буфер добавляет epsilon и степень сам.
Refresh после optimizer.step использует signed detached TD errors от этого loss.
Повторы сохраняются в loss; buffer сводит обновления приоритета по max(abs(delta)).
Beta .4 -> 1 по всему decision budget, включая первые 5000 решений.
Полный бюджет 300000 решений, warm-up 5000, один step на четыре решения,
target copy каждые 1000 optimizer steps. Эти правила исполняет общий training loop,
адаптер не запускает шаги самостоятельно. Сохраните final checkpoint и общие
две 64-unit ReLU layers, gamma=.99, Adam lr=1e-4, exploration и reward.
Один и тот же pipeline reduction/logging должен использоваться для всех методов.

`examples/learner_integration.learner_step` принимает существующие online, target
и optimizer; задайте их train/eval modes в общем learner. Это проверенный адаптер,
а не новая архитектура или отдельный training loop.

## Логирование и frozen buffer

```python
from outage_replay.diagnostics import ReplayLogger
from outage_replay.frozen import export_frozen, analyze_frozen

logger = ReplayLogger(
    "logs", run_id="grouped_burst_train_seed0", seed=0,
    regime="burst_p0.25_L4", synthetic=False, sample_details=True,
    metadata={"protocol": "configs/protocol.json", "git_commit": "<real commit>",
              "replay_seed": replay_seed, "metadata_ndarray_bytes": buffer.metadata_bytes},
)
# Закройте logger в finally/context manager.
# В заранее выбранный общий checkpoint остановите mutations и выберите ОДИН buffer.
source = export_frozen(buffer, "analysis/shared_buffer", synthetic=False,
                       metadata={"collector_method": method, "decisions": decisions,
                                 "checkpoint": "<real checkpoint>", "seed": seed})
analyze_frozen(source, "analysis/shared_comparison", beta=beta_schedule(decisions))
```

Frozen analysis сохраняет один SHA-256 источника и считает все четыре вероятности
из одних tags/raw priorities. Нельзя объединять четыре отдельных learner buffers
и представлять их как один controlled comparison.
Если команда хочет свежие TD errors на всём snapshot, вычислите их одной заранее
выбранной frozen сетью для всех переходов и запишите одинаковые priorities;
документируйте источник. По умолчанию экспортируются текущие replay priorities,
которые могут быть устаревшими относительно последней сети.

Графики реальных логов:

```powershell
python -m scripts.plot_diagnostics --logs logs --output analysis/plots
python -m scripts.analyze_frozen analysis/shared_buffer/frozen.csv analysis/shared_comparison --beta 1
python -m scripts.plot_diagnostics --frozen-analysis analysis/shared_comparison --output analysis/plots
```

Per-draw JSONL полезен для TD/replay-count scatter, но даёт существенный объём.
Его можно отключить `sample_details=False`, сохранив batch-level показатели,
и договориться об одинаковых заранее выбранных diagnostic windows для всех методов.
Не меняйте на основании этого update budget. Для resume используйте новый run_id/segment
с тем же metadata, а не случайно добавляйте данные в старый лог.

## Данные для общей paired evaluation

Предлагаемый внешний CSV, одна строка на episode:
`method,train_regime,train_seed,checkpoint_decisions,eval_condition,scenario_seed,mask_seed,hit,return,missed_shots,timeout`.
`hit` и `timeout` бинарны, episode IDs/seed pairs должны быть уникальны в ячейке метода.
Leon должен сохранить 100 общих held-out scenario/mask pairs по каждой condition,
отдельно от training/pilot seeds; Areg предоставляет final checkpoints трёх seeds.

Primary: per-training-seed paired hit-rate difference **grouped minus PER**,
burst training p=.25,L=4, held-out p=.25,L=8. Hierarchical bootstrap и интервал
делает evaluation pipeline команды. Нужны и положительное улучшение, и excess
outage sampling у PER. Интервал через ноль оставляет улучшение неопределённым.
Сходный эффект mixture ослабляет объяснение через специфическую пользу grouping.
При трёх training seeds точность uncertainty ограничена.

## Проверенные официальные API

Адаптер сверялся с документацией установленной ветки PyTorch 2.10:

- [smooth_l1_loss: reduction и threshold beta](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.smooth_l1_loss.html).
- [Tensor.detach: отключение autograd](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.detach.html).
- [Optimizer.step: шаг оптимизации](https://docs.pytorch.org/docs/2.10/generated/torch.optim.Optimizer.step.html).

Сверка API не заменяет проверки: все четыре метода реально прошли CPU optimizer smoke test.

## Дополнения после пяти циклов проверки

Итог: 100 тестов прошли. Адаптер отклоняет нечисловые TD/loss до backward и
нечисловые gradients до optimizer.step, сохраняя параметры/Adam state/priorities
при этих ошибках. Sampling уже произошёл: RNG и replay counts продвинулись.
Не скрывайте исключение пропуском обновления — остановите run для исправления;
для точного повтора нужен replay checkpoint до ошибочной выборки.

Добавлен API для replay checkpoints:

```python
checkpoint["replay"] = buffer.state_dict()
buffer = ReplayBuffer.from_state_dict(checkpoint["replay"])
# Или атомарно заменить существующий буфер после успешной валидации:
buffer.load_state_dict(checkpoint["replay"])
```

state включает deep-copied payloads, priorities, measured TD, IDs/counts, FIFO
cursor и PCG64 RNG state. Payload должен поддерживать deepcopy. Сохраняйте этот
объект доверенным механизмом общего learner; отдельного pickle loader здесь нет.
Для полного resume нужны также online/target/optimizer, global decisions/steps,
exploration/environment/dropout RNG и состояние эпизода. Exact replay continuation
проверено в текущем NumPy/Python runtime; cross-version идентичность не обещается.
Аналитический frozen CSV остаётся отдельным форматом без payloads/RNG.

`batch.group_probabilities` хранит marginal probabilities групп на момент sampling.
Новые JSONL поля `sampler_outage_probability` и `expected_count_from_sampler`
позволяют отделить ожидаемую концентрацию PER от шума draw counts.
Исходные observed allocation metrics не переопределены. Priority refresh теперь
обновляет общих предков дерева пакетно; API и результаты сохранены побитово.
