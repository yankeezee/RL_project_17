# Real ViZDoom predict_position demo

Настоящий движок ViZDoom, **необученная эвристика**, без replay/обучения.
Это наглядная демонстрация среды и observation outages, не проверка пользы grouped PER.
Основной эксперимент не изменён.

## Смотреть

- `output/run1/comparison.gif` и `comparison.mp4`: первые два эпизода.
  **Сверху clean, снизу burst; в каждой строке слева среда, справа данные агента.**
- `output/run1/clean.mp4`, `burst.mp4`: все шесть эпизодов каждого режима (~25 s).
- `output/run1/burst_outage.png`: мир продолжает двигаться, но кадр не доставлен.
- `output/run1/episodes.csv`, `summary.json`, `*.decisions.jsonl`: фактические результаты.

Все игровые пиксели получены из `GameState.screen_buffer`. Подписи, bounding box
и маска потери кадра добавлены Pillow; сама среда не нарисована и не сгенерирована.
MP4 идёт в реальном игровом времени 35 fps, comparison длится 8.63 s.
GIF ~8.4 s из-за округления длительности кадров форматом GIF.
При раннем завершении одного эпизода сравнение удерживает его последний валидный
кадр с результатом, пока второй эпизод не закончится.

## Запуск

Из общей папки RL (Python 3.13 x64):

```powershell
python -m pip install --target demo/vendor -r demo/requirements.txt
python demo/run_demo.py --episodes 6 --output demo/output/new_run
python demo/validate_outputs.py demo/output/new_run
python -m unittest discover -s demo -p test_demo.py -v
```

Фактически выполнено:

```powershell
python demo/run_demo.py --episodes 6 --output demo/output/run1
```

Используйте новую output directory: существующие результаты не перезаписываются.
ViZDoom 1.3.3, imageio 2.38.2 и imageio-ffmpeg 0.8.1 установлены в `demo/vendor`.
NumPy 2.4.2, Pillow 12.1.1, OpenCV 4.13.0.92 уже были доступны; проверенные версии
записаны в requirements. Пакет ViZDoom включает engine/default scenarios/Freedoom.
Используется raw API, Gymnasium wrapper в демо не участвует.

Первоначальные ограничения: sandbox DNS не позволял pip download, а engine init
в sandbox отклонялся. Установка и реальный запуск успешно выполнены с разрешённым
выходом из песочницы. Окно скрыто, звук выключен.
Windows wheel не загрузил config из Unicode/emoji path; `engine_runtime.py`
копирует неизменённые файлы во временную ASCII-папку. Asset SHA-256 сохранены.
Загруженный .pyd Windows может держать до выхода Python; остальные временные
файлы удаляются при завершении context manager.

## Агент и информация

Эвристика оценивает движение по двум прошлым **доставленным** positions видимой
цели из labels, делает фиксированное упреждение на 24 tics, поворачивается при
ошибке >10 градусов, начиная с decision 16 стреляет при приблизительном наведении.
После единственной ракеты ждёт исхода. Эти параметры записаны до запуска 12
эпизодов и по их результатам не подбирались. Обученной модели не найдено.

Входы: delivered target positions, память предыдущих измерений, собственные
heading и фиксированная spawn position. Это demo policy, не точное воспроизведение
four-step RL input vector и не новая архитектура основного эксперимента.
Hidden target positions, engine target velocity, observer-only image при outage
и outcome counters не передаются политике. Velocity estimate вычисляется из
доставленных позиций; ammo/kills читаются только для telemetry.

Clean доставляет все кадры. Burst следует proposal: p=.25, mean L=4 decisions,
recovery=1/4, onset=p/[L*(1-p)]=1/12, начало каждого эпизода available.
Masks предгенерированы отдельным RNG независимо от действий.
Решение удерживается четыре tics. Запись делается каждый tic без дополнительных
наблюдений/решений политики. Поэтому видео показывает дробный age в decisions,
а логи — age в момент самого решения.

`UNKNOWN - no frame` при outage отличается от `NO - target not visible` на
доставленном кадре. При outage agent view пуст, остаётся только память измерения.
В этих 12 эпизодах во всех доставленных decision frames цель была видна;
доставленный кадр без цели проверен unit test, но не выдаётся за наблюдавшийся случай.

## Реальные результаты run1

| Режим | Эпизоды | Попадания | Промахи | Timeouts | Выстрелы |
|---|---:|---:|---:|---:|---:|
| Clean | 6 | 1 | 5 | 0 | 6 |
| Burst p=.25, L=4 | 6 | 0 | 6 | 0 | 6 |

Scenario seeds 20261125…20261130, mask seeds 20271125…20271130.
Для каждой пары совпали SHA-256 начальных настоящих screen buffers.
В burst потеряно 53/184 decision observations (**28.80%**), самая длинная серия
12 решений (~1.37 s); в первом коротком эпизоде outage вообще не случился.
25% — stationary probability, не квота для каждого короткого эпизода.

Hit = KILLCOUNT > 0. Выстрелы считаются по расходу единственной ракеты.
Missed shot — выпущенная ракета без kill при обычном terminal; выпущенная, но
не разрешившаяся к timeout ракета учитывалась бы отдельно как unresolved.
Timeout читается через `is_episode_timeout_reached()`, не выводится из reward.
Сохранены default reward -.001/tic и +1/kill, timeout=300 tics, start=16.

Проверено: 4 unit tests, все 12 real episode logs, память при outages,
6 пар начальных кадров, reward=hit-.001*executed_tics, conservation shots,
декодирование каждого кадра трёх MP4 и GIF. Отчёт `output/run1/validation.json`.

1/6 против 0/6 на необученной эвристике — **не свидетельство эффективности RL
или grouped PER**. Это демонстрация того, как мир движется, пока наблюдение стареет.

Официальные источники:
[scenario](https://vizdoom.farama.org/environments/default/#predict-position),
[GameState/Label](https://vizdoom.farama.org/api/python/game_state/),
[DoomGame](https://vizdoom.farama.org/api/python/doom_game/),
[Windows installation](https://vizdoom.farama.org/introduction/python_quickstart/#windows).
