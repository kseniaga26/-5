# Интеграция экономической системы в проект Unity и обучение ML-Agent. 
Лабораторная работа №5 для АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ

Отчет по лабораторной работе #5 выполнил(а):
- Голубятникова Ксения Александровна
- РИ210939

Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 80 |
| Задание 2 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;


## Задание 1
### Интегрировать экономическую систему в проект Unity и обучить ML-Agent.

- Откроем проект юнити:

![image](https://user-images.githubusercontent.com/114469025/208484905-9c6c54c4-b182-44a1-b680-89ea50efa337.png)

- Перед тем как перейти к началу обучения, запустим Anaconda Prompt и создадим виртуальное пространство с помощью следующих команд:
```
conda create -n MLAgents python=3.6
conda activate MLAgents
```

- Устанавливаем нужные библиотеки:
```
pip install mlagents==0.28.0
pip install torch~=1.7.1 -f https://download.pytorch.org/whl/torch_stable.html

```
- Добавим Economic.yaml в папку с проектом. Содержимое файла Economic.yaml:

```yaml
behaviors:
  Economic:
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 10240
      learning_rate: 3.0e-4
      learning_rate_schedule: linear
      beta: 1.0e-2
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3      
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    checkpoint_interval: 500000
    max_steps: 750000
    time_horizon: 64
    summary_freq: 5000
    self_play:
      save_steps: 20000
      team_change: 100000
      swap_steps: 10000
      play_against_latest_model_ratio: 0.5
      window: 10
```

- Далее запускаем обучение модели:

![image](https://user-images.githubusercontent.com/114469025/208480239-9cedf40c-0e9d-4a1b-9a12-dc7761f253e4.png)

Шарик начинает двигаться от одного кубика к другому.

![image](https://user-images.githubusercontent.com/114469025/208475491-2d849baa-c793-4f5f-86ec-dee72eed7d4d.png)

![image](https://user-images.githubusercontent.com/114469025/208475542-d6e0ceb4-049c-493f-bc84-1036e304bcd7.png)

- Чтобы ускорить процесс обучения – увеличим количество префабов TargetAreaEconomic до 12 и снова запустим обучение:

![image](https://user-images.githubusercontent.com/114469025/208480438-b089ed6f-83cb-4b0f-a864-2b6a0ed5aff4.png)
![image](https://user-images.githubusercontent.com/114469025/208480471-2a789d11-2921-448e-a78f-c3bb5048c00b.png)

- По завершении обучения файлы сохранились.

![image](https://user-images.githubusercontent.com/114469025/208476018-2bba1695-bfa1-492e-82c8-ea0e56bb45e2.png)


- Далее построим графики для оценки результатов обучения. Для этого установим библиотеку TensorBoard с помощью следующей команды:

```
pip install tensorflow
```
![image](https://user-images.githubusercontent.com/114469025/208482921-a6fa8681-5d90-403e-ac34-217d7d43da7c.png)
![image](https://user-images.githubusercontent.com/114469025/208482952-c0be5dbf-85a0-4ccb-b234-7810805239e3.png)

- После завершения установки запустим TensorBoard и рассмотрим полученные графики:

![image_6](https://user-images.githubusercontent.com/103308669/204276023-8af189e6-118f-4435-988a-fd7727484f35.png)
![image_7](https://user-images.githubusercontent.com/103308669/204276045-8c6aa36e-d6b3-44c4-b1ac-42412c32ed61.png)

### Изменить параметры файла yaml-агента, определить какие параметры и как влияют на обучение модели. Описать результаты, выведенные в TensorBoard.

- При изменении параметра num_layers с 2 на 3:
Получаем такие графики: 

![image_10](https://user-images.githubusercontent.com/103308669/204276865-43d751a2-b9b0-4a73-b412-48fa62e2fc76.png)

- При изменении параметра batch_size с 1024 на 2048: 
Получаем графики: 

![image_13](https://user-images.githubusercontent.com/103308669/204277035-27aeb253-b186-43f0-99c2-08dc8d82db01.png)

- При изменении параметра epsilon с 0.2 на 0.3:
Получаем графики:

![image_16](https://user-images.githubusercontent.com/103308669/204277237-ead67d96-1296-493b-8489-86133fd8a419.png)

- При изменении параметра buffer_size с 10240 до 100:
Получаем графики:

![buffer_size](https://user-images.githubusercontent.com/103308669/204294656-ab36e157-56cf-4ec0-b964-f0347c3b83b9.png)

- При изменении параметра lambd с 0.95 на 0.8:
Получаем графики:

![image_19](https://user-images.githubusercontent.com/103308669/204299228-cb51112e-8b51-4e8a-8751-c84a8fb02ad0.png)

## Задание 2
### Опишите результаты, выведенные в TensorBoard.

#### Environment
- Cumulative Reward - среднее общее вознаграждение за эпизод для всех агентов. Увеличивается, когда эпизод обучения успешен. График должен постоянно увеличиваться, но может вести себя скачкообразно.

- Episode Length - средняя продолжительность эпизода обучения в среде для агентов.

#### Losses
- Policy Loss - средняя величина функции потери политики, где политика - процесс принятия решений. График должен идти вниз во время успешного эпизода.

- Value Loss - средняя потеря функции значения. Она моделирует, насколько хорошо агент прогнозирует значение своего следующего состояния. Должна увеличиваться, пока агент обучается, а затем уменьшаться, когда вознаграждение стабилизируется.

#### Policy

- Entropy - график случайности решений модели. Должен уменьшаться во время успешного эпизода. 
- Beta - гиперпараметр для настройки Entropy.
- Epsilon - гиперпараметр, влияет на скорость развития политики.
- Extrinsic Reward - соответствует среднему совокупному вознаграждению, полученному от окружающей среды за эпизод.
- Value Estimate - это среднее значение, посещённое всеми состояниями агента. Чтобы отражать увеличение знаний агента, это значение должно расти, а затем стабилизироваться.
- Learning Rate - показывает величину шага при поиске оптимальной политики. Должен уменьшаться линейно.

#### Self play
- ELO - показывает силу сети.

## Выводы
В ходе выполнения данной лабораторной работы, я научилась интегрировать экономическую систему в проект Unity в связке с MLAgent и наблюдать за результатами его обучения при изменении параметров в Yaml файле, а также научилась создавать графики на основе результатов обучения MLAgent и анализировать их.
