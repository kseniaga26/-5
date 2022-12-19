# Интеграция экономической системы в проект Unity и обучение ML-Agent. 
Лабораторная работа №5 для АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ

Отчет по лабораторной работе #3 выполнил(а):
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

По завершении обучения файлы сохранились.

![image](https://user-images.githubusercontent.com/114469025/208476018-2bba1695-bfa1-492e-82c8-ea0e56bb45e2.png)

- Чтобы ускорить процесс обучения – увеличим количество префабов TargetAreaEconomic до 12 и снова запустим обучение:

![image](https://user-images.githubusercontent.com/114469025/208480438-b089ed6f-83cb-4b0f-a864-2b6a0ed5aff4.png)
![image](https://user-images.githubusercontent.com/114469025/208480471-2a789d11-2921-448e-a78f-c3bb5048c00b.png)

- Далее построим графики для оценки результатов обучения. Для этого установим библиотеку TensorBoard с помощью следующей команды:
```
pip install tensorflow

```
- После завершения установки запустим TensorBoard и рассмотрим полученные графики стандартного агента:


![image](https://user-images.githubusercontent.com/114469025/208328134-83c29e46-7c6d-45b9-a279-f3e95efe8203.png)
