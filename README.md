# Object Detection Problem
Test Assignment

<!-- TOC -->
* [Object Detection Problem](#object-detection-problem)
* [EDA](#eda)
  * [How to run dashboard](#how-to-run-dashboard)
* [Training Results](#training-results)
* [How to Train and Inference Net](#how-to-train-and-inference-net)
  * [Kaggle](#kaggle)
  * [Colab](#colab)
<!-- TOC -->

# EDA
- Был подготовлен dashboard с интерактивными графиками для более удобного анализа данных:

![](dashboard_gifs/1.gif)

- также в нем есть возможность просмотреть все изображения из датасета с аннотацией или без нее:

![](dashboard_gifs/2.gif)

## How to run dashboard
- dashboard запускается локально, для этого необходимо:
  - склонировать проект
  - в конфиге config.json изменить параметр ["dataset"]["data_path"] на путь до датасета (папки "dataset" из 
  описания тестового задания)
  - запустить main.py

- Графики для EDA

![img.png](eda_plots/1.png)

![img_1.png](eda_plots/2.png)

![img_2.png](eda_plots/3.png)

![img_3.png](eda_plots/4.png)

![img_4.png](eda_plots/5.png)


# Training Results
- в dashboard также содержится информация о процессе обучения нейронной сети и о полученных результатах:

![](dashboard_gifs/3.gif)

- также можно посмотреть примеры предсказаний обученной сети на случайных тестовых изображениях:

![](dashboard_gifs/4.gif)

- Общая информация об обучении (скрины...)


- metrics
- images examples


# How to Train and Inference Net

- общая информация об обучении (гиперпараметры, время, гпу и тд)
- ссылки на ноутбуки с инструкциями
- графики с обучения
- что пробовала менять
- что бы хотела попробовать дальше

## Kaggle 
- Tesla P100 16gb
- Training time: ~ 25 hours (6000 iterations)
- https://www.kaggle.com/railyavaliullina/testassignment

## Colab
- Tesla T4 16gb
- Training time: ~ 27 hours (6000 iterations)
- https://colab.research.google.com/drive/1NtEQRal8Yu_0k1TlSTvFIen5VBGEiEpj?usp=sharing
