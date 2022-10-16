# Object Detection Problem

<!-- TOC -->
* [Object Detection Problem](#object-detection-problem)
  * [EDA](#eda)
    * [Запуск dashboard](#-dashboard)
  * [Training Results](#training-results)
  * [3) Запуск обучения и инференса сети](#3------)
    * [Kaggle](#kaggle)
    * [Colab](#colab)
<!-- TOC -->


## EDA
- Был подготовлен dashboard с интерактивными графиками для более удобного анализа данных:

![](../../../Downloads/videos_2_convert_2_gif/1.gif)

- также в нем есть возможность просмотреть все изображения из датасета с аннотацией или без нее:

![](../../../Downloads/videos_2_convert_2_gif/2.gif)

### Run dashboard
- dashboard запускается локально, для этого необходимо:
  - склонировать проект
  - в конфиге config.json изменить параметр ["dataset"]["data_path"] на путь до датасета (папки "dataset" из 
  описания тестового задания)
  - запустить main.py

- Графики для EDA




## Training Results
- в dashboard также содержится информация о процессе обучения нейронной сети и о полученных результатах:

![](../../../Downloads/videos_2_convert_2_gif/3.gif)

- также можно посмотреть примеры предсказаний обученной сети на случайных тестовых изображениях:

![](../../../Downloads/videos_2_convert_2_gif/4.gif)

- Общая информация об обучении (скрины...)


- metrics
- images examples


## 3) Запуск обучения и инференса сети

- общая информация об обучении (гиперпараметры, время, гпу и тд)
- ссылки на ноутбуки с инструкциями
- графики с обучения
- что пробовала менять
- что бы хотела попробовать дальше

### Kaggle 
- Tesla P100 16gb
- Training time: ~ 25 hours (6000 iterations)
- https://www.kaggle.com/railyavaliullina/testassignment

### Colab
- Tesla T4 16gb
- Training time: ~ 27 hours (6000 iterations)
- https://colab.research.google.com/drive/1NtEQRal8Yu_0k1TlSTvFIen5VBGEiEpj?usp=sharing
