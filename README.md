# Документация проекта: "Распознование рукописных цифр на основе Pytorch и Sklearn"

## Описание проекта
## Предварительная подготвка
### 1. Python установка
Изначально необходмо скачать Python. Подробная инструкция установки и сам установочный файл можно найти [здесь](https://www.python.org/downloads/)
Далее открываем консоль и с помощью ***pip*** устанавливаем необходимые модули.
![Вид консоли](https://gitlab.com/Alina111/project/blob/master/mnist_classification/image/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA00.PNG)
```
pip install <название модуля>;
```
> [!TIP]
> Желательно загрузить 3.6.8 Python. Это улучшит и упростит установку компонентов, и также не вызовет дополнительных технических ошибок.

### 2. Устанавливаем данные модули:
```
pip install torch
pip install tkinter
pip install os
pip install numpy
pip install PIL
pip install torchvision
pip install tqdm
pip install argparse
pip install matplotlib
pip install pandas
pip install sklearn
pip install pickle
pip install time
```
Если не уверены правильно ли загрузились модули, то в консоли прописываем:
```
ipython
//Так выглядят поля для ввода строки,она появится,когда загрузка интерактивной консоли закончится.
In [1]:
```
Здесь уже можем проверить наличие необходимых нам модулей с помощью строки:
![Интерактивная консоль](https://gitlab.com/Alina111/project/blob/master/mnist_classification/image/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA01.PNG)

```
import <название модуля>
//Для выхода прописываем команду:
exit
```
### 3. Скачивание файлов

 Проследите за тем куда вы устанавливайте файлы, и далее измените код вствавив в нееоторые области пути к файлам.

> [!WARNING]
> Возможна проблема с использованием команды ***pip***. Тогда слдует её заменить на ***pip3***, ***python*** или ***python3***

## Структура
- ### mnist_dataset.py
- ### conv_net_model.pth
- ### model.pkl
- ### net.py
- ### train.py
- ### platform.py
- ### dowbload_mnist.py
- ### classify.py

## Подробное описание

### classify.py
Здесь находится самый главный класс ***LinearSVM()***. 
Для его запуска необходимы два файла **train.csv**, **test.csv**.
В этих файлах находятся диретории к каринкам и лейблы.
Модель загружается в файл **model.pkl**
Позже это этот класс будет использоваться в ***Platform.py***.

### dowbload_mnist.py
Необходим для загрузки двух папок - набора картинок.

### mnist_dataset.py
В PyTorch уже интегрирован датасет MNIST, который мы можем использовать при помощи функции DataLoader.
Но мы создаём собственный и называем ***MNISTdataset***
С помощью ***argparse*** добавляем переменные, которые пользователь может вносить самостоятельно ,но для удобства мы присвоили некоторые **default**-значения.
Далее задаем преобразование для применения к MNIST и переменные для данных ***transform.Compose***
### conv_net_model.pth
Это файл для сохранения обученной модели **Pytorch**.
### model.pkl
Это файл для сохранения обученной модели **Sklearn**.
> [!IMPORTANT]
> Без этих файлов код работать не будет.

### net.py
На этом шаге задаём класс ***nn.Module***, определяющий сверточную нейронную сеть, которую мы хотим обучить.
Класс nn.Module очень полезен в PyTorch, он содержит всё необходимое для конструирования типичной глубокой нейронной сети.

***def forward***,задача этой функции - определить потоки данных через слои при прямом прохождении через сеть.
* Здесь важно назвать функцию “forward”, так как она будет использовать базовую функцию прямого распространения в nn.Module
и позволит всему функционалу nn.Module работать корректно.
Как можно видеть, функция принимает на вход аргумент х, представляющий собой данные, которые должны проходить через модель (например, партия данных).
Направляем эти данные в первый слой (self.layer1) и возвращаем выходные данные как out.
Эти выходные данные поступают на следующий слой и так далее.
Отметим, после self.layer2 применяем функцию преобразования формы к out, которая разглаживает размеры данных с 7х7х64 до 3164х1.
После двух полносвязных слоев применяется dropout-слой и из функции возвращается финальное предсказание.

### train.py
Нейронная сеть обучается лучше, когда входные данные нормализованы так, что их значения находятся в диапазоне от -1 до 1 или от 0 до 1.
Чтобы это сделать с помощью нормализации PyTorch, необходимо указать среднее и стандартное отклонение MNIST датасета, которые в этом случае равны 0.1307 и 0.3081 соответственно.
Отметим, среднее значение и стандартное отклонение должны быть установлены для каждого входного канала. 
Далее создаём объекты  train_dataset и test_dataset, которые будут последовательно проходить через загрузчик данных. 
Перед тренировкой модели мы сначала создаём экземпляр нашего класса ***ConvNet()***, определяем функцию потерь и оптимизатор.
Экземпляр класса ConvNet() создается под названием  model.
Потери добавляются в список, который будет использован позже для отслеживания прогресса обучения. 
PyTorch делает процесс обучения модели очень легким и интуитивным.
Устанавливаем режим оценки используя model.eval().
Это удобная функция запрещает любые исключения или слои нормализации партии в модели, которые будут мешать объективной оценке.
* Наконец, результат выводится в консоль, а модель сохраняется при помощи функции torch.save().
Можно видеть, сеть достаточно быстро достигает высокого уровня точности на тренировочном сете.
```
//Так выглядит загрузка
  0%|          | 0/118 [00:00<?, ?it/s]
//Спустя 9 минут
  Loss: 0.10 Acc: 96.48:  60%|██████    | 71/118 [09:10<03:46,  4.82s/it]
//Конечный результат
  Loss: 0.07 Acc: 97.92: 100%|██████████| 118/118 [12:44<00:00,  3.29s/it]
  Test Accuracy: 98.62
```
Последней строкой выводится оценка точности, которая дошла до **98.62%**.
Можете увеличивать кол-во `epoch` в функции ***argparse***, рекумендуется запуск миксимум 10 эпох.
Но большее кол-во также будет осуществлено. 
![Вывод результата точности и потери](https://gitlab.com/Alina111/project/blob/master/mnist_classification/image/bokeh_plot.png)

### platform.py
Здесь хранится класс ***Paint*** для визуализации нашей главной пользовательской рабочей среды. 
За основу взят простой **Paint** на **Python**.

***Background*** установлен цвет `black` и цвет кисти `white`. 
Т.к. мы тренировали нашу модель на изображениях Белых цифр на чёрном фоне. 
Изменения данных настроек приведут к неправильной точности ***predict***-функции.
Здесь используем ***transform.Compose*** как и в ***train.py***  с той же целью. 
Также добавляем в наш интерфейс некоторые кнопки, такие как `Изменение размера кисти`, `Очищение рабочей среды`, `Загрузка картинки`, `Определение цифры на изображении`
(С данной функцией будут 2 кнопки, т.к. способа распознования цифры у нас 2).
![Вид пользовательской рабочей среды](https://gitlab.com/Alina111/project/blob/master/mnist_classification/image/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA.PNG)
Также здесь присутствует два вида размера кисти.
![Два типа кисти](https://gitlab.com/Alina111/project/blob/master/mnist_classification/image/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA1.PNG)
Также в этом файле хранится главный запуск всей нашей программы.
После запуска появляется окно нашего ***Paint*** на чёрной области можнорисовать путём зажатия кнопки.
Если ваше устройство имеет сенсорный экран, вы также можете рисовать прикосновениями.
После нажатия на кнопку **Predict** выведется **message box** c результатом распознания цифры нейронной сетью. 
То же самое произойдёт и с кнопкой **Predict2**, но распознаваться цифра будет с помощью класса **LinearSVM()** основаного на **Sklearn**
![Нарисованное число 3](https://gitlab.com/Alina111/project/blob/master/mnist_classification/image/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA2.PNG?raw=true)
![Вывод результата после Predict](https://gitlab.com/Alina111/project/blob/master/mnist_classification/image/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA3.PNG?raw=true)

> Также будет погрешность, т.к. нейронная сеть, которая была использована в данном коде довольно проста.
> Чем проще рисовка цифры,тем выше точность результата работы программы.

## Оценка работы двух **predict**-еров
### Точность на загруженных картинках
***Pytorch*** - `98,46%` на 1 эпохе
***Sklearn*** - `91,82%`
### Точность на поле для рисования
***Pytorch*** - `70%` на 1 эпохе (20 изображений)
***Sklearn*** - `50%` (20 изображений)
## Ссылки для разбора и изучения кода им данной темы
- https://neurohive.io/ru/tutorial/glubokoe-obuchenie-s-pytorch/
- https://neurohive.io/ru/tutorial/cnn-na-pytorch/
- https://github.com/adventuresinML/adventures-in-ml-code
