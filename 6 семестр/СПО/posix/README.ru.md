# Assignment **POSIX**

Ваша задача - реализовать классический паттерн **producer-consumer** с небольшими дополнительными условиями. Программа должна состоять из 3 + *N* потоков:
 1. **main** 
 2. **producer** 
 3. **interruptor** 
 4. *N* потоков **consumer**

На стандартный ввод программе подается строка - список чисел, разделённых пробелом (читать можно до конца ввода). Длина списка чисел не задаётся - считывание происходит до перевода каретки или EOF (end of file)
 - Задача **producer**-потока - получить на вход список чисел, и по очереди
   использовать каждое значение из этого списка для обновления
   переменной разделяемой между потоками
 - Задача **consumer**-потоков реагировать на уведомления от **producer** и набирать сумму полученных значений. Также этот поток должен защититься от попыток потока-**interruptor** его остановить. Дополнительные условия:
	 1. Функция, исполняющая код этого потока **consumer_routine**, должна принимать указатель на объект/переменную, из которого будет читать обновления
	 2. После суммирования переменной поток должен заснуть на случайное количество миллисекунд, верхний предел будет передан на вход приложения (0 миллисекунд также должно корректно обрабатываться). Вовремя сна поток не должен мешать другим потокам **consumer** выполнять свои задачи, если они есть
	 3. Потоки **consumer** не должны дублировать вычисления друг с другом одних и тех же значений
	 4. В качестве возвращаемого значения поток должен вернуть свою частичную посчитанную сумму
- Задача потока-**interruptor** проста: пока происходит процесс обновления значений, он должен постоянно пытаться остановить случайный поток **consumer** (вычисление случайного потока происходит перед каждой попыткой остановки). Как только поток **producer** произвел последнее обновление, этот поток завершается.

Функция **run_threads** должна запускать все потоки, дожидаться их выполнения, и возвращать результат общего суммирования.

Для обеспечения межпоточного взаимодействия допускается использование только pthread API. На вход приложению при старте передаётся 2 аргумента  именно в такой последовательности:
 - Число потоков consumer  	
 - Верхний предел сна **consumer** в миллисекундах

Также необходимо реализовать поддержку ключа *–debug*, при использовании которого каждый **consumer**-поток будет выводить пару (*tid*, *psum*), где *tid* реализуется с помощью функции **get_tid()**, а *psum* это сумма которую посчитал поток. Вывод значений *psum* происходит при каждом изменении.

Функция **get_tid()** возвращает идентификатор потока. Идентификатор потока это не просто **pthread_self()**, а уникальное для каждого потока число в диапазоне *1 .. 3+N*. Значение этого числа предполагается хранить в *TLS*. Память под сохраняемое значение должна выделяться в heap, а указатель на нее в *TLS*. Так-же функция **get_tid** должна быть самодостаточной (Для использования ее в другом проекте должно быть достаточно только скопировать **get_tid** и использовать. Но при этом можно добавлять системные заголовочные файлы)

В поток вывода должно попадать только результирующее значение, по умолчанию никакой отладочной или запросной информации выводиться не должно.

Наличие тестов в программе обязательно! Тесты должны покрывать основную функциональность вашего проекта. Тесты нужно размещать в файле tests/tests.cpp и использовать библиотеку doctest.h которая размещена в директории tests. Менять сигнатуру функции **run_threads** разрешено по вашему усмотрению.

Структура каталогов проекта выгляди следующим образом:
```
├── Dockerfile
├── main.cpp
├── Makefile
├── producer_consumer.cpp
├── producer_consumer.h
├── README.md
└── tests
    ├── doctest.h
    └── tests.cpp
```

В этом каталоге запрещено изменять **Makefile/README.md/doctest.h** и скрытые файлы. Остальные файлы можете менять по вашему усмотрению. Также можно добавлять новые cpp/h файлы если это необходимо. Если вы хотите использовать CMake вместо GMake, то можете добавить **CMakeLists.txt**, но при этом не забывайте что **Makefile** трогать запрещено.

Шаблон для тестов:
```cpp
/******* Template (tests/tests.cpp) **********/
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <producer_consumer.h>

TEST_CASE("just_example") { CHECK(4 == 4); }
```

Шаблон для реализации проекта:
```cpp
/******* Template (producer_consumer.cpp) **********/
#include <pthread.h>
#include <iostream>

int get_tid() {
 // 1 to 3+N thread ID
 return 0;
}

void* producer_routine(void* arg) {
 (void)arg;
 // read data, loop through each value and update the value, notify consumer,
 // wait for consumer to process
 return nullptr;
}

void* consumer_routine(void* arg) {
 (void)arg;
 // for every update issued by producer, read the value and add to sum
 // return pointer to result (for particular consumer)
 return nullptr;
}

void* consumer_interruptor_routine(void* arg) {
 (void)arg;
 // interrupt random consumer while producer is running
 return nullptr;
}

// the declaration of run threads can be changed as you like
int run_threads() {
 // start N threads and wait until they're done
 // return aggregated sum of values
 return 0;
}
```

# Setup Environment
Все тесты запускаются под `ubuntu:18.04`. Вы можете выбрать другую версию ubuntu или другой дистрибутив. Но тогда работоспособность тестов на `ubuntu:18.04` не гарантируется. Для использования `ubuntu:18.04` существует как минимум три способа: 
 - нативная установка ОС
 - установка с использованием Virtual Box
 - установка с использованием docker

Рекомендуемые способ с использованием docker. Если вы хотите попробовать другой способ, то это под вашу собственную настройку. Пример использования docker:

 1. Сборка образа
 ```bash
$ sudo docker build -t posix:latest .
Sending build context to Docker daemon  26.03MB
Step 1/5 : FROM ubuntu:18.04
 ---> 3339fde08fc3
Step 2/5 : LABEL name="posix"
 ---> Using cache
 ---> 18d4d446880d
Step 3/5 : RUN apt update
 ---> Using cache
 ---> 7b8914da4b6a
Step 4/5 : RUN apt install -y --allow-unauthenticated     clang     g++     clang-tidy     clang-format-7     make
 ---> Using cache
 ---> 73647810ca79
Step 5/5 : RUN ln -s /usr/bin/clang-format-7 /usr/bin/clang-format
 ---> Using cache
 ---> d149e813ff12
Successfully built d149e813ff12
Successfully tagged posix:latest
 ```
 2. Запуска инстанса
 ```bash
$ sudo docker run -v $(pwd):/home/posix --rm -it posix
root@352d2930048c:/#
 ```
3. Пример запуска тестов
```bash
root@352d2930048c:/# cd /home/posix/
root@352d2930048c:/home/posix# make build_tests
g++ producer_consumer.cpp tests/*.cpp -I tests -I . -std=c++17 -g -O3 -Werror -Wall -Wextra -pthread -pedantic -o tests/tests
root@352d2930048c:/home/posix# make run_tests
./tests/tests -s -d
[doctest] doctest version is "2.4.8"
[doctest] run with "--help" for options
===============================================================================
tests/tests.cpp:5:
TEST CASE:  just_example

tests/tests.cpp:5: SUCCESS: CHECK( 4 == 4 ) is correct!
  values: CHECK( 4 == 4 )

0.000118 s: just_example
===============================================================================
[doctest] test cases: 1 | 1 passed | 0 failed | 0 skipped
[doctest] assertions: 1 | 1 passed | 0 failed |
[doctest] Status: SUCCESS!
```
# Run Tests
В проекте присутствует 4 типа тестов: **build**, **doctest (builtin)**, **tidy**, **tests**. В вашем локальном окружении доступны только doctest (builtin), tidy. Остальные тесты запускаются через pipeline. Про pipeline описано в последнем разделе.

Запуск tidy тестов (статический анализатор):
```bash
$ make clang_tidy
clang-tidy *.cpp tests/*.cpp -checks=-*,clang-analyzer-*,-clang-analyzer-cplusplus* -- -I tests -I . -std=c++17 -g -O3 -Werror -Wall -Wextra -pthread -pedantic
```

Ваш код должен соответствовать [google style guide](https://google.github.io/styleguide/cppguide.html). Для проверки этого условия нужно запустить:

```bash
$ make clang_format
diff -u <(cat *.cpp *.h tests/*.cpp) <(clang-format *.cpp *.h tests/*.cpp)
```

Если clang-format выдает ошибку, то вы можете исправить код под style guide с помощью команды:
```bash
$ make clang_format_fix
clang-format -i *.cpp *.h tests/*.cpp
```

Для запуска unit тестов основанных на doctest нужно использовать команды build_tests/run_tests
```bash
$ CXX=g++ make build_tests
g++ producer_consumer.cpp tests/*.cpp -I tests -I . -std=c++17 -g -O3 -Werror -Wall -Wextra -pthread -pedantic -o tests/tests
$ make run_tests
./tests/tests -s -d
[doctest] doctest version is "2.4.8"
[doctest] run with "--help" for options
===============================================================================
tests/tests.cpp:5:
TEST CASE:  just_example

tests/tests.cpp:5: SUCCESS: CHECK( 4 == 4 ) is correct!
  values: CHECK( 4 == 4 )

0.000080 s: just_example
===============================================================================
[doctest] test cases: 1 | 1 passed | 0 failed | 0 skipped
[doctest] assertions: 1 | 1 passed | 0 failed |
[doctest] Status: SUCCESS!
```

# Publish Results

 - В этом репозитории необходимо вести разработку в ветке development
 - После того как вы решите что задание готово, то нужно создать merge request в ветку master. В reviewers нужно добавить одного из преподавателей курса
 - Дальнейшее общение, code review, тестирование будет проходить через этот merge request
 - Все исправления замечаний должны проходить через один merge request. **Не нужно создавать отдельные merge requests после каждого исправления.**
 - После тогда как merge request заапрувлен и замержен в master задание считается выполненным

# Pipeline
После создания merge request в вашей ветке появится pipeline состоящий из 4-х стадий **build**, **builtin-test**, **tidy**, **test**.
 - **build** - проверка сборки проекта под разными компиляторами clang++/g++
 - **builtin-test** - запуска ваших unit тестов написанных с использованием doctest
 - **tidy** - статический анализатор и проверка соответствия style guide
 - **test** - интеграционные тесты проверяющие работоспособность вашего проекта

Помимо стадий pipeline в merge request будут отображаться тесты и их результаты выполнения. Для упавших тестов можно посмотреть подробную причину падений. Подробную информацию о gitlab pipeline можно найти [здесь](https://docs.gitlab.com/ee/ci/pipelines/)
