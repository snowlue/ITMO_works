# Assignment **POSIX**

The task is to implement the classic **producer-consumer** pattern with a few additional conditions. The program should consist of 3 + *N* threads:
 1. **main** 
 2. **producer** 
 3. **interruptor** 
 4. *N* threads **consumer**

A string in the form of a list of numbers separated by a space is passed to the standard input to the program (it can be read until the end of the input). The length of the list of numbers is not specified - reading is performed up to the newline character or EOF (end of file)
 - The task of the **producer**-thread is to get a list of numbers as input and use each value from this list in turn to update the variable shared between the threads
 - The task of **consumer** is to respond to notifications from **producer** and collect the sum of the received values. Also, this thread should protect itself from attempts by the thread-**interruptor** to stop it. Additional conditions:
	 1. The function executing the code of this thread **consumer_routine** must accept a pointer to an object/variable from which it will read updates
	 2. After summing the variable, the thread should fall asleep for a random number of milliseconds, the upper limit will be passed to the application input (0 milliseconds should also be processed correctly). During sleep, the thread should not interfere with other **consumer** threads to perform their tasks, if they exist
	 3. **consumer** threads should not duplicate calculations of the same values with each other
	 4. As a return value, the thread should return its partial calculated sum
- The task of the thread-**interruptor** is simple: while the process of updating values is taking place, it must constantly try to stop the random thread **consumer** (the calculation of the random thread occurs before each attempt to stop). As soon as the **producer** thread has made the last update, this thread ends.

The **run_threads** function should start all threads, wait for their execution, and return the result of the total summation.

Only the pthread API can be used to ensure interthreaded communication. At the start, 2 arguments are passed to the application as input in exactly this sequence:
 - Number of consumer threads 	
 - Upper limit of sleep **consumer** in milliseconds

It is also necessary to implement support for the *–debug* key, when using which each **consumer** thread will output a pair (*tid*, *psum*), where *tid* is implemented using the **get_tid()** function, and *psum* is the sum that the thread calculated. The output of *psum* values occurs with each change.

The **get_id()** function returns the thread ID. The thread ID is not just **pthread_self()**, but a unique number for each thread in the range *1.. 3+N*. The value of this number is supposed to be stored in *TLS*. The memory for the stored value should be allocated in heap, and the pointer to it in *TLS*. Also, the **get_tid** function should be self-sufficient (To use it in another project, it should only be enough to copy **get_tid** and use. But at the same time, you can add system header files)

Only the resulting value should get into the output stream, by default no debugging or request information should be output.

The presence of tests in the program is mandatory! The tests should cover the core functionality of project. Tests should be placed in a file tests/tests.cpp and use the doctest.h library, which is located in the tests directory. Changing the signature of the **run_threads** function is allowed at your discretion.

The directory structure of the project looks like this:
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

It is forbidden to modify **Makefile/README.md/doctest.h** and hidden files in this directory. The rest of the files can be changed at your discretion. You can also add new cpp/h files if necessary. If you want to use CMake instead of GMake, you can add **CMakeLists.txt **, but do not forget that ** Makefile** is forbidden to touch.

Template for tests:
```cpp
/******* Template (tests/tests.cpp) **********/
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <producer_consumer.h>

TEST_CASE("just_example") { CHECK(4 == 4); }
```

Template for project implementation:
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
All tests are run under `ubuntu:18.04`. You can choose a different version of ubuntu or a different distribution. But then the tests work on `ubuntu:18.04` is not guaranteed. To use `ubuntu:18.04` there are at least three ways:
 - native OS installation
 - installation using Virtual Box
 - installation using docker

Recommended method using docker. If you want to try another way, then it's under your own setup. Docker usage example:

 1. Building an image
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
 2. Running an instance
 ```bash
$ sudo docker run -v $(pwd):/home/posix --rm -it posix
root@352d2930048c:/#
 ```
3. Example of running tests
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
There are 4 types of tests in the project: **build**, **doctest (builtin)**, **tidy**, **tests**. Only doctest (builtin), tidy are available in your local environment. The rest of the tests are run via pipeline. The pipeline is described in the last section.

Running tidy tests (static analyzer):
```bash
$ make clang_tidy
clang-tidy *.cpp tests/*.cpp -checks=-*,clang-analyzer-*,-clang-analyzer-cplusplus* -- -I tests -I . -std=c++17 -g -O3 -Werror -Wall -Wextra -pthread -pedantic
```

Your code must comply with [google style guide](https://google.github.io/styleguide/cppguide.html ). To check this condition, you need to run:

```bash
$ make clang_format
diff -u <(cat *.cpp *.h tests/*.cpp) <(clang-format *.cpp *.h tests/*.cpp)
```

If clang-format gives an error, then you can fix the code under style guide using the command:
```bash
$ make clang_format_fix
clang-format -i *.cpp *.h tests/*.cpp
```

To run unit tests based on doctest, use the build_tests/run_tests commands
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
 - In this repository, you need to develop in the development branch
 - After you decide that the task is ready, you need to create a merge request in the master branch. You need to add one of the course teachers to reviewers
 - Further communication, code review, testing will take place through this merge request
 - All corrections of comments should go through one merge request. **No need to create separate merge requests after each fix.**
 - After the merge request is approved and merged in the master, the task is considered completed

# Pipeline
After creating a merge request, a pipeline will appear in your branch consisting of 4 stages **build**, **builtin-test**, **tidy**, **test**.
 - **build** - checking the build of the project under different compilers clang++/g++
 - **builtin-test** - run your unit tests written using doctest
 - **tidy** - static analyzer and style guide compliance check
 - **test** - integration tests that check the performance of your project

In addition to pipeline stages, merge request will display tests and their execution results. For dropped tests, you can see the detailed reason for the falls. Detailed information about the gitlab pipeline can be found [here](https://docs.gitlab.com/ee/ci/pipelines/)
