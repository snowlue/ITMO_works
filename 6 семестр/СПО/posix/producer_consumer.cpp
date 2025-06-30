#include <pthread.h>
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;

pthread_mutex_t value_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t read_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t data_updated = PTHREAD_COND_INITIALIZER;
pthread_cond_t data_processed = PTHREAD_COND_INITIALIZER;

bool terminated = false;
bool read = false;

struct config {
  size_t thread_count;
  long max_delay;
  bool debug_mode = false;
};

struct config runtime_config;

struct producer_args {
  int* value_ptr;
  int* data_ptr;
  size_t data_size;
};

int get_tid() {
  static atomic<int> counter{0};
  thread_local static int* tid_ptr;
  if (!tid_ptr) {
    counter.fetch_add(1, memory_order_relaxed);
    tid_ptr = new int(counter.load());
  }
  return *tid_ptr;
}

void* producer_routine(void* arg) {
  producer_args* args = static_cast<producer_args*>(arg);

  read = false;
  terminated = false;

  for (size_t i = 0; i < args->data_size; ++i) {
    pthread_mutex_lock(&value_mutex);
    *args->value_ptr = args->data_ptr[i];
    pthread_cond_signal(&data_updated);
    pthread_mutex_lock(&read_mutex);
    read = false;
    pthread_mutex_unlock(&read_mutex);
    while (!read) {
      pthread_cond_wait(&data_processed, &value_mutex);
    }
    pthread_mutex_unlock(&value_mutex);
  }

  terminated = true;
  pthread_cond_broadcast(&data_updated);
  return nullptr;
}

void* consumer_routine(void* arg) {
  int* shared_value = static_cast<int*>(arg);
  pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, nullptr);

  int* sum = new int(0);

  while (!terminated) {
    pthread_mutex_lock(&value_mutex);
    while (read && !terminated) {
      pthread_cond_wait(&data_updated, &value_mutex);
    }

    if (terminated) {
      pthread_mutex_unlock(&value_mutex);
      break;
    }

    pthread_mutex_lock(&read_mutex);
    *sum += *shared_value;
    read = true;
    pthread_mutex_unlock(&read_mutex);

    if (runtime_config.debug_mode) {
      cout << "Thread " << get_tid() << ": " << *sum << endl;
    }

    pthread_cond_signal(&data_processed);
    pthread_mutex_unlock(&value_mutex);

    if (runtime_config.max_delay > 0) {
      this_thread::sleep_for(
          chrono::milliseconds(rand() % (runtime_config.max_delay + 1)));
    }
  }
  pthread_exit((void*)sum);
}

void* consumer_interruptor_routine(void* arg) {
  pthread_t* consumers = static_cast<pthread_t*>(arg);
  while (!terminated) {
    int target = rand() % runtime_config.thread_count;
    pthread_cancel(consumers[target]);
  }

  return nullptr;
}

int run_threads(size_t thread_count, long max_delay, vector<int> nums,
                bool debug) {
  runtime_config.thread_count = thread_count;
  runtime_config.max_delay = max_delay;
  runtime_config.debug_mode = debug;
  int shared_data = 0;

  producer_args* args =
      new producer_args{&shared_data, nums.data(), nums.size()};

  pthread_t producer;
  pthread_create(&producer, nullptr, producer_routine, args);

  pthread_t* consumers = new pthread_t[thread_count];
  for (size_t i = 0; i < thread_count; ++i) {
    pthread_create(&consumers[i], nullptr, consumer_routine, &shared_data);
  }

  pthread_t interruptor;
  pthread_create(&interruptor, nullptr, consumer_interruptor_routine,
                 consumers);
  int total_sum = 0;
  for (size_t i = 0; i < thread_count; ++i) {
    int* con_sum;
    pthread_join(consumers[i], reinterpret_cast<void**>(&con_sum));
    total_sum += *con_sum;
    delete con_sum;
  }

  pthread_join(producer, nullptr);
  pthread_join(interruptor, nullptr);

  delete args;
  delete[] consumers;

  return total_sum;
}
