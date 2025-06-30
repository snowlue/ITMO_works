#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <producer_consumer.h>
#include <cstdlib>
#include <ctime>
#include <vector>
using namespace std;

TEST_CASE("Positive test case") {
  vector<int> nums{1, 2, 3, 4, 5};
  CHECK(run_threads(2, 1, nums, false) == 15);
}

TEST_CASE("Large threads count") {
  vector<int> nums{2, 4, 6, 8, 10};
  CHECK(run_threads(1000, 1, nums, false) == 30);
}

TEST_CASE("Single consumer thread") {
  vector<int> nums{1, 2, 3, 4, 5};
  CHECK(run_threads(1, 5, nums, false) == 15);
}

TEST_CASE("Empty array") {
  vector<int> nums{};
  CHECK(run_threads(4, 10, nums, false) == 0);
}

TEST_CASE("Negative values") {
  vector<int> nums{-1, -2, -3, -4, -5};
  CHECK(run_threads(3, 5, nums, false) == -15);
}

TEST_CASE("Mixed positive and negative values") {
  vector<int> nums{-10, 20, -30, 40, -50};
  CHECK(run_threads(2, 10, nums, false) == -30);
}

TEST_CASE("Zero delay") {
  vector<int> nums{1, 2, 3, 4, 5};
  CHECK(run_threads(3, 0, nums, false) == 15);
}

TEST_CASE("Large values") {
  vector<int> nums{1000000, 2000000, 3000000};
  CHECK(run_threads(2, 5, nums, false) == 6000000);
}

TEST_CASE("Single value") {
  srand(time(nullptr));
  int random_value = rand() % 10000 + 1;
  vector<int> nums{random_value};
  CHECK(run_threads(5, 10, nums, false) == random_value);
}

TEST_CASE("Many elements") {
  vector<int> nums;
  int expected_sum = 0;

  for (int i = 1; i <= 10000; i++) {
    nums.push_back(i);
    expected_sum += i;
  }

  CHECK(run_threads(4, 2, nums, false) == expected_sum);
}