#include <iostream>
#include <sstream>
#include <vector>
#include "producer_consumer.h"
using namespace std;

int main(int argc, char* argv[]) {
  if ((argc != 4) && (argc != 3)) {
    return 1;
  }

  int special_i = 1;
  bool verbose_mode = false;

  if ((string(argv[special_i]) == "-debug") && (argc == 4)) {
    ++special_i;
    verbose_mode = true;
  }

  size_t thread_count = atoi(argv[special_i]);
  ++special_i;
  long max_delay = atoi(argv[special_i]);

  int m;
  string str;

  getline(cin, str);
  stringstream inp(str);

  vector<int> nums;

  while (inp >> m) nums.push_back(m);

  cout << run_threads(thread_count, max_delay, nums, verbose_mode) << endl;
  return 0;
}
