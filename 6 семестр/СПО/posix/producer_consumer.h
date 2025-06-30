#pragma once

#include <vector>

int run_threads(std::size_t thread_count, long max_delay, std::vector<int> nums,
                bool debug);
