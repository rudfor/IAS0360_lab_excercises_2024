#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <string>
#include <utility>

class Common {
public:
    int taskNumber = -1;  // Initialize to a default value
    double learningRate = 0.001; // Initialize to a default value
    bool runAll = false;
    bool verbose = false;
    bool assertion = false;
    // Load the dataset from a CSV file
    void print_usage();
    int parse_options(int argc, char* argv[]);
};

#endif // COMMON_H
