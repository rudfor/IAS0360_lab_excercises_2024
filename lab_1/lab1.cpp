#include <iostream>
#include <iomanip>  // For std::fixed and std::setprecision
#include <vector>
#include <getopt.h>  // For getopt
#include <cassert>  // For assert
#include "../lib/includes/NeuralNetwork.h"

void print_usage() {
    std::cout << "Usage: program [options]\n"
              << "Options:\n"
              << "  -t, --task <number>   Specify task number (1-4) to execute\n"
              << "  -a, --all             Run all tasks (1-4)\n"
              << "  -v, --verbose         Set verbosity\n"
              << "  -s, --assert          Set assertion\n"
              << "  -h, --help            Show this help message\n";
}

void task1(bool assert_only=false, bool verbose=false) {
    NeuralNetwork nn;
    std::vector<double> input = {12, 23, 47};
    std::vector<double> weight = {-3, -2, -3};
    double result = 0;
    double bias = 0;

    // Expected results for task1 (you need to fill these in based on actual expectations)
    std::vector<double> expectedResults = {-36, -46, -141};  // Example expected results
    
    for (size_t i = 0; i < input.size(); i++) {
        result = nn.singleNeuron(input[i], weight[i]) + bias;
        if(!assert_only) {
            std::cout << "Single Neuron Result: " << result
                      << (verbose?(result == expectedResults[i] ? " - Pass" : " - Fail (expected " + std::to_string(expectedResults[i]) + ")"):"")
                      << "\n";
        }
        // Assertion to validate the result
        assert(result == expectedResults[i] && "Assertion failed for task1: unexpected result");
    }
}

void task2() {
    NeuralNetwork nn;
    std::vector<double> temperature = {12, 23, 50, -10, 16};
    std::vector<double> humidity = {60, 67, 45, 65, 63};
    std::vector<double> airQuality = {60, 47, 157, 187, 94};
    std::vector<double> weights = {-2, 2, 1};
    double result = 0;
    double bias = 0;

    for (size_t i = 0; i < temperature.size(); i++) {
        std::vector<double> inputs = {temperature[i], humidity[i], airQuality[i]};
        result = nn.multipleInputSingleOutput(inputs, weights, bias);
        std::cout << "Multiple Input Single Output Result: " << result << "\n";
    }
}

void task3() {
    NeuralNetwork nn;
    double input = 0.9;
    std::vector<double> weights = {-20.2, 95, 201.0};
    std::vector<double> outputs(3);
    double bias = 0;
    nn.singleInputMultipleOutput(input, weights, bias, outputs);
    std::cout << "Single Input Multiple Outputs Result: " << outputs[0] << ", " << outputs[1] << ", " << outputs[2] << "\n";
}

void task4() {
    NeuralNetwork nn;
    int inputSize = 3, outputSize = 3;
    std::vector<double> inputs = {30.0, 87.0, 110.0};
    std::vector<double> biases = {0, 0, 0};
    std::vector<double> multiOutputs(3);
    std::vector<double> multiWeights = {
        -2.0, 9.5, 2.0,  // Weights for output 1
        -0.8, 7.2, 6.3,  // Weights for output 2
        -0.5, 0.4, 0.9   // Weights for output 3
    };

    nn.multipleInputMultipleOutput(inputs, multiWeights, biases, multiOutputs, inputSize, outputSize);
    std::cout << "Multiple Input Multiple Outputs Result: " << multiOutputs[0] << ", " << multiOutputs[1] << ", " << multiOutputs[2] << "\n";
}

int main(int argc, char* argv[]) {
    int option;
    int taskNumber = -1;
    bool runAll = false;
    bool verbose = false;
    bool assertion = false;

    // Define long options for getopt
    struct option long_options[] = {
        {"task", required_argument, 0, 't'},
        {"all", no_argument, 0, 'a'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}  // End of options
    };

    // Parse command-line options
    while ((option = getopt_long(argc, argv, "t:ahsv", long_options, nullptr)) != -1) {
        switch (option) {
            case 't':
                taskNumber = std::atoi(optarg);  // Convert task number from string to int
                break;
            case 'a':
                runAll = true;  // Set flag to run all tasks
                break;
            case 'v':
                verbose = true;  // Set flag to run verbose
                break;
            case 's':
                assertion = true;  // Set flag to run regression tests
                break;
            case 'h':
                print_usage();
                return 0;
            default:
                print_usage();
                return 1;
        }
    }

    // Print the results using std::cout with 2 decimal precision
    std::cout << std::fixed << std::setprecision(2);  // Set decimal precision to 2

    // Handle the "run all tasks" option
    if (runAll) {
        for (int i = 1; i <= 4; ++i) {
            switch (i) {
                case 1: task1(assertion, verbose); break;
                case 2: task2(); break;
                case 3: task3(); break;
                case 4: task4(); break;
            }
        }
        return 0;
    }

    // Handle single task execution
    if (taskNumber == -1) {
        std::cout << "Error: Task number not specified.\n";
        print_usage();
        return 1;
    }

    // Execute the appropriate task
    switch (taskNumber) {
        case 1:
            task1(assertion, verbose);
            break;
        case 2:
            task2();
            break;
        case 3:
            task3();
            break;
        case 4:
            task4();
            break;
        default:
            std::cout << "Invalid task number! Please enter a number between 1 and 4.\n";
            return 1;
    }

    return 0;
}