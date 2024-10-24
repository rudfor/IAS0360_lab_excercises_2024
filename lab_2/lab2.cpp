#include <iostream>
#include <iomanip>  // For std::fixed and std::setprecision
#include <vector>
#include <getopt.h>  // For getopt
#include <cassert>  // For assert
#include "../lib/includes/NeuralNetwork.h"

void print_usage() {
    std::cout << "Usage: program [options]\n"
              << "Options:\n"
              << "  -t, --task <number>   Specify task number (1-3) to execute\n"
              << "  -a, --all             Run all tasks (1-3)\n"
              << "  -v, --verbose         Set verbosity\n"
              << "  -s, --assert          Set assertion\n"
              << "  -h, --help            Show this help message\n";
}

// Define constants for the layer sizes
const int OUT_LEN = 3;
const int IN_LEN = 3;
const int HID_LEN = 3;

std::vector<double> inputVector = {30.0, 87.0, 110.0};  // Input values: temp, hum, air_q

// Weights from input layer to hidden layer
std::vector<std::vector<double>> inputToHiddenWeights = {
    {-2.0, 9.5, 2.01},   // Hidden neuron 1
    {-0.8, 7.2, 6.3},    // Hidden neuron 2
    {-0.5, 0.4, 0.9}     // Hidden neuron 3
};

// Biases for hidden layer neurons
std::vector<double> hiddenBiases = {0, 0, 0};

// Weights from hidden layer to output layer
std::vector<std::vector<double>> hiddenToOutputWeights = {
    {-1.0, 1.15, 0.11},  // Sad prediction
    {-0.18, 0.15, -0.01},// Sick prediction
    {0.25, -0.25, -0.1}  // Active prediction
};

// Biases for output layer neurons
std::vector<double> outputBiases = {0, 0, 0};

// Arrays to store the outputs of the hidden layer and final output
std::vector<double> hiddenOutputs(HID_LEN);
std::vector<double> predictedOutput(OUT_LEN);
std::vector<double> error(OUT_LEN);

// Ground truth values for the output
std::vector<double> groundTruth = {600, 10, -80};

// Task 1: Compute predictions based on the input and weights
void task1(bool assert_only=false, bool verbose=false) {
    NeuralNetwork nn;

    // Step 1: Compute the hidden layer outputs
    std::vector<double> flattenedInputToHiddenWeights;
    for (const auto& row : inputToHiddenWeights) {
        flattenedInputToHiddenWeights.insert(flattenedInputToHiddenWeights.end(), row.begin(), row.end());
    }
    nn.hiddenLayer(inputVector, flattenedInputToHiddenWeights, hiddenBiases, hiddenOutputs, IN_LEN, HID_LEN);

    // Step 2: Compute the final output predictions using the hidden layer outputs
    std::vector<double> flattenedHiddenToOutputWeights;
    for (const auto& row : hiddenToOutputWeights) {
        flattenedHiddenToOutputWeights.insert(flattenedHiddenToOutputWeights.end(), row.begin(), row.end());
    }
    nn.multipleInputMultipleOutput(hiddenOutputs, flattenedHiddenToOutputWeights, outputBiases, predictedOutput, HID_LEN, OUT_LEN);

    // Print the predictions
    std::cout << "Sad prediction: " << predictedOutput[0] << std::endl;
    std::cout << "Sick prediction: " << predictedOutput[1] << std::endl;
    std::cout << "Active prediction: " << predictedOutput[2] << std::endl;
}

// Task 2: Compute predictions and calculate errors compared to ground truth
void task2() {
    NeuralNetwork nn;

    // Step 1: Compute the hidden layer outputs
    std::vector<double> flattenedInputToHiddenWeights;
    for (const auto& row : inputToHiddenWeights) {
        flattenedInputToHiddenWeights.insert(flattenedInputToHiddenWeights.end(), row.begin(), row.end());
    }
    nn.hiddenLayer(inputVector, flattenedInputToHiddenWeights, hiddenBiases, hiddenOutputs, IN_LEN, HID_LEN);

    // Step 2: Compute the final output predictions using the hidden layer outputs
    std::vector<double> flattenedHiddenToOutputWeights;
    for (const auto& row : hiddenToOutputWeights) {
        flattenedHiddenToOutputWeights.insert(flattenedHiddenToOutputWeights.end(), row.begin(), row.end());
    }
    nn.multipleInputMultipleOutput(hiddenOutputs, flattenedHiddenToOutputWeights, outputBiases, predictedOutput, HID_LEN, OUT_LEN);

    // Step 3: Calculate the errors compared to the ground truth
    nn.calculateError(predictedOutput, groundTruth, error);

    std::cout << std::fixed << std::setprecision(10);  // Set decimal precision to 2

    // Step 4: Print the errors in the required format with high precision
    std::cout << "\nError Analysis:\n";
    for (int i = 0; i < OUT_LEN; i++) {
        std::cout << "For ground truth " << groundTruth[i] << ", predicted output is " << predictedOutput[i] 
                  << ", error is " << error[i] << "\n";
    }
}

// Task 3: Brute force learning to adjust weight
void task3(double learningRate = 0.001, bool assert_only=false, bool verbose=false) {
    NeuralNetwork nn;
    double input = 0.5;           // Example input value
    double weight = 0.5;          // Initial weight
    double expectedValue = 0.8;   // Target (expected) output value
    //double learningRate = 0.001;  // Learning rate
    int maxEpochs = 1500;         // Maximum number of epochs

    std::cout << std::fixed << std::setprecision(6);  // Set decimal precision to 2

    // Perform brute-force learning to adjust the weight
    nn.bruteForceLearning(input, weight, expectedValue, learningRate, maxEpochs);
}

// int main() {
//     task1();
//     task2();
//     // task3();
//     return 0;
// }

int main(int argc, char* argv[]) {
    int option;
    int taskNumber = -1;
    double learningRate = 0.5;
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
        for (int i = 1; i <= 3; ++i) {
            switch (i) {
                case 1: task1(assertion, verbose); break;
                case 2: task2(); break;
                case 3: task3(learningRate); break;
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
            task3(learningRate);
            break;
        default:
            std::cout << "Invalid task number! Please enter a number between 1 and 3.\n";
            return 1;
    }

    return 0;
}
