#include <iostream>
#include <iomanip>  // For std::fixed and std::setprecision
#include <vector>
#include "../lib/includes/NeuralNetwork.h"

void task1() {
    NeuralNetwork nn;
    std::vector<double> input = {12, 23, 47};
    std::vector<double> weight = {-3, -2, -3};
    double result = 0;
    double bias = 0;

    for (size_t i = 0; i < input.size(); i++) {
        result = nn.singleNeuron(input[i], weight[i]) + bias;
        std::cout << "Single Neuron Result: " << result << "\n";
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

int main() {
    task1();
    // Print the results using std::cout with 2 decimal precision
    std::cout << std::fixed << std::setprecision(2);  // Set decimal precision to 2
    task2();
    task3();
    task4();
    return 0;
}
