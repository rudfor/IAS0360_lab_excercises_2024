#include <iostream>
#include <vector>
#include "../lib/includes/NeuralNetwork.h"

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
void task1() {
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

    // Step 4: Print the errors in the required format with high precision
    std::cout << "\nError Analysis:\n";
    for (int i = 0; i < OUT_LEN; i++) {
        std::cout << "For ground truth " << groundTruth[i] << ", predicted output is " << predictedOutput[i] 
                  << ", error is " << error[i] << "\n";
    }
}

// Task 3: Brute force learning to adjust weight
void task3() {
    NeuralNetwork nn;
    double input = 0.5;           // Example input value
    double weight = 0.5;          // Initial weight
    double expectedValue = 0.8;   // Target (expected) output value
    double learningRate = 0.001;  // Learning rate
    int maxEpochs = 1500;         // Maximum number of epochs

    // Perform brute-force learning to adjust the weight
    nn.bruteForceLearning(input, weight, expectedValue, learningRate, maxEpochs);
}

int main() {
    task1();
    // task2();
    // task3();
    return 0;
}
