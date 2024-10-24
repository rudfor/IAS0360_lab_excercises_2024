#include <iostream>
#include <iomanip>  // For std::fixed and std::setprecision
#include <vector>
#include <cmath>
#include "../lib/includes/NeuralNetwork.h"

int main() {
    NeuralNetwork nn;

    // Print the results using std::cout with 2 decimal precision
    std::cout << std::fixed << std::setprecision(8);  // Set decimal precision to 2

    // Example input (temperature, humidity, air quality)
    std::vector<double> inputVector = {30.0, 87.0, 110.0};

    // Example expected output (target value)
    std::vector<double> expectedOutput = {0.8};  // Target output for this example

    // Network weights and biases
    std::vector<std::vector<double>> inputToHiddenWeights = {
        {0.5, -0.2, 0.8}, 
        {-0.3, 0.9, 0.1}, 
        {0.7, -0.5, 0.2}
    };
    std::vector<double> hiddenBiases = {0.0, 0.0, 0.0};
    std::vector<std::vector<double>> hiddenToOutputWeights = {
        {0.3, -0.6, 0.9, 0.1}
    };
    std::vector<double> outputBiases = {0.0};

    // Hyperparameters
    double learningRate = 0.01;
    int epochs = 1000;

    // Perform backpropagation learning
    nn.backpropagation(inputVector, expectedOutput, inputToHiddenWeights, hiddenBiases, hiddenToOutputWeights, outputBiases, learningRate, epochs);

    // Test the final output after training
    std::vector<double> hiddenLayerOutput(inputToHiddenWeights.size(), 0.0);
    std::vector<double> finalOutput(hiddenToOutputWeights.size(), 0.0);

    // Forward pass to compute final output
    for (int i = 0; i < inputToHiddenWeights.size(); ++i) {
        double z = hiddenBiases[i];
        for (int j = 0; j < inputVector.size(); ++j) {
            z += inputVector[j] * inputToHiddenWeights[i][j];
        }
        hiddenLayerOutput[i] = 1.0 / (1.0 + exp(-z)); // Sigmoid activation
    }

    for (int i = 0; i < hiddenToOutputWeights.size(); ++i) {
        double z = outputBiases[i];
        for (int j = 0; j < hiddenLayerOutput.size(); ++j) {
            z += hiddenLayerOutput[j] * hiddenToOutputWeights[i][j];
        }
        finalOutput[i] = 1.0 / (1.0 + exp(-z)); // Sigmoid activation
    }

    // Print final output and compare to expected output
    std::cout << "Final output after training: " << finalOutput[0] << std::endl;
    std::cout << "Expected output: " << expectedOutput[0] << std::endl;

    return 0;
}
