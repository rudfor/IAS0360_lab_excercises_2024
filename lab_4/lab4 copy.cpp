#include <iostream>
#include <vector>
#include "../lib/includes/NeuralNetwork.h"

#define NUM_OF_FEATURES    3   // Number of input features (e.g., temperature, humidity, air quality)
#define NUM_OF_HIDDEN_NODES 3  // Number of neurons in the hidden layer
#define NUM_OF_OUTPUT_NODES 1  // Number of output nodes (e.g., predicted class)

double learning_rate = 0.01;  // Learning rate for updating weights (not used directly in this example)

// Intermediate outputs and storage for the hidden layer
std::vector<double> hiddenLayerOutput(NUM_OF_HIDDEN_NODES);  // Output of the hidden layer (for each example)
std::vector<double> hiddenLayerBias = {0, 0, 0};  // Initialize biases for the hidden layer neurons
std::vector<double> hiddenLayerWeightedSum(NUM_OF_HIDDEN_NODES);  // Weighted sum (z1) before applying activation function

// Weights from input layer to hidden layer
std::vector<std::vector<double>> inputToHiddenWeights = 
{
    {0.25, 0.5, 0.05},  // Weights for hidden neuron 1
    {0.8, 0.82, 0.3},   // Weights for hidden neuron 2
    {0.5, 0.45, 0.19}   // Weights for hidden neuron 3
};

// Intermediate outputs and storage for the output layer
std::vector<double> outputLayerBias = {0};  // Initialize bias for the output neuron
std::vector<double> outputLayerWeightedSum(NUM_OF_OUTPUT_NODES);  // Weighted sum (z2) before applying activation function

// Weights from hidden layer to output layer
std::vector<std::vector<double>> hiddenToOutputWeights = 
{
    {0.48, 0.73, 0.03}  // Weights for the output neuron
};

// Predicted values after applying the sigmoid activation function
std::vector<double> predictedOutput(NUM_OF_OUTPUT_NODES);  // yhat (predicted values)

// Training data (normalized input features and expected output)
std::vector<std::vector<double>> normalizedInput(2, std::vector<double>(NUM_OF_FEATURES));  // Normalized input features for training
std::vector<std::vector<double>> expectedOutput = {{1}};  // Expected output (labels) for each training example

// Task 1: Perform a forward pass through the network
void task1() 
{
    NeuralNetwork nn;

    // Raw input features before normalization
    std::vector<std::vector<double>> rawInput = {
        {23.0, 40.0, 100.0},  // Example 1: temp, hum, air_q
        {22.0, 39.0, 101.0}   // Example 2
    };

    // Normalize the raw input data
    nn.normalizeData2D(rawInput, normalizedInput);
    
    // Debugging output for normalized input
    std::cout << "Normalized training input:\n";
    for (const auto& input : normalizedInput) {
        for (const auto& feature : input) {
            std::cout << feature << " ";
        }
        std::cout << "\n";
    }

    // Step 1: Calculate the weighted sum (z1) for the hidden layer
    std::vector<double> flattenedInputToHiddenWeights;
    for (const auto& row : inputToHiddenWeights) {
        flattenedInputToHiddenWeights.insert(flattenedInputToHiddenWeights.end(), row.begin(), row.end());
    }
    nn.multipleInputMultipleOutput(normalizedInput[0], flattenedInputToHiddenWeights, hiddenLayerBias, hiddenLayerWeightedSum, NUM_OF_FEATURES, NUM_OF_HIDDEN_NODES);
    
    // Debugging output for weighted sum (z1)
    std::cout << "Weighted sum (z1) before ReLU:\n";
    for (double val : hiddenLayerWeightedSum) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    // Step 2: Apply ReLU activation to the hidden layer's weighted sum
    nn.vectorReLU(hiddenLayerWeightedSum, hiddenLayerOutput);
    
    // Debugging output for hidden layer output
    std::cout << "Hidden layer output after ReLU:\n";
    for (double val : hiddenLayerOutput) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    // Step 3: Calculate the weighted sum (z2) for the output layer
    std::vector<double> flattenedHiddenToOutputWeights;
    for (const auto& row : hiddenToOutputWeights) {
        flattenedHiddenToOutputWeights.insert(flattenedHiddenToOutputWeights.end(), row.begin(), row.end());
    }
    nn.multipleInputMultipleOutput(hiddenLayerOutput, flattenedHiddenToOutputWeights, outputLayerBias, outputLayerWeightedSum, NUM_OF_HIDDEN_NODES, NUM_OF_OUTPUT_NODES);
    
    // Debugging output for weighted sum (z2)
    std::cout << "Weighted sum (z2) before Sigmoid:\n";
    std::cout << outputLayerWeightedSum[0] << "\n";

    // Step 4: Apply Sigmoid activation to the output layer's weighted sum
    nn.vectorSigmoid(outputLayerWeightedSum, predictedOutput);
    
    // Debugging output for predicted output
    std::cout << "Predicted output (after Sigmoid):\n";
    std::cout << predictedOutput[0] << "\n";

    // Step 5: Compute the cost (logistic regression cost function)
    double cost = nn.computeCost(1, {predictedOutput}, expectedOutput);
    std::cout << "Cost: " << cost << "\n";
}


// Task 2: Save and load the network's state
void task2() 
{
    NeuralNetwork nn;
    const std::string filename = "network_save.txt";

    // Save the network to a file
    nn.saveNetwork(filename, NUM_OF_FEATURES, NUM_OF_HIDDEN_NODES, NUM_OF_OUTPUT_NODES, inputToHiddenWeights, hiddenLayerBias, hiddenToOutputWeights, outputLayerBias);

    // Clear the weights and biases to simulate loading from a file
    for (auto& row : inputToHiddenWeights) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    std::fill(hiddenLayerBias.begin(), hiddenLayerBias.end(), 0.0);

    for (auto& row : hiddenToOutputWeights) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    std::fill(outputLayerBias.begin(), outputLayerBias.end(), 0.0);

    std::cout << "Network weights and biases cleared to zero.\n";

    // Load the network from the file
    nn.loadNetwork(filename, NUM_OF_FEATURES, NUM_OF_HIDDEN_NODES, NUM_OF_OUTPUT_NODES, inputToHiddenWeights, hiddenLayerBias, hiddenToOutputWeights, outputLayerBias);

    // Execute the network after loading the saved state
    task1();
}

int main() {
    task1();
    std::cout << "\n";
    task2();
    return 0;
}
