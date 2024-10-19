#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <limits>
#include "../includes/NeuralNetwork.h"


double NeuralNetwork::singleNeuron(double input, double weight) 
{
    return (input * weight);
}

double NeuralNetwork::multipleInputSingleOutput(std::vector<double> inputs, std::vector<double> weights, double bias) 
{
    double output = 0;

    // Calculate weighted sum of inputs and add bias
    for (size_t i = 0; i < inputs.size(); ++i) {
        output += NeuralNetwork::singleNeuron(inputs[i], weights[i]);
    }

    output += bias;  // Add the bias to the final result
    return output;
}

void NeuralNetwork::singleInputMultipleOutput(double input, std::vector<double> weights, double bias, std::vector<double>& outputs) 
{
    // Clear the outputs vector to avoid appending multiple times
    outputs.clear();

    // Loop through each weight and calculate the output using singleNeuron
    for (double weight : weights) {
        double result = singleNeuron(input, weight) + bias;  // Use the singleNeuron function and add bias
        outputs.push_back(result);  // Store the result in the outputs vector
    }
}

void NeuralNetwork::multipleInputMultipleOutput(std::vector<double>& inputs, std::vector<double>& weights, std::vector<double>& biases, std::vector<double>& outputs, int inputSize, int outputSize) 
{
    // Clear the outputs vector to avoid appending multiple times
    outputs.clear();
    
    // For each output, sum the contributions from all inputs
    for (int outputIndex = 0; outputIndex < outputSize; ++outputIndex) {
        double sum = 0.0;

        // For each input, apply the corresponding weight for this output
        for (int inputIndex = 0; inputIndex < inputSize; ++inputIndex) {
            // Index into weights: weights are stored linearly for a 2D grid (flattened row-major order)
            int weightIndex = outputIndex * inputSize + inputIndex;
            sum += singleNeuron(inputs[inputIndex], weights[weightIndex]);
        }

        // Add the bias for this output
        sum += biases[outputIndex];

        // Store the computed output
        outputs.push_back(sum);
    }
}

void NeuralNetwork::hiddenLayer(std::vector<double>& inputs, std::vector<double>& hiddenWeights, std::vector<double>& hiddenBiases, std::vector<double>& hiddenOutputs, int inputSize, int hiddenSize) 
{
    return;
}

void NeuralNetwork::calculateError(std::vector<double>& predictedOutput, std::vector<double>& groundTruth, std::vector<double>& error) 
{
    return;
}

double NeuralNetwork::calculateMSE(std::vector<double>& error) 
{
    return 0;
}

double NeuralNetwork::calculateRMSE(double mse) {
    return 0;
}

void NeuralNetwork::bruteForceLearning(double input, double& weight, double expectedValue, double learningRate, int maxEpochs) 
{
    return;
}

double NeuralNetwork::relu(double x) 
{
    return 0;
}

double NeuralNetwork::sigmoid(double x) 
{
    return 0;
}

double sigmoid(double x) 
{
    return 0;
}

double sigmoidDerivative(double x) 
{
    return 0;
}

void NeuralNetwork::backpropagation(const std::vector<double>& input, const std::vector<double>& expectedOutput,
                                    std::vector<std::vector<double>>& inputToHiddenWeights, std::vector<double>& hiddenBiases,
                                    std::vector<std::vector<double>>& hiddenToOutputWeights, std::vector<double>& outputBiases,
                                    double learningRate, int epochs) 
{
    return;
}

void backpropagation2layer(const std::vector<double>& input, const std::vector<double>& expectedOutput,
                                          std::vector<std::vector<double>>& inputToHidden1Weights, std::vector<double>& hidden1Biases,
                                          std::vector<std::vector<double>>& hidden1ToHidden2Weights, std::vector<double>& hidden2Biases,
                                          std::vector<double>& hidden2ToOutputWeights, double& outputBias,
                                          double learningRate,
                                          std::vector<std::vector<double>>& inputToHidden1WeightGradients, std::vector<double>& hidden1BiasGradients,
                                          std::vector<std::vector<double>>& hidden1ToHidden2WeightGradients, std::vector<double>& hidden2BiasGradients,
                                          std::vector<double>& hidden2ToOutputWeightGradients, double& outputBiasGradient)
{
    return;
}

void NeuralNetwork::vectorReLU(std::vector<double>& inputVector, std::vector<double>& outputVector) 
{
    return;
}

void NeuralNetwork::vectorSigmoid(std::vector<double>& inputVector, std::vector<double>& outputVector) 
{
    return;
}

void NeuralNetwork::printMatrix(int rows, int cols, const std::vector<std::vector<double>>& matrix) 
{
    return;
}

double NeuralNetwork::computeCost(int m, const std::vector<std::vector<double>>& yhat, const std::vector<std::vector<double>>& y) 
{
    return 0;
}

int NeuralNetwork::normalizeData2D(const std::vector<std::vector<double>>& inputMatrix, std::vector<std::vector<double>>& outputMatrix) 
{
    int rows = inputMatrix.size();
    int cols = inputMatrix[0].size();

    if (rows <= 1) {
        std::cerr << "ERROR: At least 2 examples are required. Current dataset length is " << rows << std::endl;
        return 1;
    } else {
        for (int j = 0; j < cols; j++) {
            double max = -9999999;
            double min = 9999999;

            // Find MIN and MAX values in the given column
            for (int i = 0; i < rows; i++) {
                if (inputMatrix[i][j] > max) {
                    max = inputMatrix[i][j];
                }
                if (inputMatrix[i][j] < min) {
                    min = inputMatrix[i][j];
                }
            }

            // Normalization
            for (int i = 0; i < rows; i++) {
                outputMatrix[i][j] = (inputMatrix[i][j] - min) / (max - min);
            }
        }
    }
    return 0;
}

void NeuralNetwork::saveNetwork(const std::string& filename, int numOfFeatures, int numOfHiddenNodes, int numOfOutputNodes,
                                std::vector<std::vector<double>>& inputToHiddenWeights, std::vector<double>& hiddenLayerBias,
                                std::vector<std::vector<double>>& hiddenToOutputWeights, std::vector<double>& outputLayerBias) {

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    file << "Hidden Layer Weights:\n";
    for (int i = 0; i < numOfHiddenNodes; i++) {
        for (int j = 0; j < numOfFeatures; j++) {
            file << inputToHiddenWeights[i][j] << " ";
        }
        file << "\n";
    }

    file << "Hidden Layer Biases:\n";
    for (int i = 0; i < numOfHiddenNodes; i++) {
        file << hiddenLayerBias[i] << " ";
    }
    file << "\n";

    file << "Output Layer Weights:\n";
    for (int i = 0; i < numOfOutputNodes; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            file << hiddenToOutputWeights[i][j] << " ";
        }
        file << "\n";
    }

    file << "Output Layer Biases:\n";
    for (int i = 0; i < numOfOutputNodes; i++) {
        file << outputLayerBias[i] << " ";
    }
    file << "\n";

    file.close();
    std::cout << "Network saved to file: " << filename << "\n";
}

void NeuralNetwork::loadNetwork(const std::string& filename, int numOfFeatures, int numOfHiddenNodes, int numOfOutputNodes,
                                std::vector<std::vector<double>>& inputToHiddenWeights, std::vector<double>& hiddenLayerBias,
                                std::vector<std::vector<double>>& hiddenToOutputWeights, std::vector<double>& outputLayerBias) {

    // Clear vectors and resize to the correct dimensions
    inputToHiddenWeights.clear();
    hiddenLayerBias.clear();
    hiddenToOutputWeights.clear();
    outputLayerBias.clear();

    inputToHiddenWeights.resize(numOfHiddenNodes, std::vector<double>(numOfFeatures));
    hiddenLayerBias.resize(numOfHiddenNodes);
    hiddenToOutputWeights.resize(numOfOutputNodes, std::vector<double>(numOfHiddenNodes));
    outputLayerBias.resize(numOfOutputNodes);

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for reading.\n";
        return;
    }

    std::string temp;

    // Read "Hidden Layer Weights:" line and skip to the next line
    std::getline(file, temp);
    for (int i = 0; i < numOfHiddenNodes; i++) {
        for (int j = 0; j < numOfFeatures; j++) {
            if (!(file >> inputToHiddenWeights[i][j])) {
                std::cerr << "Error reading input-to-hidden weight at (" << i << ", " << j << ")\n";
                return;
            }
        }
    }
    // Flush remaining newline characters
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Read "Hidden Layer Biases:" line and skip to the next line
    std::getline(file, temp);
    for (int i = 0; i < numOfHiddenNodes; i++) {
        if (!(file >> hiddenLayerBias[i])) {
            std::cerr << "Error reading hidden layer bias at index " << i << "\n";
            return;
        }
    }
    // Flush remaining newline characters
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Read "Output Layer Weights:" line and skip to the next line
    std::getline(file, temp);
    for (int i = 0; i < numOfOutputNodes; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            if (!(file >> hiddenToOutputWeights[i][j])) {
                std::cerr << "Error reading hidden-to-output weight at (" << i << ", " << j << ")\n";
                return;
            }
        }
    }
    // Flush remaining newline characters
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Read "Output Layer Biases:" line and skip to the next line
    std::getline(file, temp);
    for (int i = 0; i < numOfOutputNodes; i++) {
        if (!(file >> outputLayerBias[i])) {
            std::cerr << "Error reading output layer bias at index " << i << "\n";
            return;
        }
    }

    if (file.fail()) {
        std::cerr << "File stream encountered an error.\n";
        return;
    }

    file.close();
    std::cout << "Network loaded from file: " << filename << "\n";
}
