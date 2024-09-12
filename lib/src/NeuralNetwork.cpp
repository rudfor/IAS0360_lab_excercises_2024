#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include "../includes/NeuralNetwork.h"

double NeuralNetwork::singleNeuron(double input, double weight) 
{
    return 0;
}

double NeuralNetwork::multipleInputSingleOutput(std::vector<double> inputs, std::vector<double> weights, double bias) 
{
    return 0;
}

void NeuralNetwork::singleInputMultipleOutput(double input, std::vector<double> weights, double bias, std::vector<double>& outputs) 
{
    return;
}

void NeuralNetwork::multipleInputMultipleOutput(std::vector<double>& inputs, std::vector<double>& weights, std::vector<double>& biases, std::vector<double>& outputs, int inputSize, int outputSize) 
{
    return;
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

void NeuralNetwork::vectorReLU(std::vector<double>& inputVector, std::vector<double>& outputVector) 
{
    return;
}

void NeuralNetwork::vectorSigmoid(std::vector<double>& inputVector, std::vector<double>& outputVector) 
{
    return;
}

void NeuralNetwork::saveNetwork(const std::string& filename, int numOfFeatures, int numOfHiddenNodes, int numOfOutputNodes,
                                std::vector<std::vector<double>>& inputToHiddenWeights, std::vector<double>& hiddenLayerBias,
                                std::vector<std::vector<double>>& hiddenToOutputWeights, std::vector<double>& outputLayerBias) 
{

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
                                std::vector<std::vector<double>>& hiddenToOutputWeights, std::vector<double>& outputLayerBias) 
{

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for reading.\n";
        return;
    }

    std::string temp;
    file >> temp >> temp; // Skip "Hidden Layer Weights:"
    for (int i = 0; i < numOfHiddenNodes; i++) {
        for (int j = 0; j < numOfFeatures; j++) {
            file >> inputToHiddenWeights[i][j];
        }
    }

    file >> temp >> temp; // Skip "Hidden Layer Biases:"
    for (int i = 0; i < numOfHiddenNodes; i++) {
        file >> hiddenLayerBias[i];
    }

    file >> temp >> temp; // Skip "Output Layer Weights:"
    for (int i = 0; i < numOfOutputNodes; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            file >> hiddenToOutputWeights[i][j];
        }
    }

    file >> temp >> temp; // Skip "Output Layer Biases:"
    for (int i = 0; i < numOfOutputNodes; i++) {
        file >> outputLayerBias[i];
    }

    file.close();
    std::cout << "Network loaded from file: " << filename << "\n";
}
