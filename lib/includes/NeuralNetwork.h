#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>

class NeuralNetwork {
public:
    // Single neuron calculation
    double singleNeuron(double input, double weight);

    // Multiple inputs, single output
    double multipleInputSingleOutput(std::vector<double> inputs, std::vector<double> weights, double bias);

    // Single input, multiple outputs
    void singleInputMultipleOutput(double input, std::vector<double> weights, double bias, std::vector<double>& outputs);

    // Multiple inputs, multiple outputs
    void multipleInputMultipleOutput(std::vector<double>& inputs, std::vector<double>& weights, std::vector<double>& biases, std::vector<double>& outputs, int inputSize, int outputSize);

    // Hidden layer function
    void hiddenLayer(std::vector<double>& inputs, std::vector<double>& hiddenWeights, std::vector<double>& hiddenBiases, std::vector<double>& hiddenOutputs, int inputSize, int hiddenSize);

    // Error calculation
    void calculateError(std::vector<double>& predictedOutput, std::vector<double>& groundTruth, std::vector<double>& error);

    // Mean Squared Error (MSE)
    double calculateMSE(std::vector<double>& error);

    // Root Mean Squared Error (RMSE)
    double calculateRMSE(double mse);

    // Brute-force learning
    void bruteForceLearning(double input, double& weight, double expectedValue, double learningRate, int maxEpochs);

    // Activation functions (ReLU and Sigmoid)
    double relu(double x);
    double sigmoid(double x);

    // Vectorized activation functions
    void vectorReLU(std::vector<double>& inputVector, std::vector<double>& outputVector);
    void vectorSigmoid(std::vector<double>& inputVector, std::vector<double>& outputVector);

    // Save network
    void saveNetwork(const std::string& filename, int numOfFeatures, int numOfHiddenNodes, int numOfOutputNodes,
                     std::vector<std::vector<double>>& inputToHiddenWeights, std::vector<double>& hiddenLayerBias,
                     std::vector<std::vector<double>>& hiddenToOutputWeights, std::vector<double>& outputLayerBias);

    // Load network
    void loadNetwork(const std::string& filename, int numOfFeatures, int numOfHiddenNodes, int numOfOutputNodes,
                     std::vector<std::vector<double>>& inputToHiddenWeights, std::vector<double>& hiddenLayerBias,
                     std::vector<std::vector<double>>& hiddenToOutputWeights, std::vector<double>& outputLayerBias);
};

#endif // NEURALNETWORK_H
