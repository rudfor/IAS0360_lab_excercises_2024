/**
 * @file NeuralNetwork.cpp
 * @brief Implementation of basic neural network operations.
 *
 * This file contains various functions related to neural network computations,
 * including forward propagation, backpropagation, and weight updates.
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <limits>
#include <iomanip>  // For std::fixed and std::setprecision
#include "../includes/NeuralNetwork.h"

/**
 * @brief Computes the output of a single neuron given an input and weight.
 * 
 * This function performs a simple multiplication between the input and weight:
 * 
 * \f[
 * z = x \cdot w
 * \f]
 * 
 * where \f$x\f$ is the input and \f$w\f$ is the weight.
 * 
 * @param input Input to the neuron.
 * @param weight Weight associated with the input.
 * @return The output of the neuron.
 */
double NeuralNetwork::singleNeuron(double input, double weight) 
{
    return (input * weight);
}

/**
 * @brief Computes the output for multiple inputs and weights, with a bias term.
 * 
 * This function computes the weighted sum of inputs and adds a bias term:
 * 
 * \f[
 * z = \sum_{i=1}^{n} x_i \cdot w_i + b
 * \f]
 * 
 * where \f$x_i\f$ are the inputs, \f$w_i\f$ are the weights, and \f$b\f$ is the bias.
 * 
 * @param inputs Vector of inputs.
 * @param weights Vector of weights.
 * @param bias Bias term to be added to the weighted sum.
 * @return The output of the neuron.
 */
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

/**
 * @brief Computes the output of multiple neurons from a single input.
 * 
 * For each neuron, this function computes the output by multiplying the input
 * with each corresponding weight and adding a bias:
 * 
 * \f[
 * z_i = x \cdot w_i + b
 * \f]
 * 
 * where \f$x\f$ is the input, \f$w_i\f$ is the weight for each neuron, and \f$b\f$ is the bias.
 * 
 * @param input Single input value.
 * @param weights Vector of weights.
 * @param bias Bias term to be added to each output.
 * @param outputs Vector to store the outputs of the neurons.
 */
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

/**
 * @brief Computes the output for multiple inputs and multiple outputs.
 * 
 * This function calculates the weighted sum of inputs for each output neuron
 * and adds the corresponding bias:
 * 
 * \f[
 * z_j = \sum_{i=1}^{n} x_i \cdot w_{ji} + b_j
 * \f]
 * 
 * where \f$x_i\f$ are the inputs, \f$w_{ji}\f$ are the weights for each output neuron, and \f$b_j\f$ is the bias for each output.
 * 
 * @param inputs Vector of inputs.
 * @param weights Flattened vector of weights (stored in row-major order).
 * @param biases Vector of biases for each output.
 * @param outputs Vector to store the outputs.
 * @param inputSize Number of input neurons.
 * @param outputSize Number of output neurons.
 */
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

/**
 * @brief Computes the weighted sum of inputs for a hidden layer in the neural network.
 * 
 * This function computes the weighted sum for each neuron in the hidden layer:
 * 
 * \f[
 * z_i = \sum_{j=1}^{n} x_j \cdot w_{ij} + b_i
 * \f]
 * 
 * where \f$x_j\f$ are the inputs, \f$w_{ij}\f$ are the weights for each hidden neuron, and \f$b_i\f$ is the bias for each hidden neuron.
 * 
 * @param inputs Vector of inputs.
 * @param hiddenWeights Flattened vector of weights for the hidden layer.
 * @param hiddenBiases Vector of biases for the hidden layer.
 * @param hiddenOutputs Vector to store the outputs of the hidden layer.
 * @param inputSize Number of input neurons.
 * @param hiddenSize Number of hidden neurons.
 */
void NeuralNetwork::hiddenLayer(std::vector<double>& inputs, std::vector<double>& hiddenWeights, std::vector<double>& hiddenBiases, std::vector<double>& hiddenOutputs, int inputSize, int hiddenSize) 
{
    for (int i = 0; i < hiddenSize; i++) {
        double weighted_sum = 0.0;

        // Calculate the weighted sum for each hidden neuron
        for (int j = 0; j < inputSize; j++) {
            // Use the singleNeuron function to calculate contribution of each input
            weighted_sum += singleNeuron(inputs[j], hiddenWeights[i * inputSize + j]);
        }

        // Add bias for the hidden neuron
        weighted_sum += hiddenBiases[i];

        // Store the final output for the hidden neuron
        hiddenOutputs[i] = weighted_sum; // You may want to apply an activation function here if needed
    }
}

/**
 * @brief Computes the error between predicted and actual values.
 * 
 * This function calculates the squared error for each prediction:
 * 
 * \f[
 * e_i = (y_i - \hat{y}_i)^2
 * \f]
 * 
 * where \f$y_i\f$ is the ground truth and \f$\hat{y}_i\f$ is the predicted value.
 * 
 * @param predictedOutput Vector of predicted values.
 * @param groundTruth Vector of ground truth values.
 * @param error Vector to store the calculated errors.
 */
void NeuralNetwork::calculateError(std::vector<double>& predictedOutput, std::vector<double>& groundTruth, std::vector<double>& error)
{
    int size = predictedOutput.size(); // Assuming both vectors have the same size
    error.resize(size); // Resize the error vector to hold the error values
    for (int i = 0; i < size; i++) {
        error[i] = pow(predictedOutput[i] - groundTruth[i], 2); // Calculate squared error
    }
}

/**
 * @brief Computes the Mean Squared Error (MSE) from the error vector.
 * 
 * \f[
 * \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} e_i
 * \f]
 * 
 * @param error Vector of squared errors.
 * @return Mean Squared Error.
 */
double NeuralNetwork::calculateMSE(std::vector<double>& error) 
{
    double sum = 0.0;
    for (double e : error) {
        sum += e;
    }
    return sum / error.size();
}

/**
 * @brief Computes the Root Mean Squared Error (RMSE) from the MSE.
 * 
 * \f[
 * \text{RMSE} = \sqrt{\text{MSE}}
 * \f]
 * 
 * @param mse Mean Squared Error.
 * @return Root Mean Squared Error.
 */
double NeuralNetwork::calculateRMSE(double mse) {
    return sqrt(mse);
}

void NeuralNetwork::bruteForceLearning(double input, double& weight, double expectedValue, double learningRate, int maxEpochs) 
{
    double prediction;
    std::vector<double> predictedOutput(1); // To hold predicted output
    std::vector<double> groundTruth(1, expectedValue); // Ground truth vector
    std::vector<double> error(1); // To hold error values

    for (int epoch = 0; epoch < maxEpochs; ++epoch) {
        // Make a prediction using the current weight
        prediction = singleNeuron(input, weight); // Assuming singleNeuron computes the output
        predictedOutput[0] = prediction;

        // Calculate the error using the calculateError function
        calculateError(predictedOutput, groundTruth, error);
        
        // Output learning progress
        std::cout << "Step: " << epoch
                  << "   Error: " << error[0] 
                  << "   Prediction: " << prediction 
                  << "   Weight: " << weight << "\n";

        // Adjust the weight using the error
        weight -= learningRate * (prediction - expectedValue); // Simple gradient descent
        
        // Check for convergence (small error threshold)
        if (std::abs(error[0]) < 0.000001) {
            std::cout << "Error is close to zero, stopping early.\n";
            break;
        }
    }
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
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) 
{
    return x * (1.0 - x);  // x is the output of the sigmoid function
}

void NeuralNetwork::backpropagation(const std::vector<double>& input, const std::vector<double>& expectedOutput,
                                    std::vector<std::vector<double>>& inputToHiddenWeights, std::vector<double>& hiddenBiases,
                                    std::vector<std::vector<double>>& hiddenToOutputWeights, std::vector<double>& outputBiases,
                                    double learningRate, int epochs)
{
    int inputSize = input.size();
    int hiddenSize = inputToHiddenWeights.size();
    int outputSize = expectedOutput.size();

    std::vector<double> hiddenLayerOutput(hiddenSize);
    std::vector<double> finalOutput(outputSize);
    std::vector<double> outputError(outputSize);
    std::vector<double> hiddenError(hiddenSize);

    // Sigmoid function and its derivative
    auto sigmoid = [](double x) {
        return 1.0 / (1.0 + exp(-x));
    };

    auto sigmoidDerivative = [](double x) {
        return x * (1.0 - x);  // Derivative of sigmoid function with respect to its output
    };

    // Training for the given number of epochs
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;

        // Step 1: Forward pass
        // Compute the hidden layer output
        for (int i = 0; i < hiddenSize; ++i) {
            double z = hiddenBiases[i];
            for (int j = 0; j < inputSize; ++j) {
                z += input[j] * inputToHiddenWeights[i][j];
            }
            hiddenLayerOutput[i] = sigmoid(z);  // Apply sigmoid activation
        }

        // Compute the final output (output layer)
        for (int i = 0; i < outputSize; ++i) {
            double z = outputBiases[i];
            for (int j = 0; j < hiddenSize; ++j) {
                z += hiddenLayerOutput[j] * hiddenToOutputWeights[i][j];
            }
            finalOutput[i] = sigmoid(z);  // Apply sigmoid activation
        }

        // Step 2: Calculate the output error (difference between expected and predicted)
        for (int i = 0; i < outputSize; ++i) {
            outputError[i] = expectedOutput[i] - finalOutput[i];
            totalError += outputError[i] * outputError[i];  // Accumulate squared error
        }
        totalError *= 0.5;  // Mean squared error

        // Print the error at certain epochs
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Error: " << totalError << std::endl;
        }

        // Step 3: Backward pass
        // Compute the error gradient for the output layer
        for (int i = 0; i < outputSize; ++i) {
            outputError[i] *= sigmoidDerivative(finalOutput[i]);  // Gradient of the output
        }

        // Compute the error gradient for the hidden layer
        for (int i = 0; i < hiddenSize; ++i) {
            hiddenError[i] = 0.0;
            for (int j = 0; j < outputSize; ++j) {
                hiddenError[i] += outputError[j] * hiddenToOutputWeights[j][i];
            }
            hiddenError[i] *= sigmoidDerivative(hiddenLayerOutput[i]);  // Gradient of the hidden layer
        }

        // Step 4: Update weights and biases
        // Update weights between hidden and output layers
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < hiddenSize; ++j) {
                hiddenToOutputWeights[i][j] += learningRate * outputError[i] * hiddenLayerOutput[j];
            }
            outputBiases[i] += learningRate * outputError[i];  // Update output biases
        }

        // Update weights between input and hidden layers
        for (int i = 0; i < hiddenSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                inputToHiddenWeights[i][j] += learningRate * hiddenError[i] * input[j];
            }
            hiddenBiases[i] += learningRate * hiddenError[i];  // Update hidden biases
        }
    }

    // // After training, print the final output
    // std::cout << "Final output after training: " << finalOutput[0] << std::endl;
    // std::cout << "Expected output: " << expectedOutput[0] << std::endl;
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
