/*Example code to work with the dataset*/

#include "../lib/includes/Dataset.h"
#include <iostream>

int main() {
    Dataset dataset;

    // Load the dataset
    dataset.processDataset("titanic_dataset.csv");

    // Set input columns (Pclass, Sex, Age, Fare; 1-based indexing)
    dataset.inputColumns({3, 5, 6, 10});

    // Set output column (Survived; 1-based indexing)
    dataset.outputColumn(2);

    // Shuffle the dataset
    dataset.setShuffleSeed(42);

    // Divide into train and test sets
    dataset.divideTrainAndTestData();

    // Get a sample of training data
    auto trainSample = dataset.getTrainDataSample(5);
    std::cout << "Training Data Sample:\n";
    for (const auto& row : trainSample) {
        for (const auto& col : row) {
            std::cout << col << " ";
        }
        std::cout << "\n";
    }


    // Get a sample of testing data
    auto testSample = dataset.getTestDataSample(3);
    std::cout << "Testing Data Sample:\n";
    for (const auto& row : testSample) {
        for (const auto& col : row) {
            std::cout << col << " ";
        }
        std::cout << "\n";
    }


    return 0;
}


