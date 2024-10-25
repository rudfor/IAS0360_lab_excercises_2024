#include "../includes/Common.h"
#include <iomanip>  // For std::fixed and std::setprecision
#include <getopt.h>  // For getopt
#include <iostream>

/**
 * @brief Prints usage information for the program.
 *
 * Displays the command-line options available for the user.
 */
void Common::print_usage() {
    std::cout << "Usage: program [options]\n"
              << "Options:\n"
              << "  -t, --task <number>   Specify task number (1-4) to execute\n"
              << "  -a, --all             Run all tasks (1-X)\n"
              << "  -v, --verbose         Set verbosity\n"
              << "  -s, --assert          Set assertion\n"
              << "  -h, --help            Show this help message\n";
}

int Common::parse_options(int argc, char* argv[]) {
    int option;
    struct option long_options[] = {
        {"task", required_argument, 0, 't'},
        {"all", no_argument, 0, 'a'},
        {"verbose", no_argument, 0, 'v'},
        {"assert", no_argument, 0, 's'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}  // End of options
    };
    while ((option = getopt_long(argc, argv, "t:ahsv", long_options, nullptr)) != -1) {
        switch (option) {
            case 't':
                taskNumber = std::atoi(optarg);
                break;
            case 'a':
                runAll = true;
                break;
            case 'v':
                verbose = true;
                break;
            case 's':
                assertion = true;
                break;
            case 'h':
                print_usage();
                return 0;
            default:
                print_usage();
                return 1;
        }
    }
    return 0; // Indicate successful parsing
}
