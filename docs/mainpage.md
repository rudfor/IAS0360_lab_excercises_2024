Project Build and Execution Guide

This guide explains how to build and run various components of the project using the Makefile. The Makefile provides targets for labs, homework modules, and test setups. Below are the key sections and their uses:

## 1. Basic Commands

- **Build Lab**: `make build_labX` - Builds the specified lab directory (where `X` is the lab number).
- **Run Lab**: `make labX` - Builds and executes tasks for the specified lab.
- **Push Changes**: `make push` - Pushes changes to the GitHub repository.
- **Build Homework**: `make build_hwX` - Builds a specified homework assignment.
- **Run Homework**: `make hwX` - Builds and executes a specified homework assignment.
- **Test**: `make test` - Runs all configured tests and displays testing information.

## 2. Task Control

The Makefile includes control structures for task execution based on lab or homework module:
- **TASKS**: Defines default tasks that the lab targets will run.
- **Task Variants**:
  - `TASKS_4`: Tasks for Lab 4.
  - `TASKS_3`: Tasks for Lab 3.
  - `TASKS_1`: Task for Lab 1 only.
- **TASKS_PARAM**: Controls task parameters such as learning rate.

## 3. Key Targets

- `make all`: Builds the default lab setup, which currently runs `lab1`.
- `make clean`: Cleans up generated files from the labs.
- `make debug`: Displays color-coded debug information.
- `make stat`: Outputs environment information such as user, host, and OS type.

## 4. Output Styling and Debugging

This Makefile provides color-coded output for warnings, errors, loading messages, and testing statuses. Color schemes are defined in the beginning with variables such as `ERR_COLOR`, `WRN_COLOR`, and `TS1_COLOR`.

- **Error Messages**: Highlighted in red.
- **Warnings**: Highlighted in orange.
- **Status Messages**: Different color codes for testing, loading, and general messages.

## Example Usages

To build and run `lab1`:
```shell
make lab1
```

```shell
<span style="color:orange;">-= Building lab_1 =-</span>
 -= Building lab_1 =-
make[1]: Nothing to be done for 'all'.
 -= Running lab_1 =-
 -= Running task_1 =-
 | cd lab_1 && ./lab1 1  |
Single Neuron Result: -36.00
Single Neuron Result: -46.00
Single Neuron Result: -141.00
 -= Running task_2 =-
 | cd lab_1 && ./lab1 2  |
Multiple Input Single Output Result: 156.00
Multiple Input Single Output Result: 135.00
Multiple Input Single Output Result: 147.00
Multiple Input Single Output Result: 337.00
Multiple Input Single Output Result: 188.00
 -= Running task_3 =-
 | cd lab_1 && ./lab1 3  |
Single Input Multiple Outputs Result: -18.18, 85.50, 180.90
 -= Running task_4 =-
 | cd lab_1 && ./lab1 4  |
Multiple Input Multiple Outputs Result: 986.50, 1295.40, 118.80
```
