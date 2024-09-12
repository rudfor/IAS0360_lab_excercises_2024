#include <stdio.h>
#include <iostream>
#include <vector>
#include "AnyLib.h"

void task1() 
{
    MathBasics mb;
    double a = 23, b = -2;
    double result = mb.multi_a_b(a, b);
    printf("%f\n", result);
}

int main()
{
    task1();
}