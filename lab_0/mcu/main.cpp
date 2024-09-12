#include <stdio.h>
#include <iostream>
#include <vector>
#include "pico/stdlib.h"
#include "includes/AnyLib.h"

const uint LED_PIN = 25;

void task1() 
{
    MathBasics mb;
    double a = 23, b = -2;
    double result = mb.multi_a_b(a, b);
    printf("%f\n", result);
}

int main()
{
    stdio_init_all();
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN,GPIO_OUT);
    bool value =0;
    while (true)
    {
        gpio_put(LED_PIN,value);
        sleep_ms(500);
        value =!value;
        printf("LED toggle\n");
        task1();
    }  
    return 0;
}