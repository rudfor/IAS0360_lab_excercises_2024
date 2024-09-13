#include <stdio.h>
#include <iostream>
#include <vector>
#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"
#include "includes/AnyLib.h"

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
    // Initialize the system and Wi-Fi chip (which includes the onboard LED control)
    if (cyw43_arch_init()) {
        // Failed to initialize
        return -1;  
    }
    bool value =0;
    while (true)
    {
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, value);
        sleep_ms(500);
        value =!value;
        printf("LED toggle\n");
        task1();
    }  
    return 0;
}