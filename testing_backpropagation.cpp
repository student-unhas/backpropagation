#include <stdio.h>

// Eksternal file
#include "data/data_training.txt"
#include "data/data_wb_backpropagation.txt"


int main(){

    // char huruf[7] = {'A', 'B', 'C', 'D', 'E', 'J', 'K'};
    static const char huruf[2] = {'C', 'R'};
    int array_of_target = 0; 

    for (size_t pola = 0; pola < 20; pola++)
        {
           
            // STEP 4============================================
            // Each hidden unit (Zj,j = 1, ... ,p) 
            for (size_t j = 0; j < numHiddenNodes; j++)
            {
                float total = 0;
                for (size_t i = 0; i < 1600; i++)
                {
                    total = total + (x[pola][i] * weight_v[i][j] );
                }

                // sums its weighted input signals,
                float z_in = total + bias_v[j];

                // applies its activation function to compute its output signal
                z[j] = sigmoid(z_in);

            }
            
            // STEP 5============================================
            // Each output unit (Yk , k = 1, ... , m) sums its weighted input signals,
            for (size_t k = 0; k < numOutputs; k++)
            {
                float total = 0;
                for (size_t i = 0; i < numHiddenNodes; i++)
                {
                    total = total + (z[i] * weight_w[i][k]);
                }
                // sums its weighted input signals,
                float y_in = total + bias_w[k];

                // applies its activation function to compute its output signal
                y[k] = reluCostum(y_in);
            }  


            // printf( "\n\nPola ke - %d huruf %c\t\n",pola+1, huruf[array_of_target]);
            printf( "\n\nPola ke - %lu huruf %c\t\n",pola+1, huruf[array_of_target]);
            array_of_target++;
             if (pola < 9)
            {
                array_of_target = 0;
            } 
            if (pola > 9 && pola <= 19)
            {
                array_of_target = 1;
            } 
            printf("\t(\t");
            for (size_t k = 0; k < 2; k++)
            {
                printf("%d,\t",  y[k]);
            }
            
            printf(")\n\n"); 
        }

    return 0;
}