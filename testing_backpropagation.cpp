#include <stdio.h>

// Eksternal file
#include "data/data_training_1.txt"
#include "data/data_wb_backpropagation.txt"


int main(){

   char klasifikasi[2] = {'C', 'R'};
   int array_of_target = 1; 

    for (size_t pola = 0; pola < 20; pola++)
        {
           
            // STEP 4============================================
            // Each hidden unit (Zj,j = 1, ... ,p) 
            // printf("\n-----------------------");
            for (size_t j = 0; j < numHiddenNodes; j++)
            {

                // sums its weighted input signals,
                float z_in = 0;
                for (size_t i = 0; i < numInputs; i++)
                {

                    z_in = z_in + (x[pola][i] * weight_v[i][j] );
                }

                z_in = z_in + bias_v[j];
                // applies its activation function to compute its output signal
                z[j] = sigmoid(z_in);
                
                // printf("\nz_in - sigmoid(%f) = %f", z_in, z[j]);

            }
            //  printf("\n-----------------------");
            
            // STEP 5============================================
            // Each output unit (Yk , k = 1, ... , m) sums its weighted input signals,
            for (size_t k = 0; k < numOutputs; k++)
            {
                // sums its weighted input signals,
                float y_in = 0 ;
                for (size_t j = 0; j < numHiddenNodes; j++)
                {
                    y_in = y_in + (z[j] * weight_w[j][k]);
                }
                y_in = y_in + bias_w[k];
                // applies its activation function to compute its output signal
                // printf("\n %f", y_in);
                y[k] = reluCostum(y_in);
                // printf("\ny_in - sigmoid(%f) = %f", y_in, y[k]);
            }
            // return 0;

             printf("\n========================\n\nPattern ke - %d \t  ", pola+1, klasifikasi[array_of_target]);
             array_of_target++;
            if (pola == 9  )
             {
                 array_of_target = 1;
             }
            
            printf("\t(\t");
            for (size_t k = 0; k < 2; k++)
            {
              printf("%f,\t",  y[k]);
            }
            
            printf(")\n\n"); 
        }

    return 0;
}