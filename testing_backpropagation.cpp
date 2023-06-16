#include <stdio.h>

// Eksternal file
#include "data/data_testing.txt"
#include "data/data_wb_backpropagation.txt"


int main(){

    // char klasifikasi[2] = {'C', 'R'};
    // int array_of_target = 0; 
    FILE *fpp;
    fpp = fopen("data/data_log.txt", "w");
    for (size_t pola = 0; pola < 20; pola++)
        {
            fprintf(fpp, "++++++++++++++++++++++++++++++++++++=");
            fprintf(fpp, "Pola ke - %d\t\n ", pola);
           
            // STEP 4============================================
            // Each hidden unit (Zj,j = 1, ... ,p) 
            // printf("\n-----------------------");
            for (size_t j = 0; j < numHiddenNodes; j++)
            {

                // sums its weighted input signals,
                float z_in = 0;
                // fprintf(fpp, "float z_in = bias_v[j] = %f;\n", bias_v[j]);
                for (size_t i = 0; i < numInputs; i++)
                {
                    fprintf(fpp, "z_in = z_in + (x[%d][%d] * weight_v[%d][%d] ) ||||  z_in = %f \t ", pola, i, i, j, z_in);
                    z_in = z_in + (x[pola][i] * weight_v[i][j] );
                    fprintf(fpp, "+ (%d * %f ) = %f  \n",x[pola][i], weight_v[i][j], z_in);
                }

                fprintf(fpp, "z_in = z_in + bias_v[%d] = %f + %f =", j, z_in, bias_v[j] );
                z_in = z_in + bias_v[j];
                fprintf(fpp, "%f", z_in);
                // applies its activation function to compute its output signal
                z[j] = sigmoid(z_in);
                fprintf(fpp, "sigmoid (%f) = %.15f\n", z_in, z[j]);
                                // if (epoch == max_epoch)
                fprintf(fpp, "===========================================\n");
            
            }
            fprintf(fpp, "\n************----------------------**************\n");
            fprintf(fpp, "\n************----------------------**************\n");
            fprintf(fpp, "\n************----------------------**************\n");

             printf("\n-----------------------");
            
            // STEP 5============================================
            // Each output unit (Yk , k = 1, ... , m) sums its weighted input signals,
            for (size_t k = 0; k < numOutputs; k++)
            {
                // sums its weighted input signals,
                float y_in = 0 ;
                for (size_t j = 0; j < numHiddenNodes; j++)
                {
                    fprintf(fpp, "y_in = y_in + (z[%d] *  weight_w[%d][%d]) =  y_in = %f,\t ", j, j, k, y_in);
                    y_in = y_in + (z[j] * weight_w[j][k]);
                    fprintf(fpp, "+ (%f * %f ) = %f  \n",z[j], weight_w[j][k], y_in);
                }
                y_in = y_in + bias_w[k];
                // applies its activation function to compute its output signal
                // printf("\n %f", y_in);
                y[k] = treshold(sigmoid(y_in));
                printf("\ny_in - sigmoid(%f) = %f", y_in, y[k]);
            }

            printf("\t(\t");
            for (size_t k = 0; k < 2; k++)
            {
                printf("%.0f,\t",  y[k]);
            }
            
            printf(")\n\n"); 
        }

    return 0;
}