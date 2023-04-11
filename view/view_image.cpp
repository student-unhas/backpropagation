#include <stdio.h>

// Eksternal file
#include "../data/data_training_1.txt"
#include "../data/data_target.txt"

int main()
{
    printf("\n\n");
    for (int j = 0; j < 20; j++)
    {
        int index = 0, temp = 0;
        for (int i = 1; i <= 10; i++)
        {
            for (int h = 0; h < 10; h++)
            {
                (x[j][index] >= 1)? printf(" #") : printf(" .");
                
                index+=1;
            }

            // printf("\t\t");
            // index = temp;
            // for (int h = 0; h < 7; h++)
            // {
            //     (x[j+7][index] >= 1)? printf(" #") : printf(" .");
                
            //     index+=1;
            // }
            
            // printf("\t\t");
            // index = temp;
            // for (int h = 0; h < 7; h++)
            // {
            //     (x[j+14][index] >= 1)? printf(" #") : printf(" .");
                
            //     index+=1;
            // }
            // temp = index;

            printf("\n");
           
        }
       printf("\n\n");
    }
    return 0;
}