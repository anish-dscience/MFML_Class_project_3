#include <stdio.h>
#include <math.h>

#define MAX 150
#define FEATURES 4
#define EPS 0.000001

/* Display a matrix */
void displayMatrix(double A[][FEATURES], int rows, int cols)
{
    int i, j;

    for(i = 0; i < rows; i++)
    {
        for(j = 0; j < cols; j++)
            printf("%10.4f ", A[i][j]);

        printf("\n");
    }
}

/* Display a vector */
void displayVector(double v[], int n)
{
    int i;

    for(i = 0; i < n; i++)
        printf("%10.6f ", v[i]);

    printf("\n");
}

/* Sort eigenvalues in descending order
   and arrange corresponding eigenvectors */
void sortEigen(double eigen[], double V[][FEATURES], int n)
{
    int i, j, k;
    double temp;

    for(i = 0; i < n - 1; i++)
    {
        k = i;

        for(j = i + 1; j < n; j++)
        {
            if(eigen[j] > eigen[k])
                k = j;
        }

        if(k != i)
        {
            temp = eigen[i];
            eigen[i] = eigen[k];
            eigen[k] = temp;

            for(j = 0; j < n; j++)
            {
                temp = V[j][i];
                V[j][i] = V[j][k];
                V[j][k] = temp;
            }
        }
    }
}

/* Jacobi method for eigenvalues/eigenvectors */
void jacobi(double A[][FEATURES],
            double eigen[],
            double V[][FEATURES],
            int n)
{
    int i, j, p, q, iter;
    double max, theta, c, s;
    double app, aqq, apq;

    /* Initialize V as identity matrix */
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
        {
            if(i == j)
                V[i][j] = 1.0;
            else
                V[i][j] = 0.0;
        }
    }

    for(iter = 0; iter < 100; iter++)
    {
        max = 0.0;
        p = 0;
        q = 1;

        /* Find largest off-diagonal element */
        for(i = 0; i < n; i++)
        {
            for(j = i + 1; j < n; j++)
            {
                if(fabs(A[i][j]) > max)
                {
                    max = fabs(A[i][j]);
                    p = i;
                    q = j;
                }
            }
        }

        if(max < EPS)
            break;

        theta = 0.5 * atan2(2.0 * A[p][q],
                            A[p][p] - A[q][q]);

        c = cos(theta);
        s = sin(theta);

        app = A[p][p];
        aqq = A[q][q];
        apq = A[p][q];

        A[p][p] = c*c*app - 2*s*c*apq + s*s*aqq;
        A[q][q] = s*s*app + 2*s*c*apq + c*c*aqq;

        A[p][q] = 0.0;
        A[q][p] = 0.0;

        for(i = 0; i < n; i++)
        {
            if(i != p && i != q)
            {
                double aip = A[i][p];
                double aiq = A[i][q];

                A[i][p] = c*aip - s*aiq;
                A[p][i] = A[i][p];

                A[i][q] = s*aip + c*aiq;
                A[q][i] = A[i][q];
            }
        }

        /* Update eigenvectors */
        for(i = 0; i < n; i++)
        {
            double vip = V[i][p];
            double viq = V[i][q];

            V[i][p] = c*vip - s*viq;
            V[i][q] = s*vip + c*viq;
        }
    }

    for(i = 0; i < n; i++)
        eigen[i] = A[i][i];

    sortEigen(eigen, V, n);
}

/* Matrix multiplication */
void multiply(double A[][FEATURES],
              double B[][FEATURES],
              double C[][FEATURES],
              int r1, int c1, int c2)
{
    int i, j, k;

    for(i = 0; i < r1; i++)
    {
        for(j = 0; j < c2; j++)
        {
            C[i][j] = 0;

            for(k = 0; k < c1; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    }
}

/* Transpose */
void transpose(double A[][FEATURES],
               double T[][FEATURES],
               int rows,
               int cols)
{
    int i, j;

    for(i = 0; i < rows; i++)
    {
        for(j = 0; j < cols; j++)
            T[j][i] = A[i][j];
    }
}

int main()
{
    int n, i, j, k;
    double X[MAX][FEATURES];

    double mean[FEATURES];
    double Xc[MAX][FEATURES];

    double Xt[FEATURES][FEATURES];
    double AtA[FEATURES][FEATURES];

    double eigen[FEATURES];
    double V[FEATURES][FEATURES];

    double sigma[FEATURES];

    double U[MAX][FEATURES];

    double Z[MAX][FEATURES];
    double reconstructed[MAX][FEATURES];

    double totalVariance = 0;
    double explained[FEATURES];

    double error;
    double diff;

    printf("========================================\n");
    printf(" PCA USING SVD\n");
    printf("========================================\n");

    printf("\nEnter number of data rows (maximum 150): ");
    scanf("%d", &n);

    printf("\nEnter the 4 Iris features:\n");
    printf("1. Sepal Length\n");
    printf("2. Sepal Width\n");
    printf("3. Petal Length\n");
    printf("4. Petal Width\n\n");

    for(i = 0; i < n; i++)
    {
        printf("Row %d: ", i + 1);

        for(j = 0; j < FEATURES; j++)
            scanf("%lf", &X[i][j]);
    }

    /* -------------------------------------------------
       STEP 1: Calculate mean
       ------------------------------------------------- */

    for(j = 0; j < FEATURES; j++)
        mean[j] = 0;

    for(i = 0; i < n; i++)
    {
        for(j = 0; j < FEATURES; j++)
            mean[j] += X[i][j];
    }

    for(j = 0; j < FEATURES; j++)
        mean[j] /= n;

    printf("\n\nMean Vector:\n");
    displayVector(mean, FEATURES);

    /* -------------------------------------------------
       STEP 2: Center the data
       ------------------------------------------------- */

    for(i = 0; i < n; i++)
    {
        for(j = 0; j < FEATURES; j++)
            Xc[i][j] = X[i][j] - mean[j];
    }

    printf("\nCentered Matrix:\n");

    /* Display only first 10 rows */
    for(i = 0; i < n && i < 10; i++)
    {
        for(j = 0; j < FEATURES; j++)
            printf("%10.4f ", Xc[i][j]);

        printf("\n");
    }

    if(n > 10)
        printf("... remaining rows not displayed ...\n");

    /* -------------------------------------------------
       STEP 3: Calculate Xc^T Xc
       ------------------------------------------------- */

    for(i = 0; i < FEATURES; i++)
    {
        for(j = 0; j < FEATURES; j++)
        {
            AtA[i][j] = 0;

            for(k = 0; k < n; k++)
                AtA[i][j] += Xc[k][i] * Xc[k][j];
        }
    }

    printf("\nMatrix Xc^T Xc:\n");

    for(i = 0; i < FEATURES; i++)
    {
        for(j = 0; j < FEATURES; j++)
            printf("%10.4f ", AtA[i][j]);

        printf("\n");
    }

    /* -------------------------------------------------
       STEP 4: Eigenvalues and Eigenvectors
       ------------------------------------------------- */

    jacobi(AtA, eigen, V, FEATURES);

    printf("\nEigenvalues:\n");
    displayVector(eigen, FEATURES);

    printf("\nEigenvectors (V):\n");

    for(i = 0; i < FEATURES; i++)
    {
        for(j = 0; j < FEATURES; j++)
            printf("%10.6f ", V[i][j]);

        printf("\n");
    }

    /* -------------------------------------------------
       STEP 5: Singular values
       ------------------------------------------------- */

    for(i = 0; i < FEATURES; i++)
    {
        if(eigen[i] > 0)
            sigma[i] = sqrt(eigen[i]);
        else
            sigma[i] = 0;
    }

    printf("\nSingular Values:\n");
    displayVector(sigma, FEATURES);

    /* -------------------------------------------------
       STEP 6: Calculate U = Xc V / sigma
       ------------------------------------------------- */

    for(i = 0; i < n; i++)
    {
        for(j = 0; j < FEATURES; j++)
        {
            U[i][j] = 0;

            for(k = 0; k < FEATURES; k++)
                U[i][j] += Xc[i][k] * V[k][j];

            if(sigma[j] > EPS)
                U[i][j] /= sigma[j];
            else
                U[i][j] = 0;
        }
    }

    printf("\nU Matrix (first 10 rows):\n");

    for(i = 0; i < n && i < 10; i++)
    {
        for(j = 0; j < FEATURES; j++)
            printf("%10.6f ", U[i][j]);

        printf("\n");
    }

    /* -------------------------------------------------
       STEP 7: Explained variance
       ------------------------------------------------- */

    for(i = 0; i < FEATURES; i++)
        totalVariance += eigen[i];

    printf("\nExplained Variance Ratio:\n");

    for(i = 0; i < FEATURES; i++)
    {
        explained[i] = eigen[i] / totalVariance;

        printf("PC%d = %.4f%%\n",
               i + 1,
               explained[i] * 100);
    }

    /* -------------------------------------------------
       STEP 8: PCA for k = 2
       Z = Xc Vk
       ------------------------------------------------- */

    k = 2;

    for(i = 0; i < n; i++)
    {
        for(j = 0; j < k; j++)
        {
            Z[i][j] = 0;

            for(int p = 0; p < FEATURES; p++)
                Z[i][j] += Xc[i][p] * V[p][j];
        }
    }

    printf("\nReduced Data using first 2 Principal Components:\n");

    for(i = 0; i < n && i < 10; i++)
    {
        printf("Row %d: ", i + 1);

        for(j = 0; j < k; j++)
            printf("%10.6f ", Z[i][j]);

        printf("\n");
    }

    if(n > 10)
        printf("... remaining rows not displayed ...\n");

    /* -------------------------------------------------
       STEP 9: Reconstruction
       Xreconstructed = Z Vk^T + mean
       ------------------------------------------------- */

    for(i = 0; i < n; i++)
    {
        for(j = 0; j < FEATURES; j++)
        {
            reconstructed[i][j] = mean[j];

            for(int p = 0; p < k; p++)
                reconstructed[i][j] += Z[i][p] * V[j][p];
        }
    }

    printf("\nReconstructed Data using k = 2:\n");

    for(i = 0; i < n && i < 10; i++)
    {
        for(j = 0; j < FEATURES; j++)
            printf("%10.4f ", reconstructed[i][j]);

        printf("\n");
    }

    if(n > 10)
        printf("... remaining rows not displayed ...\n");

    /* -------------------------------------------------
       STEP 10: Reconstruction error
       ------------------------------------------------- */

    error = 0;

    for(i = 0; i < n; i++)
    {
        for(j = 0; j < FEATURES; j++)
        {
            diff = X[i][j] - reconstructed[i][j];
            error += diff * diff;
        }
    }

    printf("\nReconstruction Error for k = 2: %.6f\n", error);

    /* -------------------------------------------------
       STEP 11: Compare reconstruction errors
       ------------------------------------------------- */

    printf("\nReconstruction Error for different k:\n");

    for(k = 1; k <= FEATURES; k++)
    {
        error = 0;

        for(i = 0; i < n; i++)
        {
            for(j = 0; j < FEATURES; j++)
            {
                reconstructed[i][j] = mean[j];

                for(int p = 0; p < k; p++)
                {
                    double value = 0;

                    for(int q = 0; q < FEATURES; q++)
                        value += Xc[i][q] * V[q][p];

                    reconstructed[i][j] += value * V[j][p];
                }

                diff = X[i][j] - reconstructed[i][j];

                error += diff * diff;
            }
        }

        printf("k = %d  -> Error = %.6f\n", k, error);
    }

    printf("\n========================================\n");
    printf(" Program completed successfully.\n");
    printf("========================================\n");

    return 0;
}