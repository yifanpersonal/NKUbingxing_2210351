#include<iostream>
#include <stdio.h>
#include<cstring>
#include<typeinfo>
#include <stdlib.h>
#include<cmath>
#include<mpi.h>
#include<windows.h>
#include<omp.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>

using namespace std;
#define N 500
#define NUM_THREADS 7
float** A = NULL;

long long head, tail, freq;

void A_init() 
{
    A = new float* [N];
    for (int i = 0; i < N; i++)
    {
        A[i] = new float[N];
    }
    for (int i = 0; i < N; i++) 
    {
        A[i][i] = 1.0;
        for (int j = i + 1; j < N; j++) 
        {
            A[i][j] = rand() % 5000;
        }

    }
    for (int k = 0; k < N; k++) 
    {
        for (int i = k + 1; i < N; i++) 
        {
            for (int j = 0; j < N; j++) 
            {
                A[i][j] += A[k][j];
                A[i][j] = (int)A[i][j] % 5000;
            }
        }
    }
}
void A_initAsEmpty() 
{
    A = new float* [N];
    for (int i = 0; i < N; i++)
    {
        A[i] = new float[N];
        memset(A[i], 0, N * sizeof(float));
    }

}

void deleteA()
{
    for (int i = 0; i < N; i++) 
    {
        delete[] A[i];
    }
    delete A;
}

void common()
{    
    for (int k = 0; k < N; k++) 
    {
        for (int j = k + 1; j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;

        for (int i = k + 1; i < N; i++) 
        {
            for (int j = k + 1; j < N; j++) 
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

double mpi_pipeline(int argc, char* argv[]) //流水线划分
{  
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int num = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / num * rank;
    int end = (rank == num - 1) ? N : N / num * (rank + 1);
    if (rank == 0) //0号进程初始化矩阵
    {  
        A_init();
        for (j = 1; j < num; j++) 
        {
            int b = j * (N / num), e = (j == num - 1) ? N : (j + 1) * (N / num);
            for (i = b; i < e; i++)
            {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);
            }
        }
    }
    else {
        A_initAsEmpty();
        for (i = begin; i < end; i++) 
        {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) 
    {
        if ((begin <= k && k < end))
        {
            for (j = k + 1; j < N; j++) 
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = rank + 1; j < num; j++) 
            { 
                MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
            if (k == end - 1)
                break; 
        }
        else
        {
            int s = k / (N / num);
            //MPI_Request request;
            MPI_Recv(&A[k][0], N, MPI_FLOAT, s, 0, MPI_COMM_WORLD, &status);
        }
        for (i = max(begin, k + 1); i < end; i++) 
        {
            for (j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	
    if (rank == num - 1) 
    {
        end_time = MPI_Wtime();
        printf("流水线优化耗时：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
    return end_time - start_time;
}

double mpi(int argc, char* argv[]) //块划分
{  
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int num = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / num * rank;
    int end = (rank == num - 1) ? N : N / num * (rank + 1);
    if (rank == 0) 
    {  
        A_init();
        for (j = 1; j < num; j++)
        {
            int b = j * (N / num), e = (j == num - 1) ? N : (j + 1) * (N / num);
            for (i = b; i < e; i++) 
            {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);
            }
        }
    }
    else {
        A_initAsEmpty();
        for (i = begin; i < end; i++)
        {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) 
    {
        if ((begin <= k && k < end)) 
        {
            for (j = k + 1; j < N; j++) 
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = 0; j < num; j++) 
            {
                if (j != rank)
                    MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
        }
        else {
            int s;
            if (k < N / num * num)
                s = k / (N / num);
            else
                s = num - 1;
            MPI_Recv(&A[k][0], N, MPI_FLOAT, s, 0, MPI_COMM_WORLD, &status);
        }
        for (i = max(begin, k + 1); i < end; i++) 
        {
            for (j = k + 1; j < N; j++) 
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) 
    {
        end_time = MPI_Wtime();
        printf("块划分耗时：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
    return end_time - start_time;
}

double mpi_async(int argc, char* argv[])//非阻塞通信
{  
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int num = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / num * rank;
    int end = (rank == num - 1) ? N : N / num * (rank + 1);
    if (rank == 0) 
    {  
        A_init();
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < num; j++) {
            int b = j * (N / num), e = (j == num - 1) ? N : (j + 1) * (N / num);

            for (i = b; i < e; i++) {
                MPI_Isend(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);
            }

        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); 
    }
    else
    {
        A_initAsEmpty();
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) 
        {
            MPI_Irecv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]); 
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) 
    {
        if ((begin <= k && k < end))
        {
            for (j = k + 1; j < N; j++) 
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            MPI_Request* request = new MPI_Request[num - 1 - rank]; 
            for (j = rank + 1; j < num; j++) 
            { 
                MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);
            }
            MPI_Waitall(num - 1 - rank, request, MPI_STATUS_IGNORE);
            if (k == end - 1)
                break; 
        }
        else 
        {
            int s = k / (N / num);
            MPI_Request request;
            MPI_Irecv(&A[k][0], N, MPI_FLOAT, s, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);  
        }
        for (i = max(begin, k + 1); i < end; i++) 
        {
            for (j = k + 1; j < N; j++) 
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == num - 1) 
    {
        end_time = MPI_Wtime();
        printf("平凡MPI，块划分+非阻塞耗时：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
    return end_time - start_time;
}

double mpi_plus(int argc, char* argv[]) 
{
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int num = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / num * rank;
    int end = (rank == num - 1) ? N : N / num * (rank + 1);
    if (rank == 0)
    {  
        A_init();

        for (j = 1; j < num; j++) 
        {
            int b = j * (N / num), e = (j == num - 1) ? N : (j + 1) * (N / num);
            for (i = b; i < e; i++) 
            {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);
            }
        }
    }
    else {
        A_initAsEmpty();
        for (i = begin; i < end; i++) 
        {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++)
    {
        if ((begin <= k && k < end)) 
        {
            for (j = k + 1; j < N; j++) 
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = rank + 1; j < num; j++) 
            { 
                MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
            if (k == end - 1)
                break;
        }
        else
        {
            int s = k / (N / num);
            MPI_Recv(&A[k][0], N, MPI_FLOAT, s, 0, MPI_COMM_WORLD, &status);
        }
        for (i = max(begin, k + 1); i < end; i++) 
        {
            for (j = k + 1; j < N; j++) 
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == num - 1) 
    {
        end_time = MPI_Wtime();
        printf("平凡MPI，块划分优化耗时：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
    return end_time - start_time;
}

double mpi_circle(int argc, char* argv[])
{
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int num = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) 
    { 
        A_init();
        for (j = 1; j < num; j++) 
        {
            for (i = j; i < N; i += num) 
            {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);
            }
        }
    }
    else 
    {
        A_initAsEmpty();
        for (i = rank; i < N; i += num)
        {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++)
    {
        if (k % num == rank) 
        {
            for (j = k + 1; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = 0; j < num; j++) 
            { 
                if (j != rank)
                    MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
        }
        else 
        {
            int s = k % num;
            MPI_Recv(&A[k][0], N, MPI_FLOAT, s, 0, MPI_COMM_WORLD, &status);
        }
        int begin = k;
        while (begin % num != rank)
            begin++;
        for (i = begin; i < N; i += num) 
        {
            for (j = k + 1; j < N; j++) 
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) 
    {
        end_time = MPI_Wtime();
        printf("平凡MPI,循环划分耗时：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
    return end_time - start_time;
}

double mpi_async_multithread(int argc, char* argv[]) 
{ 
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    cout << MPI_Wtick();
    int num = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / num * rank;
    int end = (rank == num - 1) ? N : N / num * (rank + 1);
    if (rank == 0) 
    {
        A_init();
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < num; j++) 
        {
            int b = j * (N / num), e = (j == num - 1) ? N : (j + 1) * (N / num);
            for (i = b; i < e; i++)
            {
                MPI_Isend(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);
            }
        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); 
    }
    else 
    {
        A_initAsEmpty();
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) 
        {
            MPI_Irecv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
#pragma omp parallel num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) 
    {
#pragma omp single
        {
            if ((begin <= k && k < end)) 
            {
                __m256 t1 = _mm256_set1_ps(A[k][k]);
                for (j = k + 1; j + 8 <= N; j += 8) {
                    __m256 t2 = _mm256_loadu_ps(&A[k][j]);
                    t2 = _mm256_div_ps(t2, t1);
                    _mm256_storeu_ps(&A[k][j], t2);
                }
                for (; j < N; j++) 
                {
                    A[k][j] = A[k][j] / A[k][k];
                }
                A[k][k] = 1.0;
                MPI_Request* request = new MPI_Request[num - 1 - rank]; 
                for (j = 0; j < num; j++) 
                { 
                    MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);
                }
                MPI_Waitall(num - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else 
            {
                int s;
                if (k < N / num * num)
                    s = k / (N / num);
                else
                    s = num - 1;
                MPI_Request request;
                MPI_Irecv(&A[k][0], N, MPI_FLOAT, s, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
        }
#pragma omp for schedule(guided) 
        for (i = max(begin, k + 1); i < end; i++) 
        {
            __m256 vik = _mm256_set1_ps(A[i][k]); 
            for (j = k + 1; j + 8 <= N; j += 8)
            {
                __m256 vkj = _mm256_loadu_ps(&A[k][j]);
                __m256 vij = _mm256_loadu_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&A[i][j], vij);
            }
            for (; j < N; j++) 
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == num - 1) 
    {
        end_time = MPI_Wtime();
        printf("平凡MPI，块划分+非阻塞+OpenMP+AVX耗时：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
    return end_time - start_time;
}

void call(void(*func)()) 
{
    A_init();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    func();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
}


int main(int argc, char* argv[]) {
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    /*call(common);
    cout << "平凡算法耗时：" << (tail - head) * 1000 / freq << "ms" << endl;
    deleteA();*/
    //mpi_pipeline(argc, argv);
    //mpi_plus(argc, argv);
    //mpi_async_multithread(argc, argv);
    //mpi_circle(argc, argv);
    mpi(argc,argv);
    //deleteA();
    //mpi_async(argc, argv);
}


