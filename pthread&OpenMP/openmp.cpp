#include<iostream>
#include<windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
using namespace std;

#define N 500

#define NUM_THREADS 8
float A[N][N];

void init() //初始化
{
    for (int i = 0; i < N; i++)
    {
        A[i][i] = 1.0;

        for (int j = i + 1; j < N; j++)
        {
            A[i][j] = rand();
        }
    }
    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                A[i][j] += A[k][j];
            }
        }
    }
}

void common() //普通高斯消元算法
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

void omp_static()
{
    int i = 0, j = 0, k = 0;
    float tmp = 0;
    #pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++)
    {
    #pragma omp single
        {
            tmp = A[k][k];
            for (j = k + 1; j < N; j++)
            {
                A[k][j] = A[k][j] / tmp;
            }
            A[k][k] = 1.0;
        }

        #pragma omp for
        for (i = k + 1; i < N; i++)
        {
            for (j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void omp_guided()
{
    int i = 0, j = 0, k = 0;
    float tmp = 0;
    #pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++)
    {
        #pragma omp single
        {
            tmp = A[k][k];
            for (j = k + 1; j < N; j++)
            {
                A[k][j] = A[k][j] / tmp;
            }
        }
        A[k][k] = 1.0;
        #pragma omp for schedule(guided)
        for (i = k + 1; i < N; i++)
        {
            for (j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void omp_dynamic()
{
    int i = 0, j = 0, k = 0;
    float tmp = 0;
    #pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++)
    {
        #pragma omp single
        {
            tmp = A[k][k];
            for (j = k + 1; j < N; j++)
            {
                A[k][j] = A[k][j] / tmp;
            }
        }
        A[k][k] = 1.0;
        #pragma omp for schedule(dynamic)
        for (i = k + 1; i < N; i++)
        {
            for (j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void avx_omp_static()
{
    int i = 0, j = 0, k = 0;

    #pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++)
    {
        #pragma omp single
        {
            __m256 t1 = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= N; j += 8)
            {
                __m256 t2 = _mm256_loadu_ps(&A[k][j]);
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(&A[k][j], t2);
            }
            for (; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
        #pragma omp for schedule(static)
        for (i = k + 1; i < N; i++)
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
}

void avx_omp_dynamic()
{
    int i = 0, j = 0, k = 0;

    #pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++)
    {
        #pragma omp single
        {
            __m256 t1 = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= N; j += 8)
            {
                __m256 t2 = _mm256_loadu_ps(&A[k][j]);
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(&A[k][j], t2);
            }
            for (; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
        #pragma omp for schedule(dynamic)
        for (i = k + 1; i < N; i++)
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
}

void avx_optimized()
{
    for (int k = 0; k < N; k++)
    {
        __m256 t1 = _mm256_set1_ps(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= N; j += 8)
        {
            __m256 t2 = _mm256_loadu_ps(&A[k][j]);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(&A[k][j], t2);
        }
        for (; j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
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
}

void avx_omp_guided()
{
    int i = 0, j = 0, k = 0;

    #pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++)
    {
        #pragma omp single
        {
            __m256 t1 = _mm256_set1_ps(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= N; j += 8)
            {
                __m256 t2 = _mm256_loadu_ps(&A[k][j]);
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(&A[k][j], t2);
            }
            for (; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
        #pragma omp for schedule(guided)
        for (i = k + 1; i < N; i++)
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
}

long long head, tail, freq;
double time(void (*Func)())
{
    double sum;
    for(int m=0;m<10;m++)
    {
        init();
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        Func();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        sum+=(tail - head) * 1000 / freq;
    }
    return sum/10;
}


int main() {
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    cout << "平凡算法" << time(common) << "ms" << endl;

    cout << "static" << time(omp_static) << "ms" << endl;

    cout << "dynamic" << time(omp_dynamic) << "ms" << endl;

    cout << "guided" << time(omp_guided) << "ms" << endl;

    cout << "avx" << time(avx_optimized) << "ms" << endl;

    cout << "avx_OpenMP_static" << time(avx_omp_static) << "ms" << endl;

    cout << "avx_OpenMP_dynamic" << time(avx_omp_dynamic) << "ms" << endl;

    cout << "avx_OpenMP_guided" << time(avx_omp_guided) << "ms" << endl;
}

