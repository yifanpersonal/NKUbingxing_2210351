#include<iostream>
#include<windows.h>
#include <stdio.h>
#include<typeinfo>
#include <stdlib.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
using namespace std;
const int N=500;
float a[N][N];  // 未对齐的数组
float b[N][N];  //对齐的数组


void unaligned_array_init() //未对齐的数组的初始化,生成测试用例，现将对角线都设置为1，上三角矩阵随机数，然后再初始化其他元素
{
    for (int i = 0; i < N; i++)
    {
        a[i][i] = 1.0;

        for (int j = i + 1; j < N; j++)
        {
            a[i][j] = rand();
        }
    }
    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                a[i][j] += a[k][j];
            }
        }
    }
}

void aligned_array_init(int alignment) //对齐数组初始化
{
    float** b = (float**)_aligned_malloc(sizeof(float*) * N , alignment);
    for (int i = 0; i < N; i++)
    {
        b[i] = (float*)_aligned_malloc(sizeof(float) * N, alignment);
    }
    for (int i = 0; i < N; i++)
    {
        b[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
        {
            b[i][j] = rand();
        }

    }
    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                b[i][j] += b[k][j];
            }
        }
    }
}

void common(float m[N][N]) //普通消元算法
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

void avx_unalined(float m[N][N]) //AVX未对齐
{
    for (int k = 0; k < N; k++)
    {
        __m256 t1 = _mm256_set1_ps(m[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= N; j += 8)
        {
            __m256 t2 = _mm256_loadu_ps(&m[k][j]);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(&m[k][j], t2);
        }
        for (j; j < N; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            __m256 vik = _mm256_set1_ps(m[i][k]);
            for (j = k + 1; j + 8 <= N; j += 8)
            {
                __m256 vkj = _mm256_loadu_ps(&m[k][j]);
                __m256 vij = _mm256_loadu_ps(&m[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&m[i][j], vij);
            }
            for (j; j < N; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

void avx_aligned(float m[N][N])//AVX对齐
{
    for (int k = 0; k < N; k++)
    {
        __m256 t1 = _mm256_set1_ps(m[k][k]);
        int j = k + 1;
        while ((int)(&m[k][j]) % 32)
        {
            m[k][j] = m[k][j] / m[k][k];
            j++;
        }
        for (j; j + 8 <= N; j += 8)
        {
            __m256 t2 = _mm256_load_ps(&m[k][j]);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_store_ps(&m[k][j], t2);
        }
        for (j; j < N; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            __m256 vik = _mm256_set1_ps(m[i][k]);
            j = k + 1;
            while ((int)(&m[k][j]) % 32)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
                j++;
            }
            for (j; j + 8 <= N; j += 8)
            {
                __m256 vkj = _mm256_load_ps(&m[k][j]);
                __m256 vij = _mm256_loadu_ps(&m[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&m[i][j], vij);
            }
            for (j; j < N; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}


void sse_unalined(float m[N][N]) //SSE未对齐
{
    for (int k = 0; k < N; k++)
    {
        __m128 t1 = _mm_set1_ps(m[k][k]);
        int j = 0;
        for (j = k + 1; j + 4 <= N; j += 4)
        {
            __m128 t2 = _mm_loadu_ps(&m[k][j]);
            t2 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(&m[k][j], t2);
        }
        for (j; j < N; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            __m128 vik = _mm_set1_ps(m[i][k]);
            int j = 0;
            for (j = k + 1; j + 4 <= N; j += 4)
            {
                __m128 vkj = _mm_loadu_ps(&m[k][j]);
                __m128 vij = _mm_loadu_ps(&m[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_storeu_ps(&m[i][j], vij);
            }
            for (j; j < N; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

void sse_aligned(float m[N][N]) //SSE对齐
{
    for (int k = 0; k < N; k++)
    {
        __m128 t1 = _mm_set1_ps(m[k][k]);
        int j = k + 1;

        //cout << &m[k][j];
        while ((int)(&m[k][j]) % 16)
        {
            m[k][j] = m[k][j] / m[k][k];
            j++;
        }
        //cout << &m[k][j]<<endl;
        for (j; j + 4 <= N; j += 4)
        {
            __m128 t2 = _mm_load_ps(&m[k][j]);
            t2 = _mm_div_ps(t2, t1);
            _mm_store_ps(&m[k][j], t2);
        }
        for (j; j < N; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            __m128 vik = _mm_set1_ps(m[i][k]);
            j = k + 1;
            while ((int)(&m[k][j]) % 16)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
                j++;
            }
            for (j; j + 4 <= N; j += 4)
            {
                __m128 vkj = _mm_load_ps(&m[k][j]);
                __m128 vij = _mm_loadu_ps(&m[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_storeu_ps(&m[i][j], vij);
            }
            for (j; j < N; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

int main()
{
    long long head, tail, freq;
    float time[30];
    float sum=0;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    unaligned_array_init();
    for(int i=0;i<30;i++)
    {
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        common(a);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        time[i]=(tail - head) * 1000 / freq ;
        unaligned_array_init();
    }
    for(int j=0;j<30;j++)
    {
        sum+=time[j];
    }
    sum/=30;
    cout << "common time:" << sum << "ms" << endl;


    sum=0;
    unaligned_array_init();
    for(int i=0;i<30;i++)
    {
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        avx_unalined(a);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        time[i]=(tail - head) * 1000 / freq ;
        unaligned_array_init();
    }
    for(int j=0;j<30;j++)
    {
        sum+=time[j];
    }
    sum/=30;
    cout << "AVX(未对齐) time:" << sum << "ms" << endl;



    sum=0;
    aligned_array_init(32);
    for(int i=0;i<30;i++)
    {
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        avx_aligned(b);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        time[i]=(tail - head) * 1000 / freq ;
        aligned_array_init(32);
    }
    for(int j=0;j<30;j++)
    {
        sum+=time[j];
    }
    sum/=30;
    cout << "AVX(对齐) time:" << sum << "ms" << endl;


    sum=0;
    unaligned_array_init();
    for(int i=0;i<30;i++)
    {
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        sse_unalined(a);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        time[i]=(tail - head) * 1000 / freq ;
        unaligned_array_init();
    }
    for(int j=0;j<30;j++)
    {
        sum+=time[j];
    }
    sum/=30;
    cout << "SSE(未对齐) time:" << sum << "ms" << endl;


    sum=0;
    aligned_array_init(16);    //SSE指令需要16位对齐
    for(int i=0;i<30;i++)
    {
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        sse_aligned(b);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        time[i]=(tail - head) * 1000 / freq ;
        aligned_array_init(16);
    }
    for(int j=0;j<30;j++)
    {
        sum+=time[j];
    }
    sum/=30;
    cout << "SSE(对齐) time:" << sum << "ms" << endl;


}
