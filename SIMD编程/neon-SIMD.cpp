#include<iostream>
#include<sys/time.h>
#include<arm_neon.h>
using namespace std;
const int N = 1000;
float a[N][N];
void init() //生成数组
{
    for (int i = 0, j = 0; i < N; i++) 
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


void common(float a[N][N])//平凡算法
{      
    for (int k = 0; k < N; k++) 
    {
        for (int j = k + 1; j < N; j++) 
        {
            a[k][j] = a[k][j] / a[k][k];
        }
        a[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) 
        {
            for (int j = k + 1; j < N; j++) 
            {
                a[i][j] = a[i][j] - a[i][k] * a[k][j];
            }
            a[i][k] = 0;
        }
    }
}

void neon(float a[N][N])//neon优化算法
{            
    for (int k = 0; k < N; k++) 
    {
        float32x4_t head = vdupq_n_f32(a[k][k]);
        int j = 0;
        for (j = k + 1; j + 4 <= N; j += 4)
        {
            float32x4_t tmp = vld1q_f32(&a[k][j]);
            tmp = vdivq_f32(tmp, head);
            vst1q_f32(&a[k][j], tmp);
        }
        for (j; j < N; j++) 
        {
            a[k][j] = a[k][j] / a[k][k];
        }
        a[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float32x4_t ik = vdupq_n_f32(a[i][k]);
            for (j = k + 1; j + 4 <= N; j += 4)
            {
                float32x4_t kj = vld1q_f32(&a[k][j]);
                float32x4_t ij = vld1q_f32(&a[i][j]);
                float32x4_t mul = vmulq_f32(kj, ik);
                ij = vsubq_f32(ij, mul);
                vst1q_f32(&a[i][j], ij);
            }
            for (j; j < N; j++) 
            {
                a[i][j] = a[i][j] - a[i][k] * a[k][j];
            }
            a[i][k] = 0;
        }
    }
}


int main() {
    init();
    struct timeval start;
    struct timeval end;
    float time[30];
    float sum = 0;
    for (int i = 0; i < 30; i++)
    {
        gettimeofday(&start, NULL);
        common(a);
        gettimeofday(&end, NULL);
        time[i] = ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
        init();
    }
    for (int j = 0; j < 30; j++)
    {
        sum += time[j];
    }
    sum /= 30;
    cout << "common：" << sum << "ms" << endl;

    sum = 0;
    for (int i = 0; i < 30; i++)
    {
        gettimeofday(&start, NULL);
        neon(a);
        gettimeofday(&end, NULL);
        time[i] = ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
        init();
    }
    for (int j = 0; j < 30; j++)
    {
        sum += time[j];
    }
    sum /= 30;
    cout << "neon优化算法：" << sum << "ms" << endl;

}
