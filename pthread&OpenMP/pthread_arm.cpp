#include<iostream>
#include <stdio.h>
#include<typeinfo>
#include <stdlib.h>
#include<sys/time.h>
#include<semaphore.h>
#include<pthread.h>
#include<arm_neon.h>
using namespace std;
#define N 2000

#define NUM_THREADS 8
float A[N][N];

sem_t sem_main;  //信号量
sem_t sem_workstart[NUM_THREADS];
sem_t sem_workend[NUM_THREADS];

sem_t sem_leader;
sem_t sem_Division[NUM_THREADS];
sem_t sem_Elimination[NUM_THREADS];

pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;

struct threadParam_t //线程数据结构
{
    int k;
    int t_id;
};

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

void neon()//neon优化算法
{
    for (int k = 0; k < N; k++)
    {
        float32x4_t head = vdupq_n_f32(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 4 <= N; j += 4)
        {
            float32x4_t tmp = vld1q_f32(&A[k][j]);
            tmp = vdivq_f32(tmp, head);
            vst1q_f32(&A[k][j], tmp);
        }
        for (j; j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float32x4_t ik = vdupq_n_f32(A[k][k]);
            for (j = k + 1; j + 4 <= N; j += 4)
            {
                float32x4_t kj = vld1q_f32(&A[k][j]);
                float32x4_t ij = vld1q_f32(&A[i][j]);
                float32x4_t mul = vmulq_f32(kj, ik);
                ij = vsubq_f32(ij, mul);
                vst1q_f32(&A[i][j], ij);
            }
            for (j; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void* dynamic_threadFunc(void* param) //动态线程线程函数
{
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;           //消去的轮次
    int t_id = p->t_id;     //线程
    int i = k + t_id + 1;   //获取任务

    for (int j = k + 1; j < N; j++)
    {
        A[i][j] = A[i][j] - A[i][k] * A[k][j];
    }
    A[i][k] = 0;
    pthread_exit(NULL);
}

void dynamic_thread() //动态线程pthread
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;

        int worker_count = N - 1 - k;//线程数
        pthread_t* handle = (pthread_t*)malloc(worker_count * sizeof(pthread_t));//创建对应的handle
        threadParam_t* param = (threadParam_t*)malloc(worker_count * sizeof(threadParam_t));//创建对应的线程数据结构

        for (int t_id = 0; t_id < worker_count; t_id++) //分配任务
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }

        for (int t_id = 0; t_id < worker_count; t_id++)//创建线程
        {
            pthread_create(&handle[t_id], NULL, dynamic_threadFunc, &param[t_id]);
        }

        for (int t_id = 0; t_id < worker_count; t_id++)//主线程挂起等待所有的工作线程完成此轮消去工作
        {
            pthread_join(handle[t_id], NULL);
        }
        free(handle);
        free(param);
    }

}

void* dynamic_threadFunc_NUM_THREADS(void* param) //动态8线程线程函数
{
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;           //消去的轮次
    int t_id = p->t_id;     //线程
    int i = k + t_id + 1;   //获取任务
    for (i; i < N; i += NUM_THREADS)
    {
        for (int j = k + 1; j < N; j++)
        {
            A[i][j] = A[i][j] - A[i][k] * A[k][j];
        }
        A[i][k] = 0;
    }
    pthread_exit(NULL);
}

void dynamic_thread_NUM_THREADS() //动态8线程pthread
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;

        int thread_cnt = NUM_THREADS;
        pthread_t* handle = (pthread_t*)malloc(thread_cnt * sizeof(pthread_t));
        threadParam_t* param = (threadParam_t*)malloc(thread_cnt * sizeof(threadParam_t));

        for (int t_id = 0; t_id < thread_cnt; t_id++)
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }

        for (int t_id = 0; t_id < thread_cnt; t_id++)
        {
            pthread_create(&handle[t_id], NULL, dynamic_threadFunc_NUM_THREADS, &param[t_id]);
        }

        for (int t_id = 0; t_id < thread_cnt; t_id++)
        {
            pthread_join(handle[t_id], NULL);
        }
        free(handle);
        free(param);
    }

}

void* neon_dynamic_threadFunc_NUM_THREADS(void* param) //动态8线程SIMD优化线程函数
{
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;           //消去的轮次
    int t_id = p->t_id;     //线程
    int i = k + t_id + 1;   //获取任务

    for (i; i < N; i += NUM_THREADS)
    {
        float32x4_t ik = vdupq_n_f32(A[k][k]);
        int j;
        for (j = k + 1; j + 8 <= N; j += 8)
        {
            float32x4_t kj = vld1q_f32(&A[k][j]);
            float32x4_t ij = vld1q_f32(&A[i][j]);
            float32x4_t mul = vmulq_f32(kj, ik);
            ij = vsubq_f32(ij, mul);
            vst1q_f32(&A[i][j], ij);
        }
        for (j; j < N; j++)
        {
            A[i][j] = A[i][j] - A[i][k] * A[k][j];
        }
        A[i][k] = 0;
    }
    pthread_exit(NULL);

}

void neon_dynamic_thread_NUM_THREADS() //动态8线程SIMD优化线程
{
    for (int k = 0; k < N; k++)
    {
        float32x4_t head = vdupq_n_f32(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= N; j += 8)
        {
            float32x4_t tmp = vld1q_f32(&A[k][j]);
            tmp = vdivq_f32(tmp, head);
            vst1q_f32(&A[k][j], tmp);
        }
        for (j; j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;

        int thread_cnt = NUM_THREADS;
        pthread_t* handle = (pthread_t*)malloc(thread_cnt * sizeof(pthread_t));
        threadParam_t* param = (threadParam_t*)malloc(thread_cnt * sizeof(threadParam_t));

        for (int t_id = 0; t_id < thread_cnt; t_id++)
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }

        for (int t_id = 0; t_id < thread_cnt; t_id++)
        {
            pthread_create(&handle[t_id], NULL, neon_dynamic_threadFunc_NUM_THREADS, &param[t_id]);
        }

        for (int t_id = 0; t_id < thread_cnt; t_id++)
        {
            pthread_join(handle[t_id], NULL);
        }
        free(handle);
        free(param);
    }
}

void* neon_dynamic_threadFunc(void* param)//SIMD优化动态线程函数
{
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;           //消去的轮次
    int t_id = p->t_id;     //线程
    int i = k + t_id + 1;   //获取任务

    float32x4_t ik = vdupq_n_f32(A[k][k]);
    int j;
    for (j = k + 1; j + 8 <= N; j += 8)
    {
        float32x4_t kj = vld1q_f32(&A[k][j]);
        float32x4_t ij = vld1q_f32(&A[i][j]);
        float32x4_t mul = vmulq_f32(kj, ik);
        ij = vsubq_f32(ij, mul);
        vst1q_f32(&A[i][j], ij);
    }
    for (j; j < N; j++)
    {
        A[i][j] = A[i][j] - A[i][k] * A[k][j];
    }
    A[i][k] = 0;
    pthread_exit(NULL);
}

void neon_dynamic_thread()//SIMD优化动态线程pthread
{
    for (int k = 0; k < N; k++)
    {
        float32x4_t head = vdupq_n_f32(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= N; j += 8)
        {
            float32x4_t tmp = vld1q_f32(&A[k][j]);
            tmp = vdivq_f32(tmp, head);
            vst1q_f32(&A[k][j], tmp);
        }
        for (j; j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;

        int worker_count = N - 1 - k;
        pthread_t* handle = (pthread_t*)malloc(worker_count * sizeof(pthread_t));
        threadParam_t* param = (threadParam_t*)malloc(worker_count * sizeof(threadParam_t));

        for (int t_id = 0; t_id < worker_count; t_id++) //分配任务
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }

        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            pthread_create(&handle[t_id], NULL, neon_dynamic_threadFunc, &param[t_id]);
        }

        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            pthread_join(handle[t_id], NULL);
        }
        free(handle);
        free(param);
    }


}

void* static_threadFunc(void* param) //静态线程+信号量线程函数
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++) 
    {
        sem_wait(&sem_workstart[t_id]);//阻塞，等待主线程除法完成

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
        {
            for (int j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }

        sem_post(&sem_main);        //唤醒主线程
        sem_wait(&sem_workend[t_id]);  //阻塞，等待主线程唤醒进入下一轮

    }
    pthread_exit(NULL);
}

void static_thread() //静态线程+信号量pthread
{
    sem_init(&sem_main, 0, 0);

    for (int i = 0; i < NUM_THREADS; i++)
    {
        sem_init(&sem_workend[i], 0, 0);
        sem_init(&sem_workstart[i], 0, 0);
    }

    pthread_t* handle = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));

    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handle[t_id], NULL, static_threadFunc, &param[t_id]);

    }

    for (int k = 0; k < N; k++)
    {

        for (int j = k + 1; j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;

        for (int t_id = 0; t_id < NUM_THREADS; t_id++) //唤起工作线程
        {
            sem_post(&sem_workstart[t_id]);
        }

        for (int t_id = 0; t_id < NUM_THREADS; t_id++) //主线程睡眠
        {
            sem_wait(&sem_main);
        }

        for (int t_id = 0; t_id < NUM_THREADS; t_id++) //再次唤起工作线程，进入下一轮消去任务
        {
            sem_post(&sem_workend[t_id]);
        }

    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handle[t_id], NULL);
    }

    sem_destroy(&sem_main);    //销毁线程
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);

    free(handle);
    free(param);
}

void* neon_static_threadFunc(void* param) //SIMD优化静态线程+信号量线程函数
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++)
    {
        sem_wait(&sem_workstart[t_id]);//阻塞，等待主线程除法完成

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
        {
            float32x4_t ik = vdupq_n_f32(A[k][k]);
            int j;
            for (j = k + 1; j + 8 <= N; j += 8)
            {
                float32x4_t kj = vld1q_f32(&A[k][j]);
                float32x4_t ij = vld1q_f32(&A[i][j]);
                float32x4_t mul = vmulq_f32(kj, ik);
                ij = vsubq_f32(ij, mul);
                vst1q_f32(&A[i][j], ij);
            }
            for (j; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }

        sem_post(&sem_main);        //唤醒主线程
        sem_wait(&sem_workend[t_id]);  //阻塞，等待主线程唤醒进入下一轮

    }
    pthread_exit(NULL);
}

void neon_static_thread() //SIMD优化静态线程+信号量pthread
{
    sem_init(&sem_main, 0, 0); //初始化信号量
    for (int i = 0; i < NUM_THREADS; i++)
    {
        sem_init(&sem_workend[i], 0, 0);
        sem_init(&sem_workstart[i], 0, 0);
    }
    pthread_t* handle = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));

    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handle[t_id], NULL, neon_static_threadFunc, &param[t_id]);

    }

    for (int k = 0; k < N; k++)
    {

        float32x4_t head = vdupq_n_f32(A[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= N; j += 8)
        {
            float32x4_t tmp = vld1q_f32(&A[k][j]);
            tmp = vdivq_f32(tmp, head);
            vst1q_f32(&A[k][j], tmp);
        }
        for (j; j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;

        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            sem_post(&sem_workstart[t_id]);
        }

        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            sem_wait(&sem_main);
        }

        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            sem_post(&sem_workend[t_id]);
        }

    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handle[t_id], NULL);
    }
    sem_destroy(&sem_main);    //销毁线程
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);

    free(handle);
    free(param);

}

void* static_whole_threadFunc(void* param) //静态线程+信号量同步+三重循环线程函数
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++)  //t_id 为 0 的线程做除法操作，其它工作线程先等待
    {

        if (t_id == 0)
        {
            for (int j = k + 1; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
        else
            sem_wait(&sem_Division[t_id - 1]);// 阻塞，等待完成除法操作

        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; i++) // t_id 为 0 的线程唤醒其它工作线程，进行消去操作
            {
                sem_post(&sem_Division[i]);
            }
        }

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) //循环划分任务
        {
            for (int j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }

        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; i++) // 等待其它 worker 完成消去
            {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < NUM_THREADS - 1; i++)  // 通知其它 worker 进入下一轮
            {
                sem_post(&sem_Elimination[i]);
            }
        }
        else
        {
            sem_post(&sem_leader);// 通知 leader, 已完成消去任务
            sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
        }

    }

    pthread_exit(NULL);
}

void static_whole_thread()//静态线程+信号量同步+三重循环pthread
{
    sem_init(&sem_leader, 0, 0);//初始化信号量
    for (int i = 0; i < NUM_THREADS; i++)
    {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }

    pthread_t* handle = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));

    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handle[t_id], NULL, static_whole_threadFunc, &param[t_id]);

    }

    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handle[t_id], NULL);
    }

    sem_destroy(&sem_main);    //销毁线程
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);

    free(handle);
    free(param);
}

void* neon_static_whole_threadFunc(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++)
    {

        if (t_id == 0) {
            float32x4_t head = vdupq_n_f32(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= N; j += 8)
            {
                float32x4_t tmp = vld1q_f32(&A[k][j]);
                tmp = vdivq_f32(tmp, head);
                vst1q_f32(&A[k][j], tmp);
            }
            for (j; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
        else
            sem_wait(&sem_Division[t_id - 1]);

        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; i++)
            {
                sem_post(&sem_Division[i]);
            }
        }

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
        {
            float32x4_t ik = vdupq_n_f32(A[k][k]);
            int j;
            for (j = k + 1; j + 8 <= N; j += 8)
            {
                float32x4_t kj = vld1q_f32(&A[k][j]);
                float32x4_t ij = vld1q_f32(&A[i][j]);
                float32x4_t mul = vmulq_f32(kj, ik);
                ij = vsubq_f32(ij, mul);
                vst1q_f32(&A[i][j], ij);
            }
            for (j; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }

        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; i++)
            {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < NUM_THREADS - 1; i++)
            {
                sem_post(&sem_Elimination[i]);
            }
        }
        else
        {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }

    }

    pthread_exit(NULL);
}

void neon_static_whole_thread()
{
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < NUM_THREADS; i++)
    {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }

    pthread_t* handle = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));

    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handle[t_id], NULL, neon_static_whole_threadFunc, &param[t_id]);

    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handle[t_id], NULL);
    }

    sem_destroy(&sem_main);    //销毁线程
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);

    free(handle);
    free(param);
}

void print()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << A[i][j] << " ";
        }
        cout << endl;
    }
}

void* static_barrier_threadFunc(void* param)//静态线程 +barrier线程函数
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++)
    {
        if (t_id == 0)
        {
            for (int j = k + 1; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }

        pthread_barrier_wait(&barrier_Division);//第一个同步点

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
        {
            for (int j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }

        pthread_barrier_wait(&barrier_Elimination);//第二个同步点


    }
    pthread_exit(NULL);
}

void static_barrier_thread()//静态线程 +barrier线程pthread
{
    pthread_barrier_init(&barrier_Division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

    pthread_t* handle = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));

    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handle[t_id], NULL, static_barrier_threadFunc, &param[t_id]);

    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handle[t_id], NULL);
    }

    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);

    free(handle);
    free(param);
}

void* neon_static_barrier_threadFunc(void* param)//静态线程 +barrier线程SIMD优化
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++)
    {
        if (t_id == 0)
        {
            float32x4_t head = vdupq_n_f32(A[k][k]);
            int j = 0;
            for (j = k + 1; j + 8 <= N; j += 8)
            {
                float32x4_t tmp = vld1q_f32(&A[k][j]);
                tmp = vdivq_f32(tmp, head);
                vst1q_f32(&A[k][j], tmp);
            }
            for (j; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }

        pthread_barrier_wait(&barrier_Division);//第一个同步点

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
        {
            float32x4_t ik = vdupq_n_f32(A[k][k]);
            int j;
            for (j = k + 1; j + 8 <= N; j += 8)
            {
                float32x4_t kj = vld1q_f32(&A[k][j]);
                float32x4_t ij = vld1q_f32(&A[i][j]);
                float32x4_t mul = vmulq_f32(kj, ik);
                ij = vsubq_f32(ij, mul);
                vst1q_f32(&A[i][j], ij);
            }
            for (j; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }

        pthread_barrier_wait(&barrier_Elimination);//第二个同步点

    }
    pthread_exit(NULL);

}

void neon_static_barrier_thread()
{
    pthread_barrier_init(&barrier_Division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

    pthread_t* handle = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));

    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handle[t_id], NULL, neon_static_barrier_threadFunc, &param[t_id]);

    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handle[t_id], NULL);
    }

    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);

    free(handle);
    free(param);
}

double time(void (*Func)())
{
    struct timeval start;
    struct timeval end;
    double sum=0;
    for (int m = 0; m < 20; m++)
    {
        init();
        gettimeofday(&start, NULL);
        Func();
        gettimeofday(&end, NULL);
        sum += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    return sum / 20;
}
int main()
{
    cout << "平凡算法" << time(common) << "ms" << endl;

    cout << "SIMD优化" << (time(neon)) << "ms" << endl;
    
    cout <<"动态线程"<<time(dynamic_thread_NUM_THREADS)<< "ms" << endl;
    
    cout <<"动态线程SIMD优化"<<time(neon_dynamic_thread_NUM_THREADS) << "ms" << endl;
    
    cout <<"静态线程+信号量同步"<<time(static_thread) << "ms" << endl;
    
    cout <<"静态线程+信号量同步SIMD优化"<<time(neon_static_thread) << "ms" << endl;
    
    cout <<"静态线程+信号量同步+三重循环"<<time(static_whole_thread) << "ms" << endl;
    
    cout <<"静态线程+信号量同步+三重循环SIMD优化"<<time(neon_static_whole_thread) << "ms" << endl;

    cout << "静态线程+barrier线程" << time(static_barrier_thread) << "ms" << endl;

    cout << "静态线程+barrier线程SIMD优化" << time(neon_static_barrier_thread) << "ms" << endl;
}


