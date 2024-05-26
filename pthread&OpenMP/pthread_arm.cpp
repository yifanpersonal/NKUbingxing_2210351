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

sem_t sem_main;  //�ź���
sem_t sem_workstart[NUM_THREADS];
sem_t sem_workend[NUM_THREADS];

sem_t sem_leader;
sem_t sem_Division[NUM_THREADS];
sem_t sem_Elimination[NUM_THREADS];

pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;

struct threadParam_t //�߳����ݽṹ
{
    int k;
    int t_id;
};

void init() //��ʼ��
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

void common() //��ͨ��˹��Ԫ�㷨
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

void neon()//neon�Ż��㷨
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

void* dynamic_threadFunc(void* param) //��̬�߳��̺߳���
{
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;           //��ȥ���ִ�
    int t_id = p->t_id;     //�߳�
    int i = k + t_id + 1;   //��ȡ����

    for (int j = k + 1; j < N; j++)
    {
        A[i][j] = A[i][j] - A[i][k] * A[k][j];
    }
    A[i][k] = 0;
    pthread_exit(NULL);
}

void dynamic_thread() //��̬�߳�pthread
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;

        int worker_count = N - 1 - k;//�߳���
        pthread_t* handle = (pthread_t*)malloc(worker_count * sizeof(pthread_t));//������Ӧ��handle
        threadParam_t* param = (threadParam_t*)malloc(worker_count * sizeof(threadParam_t));//������Ӧ���߳����ݽṹ

        for (int t_id = 0; t_id < worker_count; t_id++) //��������
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }

        for (int t_id = 0; t_id < worker_count; t_id++)//�����߳�
        {
            pthread_create(&handle[t_id], NULL, dynamic_threadFunc, &param[t_id]);
        }

        for (int t_id = 0; t_id < worker_count; t_id++)//���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
        {
            pthread_join(handle[t_id], NULL);
        }
        free(handle);
        free(param);
    }

}

void* dynamic_threadFunc_NUM_THREADS(void* param) //��̬8�߳��̺߳���
{
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;           //��ȥ���ִ�
    int t_id = p->t_id;     //�߳�
    int i = k + t_id + 1;   //��ȡ����
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

void dynamic_thread_NUM_THREADS() //��̬8�߳�pthread
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

void* neon_dynamic_threadFunc_NUM_THREADS(void* param) //��̬8�߳�SIMD�Ż��̺߳���
{
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;           //��ȥ���ִ�
    int t_id = p->t_id;     //�߳�
    int i = k + t_id + 1;   //��ȡ����

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

void neon_dynamic_thread_NUM_THREADS() //��̬8�߳�SIMD�Ż��߳�
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

void* neon_dynamic_threadFunc(void* param)//SIMD�Ż���̬�̺߳���
{
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;           //��ȥ���ִ�
    int t_id = p->t_id;     //�߳�
    int i = k + t_id + 1;   //��ȡ����

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

void neon_dynamic_thread()//SIMD�Ż���̬�߳�pthread
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

        for (int t_id = 0; t_id < worker_count; t_id++) //��������
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

void* static_threadFunc(void* param) //��̬�߳�+�ź����̺߳���
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++) 
    {
        sem_wait(&sem_workstart[t_id]);//�������ȴ����̳߳������

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
        {
            for (int j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }

        sem_post(&sem_main);        //�������߳�
        sem_wait(&sem_workend[t_id]);  //�������ȴ����̻߳��ѽ�����һ��

    }
    pthread_exit(NULL);
}

void static_thread() //��̬�߳�+�ź���pthread
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

        for (int t_id = 0; t_id < NUM_THREADS; t_id++) //�������߳�
        {
            sem_post(&sem_workstart[t_id]);
        }

        for (int t_id = 0; t_id < NUM_THREADS; t_id++) //���߳�˯��
        {
            sem_wait(&sem_main);
        }

        for (int t_id = 0; t_id < NUM_THREADS; t_id++) //�ٴλ������̣߳�������һ����ȥ����
        {
            sem_post(&sem_workend[t_id]);
        }

    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join(handle[t_id], NULL);
    }

    sem_destroy(&sem_main);    //�����߳�
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);

    free(handle);
    free(param);
}

void* neon_static_threadFunc(void* param) //SIMD�Ż���̬�߳�+�ź����̺߳���
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++)
    {
        sem_wait(&sem_workstart[t_id]);//�������ȴ����̳߳������

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

        sem_post(&sem_main);        //�������߳�
        sem_wait(&sem_workend[t_id]);  //�������ȴ����̻߳��ѽ�����һ��

    }
    pthread_exit(NULL);
}

void neon_static_thread() //SIMD�Ż���̬�߳�+�ź���pthread
{
    sem_init(&sem_main, 0, 0); //��ʼ���ź���
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
    sem_destroy(&sem_main);    //�����߳�
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);

    free(handle);
    free(param);

}

void* static_whole_threadFunc(void* param) //��̬�߳�+�ź���ͬ��+����ѭ���̺߳���
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++)  //t_id Ϊ 0 ���߳����������������������߳��ȵȴ�
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
            sem_wait(&sem_Division[t_id - 1]);// �������ȴ���ɳ�������

        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; i++) // t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
            {
                sem_post(&sem_Division[i]);
            }
        }

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) //ѭ����������
        {
            for (int j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }

        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; i++) // �ȴ����� worker �����ȥ
            {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < NUM_THREADS - 1; i++)  // ֪ͨ���� worker ������һ��
            {
                sem_post(&sem_Elimination[i]);
            }
        }
        else
        {
            sem_post(&sem_leader);// ֪ͨ leader, �������ȥ����
            sem_wait(&sem_Elimination[t_id - 1]); // �ȴ�֪ͨ��������һ��
        }

    }

    pthread_exit(NULL);
}

void static_whole_thread()//��̬�߳�+�ź���ͬ��+����ѭ��pthread
{
    sem_init(&sem_leader, 0, 0);//��ʼ���ź���
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

    sem_destroy(&sem_main);    //�����߳�
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

    sem_destroy(&sem_main);    //�����߳�
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

void* static_barrier_threadFunc(void* param)//��̬�߳� +barrier�̺߳���
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

        pthread_barrier_wait(&barrier_Division);//��һ��ͬ����

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
        {
            for (int j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }

        pthread_barrier_wait(&barrier_Elimination);//�ڶ���ͬ����


    }
    pthread_exit(NULL);
}

void static_barrier_thread()//��̬�߳� +barrier�߳�pthread
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

void* neon_static_barrier_threadFunc(void* param)//��̬�߳� +barrier�߳�SIMD�Ż�
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

        pthread_barrier_wait(&barrier_Division);//��һ��ͬ����

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

        pthread_barrier_wait(&barrier_Elimination);//�ڶ���ͬ����

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
    cout << "ƽ���㷨" << time(common) << "ms" << endl;

    cout << "SIMD�Ż�" << (time(neon)) << "ms" << endl;
    
    cout <<"��̬�߳�"<<time(dynamic_thread_NUM_THREADS)<< "ms" << endl;
    
    cout <<"��̬�߳�SIMD�Ż�"<<time(neon_dynamic_thread_NUM_THREADS) << "ms" << endl;
    
    cout <<"��̬�߳�+�ź���ͬ��"<<time(static_thread) << "ms" << endl;
    
    cout <<"��̬�߳�+�ź���ͬ��SIMD�Ż�"<<time(neon_static_thread) << "ms" << endl;
    
    cout <<"��̬�߳�+�ź���ͬ��+����ѭ��"<<time(static_whole_thread) << "ms" << endl;
    
    cout <<"��̬�߳�+�ź���ͬ��+����ѭ��SIMD�Ż�"<<time(neon_static_whole_thread) << "ms" << endl;

    cout << "��̬�߳�+barrier�߳�" << time(static_barrier_thread) << "ms" << endl;

    cout << "��̬�߳�+barrier�߳�SIMD�Ż�" << time(neon_static_barrier_thread) << "ms" << endl;
}


