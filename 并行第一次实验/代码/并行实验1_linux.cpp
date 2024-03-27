#include <iostream>
#include <sys/time.h>
using namespace std;

#define lli unsigned long long int 

const int n = 100;

lli a[n];
lli b[n][n];
lli sum[n];

void initial()
{
    for (int i = 0; i < n; i++)
        a[i] = i;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            b[i][j] = j % 10;
}

void common()
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < n; i++) {
        sum[i] = 0;
        for (int j = 0; j < n; j++)
            sum[i] += b[j][i] * a[j];
    }
    gettimeofday(&end, NULL);
    cout << "common:" << ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000 << "ms" << endl;
}


void optimize()
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < n; i++)
        sum[i] = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            sum[i] += b[j][i] * a[j];
    gettimeofday(&end, NULL);
    cout << "optimize:" << ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000 << "ms" << endl;
}

void unrollcommon()
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    for (int j = 0; j < n; j++)
    {
        sum[j] = 0;
        for (int i = 0; i < n; i += 10)
        {
            sum[j] += b[i + 0][j] * a[i + 0];
            sum[j] += b[i + 1][j] * a[i + 1];
            sum[j] += b[i + 2][j] * a[i + 2];
            sum[j] += b[i + 3][j] * a[i + 3];
            sum[j] += b[i + 4][j] * a[i + 4];
            sum[j] += b[i + 5][j] * a[i + 5];
            sum[j] += b[i + 6][j] * a[i + 6];
            sum[j] += b[i + 7][j] * a[i + 7];
            sum[j] += b[i + 8][j] * a[i + 8];
            sum[j] += b[i + 9][j] * a[i + 9];
        }
    }
    gettimeofday(&end, NULL);
    cout << "unrollcommon:" << ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000 << "ms" << endl;
}

void unrolloptimize()
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < n; i++)
        sum[i] = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i += 10)
        {
            sum[i + 0] += b[j][i + 0] * a[j];
            sum[i + 1] += b[j][i + 1] * a[j];
            sum[i + 2] += b[j][i + 2] * a[j];
            sum[i + 3] += b[j][i + 3] * a[j];
            sum[i + 4] += b[j][i + 4] * a[j];
            sum[i + 5] += b[j][i + 5] * a[j];
            sum[i + 6] += b[j][i + 6] * a[j];
            sum[i + 7] += b[j][i + 7] * a[j];
            sum[i + 8] += b[j][i + 8] * a[j];
            sum[i + 9] += b[j][i + 9] * a[j];
        }
    gettimeofday(&end, NULL);
    cout << "unrolloptimize:" << ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000 << "ms" << endl;
}

int main()
{
    initial();
    common();
    initial();
    optimize();
    initial();
    unrollcommon();
    initial();
    unrolloptimize();
}

