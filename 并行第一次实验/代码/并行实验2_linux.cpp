#include <iostream>
#include <sys/time.h>
using namespace std;

#define lli unsigned long long int 

lli n = 524288;
int a[524288];
int sum = 0;
int sum1 = 0;
int sum2 = 0;
void initial()
{
    n = 524288;
    for (int i = 0; i < n; i++)
    {
        a[i] = i % 10;
    }
}

void common()
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < n; i++)
        sum += a[i];
    gettimeofday(&end, NULL);
    cout << "common:" << ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000 << "ms" << endl;
}


void optimize()
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < n; i += 2)
    {
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    sum = sum1 + sum2;
    gettimeofday(&end, NULL);
    cout << "optimize:" << ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000 << "ms" << endl;
}

int main()
{
    initial();
    common();
    initial();
    optimize();
}
