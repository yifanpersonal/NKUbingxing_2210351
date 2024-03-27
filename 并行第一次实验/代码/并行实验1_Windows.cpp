#include<iostream>
#include<time.h>
//#include <ratio>
#include <chrono>
#include<thread>
using namespace std::chrono;
//high_resolution_clock::time_point t1 = high_resolution_clock::now();


//high_resolution_clock::time_point t2 = high_resolution_clock::now();
//duration<double, std::milli> time_span = t2 - t1;

using namespace std;
const int n = 800;
int b[n][n];
int a[n];
int sum[n];
int loop = 10;
void initial()
{
	for (int i = 0; i < n; i++)
		a[i] = 0;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			b[i][j] = j % 10;
	}
}
void common()
{
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (int m = 0; m < loop; m++)
	{
		for (int i = 0; i < n; i++) {
			sum[i] = 0;
			for (int j = 0; j < n; j++)
				sum[i] += b[j][i] * a[j];
		}
	}
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double, std::milli> time_span = t2 - t1;
	cout << "common:" << time_span.count()/loop << "ms" << endl;
}
void optimize()
{
	high_resolution_clock::time_point t3 = high_resolution_clock::now();
	for (int m = 0; m < loop; m++)
	{
		for (int i = 0; i < n; i++)
			sum[i] = 0;
		for (int j = 0; j < n; j++)
			for (int i = 0; i < n; i++)
				sum[i] += b[j][i] * a[j];
	}
	high_resolution_clock::time_point t4 = high_resolution_clock::now();
	duration<double, std::milli> time_span = t4 - t3;
	cout << "optimize:" << time_span.count() / loop << "ms" << endl;
}
void unrollcommon()
{
	high_resolution_clock::time_point t5 = high_resolution_clock::now();
	for (int m = 0; m < loop; m++)
	{
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
	}
	high_resolution_clock::time_point t6 = high_resolution_clock::now();
	duration<double, std::milli> time_span = t6 - t5;
	cout << "unrollcommon:" << time_span.count() / loop << "ms" << endl;
}
void unrolloptimize()
{
	high_resolution_clock::time_point t7 = high_resolution_clock::now();
	for (int m = 0; m < loop; m++)
	{
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
	}
	high_resolution_clock::time_point t8 = high_resolution_clock::now();
	duration<double, std::milli> time_span = t8 - t7;
	cout << "unrolloptimize:" << time_span.count() / loop << "ms" << endl;
}
int main()
{
	initial();
	common();
	optimize();
	unrollcommon();
	unrolloptimize();
}
