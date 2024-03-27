#include<iostream>
#include<time.h>
#include <ratio>
#include<chrono>
using namespace std::chrono;
//high_resolution_clock::time_point t1 = high_resolution_clock::now();

//运动代码块

//high_resolution_clock::time_point t2 = high_resolution_clock::now();
//duration<double, std::milli> time_span = t2 - t1;


using namespace std;
long long int n = pow(2, 10);
int a[1024];
int sum = 0;
int sum1 = 0;
int sum2 = 0;
void initial()
{
	n = pow(2,10);
	for (int i = 0; i < n; i++)
	{
		a[i] = i % 10;
	}
}
void common()
{
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (int i = 0; i < n; i++)
		sum += a[i];
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double, std::milli> time_span = t2 - t1;
	cout << "common:" << time_span.count() << endl;
}
//多链路式
void chain()
{
	high_resolution_clock::time_point t3 = high_resolution_clock::now();
	for (int i = 0; i < n; i += 2) {
		sum1 += a[i];
		sum2 += a[i + 1];
	}
	sum = sum1 + sum2;
	high_resolution_clock::time_point t4 = high_resolution_clock::now();
	duration<double, std::milli> time_span = t4 - t3;
	cout << "chain:" << time_span.count() << endl;
}
//递归函数
int recursion()
{
	if (n == 1)
		return a[n];
	else
	{
		for (int i = 0; i < n / 2; i++)
			a[i] += a[n - i - 1];
		n = n / 2;
		recursion();
	}
}
void mrecursion()
{
	high_resolution_clock::time_point t5 = high_resolution_clock::now();
	recursion();
	high_resolution_clock::time_point t6 = high_resolution_clock::now();
	duration<double, std::milli> time_span = t6 - t5;
	cout << "recursion:" << time_span.count() << endl;
}
//二重循环
void loop()
{
	high_resolution_clock::time_point t7 = high_resolution_clock::now();
	for (int m = n; m > 1; m /= 2) // log(n)个步骤
		for (int i = 0; i < m / 2; i++)
			a[i] = a[i * 2] + a[i * 2 + 1];
	high_resolution_clock::time_point t8 = high_resolution_clock::now();
	duration<double, std::milli> time_span = t8 - t7;
	cout << "loop:" << time_span.count() << endl;
}
void main()
{
	initial();
	common();
	initial();
	chain();
	initial();
	mrecursion();
	initial();
	loop();
}
