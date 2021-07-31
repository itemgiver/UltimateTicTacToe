// 코드 실행시키기 전에 M*MAX_batch_size 같은게 int 범위 안 넘는지 반드시 확인해보자.
#include <random>
#include <algorithm>
#include <queue>
#include <deque>
#include <stack>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <thrust/execution_policy.h>

#define N 9469 // N = (162)+(1024*9)+(81+9)+1;
#define M 8655962 // M = (162*1024)+(1024*1024*8)+(1024*(81+9))+9306;
#define learning_rate 0.0001
#define l2_constant 0.0000001
#define epsilon 0.00000001
#define beta1 0.9
#define beta2 0.999
#define MAX_batch_size 100
#define MAX_train_data_cnt 10000
// selu function values
#define lambda 1.050700987355480
#define alphalambda 1.758099340847375

int adam_t, nn_num = 0, batch_size, training_size, gpu_num = 1;
int simulation_mcts,cpu_cnt,cut_lev,cutting;
int raw_data[MAX_batch_size][9][9];
double policy_answer[MAX_batch_size][9][9],value_answer[MAX_batch_size][9];

unsigned int win_tbl[16];
const unsigned short _popcnt16_[512] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8, 5, 6, 6, 7, 6, 7, 7, 8, 6, 7, 7, 8, 7, 8, 8, 9 };
const unsigned short tzcnt_u32_[512] = { 0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 8, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0 };
const unsigned short blsi_u32_[512] = { 0, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 32, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 64, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 32, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 128, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 32, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 64, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 32, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 256, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 32, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 64, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 32, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 128, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 32, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 64, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 32, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1 };

struct WeightType {
	int start, end;
	double value;
}weight[M];

double random_weight(double x) {
	std::default_random_engine generator(rand());
	std::normal_distribution<double> distribution(0, x);
	return distribution(generator);
}

inline bool win_t(unsigned short a)
{
	return win_tbl[a >> 5] & (1 << (a & 0x1F));
}

std::pair<int,std::pair<int,double>> sort_tmp[M];
void makeNN() {
	char s[100];
	sprintf(s, "weight%05d.txt",nn_num);
	{
		FILE *f = fopen(s, "w");

		fprintf(f, "%d %d\n", N, M);
		for(int i=1; i<=18*9; i++){
			for(int j=18*9+1; j<=18*9+1024; j++){
				fprintf(f,"%d %d %.15f\n",i,j,random_weight(sqrt((double)1/(double)162)));
			}
		}
		for(int i=1; i<=6; i++){
			for(int j=18*9+(i-1)*1024+1; j<=18*9+i*1024; j++){
				for(int k=18*9+i*1024+1; k<=18*9+(i+1)*1024; k++){
					fprintf(f,"%d %d %.15f\n",j,k,random_weight(sqrt((double)1/(double)1024)));
				}
			}
		}
		for(int i=18*9+6*1024+1; i<=18*9+7*1024; i++){
			for(int j=18*9+7*1024+1; j<=18*9+8*1024; j++){
				fprintf(f,"%d %d %.15f\n",i,j,random_weight(sqrt((double)1/(double)1024)));
			}	
		}
		for(int i=18*9+6*1024+1; i<=18*9+7*1024; i++){
			for(int j=18*9+8*1024+1; j<=18*9+9*1024; j++){
				fprintf(f,"%d %d %.15f\n",i,j,random_weight(sqrt((double)1/(double)1024)));
			}
		}
		for(int i=18*9+7*1024+1; i<=18*9+8*1024; i++){
			for(int j=18*9+9*1024+1; j<= 18*9+9*1024+81; j++){
				fprintf(f,"%d %d %.15f\n",i,j,random_weight(sqrt((double)1/(double)1024)));
			}
		}
		for(int i=18*9+8*1024+1; i<=18*9+9*1024; i++){
			for(int j=18*9+9*1024+81+1; j<=18*9+9*1024+81+9; j++){
				fprintf(f,"%d %d %.15f\n",i,j,random_weight(sqrt((double)1/(double)1024)));
			}
		}
		// bias setting
		for(int i=163; i<=9468; i++){
			fprintf(f,"0 %d 0\n",i);
		}
		fclose(f);
	}

	{
		FILE *in = fopen(s,"r");

		fscanf(in,"%d %d",&sort_tmp[0].first,&sort_tmp[0].second.first);
		for(int i=0; i<M; i++){
			fscanf(in,"%d %d %lf",&sort_tmp[i].second.first,&sort_tmp[i].first,&sort_tmp[i].second.second);
			if(sort_tmp[i].second.first == 0){
				std::swap(sort_tmp[i].second.first,sort_tmp[i].first);
				sort_tmp[i].first = INT_MAX;
			}
		}
		fclose(in);
	}

	{
		FILE *out = fopen(s,"w");

		sort(sort_tmp,sort_tmp+M);
		for(int i=0; i<M; i++){
			if(sort_tmp[i].first == INT_MAX){
				sort_tmp[i].first = 0;
				std::swap(sort_tmp[i].first,sort_tmp[i].second.first);
			}
		}

		fprintf(out,"%d %d\n",N,M);
		for(int i=0; i<M; i++){
			fprintf(out,"%d %d %.15f\n",sort_tmp[i].second.first,sort_tmp[i].first,sort_tmp[i].second.second);
		}
		fclose(out);
	}
}

void initNN() {
	char s[100];
	sprintf(s, "weight%05d.txt",nn_num);
	FILE *in = fopen(s, "r");

	fgets(s,50,in);
	for (int i = 0; i < M; i++) {
		fscanf(in, "%d %d %lf", &weight[i].start, &weight[i].end, &weight[i].value);
	}
}

// block = (batch_size,81), thread = 1
__global__ void gpu_init_dev_add(int *dev_add){
	int blockidxx = blockIdx.x;
	dev_add[blockidxx*81+blockIdx.y] = blockidxx-MAX_batch_size;
}

// block = (batch_size,81), thread = 1
__global__ void gpu_set_dev_add(int *dev_add){
	int blockidxx = blockIdx.x;
	dev_add[blockidxx*81+blockIdx.y] += MAX_batch_size;
}

// block = (batch_size,81), thread = 1
__global__ void gpu_copy_nn_input(int *dev_rawdata,int *dev_memo_rawdata,double *dev_policy_answer,double *dev_memo_policy_answer,double *dev_value_answer,double *dev_memo_value_answer,int *dev_add){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int offset = blockidxx*81+blockIdxy;

	dev_rawdata[offset] = dev_memo_rawdata[dev_add[offset]*81+blockIdxy];
	dev_policy_answer[offset] = dev_memo_policy_answer[dev_add[offset]*81+blockIdxy];
	if(blockIdxy < 9){
		dev_value_answer[blockidxx*9+blockIdxy] = dev_memo_value_answer[dev_add[offset]*9+blockIdxy];
	}
}

// block = (N,1024), thread = 1
__global__ void gpu_update_lr(int *dev_lr_support,double *dev_lr){
	int blockidxx = blockIdx.x;
	int offset = blockidxx*1024+blockIdx.y;

	dev_lr_support[offset]++;
	dev_lr[offset] = (double)learning_rate*sqrt((double)1 - pow(beta2, dev_lr_support[offset])) / ((double)1 - pow(beta1, dev_lr_support[offset]));
}

// block = (1246,6947), thread = batch_size/2가 되어야 하지만 이유모를 버그를 고치기 위해 일단은 batch_size로
__global__ void gpu_copy_weight(double *dev_weight){
	int i;
	int threadIdxx = threadIdx.x;
	int offset = blockIdx.x+blockIdx.y*1246;
	
	i = 1;
	while(true){
		if(threadIdxx < i && threadIdxx+i < MAX_batch_size){
			dev_weight[(threadIdxx+i)*M+offset] = dev_weight[threadIdxx*M+offset];
		}
		i *= 2;
		__syncthreads();
		if(MAX_batch_size <= i) break;
	}
}

// block = batch_size, thread = 162 , dev_rawdata = (0~80) 이 반복되는 형태로 구성되어있어야 함.
__global__ void gpu_run1(int *dev_rawdata,double *dev_node){
	int i;
	int blockidxx = blockIdx.x;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[162];

	if(threadIdxx < 81){
		if(dev_rawdata[threadIdxx+blockidxx*81] == 1) dev_node[threadIdxx+blockidxx*N+1] = 1;
		else dev_node[threadIdxx+blockidxx*N+1] = 0;
	}else{
		if(dev_rawdata[threadIdxx+blockidxx*81-81] == -1) dev_node[threadIdxx+blockidxx*N+1] = -1;
		else dev_node[threadIdxx+blockidxx*N+1] = 0;
	}

	sum[threadIdxx] = dev_node[threadIdxx+blockidxx*N+1];
	i = 162;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = (i+1)/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = (i+1)/2;
		}
	}
	if(threadIdxx == 0) sum[0] /= (double)162;
	
	__syncthreads();
	i = 1;
	while(true){
		if(threadIdxx < i && threadIdxx+i < 162){
			sum[threadIdxx+i] = sum[threadIdxx];
		}
		i *= 2;
		__syncthreads();
		if(162 <= i) break;
	}
	dev_node[threadIdxx+blockidxx*N+1] -= sum[threadIdxx];
	sum[threadIdxx] = dev_node[threadIdxx+blockidxx*N+1];
	sum[threadIdxx] = sum[threadIdxx]*sum[threadIdxx];

	i = 162;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = (i+1)/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = (i+1)/2;
		}
	}
	if(threadIdxx == 0){
		sum[0] /= (double)162;
		sum[0] = sqrt(sum[0]);
	}	
	
	__syncthreads();
	i = 1;
	while(true){
		if(threadIdxx < i && threadIdxx+i < 162){
			sum[threadIdxx+i] = sum[threadIdxx];
		}
		i *= 2;
		__syncthreads();
		if(162 <= i) break;
	}

	if(sum[threadIdxx] != 0) dev_node[threadIdxx+blockidxx*N+1] /= sum[threadIdxx];
}

// block = (batch_size,1024), thread = 162
__global__ void gpu_run2(double *dev_nn_node,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[162];

	sum[threadIdxx] = dev_nn_node[threadIdxx+1+blockidxx*N] * dev_weight[threadIdxx+162*blockIdxy+blockidxx*M];
	int i = 162;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = (i+1)/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = (i+1)/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 163+blockIdxy+blockidxx*N;
		dev_nn_node[offset] = sum[0] + dev_weight[8646656+blockIdxy+blockidxx*M];
		if(dev_nn_node[offset] > 0){
			dev_nn_node[offset] = lambda * dev_nn_node[offset];
		}else{
			dev_nn_node[offset] = alphalambda*(exp(dev_nn_node[offset])-(double)1);
		}
	}
}

// block = (batch_size,1024), thread = 1024
__global__ void gpu_run4(double *dev_nn_node,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_nn_node[threadIdxx+163+blockidxx*N] * dev_weight[165888+threadIdxx+1024*blockIdxy+blockidxx*M];
	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 1187+blockIdxy+blockidxx*N;
		dev_nn_node[offset] = sum[0] + dev_weight[8647680+blockIdxy+blockidxx*M];
		if(dev_nn_node[offset] > 0){
			dev_nn_node[offset] = lambda * dev_nn_node[offset];
		}else{
			dev_nn_node[offset] = alphalambda*(exp(dev_nn_node[offset])-(double)1);
		}
	}
}

// block = (batch_size,1024), thread = 1024
__global__ void gpu_run6(double *dev_nn_node,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_nn_node[threadIdxx+1187+blockidxx*N] * dev_weight[1214464+threadIdxx+1024*blockIdxy+blockidxx*M];
	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 2211+blockIdxy+blockidxx*N;
		dev_nn_node[offset] = sum[0] + dev_weight[8648704+blockIdxy+blockidxx*M];
		if(dev_nn_node[offset] > 0){
			dev_nn_node[offset] = lambda * dev_nn_node[offset];
		}else{
			dev_nn_node[offset] = alphalambda*(exp(dev_nn_node[offset])-1);
		}
	}
}

// block = (batch_size,1024), thread = 1024
__global__ void gpu_run8(double *dev_nn_node,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_nn_node[threadIdxx+2211+blockidxx*N] * dev_weight[2263040+threadIdxx+1024*blockIdxy+blockidxx*M];
	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 3235+blockIdxy+blockidxx*N;
		dev_nn_node[offset] = sum[0] + dev_weight[8649728+blockIdxy+blockidxx*M];
		if(dev_nn_node[offset] > 0){
			dev_nn_node[offset] = lambda * dev_nn_node[offset];
		}else{
			dev_nn_node[offset] = alphalambda*(exp(dev_nn_node[offset])-1);
		}
	}
}

// block = (batch_size,1024), thread = 1024
__global__ void gpu_run10(double *dev_nn_node,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_nn_node[threadIdxx+3235+blockidxx*N] * dev_weight[3311616+threadIdxx+1024*blockIdxy+blockidxx*M];
	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 4259+blockIdxy+blockidxx*N;
		dev_nn_node[offset] = sum[0] + dev_weight[8650752+blockIdxy+blockidxx*M];
		if(dev_nn_node[offset] > 0){
			dev_nn_node[offset] = lambda * dev_nn_node[offset];
		}else{
			dev_nn_node[offset] = alphalambda*(exp(dev_nn_node[offset])-1);
		}
	}
}

// block = (batch_size,1024), thread = 1024
__global__ void gpu_run12(double *dev_nn_node,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_nn_node[threadIdxx+4259+blockidxx*N] * dev_weight[4360192+threadIdxx+1024*blockIdxy+blockidxx*M];
	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 5283+blockIdxy+blockidxx*N;
		dev_nn_node[offset] = sum[0] + dev_weight[8651776+blockIdxy+blockidxx*M];
		if(dev_nn_node[offset] > 0){
			dev_nn_node[offset] = lambda * dev_nn_node[offset];
		}else{
			dev_nn_node[offset] = alphalambda*(exp(dev_nn_node[offset])-1);
		}
	}
}

// block = (batch_size,1024), thread = 1024
__global__ void gpu_run14(double *dev_nn_node,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_nn_node[threadIdxx+5283+blockidxx*N] * dev_weight[5408768+threadIdxx+1024*blockIdxy+blockidxx*M];
	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 6307+blockIdxy+blockidxx*N;
		dev_nn_node[offset] = sum[0] + dev_weight[8652800+blockIdxy+blockidxx*M];
		if(dev_nn_node[offset] > 0){
			dev_nn_node[offset] = lambda * dev_nn_node[offset];
		}else{
			dev_nn_node[offset] = alphalambda*(exp(dev_nn_node[offset])-1);
		}
	}
}

// block = (batch_size,1024), thread = 1024
__global__ void gpu_run16(double *dev_nn_node,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_nn_node[threadIdxx+6307+blockidxx*N] * dev_weight[6457344+threadIdxx+1024*blockIdxy+blockidxx*M];
	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 7331+blockIdxy+blockidxx*N;
		dev_nn_node[offset] = sum[0] + dev_weight[8653824+blockIdxy+blockidxx*M];
		if(dev_nn_node[offset] > 0){
			dev_nn_node[offset] = lambda * dev_nn_node[offset];
		}else{
			dev_nn_node[offset] = alphalambda*(exp(dev_nn_node[offset])-1);
		}
	}
}

// block = (batch_size,1024), thread = 1024
__global__ void gpu_run18(double *dev_nn_node,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_nn_node[threadIdxx+6307+blockidxx*N] * dev_weight[7505920+threadIdxx+1024*blockIdxy+blockidxx*M];
	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 8355+blockIdxy+blockidxx*N;
		dev_nn_node[offset] = sum[0] + dev_weight[8654848+blockIdxy+blockidxx*M];
		if(dev_nn_node[offset] > 0){
			dev_nn_node[offset] = lambda * dev_nn_node[offset];
		}else{
			dev_nn_node[offset] = alphalambda*(exp(dev_nn_node[offset])-1);
		}
	}
}

// block = (batch_size,81), thread = 1024
__global__ void gpu_run20(double *dev_nn_node,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_nn_node[threadIdxx+7331+blockidxx*N] * dev_weight[8554496+threadIdxx+1024*blockIdxy+blockidxx*M];
	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 9379+blockIdxy+blockidxx*N;
		dev_nn_node[offset] = sum[0] + dev_weight[8655872+blockIdxy+blockidxx*M];
		dev_nn_node[offset] = (double)1/((double)1+exp(-dev_nn_node[offset]));
	}
}

// block = (batch_size,9), thread = 1024
__global__ void gpu_run22(double *dev_nn_node,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_nn_node[threadIdxx+8355+blockidxx*N] * dev_weight[8637440+threadIdxx+1024*blockIdxy+blockidxx*M];
	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 9460+blockIdxy+blockidxx*N;
		dev_nn_node[offset] = sum[0] + dev_weight[8655953+blockIdxy+blockidxx*M];
		dev_nn_node[offset] = exp((double)2*dev_nn_node[offset]);
		dev_nn_node[offset] = (dev_nn_node[offset]-(double)1)/(dev_nn_node[offset]+(double)1);
	}
}

// block = (batch_size,81), thread = 1
__global__ void gpu_backpropagation1(double *dev_nn_node,double *dev_delta,double *dev_policy_answer){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int offset = 9379+blockIdxy+blockidxx*N;
	double dev_answer = dev_policy_answer[blockIdxy+blockidxx*81];

	if(dev_answer == 2){ // answer = 2 = update 안하는걸로
		dev_delta[offset] = 0;
	}else{ // 아래 식이 맞나 모르겠다
		if(dev_answer == 0) dev_delta[offset] = dev_nn_node[offset];
		else dev_delta[offset] = -(double)((double)1-dev_nn_node[offset]);
	}
}

// block = (batch_size,9), thread = 1
__global__ void gpu_backpropagation2(double *dev_nn_node,double *dev_delta,double *dev_value_answer){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int offset = 9460+blockIdxy+blockidxx*N;

	if(dev_value_answer[blockIdxy+blockidxx*9] == 2){
		dev_delta[offset] = 0;
	}else{ // 아래 식이 맞나 모르겠다
		dev_delta[offset] = (double)2*(dev_nn_node[offset]-dev_value_answer[blockIdxy+blockidxx*9]);
		dev_delta[offset] *= ((double)1-(dev_nn_node[offset]*dev_nn_node[offset]));
	}
}

// block = (batch_size,1024), thread = 9
__global__ void gpu_backpropagation3(double *dev_nn_node,double *dev_delta,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[9];

	sum[threadIdxx] = dev_weight[8637440+blockIdxy+1024*threadIdxx+blockidxx*M]*dev_delta[9460+threadIdxx+blockidxx*N];

	int i = 9;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = (i+1)/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = (i+1)/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 8355+blockIdxy+blockidxx*N;
		
		dev_delta[offset] = sum[0];
		if(dev_nn_node[offset] > 0) dev_delta[offset] *= lambda;
		else dev_delta[offset] *= (dev_nn_node[offset]+alphalambda);
	}
}

// block = (batch_size,1024), thread = 81
__global__ void gpu_backpropagation4(double *dev_nn_node,double *dev_delta,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[81];

	sum[threadIdxx] = dev_weight[8554496+blockIdxy+1024*threadIdxx+blockidxx*M]*dev_delta[9379+threadIdxx+blockidxx*N];

	int i = 81;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = (i+1)/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = (i+1)/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 7331+blockIdxy+blockidxx*N;
		
		dev_delta[offset] = sum[0];
		if(dev_nn_node[offset] > 0) dev_delta[offset] *= lambda;
		else dev_delta[offset] *= (dev_nn_node[offset]+alphalambda);
	}
}

// block = (batch_size,1024), thread = 1024, layer 8 10
__global__ void gpu_backpropagation5(double *dev_nn_node,double *dev_delta,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_weight[7505920+blockIdxy+1024*threadIdxx+blockidxx*M]*dev_delta[8355+threadIdxx+blockidxx*N];

	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		dev_delta[6307+blockIdxy+blockidxx*N] = sum[0];
	}
}

// block = (batch_size,1024), thread = 1024, layer 8 9
__global__ void gpu_backpropagation6(double *dev_nn_node,double *dev_delta,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_weight[6457344+blockIdxy+1024*threadIdxx+blockidxx*M]*dev_delta[7331+threadIdxx+blockidxx*N];

	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 6307+blockIdxy+blockidxx*N;
		
		dev_delta[offset] += sum[0]; // 여기만 += 으로 함
		if(dev_nn_node[offset] > 0) dev_delta[offset] *= lambda;
		else dev_delta[offset] *= (dev_nn_node[offset]+alphalambda);
	}
}

// block = (batch_size,1024), thread = 1024, layer 7 8
__global__ void gpu_backpropagation7(double *dev_nn_node,double *dev_delta,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_weight[5408768+blockIdxy+1024*threadIdxx+blockidxx*M]*dev_delta[6307+threadIdxx+blockidxx*N];

	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 5283+blockIdxy+blockidxx*N;
		
		dev_delta[offset] = sum[0];
		if(dev_nn_node[offset] > 0) dev_delta[offset] *= lambda;
		else dev_delta[offset] *= (dev_nn_node[offset]+alphalambda);
	}
}

// block = (batch_size,1024), thread = 1024, layer 6 7
__global__ void gpu_backpropagation8(double *dev_nn_node,double *dev_delta,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_weight[4360192+blockIdxy+1024*threadIdxx+blockidxx*M]*dev_delta[5283+threadIdxx+blockidxx*N];

	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 4259+blockIdxy+blockidxx*N;
		
		dev_delta[offset] = sum[0];
		if(dev_nn_node[offset] > 0) dev_delta[offset] *= lambda;
		else dev_delta[offset] *= (dev_nn_node[offset]+alphalambda);
	}
}

// block = (batch_size,1024), thread = 1024, layer 5 6
__global__ void gpu_backpropagation9(double *dev_nn_node,double *dev_delta,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_weight[3311616+blockIdxy+1024*threadIdxx+blockidxx*M]*dev_delta[4259+threadIdxx+blockidxx*N];

	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 3235+blockIdxy+blockidxx*N;
		
		dev_delta[offset] = sum[0];
		if(dev_nn_node[offset] > 0) dev_delta[offset] *= lambda;
		else dev_delta[offset] *= (dev_nn_node[offset]+alphalambda);
	}
}

// block = (batch_size,1024), thread = 1024, layer 4 5
__global__ void gpu_backpropagation10(double *dev_nn_node,double *dev_delta,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_weight[2263040+blockIdxy+1024*threadIdxx+blockidxx*M]*dev_delta[3235+threadIdxx+blockidxx*N];

	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 2211+blockIdxy+blockidxx*N;
		
		dev_delta[offset] = sum[0];
		if(dev_nn_node[offset] > 0) dev_delta[offset] *= lambda;
		else dev_delta[offset] *= (dev_nn_node[offset]+alphalambda);
	}
}

// block = (batch_size,1024), thread = 1024, layer 3 4
__global__ void gpu_backpropagation11(double *dev_nn_node,double *dev_delta,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_weight[1214464+blockIdxy+1024*threadIdxx+blockidxx*M]*dev_delta[2211+threadIdxx+blockidxx*N];

	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 1187+blockIdxy+blockidxx*N;
		
		dev_delta[offset] = sum[0];
		if(dev_nn_node[offset] > 0) dev_delta[offset] *= lambda;
		else dev_delta[offset] *= (dev_nn_node[offset]+alphalambda);
	}
}

// block = (batch_size,1024), thread = 1024, layer 2 3
__global__ void gpu_backpropagation12(double *dev_nn_node,double *dev_delta,double *dev_weight){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[1024];

	sum[threadIdxx] = dev_weight[165888+blockIdxy+1024*threadIdxx+blockidxx*M]*dev_delta[1187+threadIdxx+blockidxx*N];

	int i = 1024;
	while(i != 1){
		__syncthreads();
		if(threadIdxx < i/2){
			i = i/2;
			sum[threadIdxx] += sum[threadIdxx+i];
		}else{
			i = i/2;
		}
	}
	if(threadIdxx == 0){
		int offset = 163+blockIdxy+blockidxx*N;
		
		dev_delta[offset] = sum[0];
		if(dev_nn_node[offset] > 0) dev_delta[offset] *= lambda;
		else dev_delta[offset] *= (dev_nn_node[offset]+alphalambda);
	}
}

// block = (N,1024), thread = batch_size, calculate gradients
__global__ void gpu_backpropagation13(double *dev_nn_node,double *dev_delta,double *dev_weight_m,double *dev_weight_v,double *dev_weight,double *dev_lr){
	int blockidxx = blockIdx.x;
	int blockIdxy = blockIdx.y;
	int threadIdxx = threadIdx.x;
	int i,offset;
	double gradient;
	__shared__ double sum[MAX_batch_size];

	if(blockidxx <= 162){
		i = 1;
	}else if(blockidxx <= 1186){
		if(blockIdxy < 162){
			sum[threadIdxx] = dev_delta[blockidxx+threadIdxx*N]*dev_nn_node[1+blockIdxy+threadIdxx*N];
			i = MAX_batch_size;
			while(i != 1){
				__syncthreads();
				if(threadIdxx < i/2){
					i = (i+1)/2;
					sum[threadIdxx] += sum[threadIdxx+i];
				}else{
					i = (i+1)/2;
				}
			}
			if(threadIdxx == 0){
				offset = (blockidxx-163)*162+blockIdxy;
				gradient = sum[0]/(double)MAX_batch_size + l2_constant * dev_weight[offset];
				dev_weight_m[offset] = beta1 * dev_weight_m[offset] + (1-beta1) * gradient;
				dev_weight_v[offset] = beta2 * dev_weight_v[offset] + (1-beta2) * gradient * gradient;
				dev_weight[offset] -= (dev_lr[blockidxx*1024+blockIdxy] * dev_weight_m[offset] / sqrt(dev_weight_v[offset] + epsilon));
			}
		}
	}else if(blockidxx <= 8354){
		sum[threadIdxx] = dev_delta[blockidxx+threadIdxx*N]*dev_nn_node[((int)(blockidxx-1187)/1024)*1024+163+blockIdxy+threadIdxx*N];
		i = MAX_batch_size;
		while(i != 1){
			__syncthreads();
			if(threadIdxx < i/2){
				i = (i+1)/2;
				sum[threadIdxx] += sum[threadIdxx+i];
			}else{
				i = (i+1)/2;
			}
		}
		if(threadIdxx == 0){
			offset = 165888 + (blockidxx-1187)*1024+blockIdxy;
			gradient = sum[0]/(double)MAX_batch_size + l2_constant * dev_weight[offset];
			dev_weight_m[offset] = beta1 * dev_weight_m[offset] + (1-beta1) * gradient;
			dev_weight_v[offset] = beta2 * dev_weight_v[offset] + (1-beta2) * gradient * gradient;
			dev_weight[offset] -= (dev_lr[blockidxx*1024+blockIdxy] * dev_weight_m[offset] / sqrt(dev_weight_v[offset] + epsilon));
		}
	}else if(blockidxx <= 9378){
		sum[threadIdxx] = dev_delta[blockidxx+threadIdxx*N]*dev_nn_node[((int)(blockidxx-2211)/1024)*1024+163+blockIdxy+threadIdxx*N];
		i = MAX_batch_size;
		while(i != 1){
			__syncthreads();
			if(threadIdxx < i/2){
				i = (i+1)/2;
				sum[threadIdxx] += sum[threadIdxx+i];
			}else{
				i = (i+1)/2;
			}
		}
		if(threadIdxx == 0){
			offset = 165888 + (blockidxx-1187)*1024+blockIdxy;
			gradient = sum[0]/(double)MAX_batch_size + l2_constant * dev_weight[offset];
			dev_weight_m[offset] = beta1 * dev_weight_m[offset] + (1-beta1) * gradient;
			dev_weight_v[offset] = beta2 * dev_weight_v[offset] + (1-beta2) * gradient * gradient;
			dev_weight[offset] -= (dev_lr[blockidxx*1024+blockIdxy] * dev_weight_m[offset] / sqrt(dev_weight_v[offset] + epsilon));
		}
	}else if(blockidxx <= 9459){
		sum[threadIdxx] = dev_delta[blockidxx+threadIdxx*N]*dev_nn_node[7331+blockIdxy+threadIdxx*N];
		i = MAX_batch_size;
		while(i != 1){
			__syncthreads();
			if(threadIdxx < i/2){
				i = (i+1)/2;
				sum[threadIdxx] += sum[threadIdxx+i];
			}else{
				i = (i+1)/2;
			}
		}
		if(threadIdxx == 0){
			offset = 8554496 + (blockidxx-9379)*1024+blockIdxy;
			gradient = sum[0]/(double)MAX_batch_size + l2_constant * dev_weight[offset];
			dev_weight_m[offset] = beta1 * dev_weight_m[offset] + (1-beta1) * gradient;
			dev_weight_v[offset] = beta2 * dev_weight_v[offset] + (1-beta2) * gradient * gradient;
			dev_weight[offset] -= (dev_lr[blockidxx*1024+blockIdxy] * dev_weight_m[offset] / sqrt(dev_weight_v[offset] + epsilon));
		}
	}else{
		sum[threadIdxx] = dev_delta[blockidxx+threadIdxx*N]*dev_nn_node[8355+blockIdxy+threadIdxx*N];
		i = MAX_batch_size;
		while(i != 1){
			__syncthreads();
			if(threadIdxx < i/2){
				i = (i+1)/2;
				sum[threadIdxx] += sum[threadIdxx+i];
			}else{
				i = (i+1)/2;
			}
		}
		if(threadIdxx == 0){
			offset = 8637440 + (blockidxx-9460)*1024+blockIdxy;
			gradient = sum[0]/(double)MAX_batch_size + l2_constant * dev_weight[offset];
			dev_weight_m[offset] = beta1 * dev_weight_m[offset] + (1-beta1) * gradient;
			dev_weight_v[offset] = beta2 * dev_weight_v[offset] + (1-beta2) * gradient * gradient;
			dev_weight[offset] -= (dev_lr[blockidxx*1024+blockIdxy] * dev_weight_m[offset] / sqrt(dev_weight_v[offset] + epsilon));
		}
	}
}

// block = N, thread = batch_size
__global__ void gpu_backpropagation14(double *dev_delta,double *dev_weight_m,double *dev_weight_v,double *dev_weight,double *dev_lr){
	int blockidxx = blockIdx.x;
	int threadIdxx = threadIdx.x;
	__shared__ double sum[MAX_batch_size];

	if(blockidxx > 162){
		sum[threadIdxx] = dev_delta[blockidxx+threadIdxx*N];
		int i = MAX_batch_size;
		while(i != 1){
			__syncthreads();
			if(threadIdxx < i/2){
				i = (i+1)/2;
				sum[threadIdxx] += sum[threadIdxx+i];
			}else{
				i = (i+1)/2;
			}
		}
		if(threadIdxx == 0){
			int offset = 8646656+blockidxx-163;
			double gradient = sum[0]/(double)MAX_batch_size + l2_constant * dev_weight[offset];
			dev_weight_m[offset] = beta1 * dev_weight_m[offset] + (1-beta1) * gradient;
			dev_weight_v[offset] = beta2 * dev_weight_v[offset] + (1-beta2) * gradient * gradient;
			dev_weight[offset] -= (dev_lr[blockidxx] * dev_weight_m[offset] / sqrt(dev_weight_v[offset] + epsilon));
		}
	}
}

int *dev_rawdata,*dev_support_lr,*dev_add,*dev_memo_rawdata;
double *dev_nn_node,*dev_delta;
double *dev_policy_answer,*dev_value_answer;
double *dev_weight_m,*dev_weight_v,*dev_weight;
double *dev_lr;
double *zero_array;
int *copy_int;
double *copy_double,*copy_double2;
double *dev_memo_policy_answer,*dev_memo_value_answer;

void gpu_init(){
	cudaSetDevice(gpu_num);
	cudaMalloc((void**)&dev_rawdata,81*MAX_batch_size*sizeof(int));
	cudaMalloc((void**)&dev_add,MAX_batch_size*81*sizeof(int));
	cudaMalloc((void**)&dev_nn_node,(long long)N*(long long)MAX_batch_size*(long long)sizeof(double));
	cudaMalloc((void**)&dev_delta,(long long)N*(long long)MAX_batch_size*(long long)sizeof(double));
	cudaMalloc((void**)&dev_policy_answer,81*MAX_batch_size*sizeof(double));
	cudaMalloc((void**)&dev_value_answer,9*MAX_batch_size*sizeof(double));
	cudaMalloc((void**)&dev_weight_m,M*sizeof(double));
	cudaMalloc((void**)&dev_weight_v,M*sizeof(double));
	cudaMalloc((void**)&dev_weight,(long long)M*(long long)MAX_batch_size*(long long)sizeof(double));
	cudaMalloc((void**)&dev_lr,N*1024*sizeof(double));
	cudaMalloc((void**)&dev_support_lr,N*1024*sizeof(int));
	zero_array = (double*)malloc(M*sizeof(double));
	for(int i=0; i<M; i++) zero_array[i] = 0;
	copy_int = (int*)malloc(max(max(81*MAX_batch_size,N*1024),MAX_train_data_cnt*81*81)*sizeof(int));
	copy_double = (double*)malloc(max(max(81*MAX_batch_size,M),MAX_train_data_cnt*81*81)*sizeof(double));
	copy_double2 = (double*)malloc(MAX_train_data_cnt*81*9*sizeof(double));
	cudaMalloc((void**)&dev_memo_rawdata,sizeof(int)*81*MAX_train_data_cnt*81);
	cudaMalloc((void**)&dev_memo_policy_answer,sizeof(double)*81*MAX_train_data_cnt*81);
	cudaMalloc((void**)&dev_memo_value_answer,sizeof(double)*81*MAX_train_data_cnt*9);
}

bool check[N*MAX_batch_size];
double node[N*MAX_batch_size],node2[N*MAX_batch_size];
double delta[N*MAX_batch_size],weight_m[M],weight_v[M],weight2[M];
void cpu_run(){
	for(int i=0; i<N*batch_size; i++){
		check[i] = false;
		node[i] = 0;
	}
	for(int i=0; i<batch_size; i++){
		for(int j=0; j<9; j++){
			for(int k=0; k<9; k++){
				copy_int[i*81+j*9+k] = raw_data[i][j][k];
			}
		}
		for(int j=0; j<81; j++){
			if(copy_int[i*81+j] == 1) node[i*N+j+1] = 1;
			else node[i*N+j+1] = 0;
			if(copy_int[i*81+j] == -1) node[i*N+j+82] = -1;
			else node[i*N+j+82] = 0;
			check[i*N+j+1] = check[i*N+j+82] = true;
		}
		node[i*N] = 1;
		check[i*N] = true;
		
		double mean = 0,s_dev = 0;
		for(int j=1; j<=162; j++) mean += node[i*N+j];
		mean /= (double)162;
		for(int j=1; j<=162; j++){
			node[i*N+j] -= mean;
			s_dev += (node[i*N+j]*node[i*N+j]);
		}
		s_dev /= (double)162;
		s_dev = sqrt(s_dev);
		for(int j=1; j<=162; j++){
			node[i*N+j] -= mean;
			if(s_dev != 0) node[i*N+j] /= s_dev;
		}
	}
	cudaMemcpy(copy_double,dev_nn_node,N*batch_size*sizeof(double),cudaMemcpyDeviceToHost);
	for(int i=0; i<batch_size; i++) for(int j=1; j<=162; j++) node[i*N+j] = copy_double[i*N+j];
	std::sort(weight,weight+M,[&](WeightType &cmpx,WeightType &cmpy){
		if(cmpx.end != cmpy.end) return cmpx.end < cmpy.end;
		return cmpx.start < cmpy.start;
	});
	for(int i=0; i<M; i++){
		for(int j=0; j<batch_size; j++){
			int s = weight[i].start+j*N,e = weight[i].end+j*N;
			double v = weight[i].value;
		
			if(!check[s]){
				if(node[s] > 0) node[s] *= lambda;
				else node[s] = alphalambda*(exp(node[s])-(double)1);
				check[s] = true;
			}
			node[e] += node[s]*v;
		}
	}
	for(int i=0; i<batch_size; i++){
		int s;
		for(int j=9379; j<=9459; j++){
			s = i*N+j;
			node[s] = (double)1/((double)1+exp(-node[s]));
		}
		for(int j=9460; j<=9468; j++){
			s = i*N+j;
			node[s] = (exp(2*node[s])-(double)1)/(exp(2*node[s])+(double)1);
		}
	}
}

void cpu_backpropagation(){
	for(int i=0; i<N*batch_size; i++){
		check[i] = false;
		delta[i] = 0;
	}	
	for(int i=0; i<batch_size; i++){
		for(int j=9379; j<=9459; j++){
			if(policy_answer[i][(j-9379)/9][(j-9379)%9] != 2) delta[i*N+j] = ((double)1-node[i*N+j])*policy_answer[i][(j-9379)/9][(j-9379)%9] + node[i*N+j]*((double)1-policy_answer[i][(j-9379)/9][(j-9379)%9]);
			check[i*N+j] = true;
		}
		for(int j=9460; j<=9468; j++){
			if(value_answer[i][j-9460] != 2) delta[i*N+j] = (node[i*N+j]-value_answer[i][j-9460])*(double)(2);
			delta[i*N+j] *= ((double)1-node[i*N+j])*((double)1+node[i*N+j]);
			check[i*N+j] = true;
		}
	}
	std::sort(weight,weight+M,[&](WeightType &cmpx,WeightType &cmpy){
		if(cmpx.start != cmpy.start) return cmpx.start > cmpy.start;
		return cmpx.end < cmpy.end;
	});
	for(int i=0; i<M; i++){
		int s,e;
		for(int j=0; j<batch_size; j++){
			s = weight[i].start+j*N; e = weight[i].end+j*N;
			if(!check[e]){
				check[e] = true;
				if(node[e] > 0) delta[e] *= lambda;
				else delta[e] *= (node[e]+alphalambda);
			}
			delta[s] += weight[i].value * delta[e];
		}
	}
	std::sort(weight,weight+M,[&](WeightType &cmpx,WeightType &cmpy){
		if(cmpx.start == 0 && cmpy.start == 0) return cmpx.end < cmpy.end;
		else if(cmpx.start == 0) return false;
		else if(cmpy.start == 0) return true;
		if(cmpx.end != cmpy.end) return cmpx.end < cmpy.end;
		return cmpx.start < cmpy.start;
	});
	double lr = (double)learning_rate*sqrt((double)1 - pow(beta2, adam_t)) / ((double)1 - pow(beta1, adam_t));
	for(int i=0; i<M; i++){
		int s,e;
		double g = 0;
		for(int j=0; j<batch_size; j++){
			s = weight[i].start+j*N; e = weight[i].end+j*N;
			g += delta[e] * node[s];
		}
		g /= (double)batch_size;
		g += (l2_constant * weight[i].value);
		weight_m[i] = beta1 * weight_m[i] + (1-beta1)*g;
		weight_v[i] = beta2 * weight_v[i] + (1-beta2)*g*g;
		weight[i].value -= lr * weight_m[i] / (sqrt(weight_v[i]+epsilon));
	}
}

void gpu_run(){
	clock_t clock_begin = clock();
	dim3 dim_2(batch_size,1024),dim_3(batch_size,81),dim_4(batch_size,9);

	// need dev_rawdata setting
	gpu_run1<<<batch_size,162>>>(dev_rawdata,dev_nn_node);//printf("gpu_run : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_run2<<<dim_2,162>>>(dev_nn_node,dev_weight);//printf("gpu_run : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_run4<<<dim_2,1024>>>(dev_nn_node,dev_weight);//printf("gpu_run : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_run6<<<dim_2,1024>>>(dev_nn_node,dev_weight);//printf("gpu_run : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_run8<<<dim_2,1024>>>(dev_nn_node,dev_weight);//printf("gpu_run : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_run10<<<dim_2,1024>>>(dev_nn_node,dev_weight);//printf("gpu_run : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_run12<<<dim_2,1024>>>(dev_nn_node,dev_weight);//printf("gpu_run : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_run14<<<dim_2,1024>>>(dev_nn_node,dev_weight);//printf("gpu_run : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_run16<<<dim_2,1024>>>(dev_nn_node,dev_weight);//printf("gpu_run : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_run18<<<dim_2,1024>>>(dev_nn_node,dev_weight);//printf("gpu_run : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_run20<<<dim_3,1024>>>(dev_nn_node,dev_weight);//printf("gpu_run : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_run22<<<dim_4,1024>>>(dev_nn_node,dev_weight);
	printf("gpu_run : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);
	/*puts("cpu_run");
	cpu_run();
	cudaMemcpy(node2,dev_nn_node,(long long)N*batch_size*sizeof(double),cudaMemcpyDeviceToHost);
	for(int i=0; i<N*batch_size; i++){
		if(abs(node[i]-node2[i]) > epsilon && i%N != 0){
			printf("%d %d : %.15f %.15f\n",i,i%N,node[i],node2[i]);
			for(int j=0; j<9; j++){
				for(int k=0; k<9; k++){
					printf("%d ",raw_data[i/N][j][k]);
				}
				puts("");
			}
			puts("");
			//for(int j=1; j<=162; j++) printf("%d : %.15f %.15f\n",j,node[j],node2[j]);
			exit(0);
		}
	}*/
}

void gpu_backpropagation(){
	clock_t clock_begin = clock();
	dim3 dim_0(1246,6947);
	dim3 dim_1(batch_size,81),dim_2(batch_size,9),dim_3(batch_size,1024),dim_4(N,1024);

	// need dev_policy_answer,dev_value_answer setting
	gpu_backpropagation1<<<dim_1,1>>>(dev_nn_node,dev_delta,dev_policy_answer);//printf("gpu_backpropagation : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_backpropagation2<<<dim_2,1>>>(dev_nn_node,dev_delta,dev_value_answer);//printf("gpu_backpropagation : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_backpropagation3<<<dim_3,9>>>(dev_nn_node,dev_delta,dev_weight);//printf("gpu_backpropagation : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_backpropagation4<<<dim_3,81>>>(dev_nn_node,dev_delta,dev_weight);//printf("gpu_backpropagation : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_backpropagation5<<<dim_3,1024>>>(dev_nn_node,dev_delta,dev_weight);//printf("gpu_backpropagation : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_backpropagation6<<<dim_3,1024>>>(dev_nn_node,dev_delta,dev_weight);//printf("gpu_backpropagation : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_backpropagation7<<<dim_3,1024>>>(dev_nn_node,dev_delta,dev_weight);//printf("gpu_backpropagation : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_backpropagation8<<<dim_3,1024>>>(dev_nn_node,dev_delta,dev_weight);//printf("gpu_backpropagation : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_backpropagation9<<<dim_3,1024>>>(dev_nn_node,dev_delta,dev_weight);//printf("gpu_backpropagation : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_backpropagation10<<<dim_3,1024>>>(dev_nn_node,dev_delta,dev_weight);//printf("gpu_backpropagation : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_backpropagation11<<<dim_3,1024>>>(dev_nn_node,dev_delta,dev_weight);//printf("gpu_backpropagation : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_backpropagation12<<<dim_3,1024>>>(dev_nn_node,dev_delta,dev_weight);//printf("gpu_backpropagation : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_backpropagation13<<<dim_4,batch_size>>>(dev_nn_node,dev_delta,dev_weight_m,dev_weight_v,dev_weight,dev_lr);//printf("gpu_backpropagation : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_backpropagation14<<<N,batch_size>>>(dev_delta,dev_weight_m,dev_weight_v,dev_weight,dev_lr);//printf("gpu_backpropagation : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);clock_begin = clock();
	gpu_copy_weight<<<dim_0,MAX_batch_size>>>(dev_weight);

	printf("gpu_backpropagation : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);
	/*puts("cpu_backpropagation");
	cpu_run();
	cpu_backpropagation();
	cudaMemcpy(weight2,dev_weight,sizeof(double)*M,cudaMemcpyDeviceToHost);
	for(int i=0; i<M; i++){
		if(abs(weight[i].value-weight2[i]) > epsilon){
			printf("%d : %.15f %.15f\n",i,weight[i].value,weight2[i]);
			exit(0);
		}
		//printf("%d : %.15f %.15f\n",i,weight[i].value,weight2[i]);
	}*/
}

std::vector<std::pair<int,int>> memo_games[MAX_train_data_cnt];

void set_input(int pos,int x,int y){
	for(int i=0; i<9; i++) for(int j=0; j<9; j++) raw_data[pos][i][j] = 0;
	for(int i=0; i<y; i++){
		int t1,t2;

		t1 = memo_games[x][i].first;
		t2 = memo_games[x][i].second;
		raw_data[pos][t1][t2] = 1;
		for(int j=0; j<9; j++) for(int k=0; k<9; k++) raw_data[pos][j][k] = -raw_data[pos][j][k];
	}
	bool finish_game = false;
	
	if(y != 0){
		int sum1 = 0,sum2 = 0;
		for(int i=0; i<9; i++){
			if(raw_data[pos][memo_games[x][y-1].first][i] == 1) sum1 |= (1<<i);
			if(raw_data[pos][memo_games[x][y-1].first][i] == -1) sum2 |= (1<<i);
		}
		if(win_t(sum1) || win_t(sum2) || sum1+sum2 == 511) finish_game = true;
	}else{
		finish_game = true;
	}

	if(finish_game){
		for(int i=0; i<9; i++){
			for(int j=0; j<9; j++){
				if(raw_data[pos][i][j] != 0) policy_answer[pos][i][j] = 2;
				else policy_answer[pos][i][j] = 0;
			}
		}
	}else{
		for(int i=0; i<9; i++) for(int j=0; j<9; j++) policy_answer[pos][i][j] = 2;
		for(int i=0; i<9; i++){
			if(raw_data[pos][memo_games[x][y].first][i] != 0) continue;
			policy_answer[pos][memo_games[x][y].first][i] = 0;
		}
	}
	policy_answer[pos][memo_games[x][y].first][memo_games[x][y].second] = 1;
	for(int i=0; i<9; i++) value_answer[pos][i] = 2;
	value_answer[pos][memo_games[x][y].first] = (y%2 == 0) ? -memo_games[x].back().first : memo_games[x].back().first;
}

void train(int epoch,int file_start,int file_end){
	char s[100];
	dim3 dim_0(1246,6947),dim_1(N,1024),dim_2(MAX_batch_size,81);
	std::pair<int,std::pair<int,int>> *play_data;

	for(int i=0; i<M; i++) copy_double[i] = weight[i].value;
	cudaMemcpy(dev_weight,copy_double,M*sizeof(double),cudaMemcpyHostToDevice);
	gpu_copy_weight<<<dim_0,MAX_batch_size>>>(dev_weight);
	cudaMemcpy(dev_weight_m,zero_array,M*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_weight_v,zero_array,M*sizeof(double),cudaMemcpyHostToDevice);

	training_size = 0;
	play_data = (std::pair<int,std::pair<int,int>>*)malloc(sizeof(std::pair<int,std::pair<int,int>>)*(file_end-file_start+1)*81);
	for(int i=file_start; i<=file_end; i++){
		sprintf(s,"../save_result2/result%05d.txt",i);
		FILE *f = fopen(s,"r");
		fgets(s,50,f);
		fgets(s,50,f);

		memo_games[i-file_start].clear();
		int j = 0;
		while(true){
			fgets(s,50,f);
			if(!('0' <= s[2] && s[2] <= '8')){
				memo_games[i-file_start].push_back({s[0]-'0',0});
				if(memo_games[i-file_start].back().first == 2) memo_games[i-file_start].back().first = -1;
				break;
			}
			memo_games[i-file_start].push_back({s[0]-'0',s[2]-'0'});
			play_data[training_size++] = {0,{i-file_start,j++}};
		}
		fclose(f);
	}
	for(int i=0; i<training_size; i++) play_data[i].first = rand();
	sort(play_data,play_data+training_size);
	for(int i=0; i<training_size; i++){
		set_input(0,play_data[i].second.first,play_data[i].second.second);
		for(int j=0; j<81; j++){
			copy_int[i*81+j] = raw_data[0][j/9][j%9];
			copy_double[i*81+j] = policy_answer[0][j/9][j%9];
		}
		for(int j=0; j<9; j++){
			copy_double2[i*9+j] = value_answer[0][j];
		}
	}
	cudaMemcpy(dev_memo_rawdata,copy_int,sizeof(int)*81*training_size,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_memo_policy_answer,copy_double,sizeof(double)*81*training_size,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_memo_value_answer,copy_double2,sizeof(double)*9*training_size,cudaMemcpyHostToDevice);

	adam_t = 0;
	for(int i=0; i<N*1024; i++) copy_int[i] = 0;
	cudaMemcpy(dev_support_lr,copy_int,sizeof(int)*N*1024,cudaMemcpyHostToDevice);

	//training_size = MAX_batch_size * 20;
	for(int iepoch = 1; iepoch <= epoch; iepoch++){
		printf("epoch = %d\n",iepoch);
		adam_t++;
		gpu_update_lr<<<dim_1,1>>>(dev_support_lr,dev_lr);
		gpu_init_dev_add<<<dim_2,1>>>(dev_add);
		for(int i=0; i<training_size; i+=MAX_batch_size){
			//clock_t clock_begin = clock();
			printf("i = %d\n",i);
			batch_size = min(training_size-i,MAX_batch_size);
			dim3 dim_3(batch_size,81);
			gpu_set_dev_add<<<dim_3,1>>>(dev_add);
			gpu_copy_nn_input<<<dim_3,1>>>(dev_rawdata,dev_memo_rawdata,dev_policy_answer,dev_memo_policy_answer,dev_value_answer,dev_memo_value_answer,dev_add);
			//printf("gpu_nn_input : %.15f\n",(double)(clock()-clock_begin)/(double)CLOCKS_PER_SEC);

			/*cudaMemcpy(copy_int,dev_rawdata,batch_size*81*sizeof(int),cudaMemcpyDeviceToHost);
			cudaMemcpy(copy_double,dev_policy_answer,batch_size*81*sizeof(double),cudaMemcpyDeviceToHost);
			cudaMemcpy(copy_double2,dev_value_answer,batch_size*9*sizeof(double),cudaMemcpyDeviceToHost);
			for(int j=0; j<batch_size; j++){
				for(int k=0; k<81; k++) raw_data[j][k/9][k%9] = copy_int[j*81+k];
				for(int k=0; k<81; k++) policy_answer[j][k/9][k%9] = copy_double[j*81+k];
				for(int k=0; k<9; k++) value_answer[j][k] = copy_double2[j*9+k];
			}*/

			gpu_run();
			gpu_backpropagation();
		}
	}
	printf("calculating policy_accuracy and total loss\n");

	double policy_accuracy = 0,policy_error = 0,value_error = 0,weight_loss = 0;
	gpu_init_dev_add<<<dim_2,1>>>(dev_add);
	for(int i=0; i<training_size; i+=MAX_batch_size){
		batch_size = min(training_size-i,MAX_batch_size);
		dim3 dim_3(batch_size,81);
		gpu_set_dev_add<<<dim_3,1>>>(dev_add);
		gpu_copy_nn_input<<<dim_3,1>>>(dev_rawdata,dev_memo_rawdata,dev_policy_answer,dev_memo_policy_answer,dev_value_answer,dev_memo_value_answer,dev_add);
		gpu_run();
		cudaMemcpy(copy_double,dev_nn_node,N*batch_size*sizeof(double),cudaMemcpyDeviceToHost);
		cudaMemcpy(copy_double2,dev_policy_answer,81*batch_size*sizeof(double),cudaMemcpyDeviceToHost);
		for(int j=0; j<batch_size; j++){
			int max_idx = 0,correct_idx;
			for(int k=0; k<81; k++){
				if(copy_double2[j*81+k] == 2) continue;
				if(copy_double2[j*81+k] == 0) policy_error -= log((double)1-copy_double[j*N+9379+k]);
				else policy_error -= log(copy_double[j*N+9379+k]);
				if(copy_double2[j*81+k] == 1) correct_idx = k;
				if(copy_double[j*N+9379+k] > copy_double[j*N+9379+max_idx]) max_idx = k;
			}
			if(max_idx == correct_idx){
				policy_accuracy++;
			}
		}
		cudaMemcpy(copy_double2,dev_value_answer,9*batch_size*sizeof(double),cudaMemcpyDeviceToHost);
		for(int j=0; j<batch_size; j++){
			for(int k=0; k<9; k++){
				if(copy_double2[j*9+k] == 2) continue;
				value_error += pow(copy_double2[j*9+k]-copy_double[j*N+9460+k],2);
			}
		}
	}
	policy_accuracy /= (double)training_size;
	policy_accuracy *= 100;
	policy_error /= (double)training_size;
	value_error /= (double)training_size;
	printf("policy_accuracy = %.15f%\n",policy_accuracy);
	printf("policy_error = %.15f\n",policy_error);
	printf("value_error = %.15f\n",value_error);
	
	cudaMemcpy(copy_double,dev_weight,M*sizeof(double),cudaMemcpyDeviceToHost);
	for(int i=0; i<M; i++){
		weight[i].value = copy_double[i];
		weight_loss += l2_constant * pow(weight[i].value,2);
	}
	printf("weight_loss = %.15f\n",weight_loss);
	printf("saving weight...\n");
	
	sprintf(s, "weight%05d.txt",nn_num);
	FILE *out = fopen(s, "w");

	fprintf(out, "%d %d\n",N,M);
	for (int i = 0; i < M; i++) {
		fprintf(out, "%d %d %.15f\n", weight[i].start, weight[i].end, weight[i].value);
	}
	fclose(out);
	printf("saving weight complete!\n");
}

class board
{
public:
	board() :bb{ 0, }, playable{ 0 }, pb{ 9 }, t{ 1 } {}
	board(const board &) = default;
	void print() const
	{
		for (unsigned int i = 0; i != 9; ++i)
		{
			if (i && (i % 3 == 0))
				puts("-------------------------------------------------------------------------");
			for (unsigned int j = 0; j != 9; ++j)
			{
				if (j && (j % 3 == 0))
					printf("|￠");
				unsigned int A = (i / 3) * 3 + (j / 3);
				unsigned int B = (i % 3) * 3 + (j % 3);
				if (bb[1][9] & (1 << A))
					printf("￡I ");
				else if ((bb[0][9] & (1 << A)) || (bb[0][A] & (1 << B)))
					printf("￡Ø");
				else if (bb[1][A] & (1 << B))
					printf("￡I ");
				else printf("   ");
			}
			puts("");
		}
		printf("pb = %hhu\n", pb);
	}

	void printpb() const
	{
		puts("");
		if (pb == 9)
		{
			puts("¡a|￠¡a|￠¡a");
			puts("|¡|¡|≪|¡|¡|≪|¡|¡");
			puts("¡a|￠¡a|￠¡a");
			puts("|¡|¡|≪|¡|¡|≪|¡|¡");
			puts("¡a|￠¡a|￠¡a");
		}
		else
		{
			for (unsigned int i = 0; i != 3; ++i)
			{
				if (i)
					puts("|¡|¡|≪|¡|¡|≪|¡|¡");
				for (unsigned int j = 0; j != 3; ++j)
				{
					if (j)printf("|￠");
					if (i * 3 + j == pb)
						printf("¡a");
					else printf("  ");
				}
				puts("");
			}
		}
	}

	int get_log2(int x) {
		int cnt = 0;
		while (x != 1) {
			cnt++;
			x /= 2;
		}
		return cnt;
	}

	void play(unsigned char x)
	{
		if (pb != 9)
			if (((bb[0][pb] | bb[1][pb])&(1 << x)))
				return;
			else if (win_t(bb[t ^= 1][pb] |= (1 << x)))
			{
				bb[t][9] |= (1 << pb);
				playable |= (1 << pb);
			}
			else if ((bb[0][pb] | bb[1][pb]) == 0x1FF)
				playable |= (1 << pb);
		pb = (playable & (1 << x)) ? 9 : x;
	}
	inline bool win() const
	{
		return win_t(bb[t][9]);
	}
	inline bool end() const
	{
		return (playable == 0x1FF) || win();
	}
	inline unsigned char gett() const
	{
		return t;
	}
	inline std::vector<unsigned short> getbb() const
	{
		std::vector<unsigned short> x;
		for (int i = 0; i < 10; i++) x.push_back(bb[0][i]);
		for (int i = 0; i < 10; i++) x.push_back(bb[1][i]);
		return x;
	}
	void generate_moves(std::vector</*const*/ board> &v) const
	{
		if (pb == 9)
		{
			for (unsigned int i = 0; i != 9; ++i)
			{
				if (playable & (1 << i))continue;
				for (unsigned short x = 0x1FF ^ (bb[0][i] | bb[1][i]), a = 0;
					a = blsi_u32_[x]; x ^= a)
				{
					board temp = *this;
					temp.play(i, a);
					v.push_back(temp);
				}
			}
		}
		else
		{
			for (unsigned short x = 0x1FF ^ (bb[0][pb] | bb[1][pb]), a = 0;
				a = blsi_u32_[x]; x ^= a)
			{
				board temp = *this;
				temp.play(pb, a);
				v.push_back(temp);
			}
		}
	}
	inline void play(unsigned char pbb, unsigned short a)
	{
		unsigned short b = 1 << pbb;
		if (win_t(bb[t ^= 1][pbb] |= a))
		{
			bb[t][9] |= b;
			playable |= b;
		}
		else if ((bb[0][pbb] | bb[1][pbb]) == 0x1FF)
			playable |= b;
		board::pb = (playable & a) ? 9 : tzcnt_u32_[a];
	}
	unsigned short bb[2][10];
	unsigned short playable;
	unsigned char pb;
	unsigned char t;
};
std::ostream &operator<<(std::ostream& os, const board& b);

struct nodee
{
	int value_sum;
	unsigned int num_simul;
	unsigned int child_index_start, child_index_end;
	nodee() {
		value_sum = 0;
		num_simul = 0;
		child_index_start = child_index_end = 0;
	}
	nodee(int value_sum, unsigned int num_simul, unsigned int child_index_start, unsigned int child_index_end) 
		: value_sum(value_sum), num_simul(num_simul), child_index_start(child_index_start), child_index_end(child_index_end) {}
};

void gpu_nn_generate_moves(std::vector<board> &v,board *board_data){

}

double gpu_time;
class searcher
{
public:
	searcher(board x, float c) :c(c) {
		search_data = std::vector<nodee>(1),
			board_data.push_back(x);
	}
	void setting_cutting(int cutting_value) {
		cutting = cutting_value;
	}
	void search(unsigned int trials, float DRAW_REWARD)
	{
		do {
			_search(0, DRAW_REWARD, 0);
			if (search_data[0].child_index_end == 2) break;
		} while (--trials);
	}
	int search_within_time(double second, float DRAW_REWARD) // gpu_time이라는 변수를 추가함.
	{
		int num_simul = 0;
		clock_t end = clock() + second * CLOCKS_PER_SEC;
		gpu_time = 0;
		do {
			_search(0, DRAW_REWARD, 0);
			num_simul++;
			if (search_data[0].child_index_end == 2) break;
		} while (clock() < end+gpu_time);
		return num_simul;
	}
	int getresult_index() {
		unsigned int index = 1;
		int max = search_data[1].num_simul;
		for (unsigned char i = 2; i != search_data[0].child_index_end; ++i)
		{
			if (max < search_data[i].num_simul) {
				max = search_data[i].num_simul;
				index = i;
			}
		}
		return index;
	}
	board getresult()
	{
		return board_data[getresult_index()];
	}
	unsigned int numsimualation()
	{
		return search_data[0].num_simul;
	}
	void change_use_nn(bool value) {
		use_nn = value;
	}
	int get_search_data_size() {
		return (int)search_data.size();
	}
	int get_board_data_size() {
		return (int)board_data.size();
	}
	inline float UCT(unsigned int index, unsigned int parent_total_simul) const
	{
		return (float)search_data[index].value_sum / search_data[index].num_simul + 2 * sqrtf(c * log(parent_total_simul) / search_data[index].num_simul);
	}
	std::pair<unsigned int, int> _search(int index, float DRAW_REWARD, int lev)
	{
		if (search_data[index].child_index_start == search_data[index].child_index_end)
		{
			if (board_data[index].end())
			{
				search_data[index].num_simul = (INT_MAX >> 3);
				int reward = board_data[index].win() ? 1 : DRAW_REWARD;
				search_data[index].value_sum = reward * INT_MAX;
				return std::make_pair(1, reward);
			}
			search_data[index].child_index_start = board_data.size();
			{
				board temp = board_data[index];
				if (use_nn && lev < cut_lev){
					clock_t calc_time = clock();
					gpu_nn_generate_moves(board_data,&temp);
					gpu_time += (double)(clock()-calc_time);
				}else{
					temp.generate_moves(board_data);
				}
			}
			search_data[index].child_index_end = board_data.size();
			for (unsigned int i = search_data[index].child_index_start; i != search_data[index].child_index_end; ++i)
			{
				board temp = board_data[i];
				int reward = rand()%2 ? 1 : DRAW_REWARD;
				if (temp.gett() != board_data[i].gett())
					reward = -reward;
				search_data[index].value_sum -= reward;
				search_data.push_back(nodee(reward, 1, 0, 0));
			}
			search_data[index].num_simul = search_data[index].child_index_end - search_data[index].child_index_start;
			return std::make_pair( search_data[index].num_simul, search_data[index].value_sum );
		}
		unsigned int node_to_search = search_data[index].child_index_start;
		float max_UCT = UCT(search_data[index].child_index_start, search_data[index].num_simul);
		// max 값 찾기를 버블정렬 응용하면 더 효율적으로 가능할듯.
		for (unsigned int i = search_data[index].child_index_start + 1; i != search_data[index].child_index_end; ++i)
		{
			float temp = UCT(i, search_data[index].num_simul);
			if (temp > max_UCT)
			{
				max_UCT = temp;
				node_to_search = i;
			}
		}
		std::pair<unsigned int, int>  result = _search(node_to_search, DRAW_REWARD, lev + 1);
		result.second *= -1;
		search_data[index].num_simul += result.first;
		search_data[index].value_sum += result.second;
		return result;
	}
	std::vector <nodee> search_data;
	std::vector </*const*/ board> board_data;
	float c;
	bool use_nn;
};

void init()
{
	srand(time(NULL));
	memset(win_tbl,0,sizeof(win_tbl));
	for (unsigned short i = 0; i != 512; ++i){
		if ((i & (i >> 3) & (i >> 6) & 7) || (i & (i >> 1) & (i >> 2) & 0111) || (_popcnt16_[i & 0421] == 3) || (_popcnt16_[i & 0124] == 3)){
			win_tbl[i / 32] |= (1 << (i % 32));
		}
	}	
}

std::pair<int, int> find_movement(std::vector<unsigned short> b1, std::vector<unsigned short> b2) {
	for (int i = 0; i < 20; i++) {
		if (b2[i] - b1[i] == 0) continue;
		for (int j = 0; j < 10; j++) if (((1 << j)&(b2[i] - b1[i])) != 0) return std::make_pair(i, j);
	}
}

void run_cuda() {
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	printf("cudagetdevicecount = %d\n",count);
	for(int i=0; i<count; i++){
		cudaGetDeviceProperties(&prop,i);
		printf("--- general Information for device %d ---\n",i);
		printf("Name: %s\n",prop.name);
		printf("Compute capability: %d.%d\n",prop.major,prop.minor);
		printf("Clock rate : %d\n",prop.clockRate);
		printf("Device copy overlap: ");
		if(prop.deviceOverlap) printf("Enabled\n");
		else printf("Disabled\n");
		printf("Kernel execition timeout : ");
		if(prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else printf("Disabled\n");
		
		printf("--- Memory Information for device %d ---\n",i);
		printf("Total global mem: %ld\n",prop.totalGlobalMem);
		printf("Total constant Mem: %ld\n",prop.totalConstMem);
		printf("Max mem pitch: %ld\n", prop.memPitch);
		printf("Texture Alignment: %ld\n", prop.textureAlignment);
		printf("---MP information for device %d ---\n",i);
		printf("Multiprocessor count: %d\n",prop.multiProcessorCount);
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n",prop.warpSize);
		printf("Max threads per block: %d\n",prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n",prop.maxThreadsDim[0], prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
		puts("");
	}
}

void wait(double wait_second){
	int clock_tmp = 0;	
	clock_t begin = clock();
	clock_t end = clock()+(wait_second*CLOCKS_PER_SEC);

	while(clock() <= end){
		clock_tmp++;
		clock_tmp--;
	}
}

void all_run(){
	int printnum = 0;

	while (1)
	{
		board x;
		unsigned char t;
		t = 1;

		char print_filename[100];

		while(true){
			while(1){
				{
					FILE *in = fopen("record_num.txt", "r");
					if (in == NULL) continue;
					if (fscanf(in, "%d", &printnum) == EOF) continue;
					fclose(in);
					break;
				}			
			}
			if (printnum % cpu_cnt != 0 || printnum == -1 || printnum == 0){
				{
					FILE *out = fopen("record_num.txt", "w");
					fprintf(out, "%d\n", printnum + 1);
					fclose(out);
				}
				break;
			}
			wait(60); // wait 60 seconds
		}

		sprintf(print_filename, "../result/result%05d.txt", ++printnum);
		{
			FILE *out = fopen(print_filename, "w");
			fprintf(out,"%d %d\n\n",nn_num,nn_num);
			fclose(out);
		}

		double playtime = 9;
		
		while (!x.end())
		{
			std::vector<unsigned short> past_board = x.getbb();
			searcher a(x, 2);
			
			int make_mcts_node,mcts_search_num;
			if (false) {
				// nn 선공. mcts 후공
				if ((x.gett() ^ t) == 0) {
					a.change_use_nn(true);
					clock_t tmp = clock();
					mcts_search_num = a.search_within_time(playtime, 0);
					//printf("playtime = %.3f, real_playtime = %.3f\n",playtime,(double)(clock()-tmp)/(double)CLOCKS_PER_SEC);
					//printf("search_num = %d, make_node = %d\n",mcts_search_num,make_mcts_node);
				}
				else {
					a.change_use_nn(false);
					clock_t begin = clock();
					a.search(simulation_mcts, 0);
					playtime = (double)(clock()-begin) / (double)CLOCKS_PER_SEC;
					//printf("debug %.3f %.3f\n",playtime,(double)(clock()-begin) / (double)CLOCKS_PER_SEC);
					//printf("playtime = %.14f\n", playtime);
					mcts_search_num = simulation_mcts;
				}
			}
			else if (false) {
				// nn 후공, mcts 선공
				if ((x.gett() ^ t) == 0) {
					a.change_use_nn(false);
					clock_t begin = clock();
					a.search(simulation_mcts, 0);
					begin = clock() - begin;
					playtime = (double)begin / (double)CLOCKS_PER_SEC;
					mcts_search_num = simulation_mcts;
				}
				else {
					a.change_use_nn(true);
					clock_t tmp = clock();
					mcts_search_num = a.search_within_time(playtime, 0);
					//printf("playtime = %.3f, real_playtime = %.3f\n",playtime,(double)(clock()-tmp)/(double)CLOCKS_PER_SEC);
					//printf("search_num = %d, make_node = %d\n",mcts_search_num,make_mcts_node);
				}
			}
			else {
				// nn 선공, nn 후공
				int simulation_nn = simulation_mcts;
				a.change_use_nn(true);
				a.search(simulation_nn, 0);
			}
			make_mcts_node = a.board_data.size();
			//printf("search_num = %d, make_node = %d\n",mcts_search_num,make_mcts_node);

			x = a.getresult();
			//_wsystem(L"CLS");
			//int numsimulation = a.numsimualation();
			std::pair<int, int> movement = find_movement(past_board, x.getbb());
			if (movement.first >= 10) movement.first -= 10;
			{
				FILE *out = fopen(print_filename, "a");
				fprintf(out, "%d %d\n", movement.first, movement.second);
				fclose(out);
			}
			//x.print(); puts("");
		}
		FILE *out = fopen(print_filename, "a");
		if (!x.win()) {
			fprintf(out, "0 DRAW");
			//draw++;
		}
		else if ((x.gett() ^ t) == 1) {
			fprintf(out, "1 wins the game");
			//one++;
		}
		else {
			fprintf(out, "2 wins the game");
			//second++;
			//_wsystem(L"PAUSE");;
		}
		//printf("DRAW = %d,ONE = %d,TWO = %d\n",draw,one,second);
		fclose(out);
		{
			int done;
			FILE *in = fopen("done_num.txt","r");
			fscanf(in,"%d",&done);
			fclose(in);
			if(done%cpu_cnt == cpu_cnt-1){
				nn_num++;
				//nn_train();
				initNN();
				printnum = -1;
			}
			FILE *out = fopen("done_num.txt","w");
			fprintf(out,"%d\n",done+1);
			fclose(out);
		}
	}
}

int main()
{
	//run_cuda();
	init();
	gpu_init();
	//makeNN();
	initNN();
	puts("training start");
	train(200,2201,3200);
	puts("finish");

	/*nn_num = 0;
	cut_lev = 6; // cut_lev = mcts tree에서 node lev이 cut_lev보다 작을 때만 gpu_nn_generate함. 시간때문에 이렇게 설정함. value network 사용하면 해결될숟;ㅗ.
	cutting = 0.05;
	simulation_mcts = 1500000;
	cpu_cnt = 36; // 48개지만 협상을 통한 결과로 36개 배당받음.

	//run_cuda();
	srand((unsigned int)time(NULL));
	init();
	initNN(nn_num, true);
	gpu_init();
	all_run();*/

	return 0;
}
