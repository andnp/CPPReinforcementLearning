#include "../../FeedforwardNet/network.h"
#include <random>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
using namespace std;

int getHighest(vector<double> vect){
	int max = -9999999;
	int maxLoc = -1;
	for(int i = 0; i < vect.size(); i++){
		if(vect[i] > max){
			max = vect[i];
			maxLoc = i;
		}
	}
	return maxLoc;
}

vector<vector<double>> readFile(){
	vector<vector<double>> v;
	if (FILE *fp = fopen("../Tests/MNIST/mnist_train.csv", "r")) {
		char line[1024 * 256];
		char * string;
		while (fgets(line, sizeof(line), fp)){
			vector<double> vect;
			string = strtok(line, ",");
			vect.push_back(stod(string));
			while(string = strtok(NULL, ",")){
				vect.push_back(stod(string));
			}
			v.push_back(vect);
		}
		fclose(fp);
	} else {
		printf("error in loading file\n");
	}
	return v;
}

int main(){
	nnetwork n;

	int epochs = 5;

	int inputSize = 28*28;
	vector<int> layers = {100, 10};
	vector<int> types = {0, 0};
	vector<double> dropout = {0, 0};
	vector<double> lambda = {0,0};
	n.instantiate(inputSize, layers, types, dropout, lambda, .01);

	printf("------------Reading File----------\n");
	vector<vector<double>> data = readFile();
	printf("------------Training -------------\n");
	printf("%i\n", data.size()); 
	for(int k = 0; k < epochs; k++){
		for(int i = 0; i < 50000; i++){
			int target = (int)data[i][0];
			data[i].erase(data[i].begin());
			vector<double> targetVector = {0,0,0,0,0,0,0,0,0,0};
			targetVector[target] = 1;
			n.learn(targetVector, data[i]);
			// vector<vector<double>> o = n.fire(data[i]);
			// vector<double> outs = o[o.size() - 1];
			// for(int j = 0; j < outs.size(); j++){
			// 	printf("%f\n", outs[j]);
			// }
			if(i % 5000 == 0)
				printf(".");
		}
		printf("\n");

		int correct = 0;
		printf("------------Testing-------------\n");
		for(int i = 50000; i < 60000; i++){
			int target = (int)data[i][0];
			data[i].erase(data[i].begin());
			vector<vector<double>> o = n.fire(data[i]);
			int got = getHighest(o[o.size() - 1]);
			// printf("got: %i wanted: %i\n", got, target);
			if(got == target)
				correct++;
		}
		printf("correct: %f", (double)correct / 10000.0);
	}
	return 0;
}