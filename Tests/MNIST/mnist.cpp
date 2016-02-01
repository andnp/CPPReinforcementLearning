#include "../../FeedforwardNet/network.h"
#include <random>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
using namespace std;

int getHighest(vector<double> vect){
	double max = -9999999;
	int maxLoc = -1;
	for(int i = 0; i < vect.size(); i++){
		double val = vect[i];
		if(val > max){
			max = val;
			maxLoc = i;
		}
		// printf("%f\n", val);
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
	n.instantiate(inputSize, layers, types, dropout, lambda, .10);

	vector<double> targets(60000);

	printf("------------Reading File----------\n");
	vector<vector<double>> data = readFile();

	for(int i = 0; i < 60000; i++){
		targets[i] = data[i][0];
		data[i].erase(data[i].begin());
	}

	// printf("%f\n", targets[1]);
	// for(int i = 0; i < data[1].size(); i++){
	// 	printf("%f ", data[1][i]);
	// }
	// printf("\n");

	printf("------------Training -------------\n");
	printf("%i\n", data[0].size()); 
	for(int k = 0; k < epochs; k++){
		for(int i = 0; i < 1000; i++){
			double target = targets[i];
			vector<double> targetVector = {0,0,0,0,0,0,0,0,0,0};
			printf("----------------------\n");
			printf("Target: %f\n", target);
			vector<double> input = data[i];
			targetVector[target] = 1;
			vector<double> first = n.fire(input);
			n.learn(targetVector, input);
			vector<double> o = n.fire(input);
			for(int j = 0; j < first.size(); j++){
				printf("%f ", first[j]);
			}
			printf("\n");
			for(int j = 0; j < o.size(); j++){
				printf("%f ", o[j]);
			}
			printf("\n");
			// if(i % 5000 == 0)
			// 	printf(".");
		}
		printf("\n");

		int correct = 0;
		printf("------------Testing-------------\n");
		for(int i = 50000; i < 50100; i++){
			int target = targets[i];
			vector<double> o = n.fire(data[i]);
			int got = getHighest(o);
			printf("got: %i wanted: %i\n", got, target);
			for(int j = 0; j < o.size(); j++){
				printf("%f ", o[j]);
			}
			printf("\n");
			if(got == target)
				correct++;
		}
		printf("correct: %f", (double)correct / 10000.0);
	}
	return 0;
}