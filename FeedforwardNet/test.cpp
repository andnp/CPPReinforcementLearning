#include "network.h"
#include <iostream>
#include <random>
using namespace std;

int main(){
	nnetwork n;

	int inputSize = 2;
	vector<int> layers = {2, 1};
	vector<int> types = {2, 0};
	vector<double> dropout = {0.0, 0};
	vector<double> lambda = {0,0};
	n.instantiate(inputSize, layers, types, dropout, lambda, .5);

	vector<double> output(1);
	vector<double> inputs;
	vector<double> target;

	vector<int> counter = {0,0,0,0};

	std::random_device rd;
    std::mt19937 gen(rd());
	std::uniform_int_distribution<> dist(0, 1);

	cout << "-----------TRAINING----------\n";

	double err = 10;
	double err_last = 100;
	int i = 0;
	while(err > .00001 && i < 10000){
	// for(int j = 0; j < 1000; j++){
		inputs = {(double)dist(gen), (double)dist(gen)};
		output = n.fire(inputs);
		if(inputs[0]==1 ^ inputs[1]==1){
			target = {1.0};
		} else {
			target = {0.0};
		}

		if(inputs[0] == 1 && inputs[1] == 1) counter[0] += 1;
		else if(inputs[0] == 0 && inputs[1] == 1) counter[1] += 1;
		else if(inputs[0] == 1 && inputs[1] == 0) counter[2] += 1;
		else if(inputs[0] == 0 && inputs[1] == 0) counter[3] += 1;
		// for(auto i : output){
		// 	cout << "in: " << inputs[0] << " " << inputs[1] << " got: " << i << " expected: " << target[0] << "\n";
		// }
		n.learn(target);
		i++;
		err = 0;

		inputs = {0,0};
		output = n.fire(inputs);
		if(inputs[0]==1 ^ inputs[1]==1){
			target = {1};
		} else {
			target = {0};
		}
		err += pow((target[0] - output[0]), 2);
		inputs = {1,0};
		output = n.fire(inputs);
		if(inputs[0]==1 ^ inputs[1]==1){
			target = {1};
		} else {
			target = {0};
		}
		err += pow((target[0] - output[0]), 2);
		inputs = {0,1};
		output = n.fire(inputs);
		if(inputs[0]==1 ^ inputs[1]==1){
			target = {1};
		} else {
			target = {0};
		}
		err += pow((target[0] - output[0]), 2);
		inputs = {1,1};
		output = n.fire(inputs);
		if(inputs[0]==1 ^ inputs[1]==1){
			target = {1};
		} else {
			target = {0};
		}
		err += pow((target[0] - output[0]), 2);
		// if(err > err_last) break;
		err_last = err;
		cout << err << "\n";
	// }
	}
	cout << err_last << "\n";
	cout << "i: " << i << "\n";

	cout << "--------EVALUATION----------\n";

	inputs = {0,0};
	output = n.fire(inputs);
	if(inputs[0]==1 ^ inputs[1]==1){
		target = {1};
	} else {
		target = {0};
	}
	for(auto i : output){
		cout << "in: " << inputs[0] << " " << inputs[1] << " got: " << i << " expected: " << target[0] << "\n";
	}
	inputs = {1,0};
	output = n.fire(inputs);
	if(inputs[0]==1 ^ inputs[1]==1){
		target = {1};
	} else {
		target = {0};
	}
	for(auto i : output){
		cout << "in: " << inputs[0] << " " << inputs[1] << " got: " << i << " expected: " << target[0] << "\n";
	}
	inputs = {0,1};
	output = n.fire(inputs);
	if(inputs[0]==1 ^ inputs[1]==1){
		target = {1};
	} else {
		target = {0};
	}
	for(auto i : output){
		cout << "in: " << inputs[0] << " " << inputs[1] << " got: " << i << " expected: " << target[0] << "\n";
	}
	inputs = {1,1};
	output = n.fire(inputs);
	if(inputs[0]==1 ^ inputs[1]==1){
		target = {1};
	} else {
		target = {0};
	}
	for(auto i : output){
		cout << "in: " << inputs[0] << " " << inputs[1] << " got: " << i << " expected: " << target[0] << "\n";
	}

	cout << "0 0: " << counter[3] << "\n";
	cout << "1 0: " << counter[2] << "\n";
	cout << "0 1: " << counter[1] << "\n";
	cout << "1 1: " << counter[0] << "\n";

	return 0;
}
