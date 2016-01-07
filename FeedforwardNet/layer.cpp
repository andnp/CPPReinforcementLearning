#include "layer.h"
#include <iostream>
using namespace std;

vector<double> layer::compute(vector<double> inputs){
	vector<double> outputs(size);
	for(int i = 0; i < size; i++){
		outputs[i] = neurons[i].compute(inputs);
	}
	return outputs;
}

vector<vector<double>> layer::backpropOutput(vector<double> y, vector<double> prev_act){
	vector<vector<double>> errors(size);
	for(int i = 0; i < size; i++){
		errors[i] = neurons[i].backpropOutput(y[i], prev_act);
	}
	return errors;
}

vector<vector<double>> layer::backprop(vector<vector<double>> input_errors, vector<double> prev_act){
	vector<vector<double>> errors(size);
	for(int i = 0; i < size; i++){
		errors[i] = neurons[i].backprop(input_errors[i], prev_act);
	}
	return errors;
}

void layer::learning(bool learning){
	for(int i = 0; i < neurons.size(); i++){
		neurons[i].learning = learning;
	}
}

void layer::instantiate(int _size, int _input_size, int _type, double dropout, double c, double lambda){
	type = _type;
	size = _size;
	input_size = _input_size;
	neurons.resize(size);
	for(int i = 0; i < size; i++){
		neuron n;
		n.instantiate(input_size, type, dropout, c, lambda);
		neurons[i] = n;
	}
}