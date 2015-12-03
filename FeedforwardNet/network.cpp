#include "network.h"
#include <iostream>
using namespace std;

vector<vector<double>> nnetwork::fire(vector<double> inputs){
	vector<vector<double>> outputs(layers.size() + 1);
	outputs[0] = inputs;
	outputs[1] = layers[0].compute(inputs);
	for(int i = 1; i < layers.size(); i++){
		outputs[i + 1] = layers[i].compute(outputs[i]);
	}
	return outputs;
}

void nnetwork::learn(vector<double> target, vector<vector<double>> prev_outputs){
	vector<vector<vector<double>>> errors;
	errors.push_back(layers[layers.size() - 1].backpropOutput(target, prev_outputs[prev_outputs.size() - 2]));
	int length = prev_outputs.size() - 2;
	for(int i = length - 1; i >= 0; i--){
		errors.push_back(layers[i].backprop(transpose(errors[errors.size() - 1]), prev_outputs[i]));
	}
	// cout << "Prev_outputs\n";
	// for(int i = 0; i < prev_outputs.size(); i++){
	// 	for(int j = 0; j < prev_outputs[i].size(); j++){
	// 		cout << prev_outputs[i][j] << " ";
	// 	}
	// 	cout << "\n";
	// }
}

void nnetwork::instantiate(int input_size, vector<int> _layer_sizes, vector<int> _types, vector<double> dropout,vector<double> lambda, double c){
	layer_sizes = _layer_sizes;
	layer_types = _types;
	layers.resize(layer_sizes.size());
	layer input_layer;
	input_layer.instantiate(layer_sizes[0], input_size, layer_types[0], dropout[0], c, lambda[0]);
	layers[0] = input_layer;
	for(int i = 0; i < layer_sizes.size() - 1; i++){
		layer l;
		l.instantiate(layer_sizes[i + 1], layer_sizes[i], layer_types[i + 1], dropout[i + 1], c, lambda[i + 1]);
		layers[i + 1] = l;
	}
	// for(int i = 0; i < layers.size(); i++){
	// 	cout << "layer type: " << layers[i].type << "\n";
	// 	cout << "input size: " << layers[i].input_size << "\n";
	// }
}

vector<vector<double>> nnetwork::transpose(vector<vector<double>> matrix){
	vector<vector<double>> ret;
	for(int i = 0; i < matrix[0].size(); i++){
		vector<double> row;
		for(int j = 0; j < matrix.size(); j++){
			row.push_back(matrix[j][i]);
		}
		ret.push_back(row);
	}
	return ret;
}