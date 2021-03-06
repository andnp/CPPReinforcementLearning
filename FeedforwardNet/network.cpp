#include "network.h"
#include <iostream>
using namespace std;

vector<double> nnetwork::fire(vector<double> inputs){
	vector<vector<double>> outputs(layers.size());
	outputs[0] = layers[0].compute(inputs);
	for(int i = 1; i < layers.size(); i++){
		outputs[i] = layers[i].compute(outputs[i - 1]);
	}
	return outputs[outputs.size() - 1];
}

void nnetwork::learn(vector<double> target){
	// vector<vector<vector<double>>> errors;
	vector<vector<double>> error;
	error = layers[layers.size() - 1].backpropOutput(target);
	for(int i = layers.size() - 2; i >= 0; i--){
		error = layers[i].backprop(transpose(error));
	}
	// cout << "Prev_outputs\n";
	// for(int i = 0; i < prev_outputs.size(); i++){
	// 	for(int j = 0; j < prev_outputs[i].size(); j++){
	// 		cout << prev_outputs[i][j] << " ";
	// 	}
	// 	cout << "\n";
	// }
}

void nnetwork::learn(vector<double> target, vector<double> inputs){
	fire(inputs);
	learn(target);
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

void nnetwork::learning(bool learning){
	for(int i = 0; i < layers.size(); i++){
		layers[i].learning(learning);
	}
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