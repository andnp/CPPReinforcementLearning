#include <math.h>
#include <random>
#include <iostream>
#include <chrono>
#include "neuron.h"
using namespace std;

vector<double> neuron::hadamardProduct(vector<double> vect1, vector<double> vect2){
	vector<double> ret(vect1.size());
	for(int i = 0; i < vect1.size(); i++){
		ret[i] = vect1.at(i) * vect2.at(i);
	}
	return ret;
}
double neuron::vectorSum(vector<double> vector){
	double sum = 0.0;
	for(int i = 0; i < vector.size(); i++){
		sum += vector[i];
	}
	return sum;
}
double neuron::dot(vector<double> vect1, vector<double> vect2){
	return vectorSum(hadamardProduct(vect1, vect2));
}
double neuron::sigmoidFunction(double sigma){
	return 1.0 / (1.0  + exp(-sigma));
}
double neuron::sigmaPrime(double sigma){
	return sigmoidFunction(sigma) * (1.0 - sigmoidFunction(sigma));
}
double neuron::linearPrime(double z){
	return 1.0;
}
double neuron::tanh(double z){
	return tanhf(z);
}
double neuron::tanhPrime(double z){
	return 1.0 - (tanh(z) * tanh(z));
}
double neuron::softplus(double z){
	return log(1.0 + exp(z));
}
double neuron::softplusPrime(double z){
	return sigmoidFunction(z);
}
double neuron::relu(double z){
	if(z <= 0) return 0;
	return z;
}
double neuron::reluPrime(double z){
	if(z <= 0) return .01;
	return 1.0;
}
double neuron::maxout(double z1, double z2){
	if(z1 > z2) return z1;
	else return z2;
}
double neuron::maxoutPrime(double z1, double z2, int set){
	if(set == 0 && z1 > z2) return 1;
	else if(set == 1 && z1 < z2) return 1;
	return 0;
}
double neuron::delta(double errors, double previous_z, int set){
	if(type == SIGMOID)
		return errors * sigmaPrime(previous_z);
	else if(type == LINEAR)
		return errors * linearPrime(previous_z);
	else if(type == TANH)
		return errors * tanhPrime(previous_z);
	else if(type == SOFTPLUS)
		return errors * softplusPrime(previous_z);
	else if(type == RELU)
		return errors * reluPrime(previous_z);
	else if(type == MAXOUT)
		return errors * maxoutPrime(previous_z, previous_z2, set);
	else {
		cout << "ERROR d\n";
		return 0.0;
	}
}
// temp: sigmoid uses cross-entropy loss, where all others use squared error (this will be fixed in a later update);
double neuron::deltaOutput(double y, double previous_z){
	if(type == SIGMOID)
		return sigmoidFunction(previous_z) - y;
	else if(type == LINEAR)
		return (previous_z - y) * linearPrime(previous_z);
	else if(type == TANH)
		return (tanh(previous_z) - y) * tanhPrime(previous_z);
	else if(type == SOFTPLUS)
		return (softplus(previous_z) - y) * softplusPrime(previous_z);
	else if(type == RELU)
		return (relu(previous_z) - y) * reluPrime(previous_z);
	else {
		cout << "ERROR do\n";
		return 0.0;
	}
}
vector<double> neuron::backpropOutput(double y){
	double db = deltaOutput(y, previous_z);
	vector<double> errors(weights.size());
	// cout << "Backprop Output" << "\n";
	for(int i = 0; i < weights.size(); i++){
		errors[i] = db * weights[i];
		// cout << "prev_act " << prev_act[i] << "\n";
		//			      cost					   momentum				    weight decay
		double dw = (c * db * prevInputs[i]) + (momentum * prevDW[i]) - (lambda * c * weights[i]);
		prevDW[i] = dw;
		// cout << "prev " << weights[i] << "\n";
		weights[i] = weights[i] - dw;
		// cout << "now " << weights[i] << "\n";
	}
	double deltab = c * db + momentum * prevDB;
	prevDB = deltab;
	// cout << "prev_bias " << bias << "\n";
	bias = bias - deltab;
	// cout << "now_bias " << bias << "\n";
	return errors;
}
vector<double> neuron::backprop(vector<double> y){
	if(previous_z > previous_z2 || type != MAXOUT){
		double db = delta(vectorSum(y), previous_z, 0);
		vector<double> errors(weights.size());

		// cout << "Backprop" << "\n";
		for(int i = 0; i < weights.size(); i++){
			errors[i] = db * weights[i];
			double dw = (c * db  * prevInputs[i]) + (momentum * prevDW[i]) - (lambda * c * weights[i]);
			prevDW[i] = dw;
			// cout << "prev " << weights[i] << "\n";
			weights[i] = weights[i] - dw;
			// cout << "now " << weights[i] << "\n";
		}
		double deltab = c * db + momentum * prevDB;
		prevDB = deltab;
		// cout << "prev_bias " << bias << "\n";
		bias = bias - deltab;
		// cout << "now_bias " << bias << "\n";
		return errors;
	} else {
		double db = delta(vectorSum(y), previous_z, 1);
		vector<double> errors(weights2.size());

		// cout << "Backprop" << "\n";
		for(int i = 0; i < weights2.size(); i++){
			errors[i] = db * weights2[i];
			double dw = (c * db  * prevInputs[i]) + (momentum * prevDW2[i]) - (lambda * c * weights2[i]);
			prevDW2[i] = dw;
			// cout << "prev " << weights[i] << "\n";
			weights2[i] = weights2[i] - dw;
			// cout << "now " << weights[i] << "\n";
		}
		double deltab = c * db + momentum * prevDB2;
		prevDB2 = deltab;
		bias2 = bias2 - deltab;
		return errors;
	}
	
}
double neuron::compute(vector<double> inputs){
	prevInputs = inputs;
	previous_z = dot(weights, inputs) + bias;
	if(type == MAXOUT) previous_z2 = dot(weights2, inputs) + bias2;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine gen(seed);
    uniform_real_distribution<double> dist(0,1);
    if(dist(gen) < dropout && learning){
    	return 0;
    } else if(type == SIGMOID)
		return sigmoidFunction(previous_z);
	else if(type == LINEAR)
		return previous_z;
	else if(type == TANH)
		return tanh(previous_z);
	else if(type == SOFTPLUS)
		return softplus(previous_z);
	else if(type == RELU)
		return relu(previous_z);
	else if(type == MAXOUT)
		return maxout(previous_z, previous_z2);
	else{
		cout << "ERROR!\n";
		return -1.0;
	}
}

void neuron::instantiate(int num, int _type, double _dropout, double _c, double _lambda){
	type = _type;
	lambda = _lambda;
	dropout = _dropout;
	weights.resize(num);
	weights2.resize(num);
	prevDW.resize(num);
	prevDW2.resize(num);
	c = _c;
	// initialize weights using xavier's initialization
	// Glorot & Bengio 2010a
	// Modifications for rectified proposed by He, Rang, Zhen, and Sun 2015
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine gen(seed);
    double stan_dev;
    double mean = 0.0;    
    if(type == RELU || type == SOFTPLUS || type == MAXOUT){
    	stan_dev = 2.0/num;
    } else {
    	stan_dev = 1.0/num;
    }
	normal_distribution<double> dist(mean,stan_dev);

    if(type == MAXOUT){
    	for(int i = 0; i < num; i++){
			weights2[i] = dist(gen);
			prevDW2[i] = 0.0;
		}
		prevDB2 = 0.0;
		bias2 = dist(gen);
    }
	for(int i = 0; i < num; i++){
		weights[i] = dist(gen);
		prevDW[i] = 0.0;
	}
	prevDB = 0.0;
	bias = dist(gen);
}