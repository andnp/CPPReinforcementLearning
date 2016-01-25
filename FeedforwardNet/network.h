#include <vector>
#include "layer.h"
using namespace std;

class nnetwork{
private:
	vector<int> input_size;
	vector<int> layer_sizes;
	vector<layer> layers;
	vector<int> layer_types;
public:
	vector<double> fire(vector<double> inputs);
	void learn(vector<double> target);
	void learn(vector<double> target, vector<double> inputs);
	void instantiate(int input_size, vector<int> layer_sizes, vector<int> types, vector<double> useDropout, vector<double> lambda, double c);
	vector<vector<double>> transpose(vector<vector<double>> matrix);
	double c = .1;
	void learning(bool learning);
};