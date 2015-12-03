#include <vector>
#include "neuron.h"
using namespace std;

class layer{
private:
	int size;
	vector<neuron> neurons;
	double c;

public:
	int input_size;
	int type;
	
	vector<double> compute(vector<double> inputs);
	vector<vector<double>> backpropOutput(vector<double> y, vector<double> prev_act);
	vector<vector<double>> backprop(vector<vector<double>> input_errors, vector<double> prev_act);
	void instantiate(int size, int input_size, int type, double useDropout, double c, double lambda);
};