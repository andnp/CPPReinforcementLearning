#include <vector>
using namespace std;

class neuron{
private:
	vector<double> weights;
	vector<double> weights2;
	vector<double> prevInputs;
	double bias;
	double bias2 = 0;
	double previous_z;
	double previous_z2 = -99999;
	double c;
	int type;
	vector<double> prevDW;
	vector<double> prevDW2;
	double prevDB;
	double prevDB2;
	double momentum = 0.0;
	double lambda = 0.0;
	double dropout;

	vector<double> hadamardProduct(vector<double> vect1, vector<double> vect2);
	double vectorSum(vector<double> vector);
	double dot(vector<double> vect1, vector<double> vect2);
	double sigmoidFunction(double sigma);
	double sigmaPrime(double sigma);
	double delta(double errors, double previous_z, int set);
	double deltaOutput(double y, double previous_z);
	double linearPrime(double z);
	double tanh(double z);
	double tanhPrime(double z);
	double relu(double z);
	double reluPrime(double z);
	double softplus(double z);
	double softplusPrime(double z);
	double maxout(double z, double z2);
	double maxoutPrime(double z, double z2, int set);
public:
	bool learning = true;
	
	vector<double> backpropOutput(double y);
	vector<double> backprop(vector<double> y);
	double compute(vector<double> inputs);
	void instantiate(int num, int type, double dropout, double c, double lambda);
	int LINEAR = 1;
	int SIGMOID = 0;
	int TANH = 2;
	int RELU = 3;
	int SOFTPLUS = 4;
	int MAXOUT = 5;
};