#include <vector>
#include "network.h"
using namespace std;

class brain{
private:
	class p{
	public:
		int action;
		double value;
		vector<vector<double>> outputs;
	};

	// vector<experience> episodeBuffer;

	vector<vector<double>> stateWindow;
	vector<int> actionWindow;
	vector<double> rewardWindow;

	nnetwork valueNet;
public:
	class experience{
	public:
		int action1;
		double reward0;
		vector<double> state0;
		vector<double> state1;
		int action0;
	};
	vector<experience> experienceBuffer;
	unsigned int forwardPasses = 0;
	double epsilon = 1;
	double age = 0;
	double burnIn = 100000;
	double learnSteps = 1000000;
	unsigned int experienceBufferSize = 1000000;
	unsigned int batchSize = 1;
	double gamma = .9;
	unsigned int startLearnSize = 100000;

	int randomAction();
	p policy(vector<double> state);
	int forward(vector<double> state);
	void backward(double reward);
	void initialize(int inputSize);
	void learn(experience e);
};