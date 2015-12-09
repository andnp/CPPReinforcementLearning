#include "brain.h"
#include <random>
#include <iostream>
#include <fstream>
#include <math.h>

void brain::writeToFile(){
	
}

int brain::randomAction(){
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<> mdist(0,3);
	return mdist(gen);
}

void brain::initialize(int inputSize){
	vector<int> layers = {10, 5, 1};
	vector<int> types = {5, 5, 1};
	vector<double> dropout = {.01, .01, 0};
	vector<double> lambda = {0,0,0};
	valueNet.instantiate(inputSize, layers, types, dropout, lambda, .05);
}

brain::p brain::policy(vector<double> state){

	// Get q-value for all actions
	vector<double> actions = {0,0,0,0};
	vector<vector<vector<double>>> actionValues(4);
	for(int i = 0; i < actions.size(); i++){
		actions[i] = 1;
		vector<double> tmp = state;
		tmp.insert(tmp.end(), actions.begin(), actions.end());
		actionValues[i] = valueNet.fire(tmp);
		actions[i] = 0;
	}

	// get Action that maximizes q-value
	int maxLoc = 0;
	double maxVal = actionValues[0][actionValues[0].size() - 1][0];
	for(int i = 1; i < actionValues.size(); i++){
		if(actionValues[i][actionValues[0].size() - 1][0] > maxVal){
			maxLoc = i;
			maxVal = actionValues[i][actionValues[i].size() - 1][0];
		}
	}
	p ret;
	ret.action = maxLoc;
	ret.value = maxVal;
	ret.outputs = actionValues[maxLoc];
	return ret;
}

int brain::forward(vector<double> state){
	forwardPasses++;

	int action;
	epsilon = min(1.0, max(.01, 1.0 - (age - burnIn) / (learnSteps - burnIn)));
	if(!explore)
		epsilon = -1;

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dist(0,1);
	if(dist(gen) < epsilon){
		action = randomAction();
	} else {
		p maxact = policy(state);
		action = maxact.action;
	}

	if(stateWindow.size() > 2){
		stateWindow.erase(stateWindow.begin());
		actionWindow.erase(actionWindow.begin());
		rewardWindow.erase(rewardWindow.begin());
	}
	stateWindow.push_back(state);
	actionWindow.push_back(action);

	return action;
}

void brain::backward(double reward){
	rewardWindow.push_back(reward);

	age++;

	if(age > 2){
		experience e;
		e.state0 = stateWindow[1];
		e.action0 = actionWindow[1];
		e.reward0 = rewardWindow[1];
		e.state1 = stateWindow[2];

		// Q-Learning
		// Qt+1(St, At) = Qt(St, At) + learn_ratet(St, At) * (Rt + discount * maxA(Qt(St+1, A)) - Qt(St, At))

		p maxact = policy(e.state1);
		double maxA_Qt1 = maxact.value;

		vector<double> St = stateWindow[1];
		vector<double> St1 = stateWindow[2];
		int A = actionWindow[1];

		vector<double> tmp = St;
		vector<double> actions = {0,0,0,0};
		actions[A] = 1;
		tmp.insert(tmp.end(), actions.begin(), actions.end());
		vector<vector<double>> Qt_Outputs = valueNet.fire(tmp);
		double Qt = Qt_Outputs[Qt_Outputs.size() - 1][0];

		// for now assume learn rate is .01

		double Rt = rewardWindow[1];
		double Qt1 = Qt + (.01 * (Rt + (gamma * maxA_Qt1) - Qt));

		vector<double> target = {Qt1};

		valueNet.learn(target, Qt_Outputs);
		// for(int i = 0; i < target.size(); i++){
		// 	cout << target[i] << ", ";
		// }
		// cout << "\n";
		// for(int i = 0; i < maxact.outputs[maxact.outputs.size() - 1].size(); i++){
		// 	cout << maxact.outputs[maxact.outputs.size() - 1][i] << ", ";
		// }
		// cout << "\n";

		// if(experienceBuffer.size() < experienceBufferSize){
		// 	experienceBuffer.push_back(e);
		// } else {
		// 	random_device rd;
		// 	mt19937 gen(rd());
		// 	uniform_int_distribution<> dist(0,experienceBuffer.size() - 1);
		// 	int randint = dist(gen);
		// 	experienceBuffer[randint] = e;
		// }

		// if(experienceBuffer.size() >  startLearnSize){
		// 	random_device rd;
		// 	mt19937 gen(rd());
		// 	uniform_int_distribution<> dist(0,experienceBuffer.size() - 1);
		// 	for(int i = 0; i < batchSize; i++){
		// 		int randExp = dist(gen);
		// 		e = experienceBuffer[randExp];
		// 		maxact = policy(e.state1);
		// 		r = e.reward0 + gamma * maxact.value;
		// 		target = {r};
		// 		valueNet.learn(target, maxact.outputs);
		// 	}
		// }
	}
}

void brain::learn(experience e){
	p maxact = policy(e.state1);
	double r = e.reward0 + gamma * maxact.value;
	vector<double> target = {r};
	valueNet.learn(target, maxact.outputs);
}