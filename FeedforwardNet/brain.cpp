#include "brain.h"
#include <random>
#include <iostream>
#include <math.h>

int brain::randomAction(){
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<> mdist(0,3);
	return mdist(gen);
}

void brain::initialize(int inputSize){
	vector<int> layers = {50, 20, 1};
	vector<int> types = {5, 5, 1};
	vector<double> dropout = {.01, .01, 0};
	vector<double> lambda = {0,0,0};
	valueNet.instantiate(inputSize, layers, types, dropout, lambda, .005);
}

brain::p brain::policy(vector<double> state){
	vector<double> actions = {0,0,0,0};
	vector<vector<vector<double>>> actionValues(4);
	for(int i = 0; i < actions.size(); i++){
		actions[i] = 1;
		vector<double> tmp = state;
		tmp.insert(tmp.end(), actions.begin(), actions.end());
		actionValues[i] = valueNet.fire(tmp);
		actions[i] = 0;
	}

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


		p maxact = policy(e.state1); // this isn't right.
		double r = e.reward0 + gamma * maxact.value;
		vector<double> target = {r};
		valueNet.learn(target, maxact.outputs);

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