#include "brain.h"
#include <iostream>
#include <random>
#include <vector>
using namespace std;

#define SIZE 25

vector<int> takeAction(int action, int x, int y){
	if(action == 0){
		x++;
		if(x >= SIZE)
			x--;
	} else if(action == 1){
		y--;
		if(y < 0)
			y=0;
	} else if(action == 2){
		x--;
		if(x < 0)
			x = 0;
	} else if(action == 3){
		y++;
		if(y >= SIZE)
			y--;
	}
	vector<int> temp = {x, y};
	return temp;
}

vector<vector<int>> move(vector<vector<int>> grid, int sx, int sy, int x, int y){
	grid[sy][sx] = 0;
	grid[y][x] = 1;
	return grid;
}

bool won(int x, int y, int targetX, int targetY){
	return x == targetX && y == targetY;
}

vector<double> getGrid(vector<vector<int>> grid){
	vector<double> tmp;
	for(int i = 0; i < SIZE; i++){
		for(int j = 0; j < SIZE; j++){
			tmp.push_back((double)grid[j][i]);
		}
	}
	return tmp;
}

vector<vector<int>> initGrid(){
	vector<vector<int>> grid(SIZE, vector<int>(SIZE, 0));
	return grid;
}

void print(vector<vector<int>> grid){
	for(int i = 0; i < SIZE; i++){
		for(int j = 0; j < SIZE; j++){
			cout << grid[j][i];
		}
		cout << "\n";
	}
}


int main(){
	vector<vector<int>> grid = initGrid();
	int winX = SIZE - 1;
	int winY = SIZE - 1;

	int startX = 0;
	int startY = 0;

	int x = startX, y = startY;

	brain b;
	vector<int> layers = {8, 1};
	vector<int> types = {5, 1};
	vector<double> dropout = {.01, 0};
	vector<double> lambda = {0, 0};
	b.valueNet.instantiate((SIZE * SIZE) + 4, layers, types, dropout, lambda, .01);
	b.learnSteps = SIZE * SIZE  * 1000;

	vector<double> inputs;

	int totalReward = -100;

	int c = 0;

	vector<int> rewards;
	double avg = -100;

	uint games = 0;
	int ignore =  SIZE * SIZE * 10;

	while(avg < -1){
		totalReward = 0;
		x = startX;
		y = startY;
		grid = initGrid();
		while(!won(x, y, winX, winY)){
			inputs = getGrid(grid);
			int action = b.forward(inputs);
			int reward = -1;
			vector<int> coords = takeAction(action, x, y);
			grid = move(grid, x, y, coords[0], coords[1]);
			if(x == coords[0], y == coords[1])
				reward = -2;
			x = coords[0];
			y = coords[1];
			if(won(x, y, winX, winY))
				if(SIZE % 2 == 1)
					reward = SIZE * 2;
				else
					reward = (SIZE * 2) + 1;
			b.backward(reward);
			totalReward += reward;
			// cout << "location: " << x << ", " << y << " action: " << action << "\n";
			// print(grid);
			// if(c == 10000)
			// 	break;
			c++;
		}
		if(games >= ignore){
			rewards.push_back(totalReward);
			int sum = 0;
			for(int i = 0; i < rewards.size(); i++){
				sum += rewards[i];
			}

			avg = (double)sum / (double)rewards.size();
		}
		games++;
		if(games > 2000)
			b.explore = false;
		cout << "totalReward " << totalReward << " avg: " << avg << " games: " << games << " epsilon: " << b.epsilon << "\n";
	}

	return 0;
}
