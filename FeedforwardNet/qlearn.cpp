#include "brain.h"
#include <iostream>
#include <random>
#include <vector>
using namespace std;

#define SIZE 5

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
	} else if(action == 4){
		y++;
		x++;
		if(y >= SIZE)
			y--;
		if(x >= SIZE)
			x--;
	} else if(action == 5){
		y--;
		x++;
		if(y < 0)
			y = 0;
		if(x >= SIZE)
			x--;
	} else if(action == 6){
		y--;
		x--;
		if(x < 0)
			x = 0;
		if(y < 0)
			y = 0;
	} else if(action == 7){
		y++;
		x--;
		if(y >= SIZE)
			y--;
		if(x < 0)
			x = 0;
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

vector<vector<int>> initGrid(int x, int y){
	vector<vector<int>> grid(SIZE, vector<int>(SIZE, 0));
	grid[y][x] = 1;
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
	vector<vector<int>> grid;
	int winX = SIZE - 1;
	int winY = SIZE - 1;

	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<> dist(0,SIZE - 2);

	int startX;
	int startY;

	int x, y;

	brain b;
	b.numActions = 8;
	vector<int> layers = {22, 8, 1};
	vector<int> types = {5, 5, 1};
	vector<double> dropout = {.01, .02, 0};
	vector<double> lambda = {0, 0, 0};
	b.valueNet.instantiate((SIZE * SIZE) + 8, layers, types, dropout, lambda, .01);
	b.learnSteps = pow(1.35, SIZE) * SIZE  * 25000;
	b.lambda = 0;
	int afterLearn = 1000;

	vector<double> inputs;

	int totalReward = -100;

	int c = 0;

	vector<int> rewards;
	double avg = -100;

	uint games = 0;
	int ignore =  SIZE * SIZE * 100;

	bool end = false;

	while(avg < 3 || rewards.size() < 100){
		totalReward = 0;
		x = dist(gen);
		y = dist(gen);
		grid = initGrid(x, y);
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
			// if(x == SIZE - 3 && y != 1)
			// 	reward = -10;
			// if(x == SIZE/2 && y == SIZE/2){
			// 	reward = 0 - (SIZE * SIZE);
			// 	end = true;
			// }
			if(won(x, y, winX, winY))
				reward = 2 * SIZE;
			b.backward(reward);
			totalReward += reward;
			if(end) break;
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
		b.resetEpisode();
		end = false;
		if(b.epsilon == .01){
			afterLearn--;
			if(afterLearn <= 0){
				// b.explore = false;
				b.learning(false);
			}
		}
		cout << "totalReward " << totalReward << " avg: " << avg << " games: " << games << " epsilon: " << b.epsilon << "\n";
	}

	return 0;
}
