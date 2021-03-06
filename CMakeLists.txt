cmake_minimum_required(VERSION 2.8)

project(CNeuralNets)

set(FEEDFORWARD "FeedforwardNet")

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -g")

add_library(neuron FeedforwardNet/neuron.cpp FeedforwardNet/neuron.h)
add_library(layer FeedforwardNet/layer.cpp FeedforwardNet/layer.h)
add_library(network FeedforwardNet/network.cpp FeedforwardNet/network.h)
# add_library(brain FeedforwardNet/brain.cpp FeedforwardNet/brain.h)

add_executable(tester FeedforwardNet/test.cpp)
# add_executable(qlearn FeedforwardNet/qlearn.cpp)
add_executable(mnist Tests/MNIST/mnist.cpp)

target_link_libraries(tester network layer neuron)
# target_link_libraries(qlearn network layer neuron brain)
target_link_libraries(mnist network layer neuron)