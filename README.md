# Reinforcement Learning for Traveling Salesman and Bin-Packing Problems

Deep learning, specifically reinforcement learning has gained significant attention
in solving combinatorial optimization problems recently. In this paper, we implement
a framework with neural networks and reinforcement learning to tackle
combinatorial optimization problems. We focus on the traveling salesman problem
(TSP) and train a recurrent neural network that, given a set of city coordinates,
predicts a distribution over different city permutations. Using negative tour length
as the reward signal, we optimize the parameters of the recurrent neural network
with the policy gradient method. We compare learning the network parameters
on a set of training graphs against learning them on individual test graphs. We
test our implementation to solve TSP on 2D Euclidean graphs with 10 and 20
nodes. Additionally, we also apply the reinforcement learning method to solve the
bin-packing, another NP-hard problem.
