# UltimateTicTacToe

<img src="https://user-images.githubusercontent.com/87184009/127621048-b53e4a5f-e9f4-43cf-9856-badf813a333d.png" alt="drawing" width="400"/>

## Introduction

Ultimate tic-tac-toe is a board game composed of nine tic-tac-toe boards arranged in a 3 × 3 grid. Players take turns playing in the smaller tic-tac-toe boards until one of them wins in the larger tic-tac-toe board. Roughly calculating the total complexity of this game, it is about (9!)^9 = 10^50. In this project, I tried to create a game-ai that surpasses MCTS which previously showed the best performance.

## Deep Neural Network

1. Unlike the game of Go, we cannot use a convolutional neural network in the beginning. So, we need to find new input features first. To do this, I put all the logically possible input features in the neural network and removed some of the input features that degrade performance. Therefore, a total of 162 input features were obtained.
2. All neural network layers are fully connected with the previous layer. All of the layers used the ReLU activation function except for the last layer. The last layer consists of nine policy network outputs and one value network output. These values are used later in pruning the Monte Carlo tree to reduce the number of search cases.

![image](https://user-images.githubusercontent.com/87184009/136514436-cc29251f-cd42-48cc-8fae-f45bba9f6d68.png)

## Game-AI Strategy

MCTS : Monte Carlo Tree Search with 15,000,000 searches per each decision. (takes about 9 sec)\
My Game-AI : Reduce the number of search cases using policy network and value network. Networks are consists of deep neural network with 12 layers. Similar to MCTS, the game-ai thinks about 9 seconds and makes a decision.

## Result

When MCTS play games together, there are many draws. Increasing the number of searches in MCTS did not significantly change the winning rate. It means that we need to apply a better algorithm not just increasing the number of searches to win against MCTS. My game-ai used AlphaGo's algorithm to beat MCTS. I trained my game-ai by reinforcement learning in a way that to win more. As a result, I could see my game-ai beating the opponent with a 70% and 52% chance.

![image](https://user-images.githubusercontent.com/87184009/136501708-10c99107-2c81-4dae-8e3a-781a5347589e.png)

![image](https://user-images.githubusercontent.com/87184009/136501334-9d6464fe-46d1-4bb5-af85-f31aef0de8aa.png)

## Conclusion

The result showed that maybe the optimal solution to this game is that the first player wins. Looking at "MCTS vs. MCTS" game results, everyone could guess that the optimal solution is a draw. However, the result of "My Game AI vs. MCTS" shows the possibility that the optimal solution of this game may not necessarily be a draw.

## Usage

You can run the code using main.cc and weights00.txt in the src folder. However, rather than running the code, I recommend re-implementing the code with python adding your new creative ideas.

## Other

The study of this project ended in August 2018. This is because satisfactory results came out and it was no longer meaningful to proceed with the project. The project could be implemented using a library such as Tensorflow, but it was implemented in CUDA C. I chose the CUDA C programming language to practice how neural networks work in GPU. After programming and debugging my code, I improved my understanding of deep neural networks by implementing forwarding, backpropagation, and adam optimizer.
