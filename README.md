# UltimateTicTacToe

<img src="https://user-images.githubusercontent.com/87184009/127621048-b53e4a5f-e9f4-43cf-9856-badf813a333d.png" alt="drawing" width="400"/>

## Introduction


## Deep Neural Network

Unlike the game of Go, we cannot use a convolutional neural network in the beginning. We need to find a method to figure out which input features to choose. I put all the logically possible input features in the neural network and removed some of the input features that degrade performance. Therefore, a total of 162 input features were obtained. All neural network layers are fully connected with the previous layer. Except for the last layer, all of them used the ReLU activation function. The last layer consists of nine policy network outputs and one value network output. These values are used in pruning the Monte Carlo tree to reduce the number of search cases.

![image](https://user-images.githubusercontent.com/87184009/136514436-cc29251f-cd42-48cc-8fae-f45bba9f6d68.png)

## Game-AI Strategy

MCTS : Monte Carlo Tree Search with 15,000,000 searches per each decision. (takes about 9 sec)\
My Game-AI : Reduce the number of search cases using policy network and value network. Networks are consists of deep neural network with 12 layers. Similar to MCTS, the game-ai thinks about 9 seconds.

## Result

![image](https://user-images.githubusercontent.com/87184009/136501708-10c99107-2c81-4dae-8e3a-781a5347589e.png)

![image](https://user-images.githubusercontent.com/87184009/136501334-9d6464fe-46d1-4bb5-af85-f31aef0de8aa.png)

## Conclusion

The result showed that maybe the optimal solution to this game is that the first player wins. When MCTS played a game with each other, everyone could think that the optimal solution was a draw. However, the result of "My Game AI vs. MCTS" shows the possibility that the optimal solution of this game may not necessarily be a draw.

## Usage

You can run the code using main.cc and weights00.txt in the src folder. However, rather than running the code, I recommend re-implementing the code with python.

## Other

The study of this project ended in August 2018. This is because satisfactory results came out and it was no longer meaningful to proceed with the project. The project could be implemented using a library such as Tensorflow, but it was implemented in CUDA C because I was curious about the internal structure of the GPU. To practice how neural networks work in GPU, I chose the CUDA C programming language. After programming and debugging my code, I was able to improve my understanding of deep neural networks by implementing forwarding, backpropagation, and adam optimizer by myself.
