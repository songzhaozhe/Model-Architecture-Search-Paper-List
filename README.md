# Model Architecture Search Paper List

> Last updated 12/31/2017
>
> Zhaozhe Song

## Two early papers in ICLR 2017

##### **NASNet (Google)** 

Zoph B, Le Q V. Neural architecture search with reinforcement learning[J]. arXiv preprint arXiv:1611.01578, 2016. [[pdf]](https://arxiv.org/pdf/1611.01578.pdf)

> Sample architectures and train, by policy gradient (RL). On CIFAR only.

**MetaQNN**

Baker B, Gupta O, Naik N, et al. Designing neural network architectures using reinforcement learning[J]. arXiv preprint arXiv:1611.02167, 2016. [[pdf]](https://arxiv.org/pdf/1611.02167.pdf)

> Use Q-learning (RL) to choose the next cell (conv/pool)

## State-of-the-art Papers

**NASNet-A (Google)**

Zoph B, Vasudevan V, Shlens J, et al. Learning transferable architectures for scalable image recognition[J]. arXiv preprint arXiv:1707.07012, 2017. [[pdf]](https://arxiv.org/pdf/1707.07012.pdf)

> Modified version of NASNet, used RNN + PPO (RL) to train on CIFAR, and transfer the same cell to ImageNet.

**Hierarchical Representations (Google)**

Liu H, Simonyan K, Vinyals O, et al. Hierarchical representations for efficient architecture search[J]. arXiv preprint arXiv:1711.00436, 2017. [[pdf]](https://arxiv.org/pdf/1711.00436.pdf)

> An interesting hierarchical search space + Evolution algorithm.
>
> Slightly worse than the other two.

**PNASNet (Google)**

Liu C, Zoph B, Shlens J, et al. Progressive neural architecture search[J]. arXiv preprint arXiv:1712.00559, 2017. [[pdf]](https://arxiv.org/pdf/1712.00559.pdf)

> Used sequential model-based optimization (SMBO) (similar to A* algorithm). Search from simple to complex cells, also train a surrogate function to predict performance in the beginning of next step.
>
> Same accuracy, 2 times faster than NASNet-A algorithm.

**AmoebaNets (Google)**

Regularized Evolution for Image Classifier Architecture Search [[pdf]](https://arxiv.org/pdf/1802.01548.pdf)

> Same search space with NASNet-A, a more detailed comparison of search methods. Claimed that evolution is better than RL.
>
> Introduced regularized evolution, which is a simple modification to traditional evolution algorithm.

## Other papers

Real, Esteban, et al. "Large-scale evolution of image classifiers." arXiv preprint arXiv:1703.01041 (2017).

> Evolution algorithm. Inherit most weights after every mutation.

Sun, Yanan, Bing Xue, and Mengjie Zhang. "Evolving Deep Convolutional Neural Networks for Image Classification." *arXiv preprint arXiv:1710.10741* (2017).

> Genetic algorithm (Encoded networks into chromosomes).

Cai, Han, et al. "Reinforcement learning for architecture search by network transformation." *arXiv preprint arXiv:1707.04873* (2017).

> Net2Net idea + RL

Elsken, Thomas, Jan-Hendrik Metzen, and Frank Hutter. "Simple And Efficient Architecture Search for Convolutional Neural Networks." arXiv preprint arXiv:1711.04528 (2017).

> Network morphism (generalized Net2Net)

Zhong Z, Yan J, Liu C L. Practical Network Blocks Design with Q-Learning[J]. arXiv preprint arXiv:1708.05552, 2017.

> Q-learning + performance prediction.

Baker, Bowen, et al. "Accelerating neural architecture search using performance prediction." CoRR, abs/1705.10823(2017).

> Predict weights by early training stats and network architecture

Brock, Andrew, et al. "SMASH: one-shot model architecture search through hypernetworks." arXiv preprint arXiv:1708.05344 (2017).

> Fancy idea: Train a HyperNetwork to predict weights for every architecture.
>
> But I did not replicate their results using their released code.

## Some other objectives

#### Searching for LSTM cell structure

Jozefowicz, Rafal, Wojciech Zaremba, and Ilya Sutskever. "An empirical exploration of recurrent network architectures." ICML 2015.

#### Searching Activation Functions

Ramachandran, P., Zoph, B., & Le, Q. (2017). Searching for activation functions.

> Discovered Swish (x * sigmoid(x)) activation function through Brute-Force search. It is able to boost 1~2% classification accuracy on ImageNet, in my experiments.