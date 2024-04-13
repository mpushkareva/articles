# Structured pruning

### Adaptive Activation-based Structured Pruning

**Year**: 2023

**Authors**: Kaiqi Zhao, Animesh Jain, Ming Zhao

**Gist**: 
The authors propose adaptive pruning algorthm. It is suggested to  iteratively remove weights less than calculated treshhold and train pruned network on T - k epochs. If new network is better regarding to choosen criteria (FLOPS/size/accuracy drop), we update weight. Otherwise, return to previous weights and set new threshold.

<img src="images/adaptive_activation_algo.png" alt="isolated" width="300"/>

**Results**:

<img src="images/adaptive_activation_res.png" alt="isolated" width="300"/>

**Tags**: Iterative pruning, structured pruning, ResNet, VGG, MobileNet, CIFAR, Tiny-ImageNet

### ON THE ROLE OF STRUCTURED PRUNING FOR NEURAL NETWORK COMPRESSION

**Year**: 2021

**Authors**: Andrea Bragagnolo, Enzo Tartaglione, Attilio Fiandrotti, Marco Grangetto

**Gist**: The authors compared performance of structured (SeReNe) and unstructured (LOBSTER) pruning on different devices. The result of the experiments is that structured pruning enables better end-to-end compression despite lower pruning ratios.

**Results**:

<img src="images/on_the_role_res.png" alt="isolated" width="300"/>

**Tags**: Structured pruning, instructured pruning, SeReNe, LOBSTER, VGG, ResNet, CIFAR, ImageNet

### Parameterized Structured Pruning for Deep Neural
Networks

**Year**: 2019

**Authors**: Günther Schindler, Wolfgang Roth, Franz Pernkopf, and Holger Fröning

**Gist**:
Authours consider only column, filter and layer pruning. They suggest to add parameter $\nu$ of size (number of columns in layer for column pruning, number of filters in layer for filter pruning, one number for layer pruning). On each forward we collect gradients for weights, on the backward pass we don't change weights, but $\nu$. Gradient for $\nu$ is calculates as a sum of gradients in column/filter/layer. If the gradient sum is less than certain number $\epsilon$, $\nu$ gradient equals zero. To force $\nu$ to be zero in gradient update l1 regularization was added.

<img src="images/parametrized_structured_meth.png" alt="isolated" width="300"/>

**Results**:

<img src="images/parametrized_structured_res.png" alt="isolated" width="300"/>


**Tags**: (~) Iterative pruning, CIFAR, structured pruning, ResNet, DenseNet


### Revisiting Loss Modelling for Unstructured Pruning

**Year**: 2020

**Authors**: César Laurent, Camille Ballas, Thomas George, Nicolas Ballas, Pascal Vincent

**Gist**: Authors propose linear (LM) and quadratic criterions (QM) for unstructured iterative pruning. They suggest to apply this criterions instead of  optimal brain damage criterion (OBD). Finetuning is optional.

<img src="images/revisiting_algo.png" alt="isolated" width="300"/>

**Results**:

<img src="images/revisiting_res1.png" alt="isolated" width="300"/>

<img src="images/revisiting_res2.png" alt="isolated" width="300"/>

**Tags**: Unstructured pruning, MNIST, CIFAR, VGG, Pruning criterions


### A Closer Look at Structured Pruning for Neural Network Compression

**Year**: 2018

**Authors**: Elliot J. Crowley, Jack Turner, Amos Storkey, Michael O’Boyle

**Gist**: l1 norm and fisher pruning. The authors compare pruned and finetuned models with reduced networks (by depth or width) and network with fisher pruning structure trained from scratch. Models trained from scratch show better performance, fisher scratch is the best.

**Results**:

DenseNet TestError:

<img src="images/closer_look_res.png" alt="isolated" width="300"/>

<img src="images/closer_look_res2.png" alt="isolated" width="300"/>


**Tags**: Fisher pruning, structured pruning, DenseNet


### Group Fisher Pruning for Practical Network Compression

**Year**: 2021

**Authors**: Liyang Liu, Shilong Zhang, Zhanghui Kuang, Aojun Zhou, Jing-Hao Xue, Xinjiang Wang, Yimin Chen, Wenming Yang, Qingmin Liao, Wayne Zhang

**Gist**: 
1. Pruning of coupled channels (like in pyramid network). If two layers recevie info from one conv, they are coupled. If parent conv is pruned, coupled convs should be pruned accordingly. 
2. it prunes globally rather than locally. 
3. it estimates importances of all channels in one pass via the principled Fisher information instead of multiple forward passes for individual channels
4. it does not depend on specific layers like batch normalization (BN) and thus is more general.

**Results**:

<img src="images/group_fisher_res.png" alt="isolated" width="300"/>

**Tags**: Structured pruning, ImageNet.


### ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression

**Year**: 2017

**Authors**: Jian-Hao Luo, Jianxin Wu, and Weiyao Lin

**Gist**: The authors propose the method of iterative channel pruning. Their method includes choosing weak channels in a layer $i$ based on input values in a layer $i+1$. The subset of weak channels in layer $i$ is defined by a greedy algorithm. Then weak channels are removed from layer $i$. The remaining channels are multiplied by weights given by minimization of construction loss between $i+1$ output before pruning and after pruning. Afterwards, the network is finetuned. Such operation is performed for each layer.

<img src="images/thinet_method.png" alt="isolated" width="300"/>

**Results**:

<img src="images/thinet_result_1.png" alt="isolated" width="300"/>

<img src="images/thinet_result_2.png" alt="isolated" width="300"/>

**Tags**: VGG, ResNet, ImageNet, Iterative pruning, Structured Pruning

### RETHINKING THE VALUE OF NETWORK PRUNING

**Year**: 2018

**Authors**: Zhuang Liu, Mingjie Sun, Tinghui Zhou, Gao Huang, Trevor Darrell

**Gist**: The authours claims that common beliefs about pruning are not necessarily true:
- it is believed that starting with training a large, over-parameterized network is important. 
- both the pruned architecture and its associated weights are believed to be essential for obtaining the final efficient model.  

Instead, their results suggest that the value of automatic pruning algorithms may lie in identifying efficient structures and performing implicit architecture search, rather than selecting “important” weights.
It is suggested to compare existing pruning tehniques and models, which have the same pruning rate, but trained from scratch. These models (Scratch-E and Scratch-B) are trained on the same number of epochs and trained with the same FLOPS amount correspondingly.

**Results**:

<img src="images/rethinking_1.png" alt="isolated" width="300"/>

<img src="images/rethinking_2.png" alt="isolated" width="300"/>

**Tags** Structured pruning, ImageNet, CIFAR, VGG, ResNet.

### Structured Pruning Adapters

**Year**: 2023

**Authors**: Lukas Hedegaard, Aman Alok, Juby Jose, Alexandros Iosifidis

**Gist**: 
The authors proposed Structured Pruning Adapters (SPAs) as an alternative to fine-tuning during structured pruning. Instead of updating all model weights, SPAs consist of prunable lightweight add-on modules, which are learned in place of the original weights but can be fused with them at runtime to obtain the same computational enhancements as regular structured pruning with fine-tuning. The SPA was applied to transfer-learning tasks.

<img src="images/pruning_adapters_method.png" alt="isolated" width="300"/>

**Results**:

<img src="images/pruning_adapters_result.png" alt="isolated" width="300"/>

**Tags**: CIFAR, ResNet, Weight Pruning, Gradient Pruning, Taylor Pruning, LRP, Structural Pruning

### Name

**Year**: 

**Authors**:

**Gist**:

**Results**:

**Tags**: