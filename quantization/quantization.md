# Quantization

### The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits

**Year**: 2024

**Authors**: Shuming Ma, Hongyu Wang, Lingxiao Ma Lei Wang Wenhui Wang, Shaohan Huang Li Dong Ruiping Wang Jilong Xue Furu Wei⋄

**Gist**:
From the article itself, the details of training are not clear at all, except that they train from-scratch and ternarize weights using the absmax method.
But there is clarification in the previous article of this team https://arxiv.org/pdf/2310.11453.pdf:
- They teach quantization-aware: the behavior of the network is emulated if it were quantized at the moment, that is, conversion to quantized values is assigned to real weights. With forward pass, the weights are binarized (in the new article they are ternarized), activations have reduced accuracy. With backward pass, the weights are updated to full precision.
- For quantization, absmax is used, which was proposed in 2022 https://arxiv.org/pdf/2208.07339.pdf.
- They suggest using a higher learning rate (8e-4). Since the weights during ternarization change little on each backward pass.

<img src="images/the_era_idea.png" alt="isolated" width="300"/>

**Results**:

<img src="images/the_era_res.png" alt="isolated" width="300"/>

**Tags**: QAT, ternarization.

### BitNet: Scaling 1-bit Transformers for Large Language Models

**Year**: 2023

**Authors**: Hongyu Wang, Shuming Ma, Li Dong† Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, Furu Wei

**Gist**:
- The authors teach quantization-aware: the behavior of the network is emulated if it were quantized at the moment, that is, conversion to quantized values is assigned to real weights. With forward pass, the weights are binarized (in the new article they are ternarized), activations have reduced accuracy. With backward pass, the weights are updated to full precision.
- For quantization, absmax is used, which was proposed in 2022 https://arxiv.org/pdf/2208.07339.pdf.
- They suggest using a higher learning rate (8e-4). Since the weights during ternarization change little on each backward pass.

**Results**:

<img src="images/bitnet_res.png" alt="isolated" width="300"/>

**Tags**:

### Name

**Year**: 

**Authors**:

**Gist**:

**Results**:

**Tags**: