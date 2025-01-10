# LLMs-from-Scratch

The code in this repository is based on the book [Build a Large Language Model (From Scratch) by Sebastian Raschka](https://www.manning.com/books/build-a-large-language-model-from-scratch). Here, we demonstrate how to create, train, and tweak large language models (LLMs) by building them from the ground up!

## Contents

1. **Working with text data**:  
   - [Byte pair encoding (BPE) tokenizer](https://arxiv.org/abs/1508.07909v5)
   - Positional embeddings: absolute, relative, [rotary (RoPE)](https://arxiv.org/abs/2104.09864)

2. **Coding attention mechanisms**:  
   - Recurrent neural networks
   - Self-attention without trainable weights
   - Self attention with trainable weights (key, query, value)
   - Causal attention and dropout masks
   - Efficient multi-head [attention](https://arxiv.org/abs/1706.03762)
   - [Grouped query attention](https://arxiv.org/abs/2305.13245v3)

3. **Implementing LLM models from scratch to generate text**:  
   - Layer normalizations: LayerNorm, [RMSNorm](https://arxiv.org/abs/1910.07467)
   - Activation functions: [GELU](https://arxiv.org/abs/1606.08415), [Swish](https://arxiv.org/abs/1710.05941v1), [GLU](https://arxiv.org/abs/1612.08083), [SwiGLU](https://arxiv.org/pdf/2002.05202), [ReLU^2](https://arxiv.org/abs/2402.03804)

     > "We offer no explanation as to why these architectures seem to work; we attribute
     > their success, as all else, to divine benevolence (Shazeer, 2020)."

   - Feed forward neural network
   - Shortcut connections
   - [Transformers](https://arxiv.org/abs/1706.03762) with weight tying
   - [Model FLOP Utilization (MFP)](https://arxiv.org/abs/2204.02311)
     
4. **Pretraining on unlabeled data**:  
   - Temperature scaling, top-k sampling
   - [Learning rate warmup](https://arxiv.org/abs/1706.02677), [cosine annealing](https://arxiv.org/abs/1608.03983v5), gradient clipping

5. **Fine-tuning for classification**:  
   - Spam classifier (5k+ SMS messages) with 95.67% accuracy
   - Sentiment classifier (50k+ IBM movie reviews) with 91.88% accuracy  

6. **Fine-tuning to follow instructions**  
   - [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) and [Phi-3](https://arxiv.org/abs/2404.14219) input formats
   - Evaluating instruction responses using [Ollama](https://ollama.com)
   - [Direct preference optimization (DPO)](https://arxiv.org/abs/2305.18290)

## Foundation Models

### GPT-2

<img 
  src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/gpt-sizes.webp?timestamp=123"
  width="500px"
/>

### Llama 3.2

<img 
  src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/llama32.webp"
  width="700px"
/>
