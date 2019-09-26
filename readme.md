Official code for using / reproducing CDEP from the paper "[Interpretations are useful: penalizing explanations to align neural networks with prior knowledge](https://openreview.net/pdf?id=SkEqro0ctQ)" (ICLR 2019). This code produces hierarchical interpretations for any type of neural network trained in pytorch.

*Note: this repo is actively maintained. For any questions please file an issue.*

![](reproduce_figs/figs/intro.png)

# documentation

- fully-contained data/models/code for reproducing and experimenting with ACD
- see the [reproduce_figs](reproduce_figs) folder for notebooks with examples of using ACD to reproduce figures in the paper
  - try your own examples on these models with simple alterations to the notebooks
- for additional details, see the [acd](acd) folder which contains a bulk of the code for getting importance scores and aggregating them
    - the [dsets](dsets) folder contains small data/models/dataloaders for using different data sets 
    - more documentation is provided in readmes of subfolders and inline comments
- allows for different types of interpretations by changing hyperparameters (explained in examples)
- tested with python3 and pytorch 1.0 with/without gpu (although cpu doesn't work very well for vision models)

| Inspecting NLP sentiment models    | Detecting adversarial examples      | Analyzing imagenet models           |
| ---------------------------------- | ----------------------------------- | ----------------------------------- |
| ![](reproduce_figs/figs/fig_2.png) | ![](reproduce_figs/figs/fig_s3.png) | ![](reproduce_figs/figs/fig_s2.png) |



# using ACD on your own data

- to use ACD on your own model, replace the models in the examples with your own trained models. Specifically, 3 things must be altered:
  1. the pred_ims function must be replaced by a function you write using your own trained model. This function gets predictions from a model given a batch of examples.
  2. the model must be replaced with your model
  3. the current CD implementation doesn't always work for all types of networks. If you are getting an error inside of `cd.py`, you may need to write a custom function that iterates through the layers of your network (for examples see `cd.py`)

# related work

- this work is part of an overarching project on interpretable machine learning, guided by the [PDR framework](https://arxiv.org/abs/1901.04592)
- for related work, see the [github repo](https://github.com/jamie-murdoch/ContextualDecomposition) for [disentangled attribution curves](https://arxiv.org/abs/1905.07631)
- the file scores/score_funcs.py also contains simple pytorch implementations of [integrated gradients](https://arxiv.org/abs/1703.01365) and the simple interpration technique gradient * input

# reference

- feel free to use/share this code openly

- if you find this code useful for your research, please cite the following:

  ```c
  @article{singh2018hierarchical,
    title={Hierarchical interpretations for neural network predictions},
    author={Singh, Chandan and Murdoch, W James and Yu, Bin},
    journal={arXiv preprint arXiv:1806.05337},
    year={2018}
  }
  ```

  