# README

This is the codebase accompanying the paper ["Deep Generative Models through the Lens of the Manifold Hypothesis: A Survey and New Connections"](https://arxiv.org/abs/2404.02954), accepted to TMLR in August 2024. This codebase is based on the codebase from our previous paper, ["Diagnosing and Fixing Manifold Overfitting in Deep Generative Models"](https://arxiv.org/abs/2204.07172), please refer to [the original codebase](https://github.com/layer6ai-labs/two_step_zoo) for setup and general usage instructions.
Here we discuss how to run the experiments in section 5.3.2 of the paper, please note that the codebase has many other functionalities inherited from our previous codebase.

## Training the models and plotting the score norms

In order to train the latent diffusion model, run

    ./main.py --dataset cifar10 --gae-model adv_vae --de-model diffusion

and in order to train the diffusion model on ambient space, run

    ./single_main.py --dataset cifar10 --model diffusion

The resulting models will be automatically saved in the `runs/` directory. Once the models are trained, the notebook `score_norms.ipynb` in the `notebooks` directory can be used to reproduce Figure 8.


## BibTeX

    @article{
        loaiza-ganem2024deep,
        title={Deep Generative Models through the Lens of the Manifold Hypothesis: A Survey and New Connections},
        author={Loaiza-Ganem, Gabriel and Ross, Brendan Leigh and Hosseinzadeh, Rasa and Caterini, Anthony L and Cresswell, Jesse C},
        journal={Transactions on Machine Learning Research},
        year={2024}
    }

