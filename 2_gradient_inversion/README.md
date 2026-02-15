## Gradient Inversion on Aggregated Gradients
# An Intro
Many times, when we think about gradients, we think of a "black box" that transports data to update our model, but we often forget how gradients have a precise meaning; in real scenarios they convey precise and privacy-sensitive information.

It has been proven numerous times how images and data in general can be extracted from gradients, but this was considering a gradient which is not aggregated, and conveys only one precise image, making the inversion much easier.

But what happens in FL and distributed settings, where the gradients is by definition aggregated, many times with a batch containing dozens of different images?

Is it still possible to extract the features in the cohort?

The answer is yes, and we'll do so by using a strong prior to resolve a posterior Bayesian Problem.

# A Quick Overview of the experiment
The chapters in this notebook contain:
1. Context of the Federated Learning security guarantees
2. Deep Leakage from Gradients (DLG), Gradient Inversions
3. Four steps for DLG
- dummy inputs
- compute dummy gradients
- compare dummy gradients to real leaked gradients
- optimizing the dummy inputs
4. Variational AutoEncoder (VAE), Encoder and Decoder, privacy breach and systems considerations

# Other Contents
- in the src folder you can find the python scripts I used for the experiment; they are divided into inversion, vae, and utils
- in the stuff folder you can find the files I used for the inversion, which are described in more detail in the notebook; However, they are not needed for reproducibility purposes, as they "should" load directly from my drive folder (if it doesn't work, feel free to contact)
- config.json; like for the other files, it is directly loaded through my drive; anyway, it's relatively easy to change the script to load from your modified config.json if you want to change (or even add/remove) some variables, again, feel free to do so!
- in the notebook you can find a definition section

# Inspirations
I was heavily inspired by these papers I read:
- [Inverting Gradients -- How easy is it to break privacy in federated learning?](https://https://www.emergentmind.com/papers/2003.14053)
- [Gradient Inversion of Federated Diffusion Models](https://arxiv.org/abs/2405.20380)
- [GradInvDiff: Stealing Medical Privacy in Federated Learning via Diffusion-Based Gradient Inversion](https://papers.miccai.org/miccai-2025/0383-Paper1362.html)
- [SoK: On Gradient Leakage in Federated Learning](https://https://www.emergentmind.com/papers/2404.05403)

# To-Do's
There's a list of things I have yet to do; for instance, in the VAE part, while I was able to implement those formulas I wrote, I am a 1st year BSc and still haven't took my Statistical Inference, Bayesian Modelling and Optimization classes, so in the future I'll 100% do a passover of everything to check for hideous mistakes.
- Fixing NON-IID and IID data notebooks; as of right now they are still in need of some touchups (as I rushed them)
- Differential Privacy Model (I think I'll use Gaussian Noise)
- Secure Aggregation Demo
- Real Federated Learning Simulation (with client dropout, latency, limited bandwidth, convergence, privacy impact, robustness...; Maybe with Kafka or Mock Streaming?)
- OOP design for FL pipeline (?)
- Poisoning and byzantine clients, active attacks on FL