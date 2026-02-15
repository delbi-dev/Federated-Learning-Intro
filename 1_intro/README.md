## An Intro to FL

Federated Learning is a machine learning technique which aims at decentralizing the model's approach to training. This reduces the risk platforms to the client side only, instead of both the server/cloud side and client side. It also allows the server to not come into contact with what could potentially be privacy-sensitive data (but we are going to see how in reality an honest-but-curious server is enough to cause some troubles, see 2.5.DataSafety).

# What this repo contains

This repo contains two notebooks:
1. The first is the ideal case of IID data, and is the starter for the other notebooks I have in plan
2. The second is the slight worse case of NON-IID data, simulated multiple times through a Dirichlet distribution

Both of these, while different in nature, share the fact that they don't include the system's assumptions you would need to consider in a real system, such as latency, dropouts, samplings... ; these will be adressed in a still kind of abstract format in 5.FLRealImplementation.

I did not include configs, requirements files and such as they are really straightforward and can be changed easily if needed.

# Inspo
I was inspired mainly by these papers:
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://https://arxiv.org/abs/1602.05629)
- [Advances and Open Problems in Federated Learning](https://https://arxiv.org/abs/1912.04977)
- [Trustworthy Federated Learning](https://https://ieeexplore.ieee.org/document/10623386)