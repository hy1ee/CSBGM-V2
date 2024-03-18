# CS-BGM v2

This repository provides code to reproduce results reported in the paper [Uncertainty Modeling in Generative Compressed Sensing](https://proceedings.mlr.press/v162/zhang22ai.html). 

Portions of the codebase in this repository uses codes originally provided in the open-source [CS_BGM](https://github.com/mengchuxu97/CS_BGM)

Static_flow_vae uses codes provided in the open-source [flow-VAE](https://github.com/fmu2/flow-VAE)

## Description
1. Update some codes from the origin repository to pytorch
2. Added some new models (mainly improvements combined with flow_based models) in CSBGM

## Requirements: 
---

1. Python 3.6
2. Other packages when running codes. Please install them when the requirements are reported.


### Preliminaries
---

You can run [main.py] in each folder to train model:
NICE / RealNVP / VAE / AE_RealNVP / VAE_RealNVP(flow-VAE)


### Demos
---

run [compressed_sensing.py]