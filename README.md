# DoomGAN
Procedurally Generate DOOM(1993) levels using Generative Adversarial Networks (GAN)

The goal of this project is to harness the power of DNN to automate the generation of various aspects of level design by training the network on a subset of DOOM levels scraped from the internet that can be bought together to generate levels for the game DOOM. This is achieved by implementing a hybrid GAN network with a Wasserstien GAN used to generate the topological features such as the floor plan, wall structure and the height of every section of the encompassing level. Once obtained, these features are fed into a Conditional Adversarial Network (CAN) to generate a map for all the game objects in that level. At this point we have only focused on the categories of monsters, ammunitions, powerups, artifacts and weapons but more can be included.

# Requirements
This project is made using python v3.9 and requires tensorflow related application for applying GPU hardware to boost the computational
It includes the installation Cuda v11.2 and Cudnn v8.7 with tensorflow-gpu instead of native tensorflow. The remainder of the libraries are included in the 'requirements.txt' file which needs to be installed before proceeding.

# Scrap the dataset
Before you can start generating levels, you need to procure the dataset with which each of the networks have to be trained. This consists of DOOM PWADs which are downloaded in the 'dataset/scraped/' folder with its respective metadata. Details to execute the scraper can be found within the 'scraper' folder.

# Parse the dataset
Once the raw dataset of DOOM WADs are collected, it needs to be parsed into a format that can be used for training the GANs. They are converted into a set of feature maps i.e matrices through modifications to Eduardo Giacomello's WAD Parser and stored as tensorflow records in the 'dataset/parsed' folder with the relevant metadata. The pertaint files to parse the scraped WAD collection can be found in the 'WADParser' folder

# Train the network
The dataset is extracted from the records and split into a training and validation set which is used to optimize the network until they can generate viable level layouts and object category maps. This requires both the models present in the networks are trained, seperately in this case to reduce memory requirements. Model optimization, checkpoints and its evaluation can be done using the 'gan' folder


# Generate DOOM Levels
Since the project is created in an experimental setup, there exist 3 types of generators within the DOOMLevelGen.py file. This can be controlled through modifying the 'use_hybrid' and 'mod_can' flags to select the network. This requires that the 2 models are previously trained and their respective checkpoints saved to restore the parameters and generate the feature maps. To generate a DOOM WAD, run
```
python DOOMLevelGen.py
```