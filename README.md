# Curriculum-Inspired Hierarchical Learning for Image Classification

This project investigates whether hierarchical, curriculum-inspired learning improves image classification compared to standard flat CNN training. Inspired by the human visual cortex, a multi-stage pipeline was designed where successive CNN modules learn intermediate visual representations (edges → corners → contours → saliency) before final object recognition.

The hierarchical model is implemented in curriculum_learner.py, which defines the intermediate networks (EdgeNet, CornerNet, ContourNet, SaliencyNet) and the final RecognitionNet. A baseline flat CNN is implemented in flat_model.py. The training and evaluation pipeline is implemented in train_curriculum.py. Data-efficiency experiments, runtime measurements, parameter counts, and FLOPs analysis are included in experiments.py.

Experiments are conducted on the CIFAR-10 dataset and compare classification performance between the hierarchical pipeline and the flat CNN baseline under different training data sizes.
