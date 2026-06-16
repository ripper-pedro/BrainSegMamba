[readme.md](https://github.com/user-attachments/files/29007509/readme.md)
# BrainSegMamba: Mamba-based 3D Segmentation Architecture for easy parameter testing

This repository contains my research project, initially developed for the 2026 Cassiopée Project at Télécom SudParis. Our goal was to investigate the potential of selective State Space Models (Mamba architectures) for semantic segmentation of brain tumors in 3D MRI, utilizing the BraTS2021 dataset.

## Overview
Here you will find a modular, parameterizable codebase that consolidates different Mamba architectures into a single one. This implementation allows users to switch (or combine) between different design types through simple configuration flags.

## Key Features
*   **[nnU-Net framework](https://github.com/MIC-DKFZ/nnUNet) Integration:** Full pipeline compatibility (preprocessing, patch sampling, training, and inference) ensuring fair, reproducible comparisons.

## Getting Started
Please refer to the individual directory documentation for setup instructions. Ensure you have the nnU-Net environment configured according to the official guidelines before running our trainers.

## Acknowledgments
This research was initially conducted under the guidance of Professor Nicolas Rougon (Department ARTEMIS). Check our presentation poster for more informations.

## References
*   Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
