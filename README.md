# Temporary Repository

Implmentation of PointNet and Transformer inspired by [https://arxiv.org/abs/2109.08141](https://arxiv.org/abs/2109.08141)

Using only PointNet for now, since the data pointcloud is mostly uniform, no need to account for different densities

2 sampling methods:
>>
    1. FPS for global context
    2. Select N seed points from the first FPS, then using a ball query to get nearby points within LOCAL_RADIUS. The points in the raduis are downsampled with FPS to get a dense local context

**The local sampling can be disabled by setting N_SAMPLES_LOCAL=0**

Download the voxelized pottery here [https://drive.google.com/file/d/1cMvwglGkqwbldFO3LNpCTPK5eYqzjApZ/view?usp=sharing](https://drive.google.com/file/d/1cMvwglGkqwbldFO3LNpCTPK5eYqzjApZ/view?usp=sharing)
