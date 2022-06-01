# Realistic Trajectory Manipulation as Data Augmentation for Multi-Object Tracking

Course project members: Nanyan Zhu (nz2305) and Lucas Tao (lucastao).

External member: Chen Liu.

## Motivation
Multi-object tracking (MOT) aims to identify and keep track of all objects in a video. Under the mainstream formulation, MOT consists of two main stages: detection and association. Individual objects are recognized in the former stage, usually in the form of bounding boxes each with a confidence score. In the latter stage, an association algorithm is used to figure out the correspondences among the current detections and previous detections (sometimes referred to as ``tracklets'').

While the detection stage is witnessing tremendous progress as detectors gain power and efficiency, the association stage remains less attended. Intriguingly, many state-of-the-art MOT methods are still using very rudimentary approaches for association, such as the Hungarian matching algorithm. While there exist end-to-end learning-based methods for data association stage, they are not gaining enough popularity. One main reason against such data-hungry methods is the scarcity of labeled data for tracking.

In this project, we propose a data augmentation approach to generate synthetically labeled tracking datasets from existing labeled tracking data. The approach will ``manipulate the trajectories'' of persons in the annotated video stream.

## Usage
1. Clone this repository
2. Add the missing files
   - Download [`mot.tar`](https://drive.google.com/drive/folders/1P09HzEL8CDMkwqaHKeDwM1x6Yerhi5US) and unzip it at `SynMOT/datasets/`
   - Download [`human_segmenter_checkpoints.tar`](https://drive.google.com/drive/folders/1J0PDD4AhZ8WQBjZFHUWc6Qdo8xeNgRXA) and unzip it at `SynMOT/src/modules/human_segmenter/checkpoints/`.
3. Create a proper environment.
   - For docker users, [a docker image](https://drive.google.com/drive/folders/1muaVyr9s2BtPwoRibQSAZ5j_wuNvOhex) is provided.
4. Run `main.py`.
   ```
   cd src
   python main.py
   ```
   The docker provided does not work with certain GPUs. So to run the script with CPU, use:
   ```
   CUDA_VISIBLE_DEVICES=-1 python main.py
   ```

## Acknowledgements
This work has been assisted by the following repositories:
- [SiamMask](https://github.com/foolwood/SiamMask) as our human segmentation module.
- [LaMa](https://github.com/saic-mdal/lama) as our image inpainting module.
- [Image Harmonization](https://github.com/saic-vul/image_harmonization) as our image blending module.
