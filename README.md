# SynMOT

## Usage
1. Clone this repository
2. Add the missing files
   - Download [`mot.tar`](https://drive.google.com/drive/folders/1P09HzEL8CDMkwqaHKeDwM1x6Yerhi5US) and unzip it at `SynMOT/datasets/`
   - Download [`human_segmenter_checkpoints.tar`](https://drive.google.com/drive/folders/1J0PDD4AhZ8WQBjZFHUWc6Qdo8xeNgRXA) and unzip it at 'SynMOT/src/modules/human_segmenter/checkpoints/'.
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
