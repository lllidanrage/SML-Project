LightGBM reproduction notes

Run commands from the SML-Project folder.

Environment:
Python 3.11
LightGBM 4.7.0

Install Python dependencies:
python -m pip install -r requirements.txt

Required inputs:
- Seven .npz feature files in features/
- results/outer_and_inner_folds.pkl

Project.ipynb generates the feature files from the original images.
Image data and feature files are excluded from Git.

To reproduce the LightGBM experiments:
Open LightGBM_Robustness.ipynb in VS Code or Jupyter.
Select the Python environment containing the dependencies.
Run the cells from top to bottom.

The notebook includes pilot tuning trials and the final experiments.
Full reproduction may take several hours.

For report writing, use the existing outputs without retraining:
- results/lightgbm/lightgbm_results.csv
- results/lightgbm/lightgbm_inner_tuning.csv
- results/lightgbm/figures/
- results/lightgbm/spatial_clean_confusion_matrix.csv

The final experiments use four feature representations,
seven conditions, ten outer folds, and three inner folds.

Training checkpoints are kept locally and excluded from Git.
Regenerating the confusion matrix from predictions requires
the Spatial checkpoints created during training.