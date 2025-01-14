CSRNet and WHCNet share enough similarities that for presentation, the relevant files are placed in one folder.

Checkpoints removed, available on request.

WHCnet repo is borderline unusable, so WHCNet implementation is essentially the CSRNet implementation but using the WHCNet model. The two model files are left here for comparison.

To train CSRNet, open and run through the Notebooks in the following order:

1. dataset_preparation.ipynb
2. CSV_to_H5.ipynb
3. make_dataset.ipynb
4. make_json.ipynb