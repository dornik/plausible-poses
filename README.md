# Physically plausible object pose estimates
This repository implements the methods described in *Physical Plausibility of 6D Pose Estimates in Scenes of Static 
Rigid Objects*. The code and data required to reproduce the results in Figures 5, 7 and 8 are provided.

## Dependencies
The code has been tested on Ubuntu 16.04 with Python 3.6. To set-up the Python environment, use Anaconda and 
the provided YAML file:

`conda env create -f environment.yml --name plausibility`

`conda activate plausibility`

We use the pose-error functions and rendering implemented in the BOP toolkit. Please make sure that the BOP toolkit 
submodule is correctly initialized and see [this repository](https://github.com/thodan/bop_toolkit) for installation 
instructions. Finally, replace the file `bop_toolkit_lib/renderer_py.py` with `src/renderer_py.py`. It patches the 
Python-based renderer in the BOP toolkit, allowing to apply a transformation when loading objects.

## Dataset
The object models from the YCB VIDEO dataset can be downloaded on the [BOP Challenge 2020 page](https://bop.felk.cvut.cz/datasets/). 
Please change `YCBV_PATH` in `src/evaluation.py` to reflect your model directory. 

Additional meta information for the models and precomputed samples for evaluation are provided in `data`.

## Evaluation
Running `python src/evaluation.py --target [TARGET] --mode [MODE]` generates pose estimates for evaluation, computes the
baseline pose-error functions (MSSD, MSPD, VSD) as well as our pose errors. The results are visualized in an animated
plot. `TARGET` selects the artifical scene to evaluate: `bowl` for the scene in Figure 5, `marker` for Figure 7 and 
`clamp` for Figure 8 in the paper. `MODE` selects whether to apply `rotation` or `translation` error to the ground-truth
pose.

## Citation
If you use this repository in your publications, please cite

```
@article{bauer2020plausibility,
    title={Physical Plausibility of 6D Pose Estimates in Scenes of Static Rigid Objects},
    author={Bauer, Dominik and Patten, Timothy and Vincze, Markus},
    booktitle={European Conference on Computer Vision Workshops (ECCVW)},
    year={2020}
}
```