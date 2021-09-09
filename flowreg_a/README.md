# FlowReg-Affine (FlowReg-A)
A fully unsupervised framework for affine registration using deep learning.

## Training
To train using your own data, currently the framework supports `.mat` files, however, you may adapt to your specific needs.

The application is command-line based and can be run in two easy steps by using your terminal of choice.
1. Navigate the `flowreg_a` directory.
2. Run `python train.py` with the appropriate arguments explained below.

The command-line arguments are as follow. The text in *italics* is the expect data-type input:
- `-t` or `--train` the directory of volumes used in training (*string*)
- `-v` or `--validation` the directory of volumes used for validation (*string*)
- `-f` or `--fixed` the fixed volume path (*string*)
- `-b` or `--batch` the batch size used in training, default = 4 (*integer*)
- `-e` or `--epochs` number of training epocs, default = 100 (*integer*)
- `-c` or `--checkpoint` at which interval to save, default = 0 (*integer*)
- `-l` or `--save_loss` save loss value to a csv during training, default = True (*boolean*)
- `-m` or `--model_save` directory to save the final model (*string*)

Note: the `checkpoint` and `save_loss` will be saved in appropriate folders within the `flowreg_a` folder. Otherwise, it can be easily modified in the `train.py` file.

An example command could look something like:
```
python train.py \
--train "path/to/train/directory" \
--validation "path/to/validation/directory" \
--fixed "path/to/fixed/volume.mat" \
--batch 4 \
--checkpoint 1 \
--epochs 100 \
--save_loss True \
--model_save "path/to/model/save/directory"
```

## Registration
If you have a trained model, the script to register volumes can be found in `register.py`.

Similar to training, registration is done via a command-line interface with the following arguments:
- `-r` or `--register` directory of the volumes to be registered (*string*)
- `-f` or `--fixed` directory of the fixed volume (*string*)
- `-s` or `--save` directory where to save the registered volumes (*string*)
- `-m` or `--model` directory of the model weights, a .h5 file (*string*)

(OPTIONAL) Binary masks can be passed as additional arguments that will be warped with the calculated affine matrix. These masks do not have to be the 'brain', 'ventricles', or 'wml' (white matter lesions) masks as specified in the argument name. Any binary mask can be used as long as they correspond to the orientation and dimension of the moving volume.
- `-b` or `--brain` brain masks directory (*string*)
- `-v` or `--vent` ventricle masks directory (*string*)
- `-w` or `--wml` WML masks directory (*string*)

The output `.mat` file will be the registered volume and the corresponding flattened affine matrix. If masks are used, they will also be saved with `brainMask`, `ventMask`, or `wmlMask`.

## Citation
If you use any portion of our work, please cite our paper.
```
S. Mocanu, A. Moody, and A. Khademi, “FlowReg: Fast Deformable Unsupervised Medical Image Registration using Optical Flow,” Machine Learning for Biomedical Imaging, pp. 1–40, Sep. 2021.
```
Available at: https://www.melba-journal.org/article/27657-flowreg-fast-deformable-unsupervised-medical-image-registration-using-optical-flow