### How to replicate pre-processing work to train model on CT-82 Pancreas dataset:

1. The zipped data file in a Linux-friendly format is uploaded it to [this Google Drive](https://drive.google.com/drive/folders/1ZSwFGscVusxHJfpHTQourWhYMRNxAKoQ?usp=drive_linkhttps:/) folder in our team folder.

   * The compressed file for the labels is also there, which was already Linux-friendly and uncorrupted. It is the same as the version you download from the Cancer Imaging Archive.
   * [Original dataset source](https://www.cancerimagingarchive.net/collection/pancreas-ct/)
2. The data will be very messy when u unzip it in the cluster (or wherever you are working). It will have a nested directory structure and the images are unpacked as a series of hundreds of .dcm files (2D “slices” of the 3D CT scan), but the labels are already in .nii (3D) files. I’ve written **two scripts** to deal with this (both are in a directory called `data_preprocessing`):

   1. First, you should run `dem_to_nii_converter.py` in the directory that you unzip ur data, which will recursively grab the `.dcm` files and then join them into `.nii` files and label them consistently. It should also output them into train/test splits.
   2. When I tried to train the model with these `.nii` files, the axes needed to be re-permuted. This is where the script `fix_data_dims.py` comes in.
      1. The final data directory ultimately should look something like:         
```
data
|____ train
|____ test
|____ validation
```
      2. And then each folder should have image/label subdirectories:        

```
train
|___ image
|___ label
```
3. You’ll need to download **torchsample** that the old model uses by doing the following steps:

   1. `pip uninstall torchsample` (if you already tried installing it)
   2. `git clone https://github.com/ozan-oktay/torchsample.git`
   3. `cd torchsample`
   4. `pip install .`
4. When running training, use the following commands to run the multi-attention unet on 3D CT scans: `python3 train_segmentation.py --config configs/config_unet_ct_multi_att_dsv.json`and make sure you adjust the data path in the json file (or just name the folder with ur data “data”)
5. All other code changes i made to update deprecated packages etc. are in the `pancreas-CT` branch of our git repo. Pull that to start training before you do steps 1-4. **Do not overwite any changes.**

#### Common bugs
- If you get an error message saying something expected XXXX bytes but got XXXX bytes instead during pre-loading, the data conversion likely got interrupted somewhere (maybe due to OOM) -> temporary fix is to remove that image and the corresponding label from the `image` and `label` directories.
- The default for `num_workers` is 16, but I recommend setting it to 1, otherwise you may also get warnings and the process may kill itself
- For an interactive GPU session, I recommend adding this to your `~/.bashrc` on the cluster: `alias gpusession="salloc -p gpu_requeue --mem 99G -t 12:00:00 --gres=gpu:nvidia_a100-sxm4-80gb:1"`
