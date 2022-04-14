# H3D-Net

This is the main repository for the H3DS Reproducibility Project for the Deep Learning (MSc) course for group 54.

### Setup

We installed everything on a Virtual Machine on the Google Cloud Platform.
After creating the VM, you can connect it to your local VSCode:

1. Make sure you ahve a SSH key (e.g.)
2. Authorize your machine by adding your public key to the VM:
3. `(as root) sudo vim ~/.ssh/authorized_keys`
4. Add your public key
5. to get the VM ip you can run `curl icanhazip.com`
6. Check you can connect using SSH: `ssh root@<IP> -A`
7. Install the VSCode Remote extension
8. In VSCode, do ctrl + shift + p > remote-ssh add host > add `ssh root@3<IP> -A`

You can now edit files on the VM locally using VSCode!

### Install dependencies

To install conda and other necessary libraries, run the following commands:
```
 > sudo apt-get install -y gcc g++ libsm6 libxrender1 libfontconfig1 libice6 libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
 > sudo apt-get install linux-headers-$(uname -r)
 > sudo wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
 > bash Anaconda3-2021.11-Linux-x86_64.sh
 > cd /home
```

Then install the environment using `conda`:

```
conda env create -f environment.yml
conda activate idr
```

### Dataset Generation

Use the [flame-fitting repository](https://github.com/Rubikplayer/flame-fitting) by Rubikplayer.
Follow their instructions to create a dataset of 10.000 heads.

To transform the created `.obj` files to `.ply` files, use the command `ctmconv infile outfile`, with infile and outfile being the `.obj` and `.ply` file, respectively.

For training the IDR, the [H3DS dataset](https://github.com/CrisalixSA/h3ds) was used. Follow their steps to generate the dataset.

### Training

#### IGR

The IGR model can be trained using the files in the [IGR repository](https://github.com/amosgropp/IGR). We used `igr/code/shapespace/train.py` as main training file. However, to make sure it trains on the correct data, some changes need to be made:

- In `igr/code/datsets/`, a new Dataset file should be created that specifies how to load the head meshes from the FLAME dataset.
- In `igr/code/shapespace/`, a new `.conf` file should be created to specify which dataset should be used and to specify other settings.
- `igr/code/shapespace/train.py` should be modified to use the correct `.conf` file and training settings like the amount of epochs.

#### IDR

To train IDR, first activate the conda environment by running

```
conda activate idr
```

Then change the working directory to `H3D-Net/code` and run the dataset generator to create the stripped down H3DS dataset

```
cd H3D-Net/code
python generate_h3ds4idr.py
```

This will produce the h3ds dataset at `data/h3ds` with each folder containing the exact number of views for each head in line with the dataset used in H3D-Net.

To run the training, again from H3D-Net/code, run:

```
python idr_h3ds_train_runner.py
```

which will run the idr training for each view (5 views) and head (10 heads), in total 50 different trainings.

The viewable ply models are then stored in the `exps/TIMESTAMP/plots` folder.

### H3D-Net

Due to time constraints, the training for the H3D-Net was not completed in time.









