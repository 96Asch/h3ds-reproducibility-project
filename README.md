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
To install conda and basic libraries, run the following lines:
```
 > sudo apt-get install -y gcc g++ libsm6 libxrender1 libfontconfig1 libice6 libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
 > sudo apt-get install linux-headers-$(uname -r)
 > sudo wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
 > bash Anaconda3-2021.11-Linux-x86_64.sh
 > cd /home
```

### Dataset Generation

Use the [flame-fitting repository](https://github.com/Rubikplayer/flame-fitting) by Rubikplayer.
Follow their instructions to create a dataset of 10.000 heads.

To transform the created `.obj` files to `.ply` files, use the command `ctmconv infile outfile`, with infile and outfile being the `.obj` and `.ply` file, respectively.




