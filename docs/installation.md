# CUDA 13.0 + NVIDIA Driver Installation on Ubuntu 22.04

> This guide documents a **tested**, **repeatable** way to install the NVIDIA driver and CUDA 13.0 toolkit on Ubuntu 22.04.
> 
> It covers:
> - APT repository method (**recommended**)
> - Runfile method (**last resort**)
> - Common pitfalls (Secure Boot, `nouveau`, DKMS, local repo conflicts)

---

## 📌 Why This Guide
Many Ubuntu 22.04 users struggle with CUDA installation due to:
- Conflicting old NVIDIA/CUDA packages
- Local repo GPG key errors
- Secure Boot module signing issues
- Missing kernel headers/DKMS
- `nouveau` driver conflicts

This README is meant to **get you from a fresh Ubuntu to a working CUDA environment without headaches**.

---

## 🛠 Prerequisites
- Ubuntu 22.04 LTS (Jammy)
- An NVIDIA GPU supported by CUDA 13.0
- Admin access (`sudo`)
- Internet connection
- (Optional) Secure Boot disabled **OR** willingness to enroll a Machine Owner Key (MOK)

---

## 🚀 Recommended: APT Repository Install

### 0) Remove old NVIDIA/CUDA bits
```bash
sudo /usr/bin/nvidia-uninstall || true
sudo apt purge -y 'nvidia-*' 'libnvidia-*' 'cuda-*' 'nsight-*' || true
sudo rm -f /etc/apt/sources.list.d/cuda-*-local*.list
sudo rm -rf /var/cuda-repo-*
sudo apt autoremove -y
```

### 1) Install build dependencies

```bash
sudo apt update
sudo apt install -y build-essential dkms linux-headers-$(uname -r) mokutil
```

### 2) Add NVIDIA's official CUDA repo

```bash
# Set repo priority
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Add repo keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt update
```

### 3) Install NVIDIA driver (580.xx for CUDA 13.0)

```bash
sudo apt install -y cuda-drivers
```

> If **Secure Boot** is enabled, follow the on-screen MOK enrollment:
>
> * Set a password when prompted
> * On reboot, select *Enroll MOK* → enter that password

### 4) Reboot and check driver

```bash
sudo reboot
nvidia-smi
```

### 5) Install CUDA Toolkit 13.0

```bash
sudo apt install -y cuda-toolkit-13-0
# OR to track all 13.x updates:
# sudo apt install -y cuda-toolkit-13
```

### 6) Add CUDA to PATH

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' | sudo tee /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh

nvcc --version
```

---

## ⚠️ Optional: Runfile Method (Not Recommended)

Use this only if you must install without apt (e.g., offline or custom driver build).

### 0) Prep system

```bash
sudo apt update
sudo apt install -y build-essential dkms linux-headers-$(uname -r) mokutil

sudo apt purge -y 'nvidia-*' 'libnvidia-*' 'cuda-*' 'nsight-*' || true
sudo apt autoremove -y

# Blacklist nouveau
echo -e "blacklist nouveau\noptions nouveau modeset=0" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u
```

### 1) Switch to text mode

```bash
sudo systemctl isolate multi-user.target
# Stop display manager (pick your DE)
sudo systemctl stop gdm3 || sudo systemctl stop sddm || sudo systemctl stop lightdm
```

### 2) Run the installer

```bash
cd /path/to/cuda_<version>_linux.run
sudo sh cuda_<version>_linux.run
```

> Uncheck driver install if you only want the toolkit (use arrow keys + spacebar).

### 3) Reboot and test

```bash
sudo reboot
nvidia-smi
nvcc --version
```

---

## 🐛 Troubleshooting

| Symptom                                      | Likely Cause                                        | Fix                                                                                           |
| -------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `NO_PUBKEY` on update                        | Leftover local CUDA repo                            | Remove `/etc/apt/sources.list.d/cuda-*-local*.list` and `/var/cuda-repo-*`, then `apt update` |
| Driver fails to load after install           | Secure Boot enabled, no MOK enrolled                | Enroll MOK during install or disable Secure Boot                                              |
| `nouveau` still loads                        | Blacklist not applied                               | Verify `lsmod | grep nouveau` is empty; rerun blacklist +`update-initramfs -u` |
| `Unable to locate package cuda-toolkit-13-0` | Repo not added or apt not updated                   | Redo repo keyring + `sudo apt update`                                                         |
| Runfile install errors with code 256         | X/Wayland still running, missing headers, conflicts | Stop display manager, install headers, purge old packages                                     |

---

## ✅ Verification

```bash
nvidia-smi          # should show GPU, driver version, CUDA version
nvcc --version      # should show release 13.0
```

---

**References:**

* [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
* [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
