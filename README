STARTUP GUIDE (Ubuntu 22.04 VM) do these as root
(This is the exact sequence I want to keep using across new VMs. Keep it
general on drivers; Vulkan user-space + ffmpeg are the important fixes.)
# 0) Base tools
apt-get update
apt-get install -y git python3-venv python3-pip build-essential wget tar tmux
# 1) Clone project
mkdir -p /workspace/base && cd /workspace/base
git clone --recurse-submodules https://github.com/benwyman/cardreamer-carla.git
carla_project
cd /workspace/base/carla_project
# 2) Venv + deps (CarDreamer is NOT installed from requirements)
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# 3) Install CarDreamer from source via flit (editable symlink)
cd /workspace/base/carla_project/CarDreamer
export FLIT_ROOT_INSTALL=1
flit install --symlink
cd /workspace/base/carla_project
# 4) Get CARLA 0.9.15
chmod +x scripts/get_carla_0915.sh
./scripts/get_carla_0915.sh
# 5) Vulkan user-space + ffmpeg (key fix for headless CARLA + GIF/video
summaries)
apt-get update
apt-get install -y libvulkan1 vulkan-tools libomp5 mesa-vulkan-drivers ffmpeg
# 6) Create non-root user and hand off the project
adduser --disabled-password --gecos "" carlauser
usermod -aG sudo carlauser
install -d -m 700 -o carlauser -g carlauser /home/carlauser/.ssh
[ -f /root/.ssh/authorized_keys ] && install -m 600 -o carlauser -g carlauser /
root/.ssh/authorized_keys /home/carlauser/.ssh/authorized_keys
51
chown -R carlauser:carlauser /workspace/base/carla_project
# 7) Switch to non-root and load env
su - carlauser
cd /workspace/base/carla_project
source venv/bin/activate
source /workspace/base/carla_project/env.sh
