#!/usr/bin/env bash
# Set up development environment for autoware-integration/tum_prediction package
# when use this package only with autoware.

sudo apt update
sudo apt install python3-pip
sudo pip3 install --upgrade pip

cd ~/autoware-integration/
pip3 install -r requirements.txt

colcon build
# echo "source ~/autoware/install/setup.bash" >> ~/.bashrc
echo "source ~/autoware-integration/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
