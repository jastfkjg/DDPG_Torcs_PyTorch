# Using PyTorch and DDPG to play Torcs

This code is developed based on DDPG_Torcs (https://github.com/yanpanlau/DDPG-Keras-Torcs) and Gym-Torcs (https://raw.githubusercontent.com/ugo-nama-kun/gym_torcs).

The detailed explanation of original TORCS for AI research is given by Daniele Loiacono et al. (https://arxiv.org/pdf/1304.1672.pdf)

# Requirements

* Python 3
* [gym_torcs](https://github.com/ugo-nama-kun/gym_torcs)
* PyTorch 0.4.1

# How to Run?

```
git clone https://github.com/jastfkjg/DDPG_Torcs_PyTorch.git
cd DDPG_Torcs_PyTorch
python test.py

```

# training

You can train the model both on cpu or gpu.

After 100 episodes (about one hour) the car can have a good performance.

If you want to change the reward function, see gym_torcs.py .
