Here is the code used for "Making the most of your day: online learning for optimal allocation of time".

All the experiments presented in the paper are run using this code.

Packages required:
	- numpy
	- sys
	- termcolor
	- matplotlib
	- warning
	- time
	- tqdm


Presentation of the files:
	- RBT.py: implement the augmented red black tree structures
	- rewards: implement the different class of reward functions used
	- utils: auxiliary functions for running simulations and the different distribution classes (for noise and rides)
	- strategies: implement all the different rider strategies (algorithms). When possible, both vanilla and RBT (red black trees) implementations are available
	- demo.ipynb: python notebook used for the plots of behaviors on a single instance (accept/reject decisions and estimations of the profitability)
	- simus.ipynb: python notebook used for the regret plots
