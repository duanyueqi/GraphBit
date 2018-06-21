# GraphBit
GraphBit: Bitwise Interaction Mining via Deep Reinforcement Learning
This repo contains the basic code for Graphbit on CIFAR-10. 


## Quick Start

The code is based on Keras and TensorFlow. Please prepare the real-valued feature in 4096 dimensions of CIFAR-10 by downing the data through the link, and run main.py to train the network and get hashed compact features. Finally you can use bifeat_extract.m to extract the binary feature and run retrival.m to get the mAP on retrival task. 

 
## Repo organization 

The repo is organized as follows:

-	Python: the code written by python
	-	main.py: The main 
	-	US_network.py: Contains Unsupervised Network and its necessary loss function.
	-	RL_network.py: Deep Q Network, its actions, rewards and state matrix.
-	Matlab: the test file
	-	bifeat_extract.m: extract binary feature from real-valued feature.
	-	precision.m: calculate the precision.
	-	retrival.m: calculate the mAP of retrival task based on binary feature.
-	TXT: two files are the train and test label. Need to mention that  the order of features are not the same as in https://www.cs.toronto.edu/~kriz/cifar.html. 
- Link for downloading input VGG feature:
  - dropbox:
    - train: https://www.dropbox.com/s/f930w9vga62nv1i/feat16_train.npy?dl=0
    - test: https://www.dropbox.com/s/lpa7jeuel4jdo80/feat16_test.npy?dl=0
  - baidu:
    - train: https://pan.baidu.com/s/1v_Q7P388YoJ4yFjycuZ29w
    - test: https://pan.baidu.com/s/1Dlk07xqxH9nQOgqgUTbkEA
## Citation
If you find Graphbit useful, please cite it.

	
	@inproceedings{duan2018graphbit,
    title={GraphBit: Bitwise Interaction Mining via Deep Reinforcement Learning},
    author={Duan, Yueqi and Wang, Ziwei and Lu, Jiwen and Lin, Xudong and Zhou, Jie},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={8270--8279},
    year={2018}
    }
