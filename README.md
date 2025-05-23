# Learning Dynamics of RNNs in Closed-Loop Environments

Code accompanying the paper in which we characterize the learning dynamics of recurrent neural networks (RNNs) trained in **closed-loop** environments, where the RNN’s outputs influence future inputs through agent–environment interaction. We contrast this setting with the more conventional **open-loop** supervised setup, where inputs are independent of past outputs, and demonstrate how closed-loop coupling fundamentally alters the learning dynamics.

![Framework](https://github.com/yoavger/closed_loop_rnn_learning_dynamics/blob/main/figs/framework.png?raw=true)

- To reproduce the figures from the paper: ```code/figures.ipynb```
- For training a **nonlinear RNN**:```code/train_non_linear.ipynb```
- For the **theoretical analysis** of closed-loop RNNs:```code/theory.ipynb```
- Code for the **tracking task**:```code/tracking_task.ipynb```



