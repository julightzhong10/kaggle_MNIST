## Simple MLP Method

**Structure**
  | Input    | CNN1    | CNN2     | MLP1  |MLP2 | Output |
  |----------|---------|----------|-------|-----|--------|
  |28\*28\*1 | 3\*3\*64|3\*3\*128 |1024   |1024 | 10     |

**Setting**
  * DroupOut: in hidden layers with 0.5 rate
  * Init Learning Rate： 5e-2
  * Optimization Method: Momentum with m=0.9 and learning rate decay 0.97/ 20 epoches
  * Activation Function: elu in CNN, tanh in MLP, Softmax in output layer
  * Loss Function : Cross Entropy
  * Batch Size: 32
  * Total training epoches: around 1000
  * Batch Normalization: Yes with decay=0.9
  * L2 weight decay rate:5e-2

**other**
over fitting occur at around epoche 25 
