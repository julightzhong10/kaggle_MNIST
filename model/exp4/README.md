
**Structure**
  | Input    | CNN1    | CNN2     | MLP1  |MLP2 | Output |
  |----------|---------|----------|-------|-----|--------|
  |28\*28\*1 | 3\*3\*64|3\*3\*128 |1024   |1024 | 10     |

**Setting**
  * DroupOut: in MLP layers with 0.5 rate
  * Init Learning Rate： 1e-1
  * Optimization Method: Momentum with m=0.9 and learning rate decay 0.9/ 1 epoche
  * Activation Function: elu in CNN, tanh in MLP, Softmax in output layer
  * Loss Function : Cross Entropy
  * Batch Size: 16
  * Total training epoches: around 50
  * Batch Normalization: Yes with decay=0.9
  * L2 weight decay rate:5e-2

**other**
over fitting occur at around epoche 25 
