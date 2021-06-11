# ROLANN
Single layer neural network (without hidden neurons) trained with noniterative learning procedure. It employs a mean-squared loss function with a regularizated term (L2)


Novel regularized training method for one-layer neural networks (no hidden layers) that uses an L2-norm penalty term. This noniterative supervised method uses a closed-form expression and determines the optimum set of weights by solving a system of linear equations. It presents two interesting properties that differentiate it from other methods:

– Computational complexity depends on the minimum value among the number of inputs and samples of the training set, in contrasts with the majority of methods that are only computationally efficient on one side.

–Incremental  and  distributed  privacy-preserving learning is allowed, which means a perfect fit in federated learning environments.These characteristics make ROLANN very useful in many different scenar-ios, for instance, in dealing with data sets with a high number of inputs, as the majority of methods concentrate on reducing only complexity with respect to the number of training samples. 
