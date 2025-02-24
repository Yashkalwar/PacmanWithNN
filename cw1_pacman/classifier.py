# classifier.py
# Lin Li/26-Dec-2021

# Machine learning
# MSc Artificial Intelligence
# yash kalwar
# K24018103
#
# Code has been referenced from:
# [1] https://medium.com/@qempsil0914/implement-neural-network-without-using-deep-learning-libraries-step-by-step-tutorial-python3-e2aa4e5766d1
# [2] https://encord.com/blog/an-introduction-to-cross-entropy-loss-functions/
# [3] https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# [4] https://medium.com/@adila.babayevaa/simple-explanation-of-overfitting-cross-entropy-and-regularization-in-model-training-871444895e3f
# [5] most imporatantly - Content about Neural network and classfier from keats helped me the most (pattern recognitiion and machine learning)

'''
for this coursework, after researching, as a classifier i finalised and implements a fully connected neural network using NumPy.
The network is designed for multi-class classification with 3 layers: two hidden layers with ReLU activation and an output layer using softmax.
It applies He initialization for weights, mini-batch gradient descent with cross-entropy loss, and regularization to prevent overfittting.
The model also includes features like early stopping, dynamic learning rate adjustment, and  train-validation-test splitting.
Added a compute_loss function to keep track of losses and adjust the learning rate and this can also be used for visualisation.
'''

import numpy as np

class Classifier:
    def __init__(self, input_size=25, hidden_size1=128, hidden_size2=64, output_size=4, reg_strength=0.001):
        '''
        Initialize the neural network's weights and biases 
        The network consists of:
        - Input layer with size 25
        - Two hidden layers with sizes 128 and 64
        - Output layer with size 4 (represnting directions)
        Regularization is applied with a small value 
        '''
        self.weights1 = np.random.randn(input_size, hidden_size1) / np.sqrt(input_size / 2)            
        self.biases1 = np.zeros(hidden_size1)  
        self.weights2 = np.random.randn(hidden_size1, hidden_size2) / np.sqrt(hidden_size1 / 2)  
        self.biases2 = np.zeros(hidden_size2)  
        self.weights3 = np.random.randn(hidden_size2, output_size) / np.sqrt(hidden_size2 / 2)
        self.biases3 = np.zeros(output_size)  
        self.reg_strength = reg_strength  
   
    def reset(self):
        # Re-initialize weights and biases to their original values
        self.__init__()

    def fit(self, data, target, epochs=1000, lr=0.01, batch_size=64, val_data=None):
        # Convert data and target to numpy arrays for processing
        data = np.array(data)
        target = np.array(target)

         # 80% for training , 10% for validation and remaining unseen 10% for testing
        n_samples = len(data)
        n_train = int(0.8 * n_samples)  
        n_val = int(0.1 * n_samples)     

        # Randomly shuffle indices for splitting
        indices = np.random.permutation(n_samples) 
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]

        train_data = data[train_idx]
        train_target = target[train_idx]
        val_data = (data[val_idx], target[val_idx])
        test_data = (data[test_idx], target[test_idx])

        # One-hot encode the target labels for training
        num_classes = 4
        train_target_one_hot = np.zeros((train_target.size, num_classes))
        for i in range(train_target.size):
            train_target_one_hot[i, train_target[i]] = 1

        # Initialize parameters for early stopping
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        no_improvement_count = 0  # Counter for validation loss improvement
        max_no_improvement = 30  # Maximum epochs without improvement

        print("\nStarting training...")

        '''
            1) This is the main Training loop where we are performing a mini-batch training on shufled data with ReLU activation function.
            2) with a forward pass from input to 1st hidden layer then second hidden layer and to the output layer(softmax probablities).
            3) After this we are calculating the cross entropy loss using the regularisation we initialised to 0.001.
            4) Now we perform back-propogation
            5) After that we compute the gradients for each layer and finally update the weights and the biases.
        '''

        for epoch in range(epochs):
            # Shuffle data for mini-batch training
            indices = np.random.permutation(len(train_data))
            shuffled_data = train_data[indices]
            shuffled_target = train_target_one_hot[indices]

            # Mini-batch training
            for i in range(0, len(train_data), batch_size):
                batch_data = shuffled_data[i:i+batch_size]
                batch_target = shuffled_target[i:i+batch_size]

                # Forward pass
                hidden1 = np.dot(batch_data, self.weights1) + self.biases1
                hidden_activation1 = np.maximum(0, hidden1)

                hidden2 = np.dot(hidden_activation1, self.weights2) + self.biases2
                hidden_activation2 = np.maximum(0, hidden2)

                logits = np.dot(hidden_activation2, self.weights3) + self.biases3
                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

                # Compute cross-entropy loss with regularization which we initialised
                batch_loss = -np.mean(np.sum(batch_target * np.log(probs + 1e-8), axis=1))
                batch_loss += 0.5 * self.reg_strength * (np.sum(self.weights1 ** 2) + 
                                                        np.sum(self.weights2 ** 2) + 
                                                        np.sum(self.weights3 ** 2))

                # Backpropagation
                error = probs - batch_target

                # Compute gradients for each layer
                grad_weights3 = np.dot(hidden_activation2.T, error) / batch_size + self.reg_strength * self.weights3
                grad_biases3 = np.mean(error, axis=0)

                hidden_error2 = np.dot(error, self.weights3.T)
                hidden_error2[hidden_activation2 <= 0] = 0

                grad_weights2 = np.dot(hidden_activation1.T, hidden_error2) / batch_size + self.reg_strength * self.weights2
                grad_biases2 = np.mean(hidden_error2, axis=0)

                hidden_error1 = np.dot(hidden_error2, self.weights2.T)
                hidden_error1[hidden_activation1 <= 0] = 0

                grad_weights1 = np.dot(batch_data.T, hidden_error1) / batch_size + self.reg_strength * self.weights1
                grad_biases1 = np.mean(hidden_error1, axis=0)

                # Update weights and biases using gradients for all the layers
                self.weights3 -= lr * grad_weights3
                self.biases3 -= lr * grad_biases3
                self.weights2 -= lr * grad_weights2
                self.biases2 -= lr * grad_biases2
                self.weights1 -= lr * grad_weights1
                self.biases1 -= lr * grad_biases1

            #  Computing validation loss and validation loss for evaluation
            val_loss = self.compute_loss(val_data[0], val_data[1])
            train_loss = self.compute_loss(train_data, train_target)
            
            # Print losses for every epoch
            print(f"Epoch {epoch+1} ; training loss = {train_loss:.6f} ; validation loss = {val_loss:.6f} ; learning rate = {lr:.6f}")

            # checking the condition for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                no_improvement_count = 0  # Reset no improvement counter
            else:
                patience_counter += 1
                no_improvement_count += 1  # Increment no improvement counter

            # Check if we should stop due to no improvement
            if no_improvement_count >= max_no_improvement:
                print(f"\nStopping training: No improvement in validation loss for {max_no_improvement} epochs")
                break

            # Early stopping by checking patience otherwise reducing lr after every epoch
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                lr *= 0.5
                patience_counter = 0
            else:
                lr *= 0.995

            # Additional early stopping condition based on validation loss
            if train_loss > 1e6:
                print("\nTraining stopped due to exploding loss")
                break

        # Final evaluation including test set
        final_train_loss = self.compute_loss(train_data, train_target)
        final_val_loss = self.compute_loss(val_data[0], val_data[1])
        final_test_loss = self.compute_loss(test_data[0], test_data[1])
        
        print("\nFinal Results:")
        print(f"Final Training Loss: {final_train_loss:.6f}")
        print(f"Final Validation Loss: {final_val_loss:.6f}")
        print(f"Final Test Loss: {final_test_loss:.6f}")



    def predict(self, data, legal=None):
        # Convert data to numpy array and ensure it's 2D
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Forward pass
        hidden1 = np.dot(data, self.weights1) + self.biases1
        hidden_activation1 = np.maximum(0, hidden1)

        # Second hidden layer
        hidden2 = np.dot(hidden_activation1, self.weights2) + self.biases2
        hidden_activation2 = np.maximum(0, hidden2)

        # Compute logits for output layer
        logits = np.dot(hidden_activation2, self.weights3) + self.biases3
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # Softmax probabilities

        predicted_class = np.argmax(probs, axis=1)[0]
        
        #print(predicted_class)
        
        return predicted_class

    def compute_loss(self, data, target):
        '''
            This function computes the loss of the neural network by comparing the predicted probabilities with the actual target labels. 
            It helps in tracking the performance of the model during training and testing phases. 
            By monitoring the loss, we can assess the model's learning progress and make necessary adjustments.
        '''
        data = np.array(data)
        target = np.array(target)

        num_classes = self.biases3.shape[0]
        target_one_hot = np.zeros((target.size, num_classes))
        target_one_hot[np.arange(target.size), target] = 1

        hidden1 = np.dot(data, self.weights1) + self.biases1
        hidden_activation1 = np.maximum(0, hidden1)

        hidden2 = np.dot(hidden_activation1, self.weights2) + self.biases2
        hidden_activation2 = np.maximum(0, hidden2)

        logits = np.dot(hidden_activation2, self.weights3) + self.biases3
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Compute cross-entropy loss with regularization
        loss = -np.mean(np.sum(target_one_hot * np.log(probs + 1e-8), axis=1))
        loss += 0.5 * self.reg_strength * (np.sum(self.weights1 ** 2) + 
                                          np.sum(self.weights2 ** 2) + 
                                          np.sum(self.weights3 ** 2))

        return loss
