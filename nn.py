''' 
create neural network from zero to one.
@references: https://github.com/makeyourownneuralnetwork/
'''
import numpy as np
import scipy.ndimage
import scipy.special
import matplotlib.pyplot as plt

class NN:
    '''
    neural network class.
    '''
    def __init__(self, inodes, hnodes, onodes, lr):
        '''
        initialize the neural network with input, hidden and output nodes,
        and also learning rate.
        '''
        # number of nodes in input, hidden and output layer
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes

        # learning rate
        self.lr = lr

        # weights between input and hidden layer
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))

        # weights between hidden and output layer
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # activation function
        self.activate = lambda x: scipy.special.expit(x)

        # reverse activation function
        self.reverse_activate = lambda x: scipy.special.logit(x)


    def train(self, inputs_list, targets_list):
        '''
        train the neural network with training data.
        inputs_list: input data
        targets_list: target data
        '''
        # convert inputs and targets to 2d arrays
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # hidden inputs
        hi = np.dot(self.wih, inputs)
        
        # hidden outputs
        ho = self.activate(hi)

        # final inputs
        fi = np.dot(self.who, ho)

        # final outputs
        fo = self.activate(fi)

        # output errors
        oe = targets - fo

        # hidder layer errors is the output errors, split by weights, recombined at hidden layer
        he = np.dot(self.who.T, oe)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((oe * fo * (1.0 - fo)), np.transpose(ho))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((he * ho * (1.0 - ho)), np.transpose(inputs))


    def query(self, inputs_list):
        '''
        query the neural network with inputs.
        '''
        # print('inputs_list:', inputs)
        # print('wih:', self.wih)
        # print('who:', self.who)
        # convert inputs to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        # print('inputs:', inputs)

        # hidden inputs
        hi = np.dot(self.wih, inputs)
        # print('hi:', hi)

        # hidden outputs
        ho = self.activate(hi)
        # print('ho:', ho)

        # final inputs
        fi = np.dot(self.who, ho)
        # print('fi:', fi)

        # final outputs
        fo = self.activate(fi)
        # print('fo:', fo)
        return fo
    
    def back_query(self, targets_list):
        '''
        back query the neural network with targets_list.
        '''
        fo = np.array(targets_list, ndmin=2).T
        # print('fo:', fo)
        fi = self.reverse_activate(fo)
        # print('fi:', fi)
        ho = np.dot(self.who.T, fi)
        # scale the hidden outputs to 0.01 to 0.99
        ho -= np.min(ho)
        ho /= np.max(ho) # scale 0.0 to 1.0
        ho *= 0.98 # scale 0.0 to 0.98
        ho += 0.01 
        # print('ho:', ho)
        hi = scipy.special.logit(ho)
        # print('hi:', hi)
        inputs = np.dot(self.wih.T, hi)
        # scale the inputs to 0.01 to 0.99
        inputs -= np.min(inputs)
        inputs /= np.max(inputs) # scale 0.0 to 1.0
        inputs *= 0.98 # scale 0.0 to 0.98
        inputs += 0.01
        # print('inputs:', inputs)
        return inputs


if __name__ == '__main__':

    # Initialize the neural network 
    inodes = 28 * 28
    hnodes = 200
    onodes = 10
    lr = 0.2
    nn = NN(inodes,hnodes,onodes,lr)

    # Load the train data
    train_data_file = open('./mnist_dataset/mnist_train.csv', 'r')
    train_data_list = train_data_file.readlines()
    train_data_file.close()

    # Train the neural network
    epochs = 10
    for epoch in range(epochs):
        # print('epoch:', epoch)
        count = 0
        for record in train_data_list:
            all_values = record.split(',')
            label = int(all_values[0])
            inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
            targets = np.zeros(onodes) + 0.01
            targets[int(all_values[0])] = 0.99
            nn.train(inputs, targets)

            # rotate input
            input_minus_10_img = scipy.ndimage.rotate(inputs.reshape(28,28), -10, cval=0.01, reshape=False)
            input_plus_10_img = scipy.ndimage.rotate(inputs.reshape(28,28), 10, cval=0.01, reshape=False)

            # train rotated data
            nn.train(input_minus_10_img.reshape((28*28)), targets)
            nn.train(input_plus_10_img.reshape((28*28)), targets)
            print('epoch:', epoch, 'count:', count, 'label:', label)
            count += 1

    # Load the test data
    test_data_file = open('./mnist_dataset/mnist_test.csv', 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # Test the neural network
    scorecard = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
        outputs = nn.query(inputs)
        output_label = np.argmax(outputs)
        print('correct label:{}, output lable:{}'.format(correct_label, output_label))
        # image_array = np.asfarray(all_values[1:]).reshape((28, 28))
        # plt.imshow(image_array, cmap='gray', interpolation='None')
        # plt.show()
        if output_label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
    # print('scorecard:', scorecard)
    print('performance = {}%'.format(np.sum(scorecard) / len(scorecard) * 100))

    # # Back query the neural network
    # for label in range(10):
    #     print('label:', label)
    #     outputs_list = np.zeros(onodes) + 0.01
    #     outputs_list[label] = 0.99
    #     inputs = nn.back_query(outputs_list)
    #     print('inputs:', inputs)
    #     plt.imshow(inputs.reshape((28, 28)), cmap='gray', interpolation='None')
    #     plt.show()