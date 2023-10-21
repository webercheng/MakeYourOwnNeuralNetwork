# neural network class definition
import numpy
import scipy.special
import matplotlib.pyplot

class neuralNetwork:
    # initialise the neural network
    def __init__(self, inodes, onodes, hnodes, lr):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes
        # learning rate
        self.lr = lr
        # weight matrices which link the input and hidden, the hidden and output
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # print("wih size:", self.wih.size, "shape:", self.wih.shape, "ndim:", self.wih.ndim)
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # print("who size:", self.who.size, "shape:", self.who.shape, "ndim:", self.who.ndim)
        # activation function, here is sigmod, weird!
        self.act_func = lambda x: scipy.special.expit(x) 
        # print count (for debug)
        self.print_count = 0
        
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        # print("inputs:\n", inputs)
        # convert target list to 2d array
        targets = numpy.array(targets_list, ndmin=2).T
        # print("targets:\n", targets)
        # calculate signal into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # print("hidden_inputs:\n", hidden_inputs)
        # calculate signal emerging from hidden layer
        hidden_outputs = self.act_func(hidden_inputs)
        # print("hidden_outputs:\n", hidden_outputs)
        # calculate signal into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # print("final_inputs:\n", final_inputs)
        # calculate signal emerging from final output layer
        final_outputs = self.act_func(final_inputs)
        # print("final_outputs:\n", final_outputs)
        # calculate the output layer error
        output_errors = targets - final_outputs
        # print("output_errors:\n", output_errors)
        # calculate the hidden layer error
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * (final_outputs * (1.0 - final_outputs))), numpy.transpose(hidden_outputs))
        # update the weights for the links between the input and  hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * (hidden_outputs * (1.0 - hidden_outputs))), numpy.transpose(inputs))

    # query the neural network
    def query(self, input_list):
        # convert input list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T
        # print("inputs:\n", inputs)
        # calculate signal into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # print("hidden_inputs:\n", hidden_inputs)
        # calculate signal emerging from hidden layer
        hidden_outputs = self.act_func(hidden_inputs)
        # print("hidden_outputs=\n", hidden_outputs)
        # calculate signal into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # print("final_inputs:\n", final_inputs)
        # calculate signal emerging from output layer
        final_outputs = self.act_func(final_inputs)
        return final_outputs


def main():
    # number of input, hidden and output nodes
    inodes = 28*28
    hnodes = 100
    onodes = 10
    # print(targets)
    # learning rate is 0.3
    lr = 0.3
    # create instance of neural network
    n = neuralNetwork(inodes, onodes, hnodes, lr)
    train_data_file = open("mnist_train.csv", 'r')
    train_data_list = train_data_file.readlines()
    train_data_file.close()
    # print(train_data_list[0])
    del train_data_list[0]
    # print(train_data_list[0])

    print("before train: wih: \n", n.wih)
    print("before train: who: \n", n.who)

    # print(len(train_data_list))
    for record in train_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        if '\n' in all_values[-1]:
            all_values[-1].replace('\n', '')
        # print(record)
        # print(all_values)
        # print(len(all_values))
        # print(all_values[-1])
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(onodes) + 0.01
        # all values[0] is the target label for this record
        # print(inputs)
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

    # print("after train: wih: \n", n.wih)
    # print("after train: who: \n", n.who)

    # Test neraul network
    test_data_file = open("mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    # print(test_data_list[0])
    del test_data_list[0]
    # print(test_data_list[0])
    # print(len(test_data_list))
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        if '\n' in all_values[-1]:
            all_values[-1].replace('\n', '')
        # print(record)
        # print(all_values)
        # print(len(all_values))
        # print(all_values[-1])
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(onodes) + 0.01
        # all values[0] is the target label for this record
        outputs = n.query(inputs)
        print("query: in:{}, out:{}".format(all_values[0], outputs))

    # all_values = data_list[2].split(',')
    # image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
    # matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    # scaled_input = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    # print(scaled_input)


if __name__ == "__main__":
    main()