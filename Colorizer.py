import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import math
import h5py

class Colorizer():
    def __init__(self,new_img,height,width,window_size,hidden_nodes,output_nodes,weights_l1,weights_l2):
        self.new_img = new_img
        self.height = height
        self.width = width
        self.window_size = window_size
        self.input_nodes = self.window_size**2
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.a = 0.0001            # Learning rate
        self.gray_img = np.zeros((height,width))
        self.weights_l1 = weights_l1
        self.weights_l2 = weights_l2
        self.hidden_layer = np.zeros((height,width,self.hidden_nodes))
        self.output_layer = np.zeros((height,width,self.output_nodes))

    def ToGray(self,r,g,b):
        pixel = 0.21*r + 0.72*g + 0.07*b
        return pixel

    def ConvertRGBtoGray(self):
        for i in range(self.height):
            for j in range(self.width):
                pixel_val = self.new_img[i][j]
                self.gray_img[i][j] = self.ToGray(pixel_val[0],pixel_val[1],pixel_val[2])

        # Scale both the images to values between 0 and 1
        self.gray_img = np.divide(self.gray_img,255)
        self.new_img = np.divide(self.new_img,255)

    def sigmoid_derivative(self,x):
        return (sigmoid(x) * (1 - sigmoid(x)))

    def FeedForward(self):
        # Feed forward network
        sqr_error = 0
        for i in range(1,self.height-1):
            for j in range(1,self.width-1):
                input_matrix = GetNeighbourPixels(self.gray_img,i,j,self.height,self.width)
                input_matrix.insert(0,self.gray_img[i][j])      # Matrix 1 * 9

                # Input to hidden layer
                hidden_node = np.dot(input_matrix,self.weights_l1)      # Matrix (1*9).(9*5) = (1*5) hidden layer
                hidden_node = [sigmoid(x) for x in hidden_node]
                self.hidden_layer[i][j] = hidden_node

                # Hidden to output layer
                output_node = np.dot(hidden_node,self.weights_l2)       # Matrix (1*5).(5*3) = (1*3) output layer
                output_node = [sigmoid(x) for x in output_node]
                self.output_layer[i][j] = output_node

                for k in range(self.output_nodes):
                    sqr_error += (0.5* (self.new_img[i][j][k] - self.output_layer[i][j][k])**2)

        # Compute Mean square error
        mse = sqr_error / ((self.height-1) * (self.width-1) * self.output_nodes)
        return mse

    def BackPropogation(self):
        # Compute the gradient descent
        ETotal = 0
        for i in range(1,self.height-1):
            #if ETotal and not (i%10):
            #    print("Total error = %f"%ETotal)
            for j in range(1,self.width-1):
                output_node = self.output_layer[i][j]
                hidden_node = self.hidden_layer[i][j]
                input_matrix = GetNeighbourPixels(self.gray_img,i,j,self.height,self.width)
                input_matrix.insert(0,self.gray_img[i][j])      # Matrix 1 * 9
                assert(len(input_matrix) == self.input_nodes)
                
                ## Total error
                #ETotal = 0
                #for k in range(self.output_nodes):
                #    ETotal += (0.5 * ((self.new_img[i][j][k] - output_node[k])** 2))

                # Ouput to hidden layer error back propogation
                for h in range(self.hidden_nodes):
                    for k in range(self.output_nodes):
                        delta_output = output_node[k] * (1 - output_node[k]) * \
                            (output_node[k] - self.new_img[i][j][k]) * hidden_node[h]
                        # Update hidden layer weights
                        self.weights_l2[h][k] -= (self.a * delta_output)


                # Hidden layer to input layer error back propogation
                for ip in range(self.input_nodes):
                    sum_error = np.zeros(self.hidden_nodes)
                    for h in range(self.hidden_nodes):
                        for k in range(self.output_nodes):
                            sum_error[h] += (output_node[k] * (1 - output_node[k]) * \
                                (output_node[k] - self.new_img[i][j][k]) * self.weights_l2[h][k])

                        # sum error is the error in the hidden layer for each hidden node.
                        delta_output = sum_error[h] * hidden_node[h] * (1 - hidden_node[h]) * input_matrix[ip]
                        self.weights_l1[ip][h] -= (self.a * delta_output)

        return self.weights_l1, self.weights_l2

    def UpdateWeights(self,weights_l1,weights_l2):
        self.weights_l1 = weights_l1
        self.weights_l2 = weights_l2

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def GetNeighbourPixels(gray_img, row, col, height, width):
    child_nodes = []
    cell_value = []
    if col > 0:
        child_nodes.append([row,col-1])
    if col < width-1:
        child_nodes.append([row,col+1])
    if row > 0:
        child_nodes.append([row-1,col])
        if col > 0:
            child_nodes.append([row-1,col-1])
        if col < width-1:
            child_nodes.append([row-1,col+1])
    if row < height-1:
        child_nodes.append([row+1,col])
        if col > 0:
            child_nodes.append([row+1,col-1])
        if col < width-1:
            child_nodes.append([row+1,col+1])
        
    for child in child_nodes:
        cell_value.append(gray_img[child[0]][child[1]])

    return cell_value

def Train_Network():
    mean_sqr_error = float("inf")
    window_size = 3
    input_nodes = window_size**2
    hidden_nodes = 5
    output_nodes = 3
    weights_l1 = np.random.uniform(0,1,size=(input_nodes,hidden_nodes))
    weights_l2 = np.random.uniform(0,1,size=(hidden_nodes,output_nodes))
    number_of_epochs = 1000
    epoch = 0
    first_run = 0
    num_pictures = 32

    # Create class objects for all images
    Image_class = []
    for v in range(num_pictures):
        img = cv2.imread("data/img%d.jpg"%(v+1))
	#img = cv2.imread("data/train3.jpg")
        img = cv2.resize(img,(256,256))
        #display_img = False
        color = [0, 0, 0]
        new_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=color)
        height = new_img.shape[0]
        width = new_img.shape[1]
        C = Colorizer(new_img,height,width,window_size,hidden_nodes,output_nodes,weights_l1,weights_l2)
        C.ConvertRGBtoGray()
        Image_class.append(C)

    var = 0
    while epoch < number_of_epochs:
        #var = random.randint(1,9)
        Image_class[var].UpdateWeights(weights_l1,weights_l2)
        mean_sqr_error = Image_class[var].FeedForward()
        weights_l1, weights_l2 = Image_class[var].BackPropogation()
        epoch += 1
        var += 1
        if var == num_pictures:
            var = 0
        print("Epoch = %d, MSE = %f"%(epoch,mean_sqr_error))
        if first_run==0:
            array_epoch = []
            array_mean_sqr_error = []
            array_weights_l1 = []
            array_weights_l2 = []
            first_run = 1
        array_epoch.append(epoch)
        array_mean_sqr_error.append(mean_sqr_error)
        array_weights_l1.append(weights_l1)
        array_weights_l2.append(weights_l2)
        
        hf = h5py.File('output_rate0001_600.h5', 'w')
        hf.create_dataset('epoch', data=array_epoch)
        hf.create_dataset('mse', data=array_mean_sqr_error)
        hf.create_dataset('weights_l1', data=array_weights_l1)
        hf.create_dataset('weights_l2', data=array_weights_l2)
        hf.close()
        print("Saved to file")

        #print("Weight L1")
        #print(weights_l1)
        #print("Weight L2")
        #print(weights_l2)

    return weights_l1, weights_l2

def Test_Network(weights1, weights2):
    # Feed forward network
    #sqr_error = 0
    test_img = cv2.imread("data/test_image.jpg")
    # Convert to gray image
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    test_img = cv2.resize(test_img,(256,256))
    test_img = np.divide(test_img,255)
    #display_img = False
    color = [0, 0, 0]
    test_img = cv2.copyMakeBorder(test_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=color)
    height = test_img.shape[0]
    width = test_img.shape[1]
    hidden_layer = np.zeros((height,width,5))
    output_layer = np.zeros((height,width,3))
    for i in range(1,height-1):
        for j in range(1,width-1):
            input_matrix = GetNeighbourPixels(test_img,i,j,height,width)
            input_matrix.insert(0,test_img[i][j])      # Matrix 1 * 9

            # Input to hidden layer
            hidden_node = np.dot(input_matrix,weights1)      # Matrix (1*9).(9*5) = (1*5) hidden layer
            hidden_node = [sigmoid(x) for x in hidden_node]
            hidden_layer[i][j] = hidden_node

            # Hidden to output layer
            output_node = np.dot(hidden_node,weights2)       # Matrix (1*5).(5*3) = (1*3) output layer
            output_node = [sigmoid(x) for x in output_node]
            output_layer[i][j] = output_node
            #print(output_node)

    output_layer = np.multiply(output_layer,255)
    output_layer = output_layer.astype('uint8')
    #print(output_layer)
    #print(output_layer.shape)
    output_layer = cv2.resize(output_layer,(256,256))
    cv2.imwrite("RGB_image",output_layer)
    #cv2.imshow('RGB image',output_layer)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

l1_weights, l2_weights = Train_Network()
#Test_Network(l1_weights, l2_weights)
