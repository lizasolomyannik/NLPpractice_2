import numpy as np
import pandas as pd
import nltk
from nltk.corpus import twitter_samples
from os import getcwd
from utils import process_tweet, build_freqs
import math

nltk.download('twitter_samples')
nltk.download('stopwords')

filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# split the data into two pieces, one for training and one for testing (validation set)
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# combine positive and negative labels
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

# Print the shape train and test sets
print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))

# create frequency dictionary
freqs = build_freqs(train_x, train_y)

# check the output
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))

# test the function below
print('This is an example of a positive tweet: \n', train_x[0])
print('This is an example of the processed version of the tweet: \n', process_tweet(train_x[0]))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

if (sigmoid(0) == 0.5):
    print('SUCCESS!')
else:
    print('Oops!')

if (sigmoid(4.92) == 0.9927537604041685):
    print('CORRECT!')
else:
    print('Oops again!')

# verify that when the model predicts close to 1, but the actual label is 0, the loss is a large positive value
-1 * (1 - 0) * np.log(1 - 0.9999) # loss is about 9.2

def gradientDescent(x, y, theta, alpha, num_iters):
    m = len(x)

    for i in range(0, num_iters):
        # get z, the dot product of x and theta
        z = np.mat(x) * np.mat(theta)

        # get the sigmoid of z
        h = sigmoid(z)

        # calculate the cost function
        J = np.mat(-1 / m) * (np.mat(y.transpose()) * np.mat(np.log(h)) + np.mat((1 - y).transpose()) * np.mat(np.log(1 - h)))

        # update the weights theta
        theta = theta - (alpha/m)*(np.mat(x.transpose())*np.mat((h-y)))

    J = float(J)
    return J, theta

# Check the function
# Construct a synthetic test case using numpy PRNG functions
np.random.seed(1)
# X input is 10 x 3 with ones for the bias terms
tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)
# Y Labels are 10 x 1
tmp_Y = (np.random.rand(10, 1) > 0.35).astype(float)
# Apply gradient descent
tmp_J, tmp_theta = gradientDescent(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 700)
print(f"The cost after training is {tmp_J:.8f}.")
# print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(tmp_theta)]}")

# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def extract_features(tweet, freqs):
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)

    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3))

    # bias term is set to 1
    x[0, 0] = 1

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # loop through each word in the list of words
    for word in word_l:
        # increment the word count for the positive label 1
        if (word, 1.0) in freqs:
            x[0, 1] += freqs.get((word, 1.0), 0)

        # increment the word count for the negative label 0
        if (word, 0.0) in freqs:
            x[0, 2] += freqs.get((word, 0.0), 0)

    ### END CODE HERE ###
    assert (x.shape == (1, 3))
    return x

# test on training data
# tmp1 = extract_features(train_x[0], freqs)
# print(tmp1)

# check for when the words are not in the freqs dictionary
# tmp2 = extract_features('blorb bleeeeb bloooob', freqs)
# print(tmp2)

# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")
# print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}") - not working

# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def predict_tweet(tweet, freqs, theta):
    '''
    Input:
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output:
        y_pred: the probability of a tweet being positive or negative
    '''
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(x * theta)

    ### END CODE HERE ###

    return y_pred

# Run this cell to test your function
for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    print( '%s -> %f' % (tweet, predict_tweet(tweet, freqs, theta)))

# Feel free to check the sentiment of your own tweet below
my_tweet = 'help :)'
print(predict_tweet(my_tweet, freqs, theta))

# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def test_logistic_regression(test_x, test_y, freqs, theta):
    """
    Input:
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output:
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # the list for storing predictions
    y_hat = []

    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)

        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            # append 0 to the list
            y_hat.append(0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    y_hat_array = np.asarray(y_hat)
    test_y_array = np.squeeze(test_y)
    if y_hat_array.all() == test_y_array.all():
        accuracy = sum(sum(y_hat_array), sum(test_y) / len(y_hat_array))

    ### END CODE HERE ###

    return accuracy

# tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta) - error occurs :(
# print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")

# Some error analysis done for you
print('Label Predicted Tweet')
for x,y in zip(test_x,test_y):
    y_hat = predict_tweet(x, freqs, theta)

    if np.abs(y - (y_hat > 0.5)) > 0:
        print('THE TWEET IS:', x)
        print('THE PROCESSED TWEET IS:', process_tweet(x))
        print('%d\t%0.8f\t%s' % (y, y_hat, ' '.join(process_tweet(x)).encode('ascii', 'ignore')))

# Feel free to change the tweet below
my_tweet = "When I first saw them on set in full costume, I was like: 'Hooooooly shit! It's Joel & Ellie!' The @HBO adaptation of @Naughty_Dog's The Last of Us is full steam ahead! Can't wait to show you more (from all of our projects!) Happy #TLoUDay!!!"
print(process_tweet(my_tweet))
y_hat = predict_tweet(my_tweet, freqs, theta)
print(y_hat)
if y_hat > 0.5:
    print('Positive sentiment')
else:
    print('Negative sentiment')