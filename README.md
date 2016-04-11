## The Problem:
Identify a user by their walking style based on cell phone gyro and accelerometer data.  

## The Data:
https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
We restricted the data to the walking activities.  Walking upstairs, walking downstairs, or just walking.

## Requirements
The algorithm must scale to millions of classes

## Challenges
Multi class classification scaling to millions of classes.  Not all classe are known at training time.
A test subject walking upstairs may look very different from a test subject walking downstairs.

## Approach
The most successful approach was to train a convolutional neural network to do a 'semantic' embedding on
walking style, following the approach outlined in http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf.

The idea is to train a function G_W: X -> V, where V is some vector space and for x_1 and x_2 in X G_W(x_1) is 
close to G_W(x_2) iff x_1 and x_2 belong to the same class.

## Results
I trained the network on 21 test subjects and tested it on the remaining 9.  
Using the convonet we achieved 94% accuracy telling whether two observations came from the same test
subject or a different test subject.  
