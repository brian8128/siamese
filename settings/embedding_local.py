PROJECT_HOME = "/Users/Brian/workplace/projects/siamese/"

# Settings for a test run.  We just want to see that we can run without error
# no expectation of reasonable prediction accuracy

# Settings for cnn
NB_EPOCH = 2
NB_CONV_FILTERS = 20

# Dimension of the embedding space. Here it is artifically small so we can visulaize it
EMBEDDING_DIM = 3

FULLY_CONNECTED_SIZE = 512

# Learning Rate. Here we make it smaller because we seem to diverge with the normal learning
# rate and such a small embedding space
LEARNING_RATE = 0.000001
OPTIMIZER = 'sgd'

# Margin for the contrastive loss function.  How far apart two observations from different
# classes need to be before the error is zero
MARGIN = 0.1


