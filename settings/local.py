PROJECT_HOME = "/Users/Brian/workplace/projects/siamese/"

# Settings for a local run.  More computational muscle than dev but less than if we
# were actually running on a gpu

# Settings for vanilla nn
NB_EPOCH = 20

# Settings for cnn
NB_CONV_FILTERS = 64

DROPOUT = True
DROPOUT_FRACTION = 0.6

# Dimension of the embedding space. Here it is artifically small so we can visulaize it
EMBEDDING_DIM = 64

FULLY_CONNECTED_SIZE = 128

# Learning Rate. Here we make it smaller because we seem to diverge with the normal learning
# rate and such a small embedding space
LEARNING_RATE = 0.001
OPTIMIZER = 'sgd'

# Margin for the contrastive loss function.  How far apart two observations from different
# classes need to be before the error is zero
MARGIN = 0.5