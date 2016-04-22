PROJECT_HOME = "/Users/Brian/workplace/projects/siamese/"

# Settings for a test run.  We just want to see that we can run without error
# no expectation of reasonable prediction accuracy

INPUT_SHAPE = (6, 128, 1)

CONVO_DROPOUT_FRACTION = 0.2

# Settings for cnn
NB_EPOCH = 1
L1_FILTERS = 2
L2_FILTERS = 4

DROPOUT = True
DROPOUT_FRACTION = 0.2
# Dimension of the embedding space. Here it is artifically small so we can visulaize it
EMBEDDING_DIM = 8

# Learning Rate. Here we make it smaller because we seem to diverge with the normal learning
# rate and such a small embedding space
LEARNING_RATE = 0.001
OPTIMIZER = 'rms'

# Margin for the contrastive loss function.  How far apart two observations from different
# classes need to be before the error is zero
MARGIN = 1