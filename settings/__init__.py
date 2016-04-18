from __future__ import absolute_import, division, print_function

import os

stage = os.environ["STAGE"]

print("Running in stage: {}".format(stage))

if stage == 'dev':
    from .embedding_dev import *
elif stage == 'embedding_local':
    from .embedding_local import *
elif stage == 'local':
    from .local import *
elif stage == 'prod':
    from .embedding_prod import *
else:
    print("Unknown stage: {}".format(stage))
    print("use: STAGE='{dev|local|prod}' python src/siamese_graph_convo.py to run")

from .common import *
