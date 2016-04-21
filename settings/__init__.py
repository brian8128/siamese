from __future__ import absolute_import, division, print_function

import os

stage = os.environ["STAGE"]
stage = stage.lower()


print("Running in stage: {}".format(stage))

from .common import *

if stage == 'dev':
    from .dev import *
elif stage == 'local':
    from .local import *
elif stage == 'prod':
    from .prod import *
elif stage.startswith('exp'):
    from .experimental import *
else:
    print("Unknown stage: {}".format(stage))
    print("use: STAGE='{dev|local|prod}' python src/siamese_model.py to run")


