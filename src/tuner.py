import random

from src import siamese_model

from settings import PROJECT_HOME


if __name__ == '__main__':
    # Use the gpu
    #os.environ["THEANO_FLAGS"] = "FAST_RUN,device=gpu,floatX=float32"

    param_dist = {
                    "c1_filters":       [8, 16, 32],
                    "c1_W_regularizer": [0, 0.0001, 0.001, 0.01, 0.1],
                    "c1_b_regularizer": [0, 0.0001, 0.001, 0.01, 0.1],
                    "c1_dropout":       [0, 0.05, 0.1, 0.3, .5],
                    "c1_width":         [4, 8, 16],
                    "c2_filters":       [8, 16, 32],
                    "c2_W_regularizer": [0, 0.0001, 0.001, 0.01, 0.1],
                    "c2_b_regularizer": [0, 0.0001, 0.001, 0.01, 0.1],
                    "c2_dropout":       [0, 0.05, 0.1, 0.3, .5],
                    "c2_width":         [4, 8, 16],
                    "d1_size":          [32, 64, 128, 258],
                    "d1_W_regularizer": [0, 0.0001, 0.001, 0.01, 0.1],
                    "d1_b_regularizer": [0, 0.0001, 0.001, 0.01, 0.1],
                    "d1_dropout":       [0, 0.05, 0.1, 0.3],
                    "embedding_dim":    [4, 16, 64],
                    "embedding_W_regularizer": [0, 0.0001, 0.001, 0.01, 0.1],
                    "embedding_b_regularizer": [0, 0.0001, 0.001, 0.01, 0.1],
                    "margin":            [3, 5, 8, 13],
                    "epochs":            [20]
                  }

    with open("{}/tuner_output/model_info.csv".format(PROJECT_HOME), 'a') as f:
        line = []
        for k in sorted(param_dist):
            line.append(k)

        line.append("auc")
        f.write(", ".join(line) + "\n")

    print "Loading Data"

    tr_pairs, tr_y, te_pairs, te_y = siamese_model.get_data()

    for i in range(100):
        # Randomly choose a set of parameters
        params = {}
        for k, v in param_dist.iteritems():
            params[k] = random.choice(v)

        model = siamese_model.train(params, save=False)
        auc = siamese_model.compute_auc_score(model, te_pairs, te_y)

        # Write a line to a csv file including name, params (in sorted order), score
        with open("{}/tuner_output/model_info.csv".format(PROJECT_HOME), 'a') as f:
            line = []
            for k in sorted(params):
                line.append(str(params[k]))
            line.append(str(auc))
            f.write(", ".join(line) + "\n")
