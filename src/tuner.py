import random

from src import siamese_model

from settings import PROJECT_HOME


if __name__ == '__main__':
    # Use the gpu
    #os.environ["THEANO_FLAGS"] = "FAST_RUN,device=gpu,floatX=float32"

    param_dist = {
                    "embedding_dim": [4, 16, 64]
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
