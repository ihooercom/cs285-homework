import glob
import tensorflow as tf


def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y


if __name__ == '__main__':
    import glob

    eventfile = 'homework_fall2020/hw4/data_keep/hw4_q2_obstacles_singleiteration_obstacles-cs285-v0_26-02-2021_12-32-08/events.out.tfevents.1614313928.admin.cluster.local'

    X, Y = get_section_results(eventfile)
    for i, (x, y) in enumerate(zip(X, Y)):
        print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))