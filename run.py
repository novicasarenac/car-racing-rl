import argparse
from a2c.train import A2CTrainer
from params import Params
from a2c.inference import a2c_inference


def get_trainer(model_type, params):
    model_path = 'models/' + model_type + '.pt'
    if model_type == 'a2c':
        return A2CTrainer(params, model_path)
    return None


def run_training(model_type):
    params = Params('params/' + model_type + '.json')
    trainer = get_trainer(model_type, params)
    trainer.run()


def run_inference(model_type):
    params = Params('params/' + model_type + '.json')
    if model_type == 'a2c':
        score = a2c_inference(params, 'models/a2c.pt')

    print('Total score: {}'.format(score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', required=True,
                        help='Train model, or run inference.')
    parser.add_argument('-m', '--model', required=True,
                        choices=['a2c', 'a3c', 'dqn'],
                        help='Which model to run / train.')

    args = vars(parser.parse_args())

    if eval(args['train']):
        run_training(args['model'])
    else:
        run_inference(args['model'])
