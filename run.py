import argparse
from actor_critic.a2c.train import A2CTrainer
from actor_critic.a3c.train import A3CTrainer
from params import Params
from actor_critic.a2c.inference import a2c_inference
from actor_critic.a2c.evaluate import evaluate_a2c


def get_trainer(model_type, params):
    model_path = 'models/' + model_type + '.pt'
    if model_type == 'a2c':
        return A2CTrainer(params, model_path)
    elif model_type == 'a3c':
        return A3CTrainer(params, model_path)
    return None


def run_training(model_type):
    params = Params('params/' + model_type + '.json')
    trainer = get_trainer(model_type, params)
    trainer.run()


def run_inference(model_type):
    params = Params('params/' + model_type + '.json')
    if model_type == 'a2c':
        score = a2c_inference(params, 'models/a2c.pt')

    print('Total score: {0:.2f}'.format(score))


def run_evaluation(model_type):
    print('evaluating')
    params = Params('params/' + model_type + '.json')
    if model_type == 'a2c':
        score = evaluate_a2c(params, 'models/a2c.pt')

    print('Average reward after 100 episodes: {0:.2f}'.format(score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True,
                        choices=['a2c', 'a3c', 'dqn'],
                        help='Which model to run / train.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true',
                       help='Train model.')
    group.add_argument('--inference', action='store_true',
                       help='Model inference.')
    group.add_argument('--evaluate', action='store_true',
                       help='Evaluate model on 100 episodes.')

    args = vars(parser.parse_args())
    if args['train']:
        run_training(args['model'])
    elif args['evaluate']:
        run_evaluation(args['model'])
    else:
        run_inference(args['model'])
