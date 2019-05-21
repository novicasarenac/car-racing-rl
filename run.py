import argparse
from actor_critic.a2c.train import A2CTrainer
from actor_critic.a3c.train import A3CTrainer
from dqn.train import DQNTrainer
from params import Params
from actor_critic.inference import actor_critic_inference
from dqn.inference import dqn_inference
from actor_critic.evaluate import evaluate_actor_critic
from dqn.evaluate import evaluate_dqn


def get_trainer(model_type, params):
    model_path = 'models/' + model_type + '.pt'
    if model_type == 'a2c':
        return A2CTrainer(params, model_path)
    elif model_type == 'a3c':
        return A3CTrainer(params, model_path)
    elif model_type == 'dqn':
        return DQNTrainer(params, model_path)
    return None


def run_training(model_type):
    params = Params('params/' + model_type + '.json')
    trainer = get_trainer(model_type, params)
    trainer.run()


def run_inference(model_type):
    params = Params('params/' + model_type + '.json')
    if model_type == 'dqn':
        score = dqn_inference('models/' + model_type + '.pt')
    else:
        score = actor_critic_inference(params, 'models/' + model_type + '.pt')

    print('Total score: {0:.2f}'.format(score))


def run_evaluation(model_type):
    params = Params('params/' + model_type + '.json')
    if model_type == 'dqn':
        score = evaluate_dqn('models/' + model_type + '.pt')
    else:
        score = evaluate_actor_critic(params, 'models/' + model_type + '.pt')

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
