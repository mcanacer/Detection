import sys
import importlib

import os

import torch

def get_everything(config_path, args):
    module_path = config_path.replace('/', '.').replace('.py', '')
    module = importlib.import_module(module_path, package=None)
    return module.everything(args)


def main(config_path, args):
    evy = get_everything(config_path, path)

    eval_loader = evy['eval_loader']

    evaluator = evy['evaluator']

    detector = evy['detector']

    checkpoint_path = evy['epoch_checkpoint_path']

    detector.load_state_dict(torch.load(checkpoint_path, weights_only=True)['model_state_dict'])

    detector.eval()
    for inputs in eval_loader:

        with torch.no_grad():
            box_preds, class_scores, classes = detector(inputs, mode='predict')

        classes = classes + 1

        preds = box_preds, class_scores, classes
        evaluator.add(inputs, preds)

    results = evaluator.evaluate()

    print(results)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1], sys.argv[2:])
