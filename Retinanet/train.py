import sys
import importlib

import os


def get_everything(config_path, args):
    module_path = config_path.replace('/', '.').replace('.py', '')
    module = importlib.import_module(module_path, package=None)
    return module.everything(args)


def main(config_path, args):
    evy = get_everything(config_path, path)

    run = evy['run']

    train_loader = evy['train_loader']

    detector = evy['detector']

    optimizer = evy['optimizer']

    scheduler = evy['scheduler']

    step_checkpoint_path = evy['step_checkpoint_path']
    epoch_checkpoint_path = evy['step_checkpoint_path']

    epoch = 1
    while True:
        detector.train()
        curr_lr = float(optimizer.param_groups[0]['lr'])
        for step, inputs in enumerate(train_loader):
            reg_loss, class_loss, box_loss = detector(inputs)
            loss = reg_loss.mean() + class_loss.mean() + box_loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            run.log(
                {
                    "reg_loss": float(losses['l1'].mean()),
                    "class_loss": float(losses['focal'].mean()),
                    "box_loss": float(losses['Iou'].mean()),
                    "total_loss": float(loss),
                    "epoch": epoch,
                    "learning-rate": curr_lr
                }
            )

            if steps % 100 == 0:
                torch.save(
                    {
                        'model_state_dict': detector.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': optimizer.state_dict(),
                    },
                    step_checkpoint_path,
                )

        epoch += 1

        torch.save(
            {
                'model_state_dict': detector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': optimizer.state_dict(),
            },
            epoch_checkpoint_path,
        )


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    get_everything(sys.argv[1], sys.argv[2:])

