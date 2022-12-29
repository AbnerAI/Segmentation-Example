import torch
import os


def save_checkpoint(net, path, global_step, accuracy=None, info=''):
    try:
        os.makedirs(path)
        print('Created checkpoint directory')
    except OSError:
        pass

    if accuracy:
        checkpoint_name = f'CP_%d_%.4f%s.pth' % (global_step, accuracy, info)
    else :
        checkpoint_name = f'CP_{global_step}{info}.pth'
    torch.save(net.state_dict(),
               os.path.join(path, checkpoint_name))

    print(f'Checkpoint {checkpoint_name} saved !')