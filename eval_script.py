import os

commands = {
    'cascaded': 'python3 eval.py --tag Cascaded --model cascaded --load checkpoints/Cascaded/669_model.pt',
    'pure3': 'python3 eval.py --tag pure3 --model default --load checkpoints/pure3/499_model.pt',

}


def eval(which):
    os.system(commands[which])


if __name__ == '__main__':
    eval('pure3')
    # eval('cascaded')
