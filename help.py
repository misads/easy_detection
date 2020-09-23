from network import models
import misc_utils as utils
import os


def get_dirs(dir):
    res = []
    if os.path.isdir(dir):
        for file in os.listdir(dir):
            if os.path.isdir(os.path.join(dir, file)):
                res.append(file)

    return '|'.join(res)


model_choices = '|'.join(models.keys())
log_choices = get_dirs('logs')
checkpoints_choices = get_dirs('checkpoints')


help = """Usage:
Training:
    python train.py --tag your_tag --model {%s} -b 8 --gpu 0""" % model_choices + """

Finding Best Hyper Params:
    python runx.py --run

Debug:
    python train.py --model {%s} --debug""" % model_choices + """
    
Load Pre-Trained:
    python train.py --tag your_tag --load checkpoints/{%s}""" % checkpoints_choices + """/500_checkpoint.pt

Eval:
    python eval.py --tag your_tag2 --load checkpoints/{%s}""" % checkpoints_choices + """/500_checkpoint.pt

See Running Log:
    cat logs/{%s}/log.txt""" % log_choices + """

Clear(delete all files with the tag, BE CAREFUL to use):
    python clear.py {%s} """ % log_choices + """

See ALL Running Commands:
    cat run_log.txt

"""

print(help)