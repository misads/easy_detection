import argparse
import os
import random
import torch
import os
import misc_utils as utils


def parse_args():
    # experiment specifics
    parser = argparse.ArgumentParser()

    parser.add_argument('tag', type=str,  default='cache', nargs='?',
                        help='folder name to clear')

    parser.add_argument('--rm', action='store_true', help='debug mode')

    return parser.parse_args()


opt = parse_args()

paths = ['checkpoints', 'logs', 'results']

if opt.rm:
    for path in paths:
        p = os.path.join(path, opt.tag)
        if os.path.isdir(p):
            command = 'rm -r ' + p
            print(command)
            os.system(command)
else:
    for path in paths:
        tmp = os.path.join('_.trash', utils.get_time_stamp(), path)
        utils.try_make_dir(tmp)
        p = os.path.join(path, opt.tag)
        if os.path.isdir(p):
            command = 'mv %s %s' % (p, tmp)
            print(command)
            os.system(command)

utils.color_print("Directory '%s' cleared." % opt.tag, 1)