import argparse
import os
import random
import torch
import os
from misc_utils import get_time_stamp, color_print


def parse_args():
    # experiment specifics
    parser = argparse.ArgumentParser()

    parser.add_argument('tag', type=str,  default='default', nargs='?',
                        help='folder name to clear')

    parser.add_argument('--rm', action='store_true', help='debug mode')

    return parser.parse_args()


opt = parse_args()

paths = ['checkpoints', 'logs', 'results']

if opt.tag.startswith('logs/'):
    opt.tag = opt.tag[5:]
    
with open('run_log.txt', 'r') as f:
    run_logs = f.readlines()

with open('run_log.txt', 'w') as f:
    for line in run_logs:
        if f'--tag {opt.tag}' not in line:
            f.writelines(line)

    print("'tag=%s' in 'run_log.txt' cleared." % opt.tag)

if opt.rm:
    for path in paths:
        p = os.path.join(path, opt.tag)
        if os.path.isdir(p):
            command = 'rm -r ' + p
            print(command)
            os.system(command)
else:
    for path in paths:
        tmp = os.path.join('_.trash', get_time_stamp(), path)
        os.makedirs(tmp, exist_ok=True)
        p = os.path.join(path, opt.tag)
        if os.path.isdir(p):
            command = 'mv %s %s' % (p, tmp)
            print(command)
            os.system(command)

color_print("Directory '%s' cleared." % opt.tag, 1)