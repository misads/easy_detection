import datetime
import os
import time

from flask import current_app as app, Blueprint, jsonify, render_template, abort, send_file, session, request
from flask.helpers import safe_join
from app import utils
import misc_utils
import json
import random
import subprocess

views = Blueprint('views', __name__)


tensorboard = None

def parse_meta_json(metapath):
    with open(metapath, 'r') as f:
        meta = json.load(f)

    state = None
    for line in meta:
        if 'finishtime' not in line:
            if line['gpu'] == '-1':
                state = 'cpu'
            else:
                state = 'cuda:' + line['gpu']
            break
    else:
        state = 'finished'

    model = meta[0]['opt']['model']
    acc = 0.
    runtime = 0.
    for line in meta:
        acc = max(acc, line['best_acc'])
        if 'finishtime' in line:
            runtime += int(line['finishtime']) - int(line['starttime'])
        else:
            runtime += int(misc_utils.get_time_stamp()) - int(line['starttime'])

    runtime = misc_utils.format_time(runtime)

    if 'remarks' in meta[0]:
        remarks = meta[0]['remarks']
    else:
        remarks = ''

    return {
        'state': state,
        'model': model,
        'acc': acc,
        'runtime': runtime,
        'remarks': remarks
    }


def parse_runs(tag):
    template_root = os.path.abspath(os.path.join(app.root_path, '..'))
    logroot = os.path.join(template_root, 'logs')
    logpath = os.path.join(logroot, tag, 'meta.json')
    if not os.path.isfile(logpath):
        return []

    with open(logpath, 'r') as f:
        meta = json.load(f)

    runs = []
    for line in meta:
        run = {
            'command': line['command'],
            'starttime': misc_utils.get_time_str(line['starttime'], fmt='%Y-%m-%d %H:%M:%S')
        }
        if 'finishtime' in line:
            runtime = int(line['finishtime']) - int(line['starttime'])
        else:
            runtime = int(misc_utils.get_time_stamp()) - int(line['starttime'])
        run['runtime'] = misc_utils.format_time(runtime)
        runs.append(run)

    return runs



def parse_log(tag):
    template_root = os.path.abspath(os.path.join(app.root_path, '..'))
    logroot = os.path.join(template_root, 'logs')
    logpath = os.path.join(logroot, tag, 'log.txt')
    if not os.path.isfile(logpath):
        return ''

    with open(logpath, 'r') as f:
        lines = f.readlines()

    option = False
    time_color = '#777777'
    for i, line in enumerate(lines):
        if option and '===========================================' in line:
            option = False
            
        if line.startswith('[INFO]'):
            if option:
                pos_equ = line.find('=')
                lines[i] = f'<span style="color:{time_color};">' + lines[i][:26] + '</span>' + \
                '<span style="color:#457ab2;">' + lines[i][26:pos_equ] + '</span>' + '=' + \
                '<span style="color:#262420;">' + lines[i][pos_equ+1:] + '</span>'
            elif 'Eva(' in line:
                lines[i] = f'<span style="color:{time_color};">' + lines[i][:26] + '</span><span style="color:#981e23;">' + lines[i][26:] + '</span>'
            else:
                lines[i] = f'<span style="color:{time_color};">' + lines[i][:26] + '</span><span style="color:#262420;">' + lines[i][26:] + '</span>'
        else:
            lines[i] = '<span style="color:#262420;">' + lines[i] + '</span>'

        if 'train_trasforms' in lines[i]:
            pos_trans = lines[i].find('train_trasforms')
            lines[i] = lines[i][:pos_trans] + '<span style="color:#067106;">train_trasforms:</span>' + lines[i][pos_trans + len('train_trasforms:'):]

        if 'val_trasforms' in lines[i]:
            pos_trans = lines[i].find('val_trasforms')
            lines[i] = lines[i][:pos_trans] + '<span style="color:#067106;">val_trasforms:</span>' + lines[i][pos_trans + len('val_trasforms:'):]

        if 'scheduler:' in lines[i]:
            pos_trans = lines[i].find('scheduler')
            lines[i] = lines[i][:pos_trans] + '<span style="color:#067106;">scheduler:</span>' + lines[i][pos_trans + len('scheduler:'):]

        if '==================Options==================' in line:
            option = True

    log = '<br/>'.join(lines)

    return log


def get_meta(dir='logs'):
    template_root = os.path.abspath(os.path.join(app.root_path, '..'))
    logroot = os.path.join(template_root, dir)
    logs = os.listdir(logroot)
    logs.sort()
    metas = []
    for tag in logs:
        logpath = os.path.join(logroot, tag, 'meta.json')
        meta_line = {
            'tag': tag,
            'state': '-',
            'model': '-',
            'acc': '-',
            'runtime': '-',
            'remarks': ''
        }
        if os.path.isfile(logpath):
            meta_line.update(parse_meta_json(logpath))

        metas.append(meta_line)

    return metas



def opentensorboard(tag, dir='logs'):
    template_root = os.path.abspath(os.path.join(app.root_path, '..'))
    logroot = os.path.join(template_root, dir)
    logpath = os.path.join(logroot, tag)
    global tensorboard
    if tensorboard is not None:
        return False

    cmd = f'tensorboard --logdir {logpath} --host=0.0.0.0'
    tensorboard = subprocess.Popen(cmd, shell=True,bufsize = -1, stdin=subprocess.PIPE, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    # tensorboard.kill()
    return True

    
def closetensorboard():
    cmd = "ps aux|grep tensorboard|grep -v grep |awk '{print $2}'|xargs kill -9"
    os.system(cmd)
    return True


@views.route('/tensorboardon')
def tensorboardon():
    requests = request.args
    if 'val' in requests:
        dir = 'results'
    else:
        dir = 'logs'

    if 'tag' not in requests:
        if opentensorboard('.', dir):
            return 'ok'
        return 'fail'

    tag = requests['tag']
    if opentensorboard(tag, dir):
        return 'ok'
    return 'fail'


@views.route('/tensorboardoff')
def tensorboardoff():
    global tensorboard
    if closetensorboard():
        tensorboard = None
        return 'ok'
    return 'fail'


@views.route('/api')
def api():
    # session.permanent = True
    requests = request.args
    if 'tag' not in requests:
        return jsonify([])
    
    meta = parse_runs(requests['tag'])
    return jsonify(meta)


@views.route('/log')
def log():
    # session.permanent = True
    requests = request.args
    if 'tag' not in requests:
        return ''
    
    log = parse_log(requests['tag'])
    return log


def del_one_tag(tag):
    template_root = os.path.abspath(os.path.join(app.root_path, '..'))
    paths = ['checkpoints', 'logs', 'results']

    trashtmp = utils.get_time_stamp() + f'{random.randint(0,1000):04d}'
    for path in paths:
        tmp = os.path.join('_.trash', trashtmp, path)
        misc_utils.try_make_dir(tmp)
        p = os.path.join(template_root, path, tag)
        if os.path.isdir(p):
            command = 'mv %s %s' % (p, tmp)
            os.system(command)


@views.route('/del', methods=['POST'])
def deltags():
    tag = request.form['tag']
    # session.permanent = True
    if not tag:
        return 'error'

    tag = tag.strip()
    tags = tag.split(',')
    
    for tag in tags:
        del_one_tag(tag)

    return 'ok'


@views.route('/rename', methods=['POST'])
def rename():
    tag = request.form['tag']
    newtag = request.form['newtag']
    # session.permanent = True
    if not tag or not newtag:
        return 'error'

    tag = tag.strip()
    newtag = newtag.strip()

    template_root = os.path.abspath(os.path.join(app.root_path, '..'))
    logroot = os.path.join(template_root, 'logs')

    oldpath = os.path.join(logroot, tag)
    newpath = os.path.join(logroot, newtag)

    oldckptpath = os.path.join(template_root, 'checkpoints', tag)
    newckptpath = os.path.join(template_root, 'checkpoints', newtag)

    if os.path.isdir(newpath):
        return 'tag exists.'

    if not os.path.isdir(oldpath):
        return 'old tag not found.'

    cmd1 = f'mv {oldpath} {newpath}'
    cmd2 = f'mv {oldckptpath} {newckptpath}'
    os.system(cmd1)
    os.system(cmd2)

    return 'ok'


@views.route('/editremark', methods=['POST'])
def editremark():
    tag = request.form['tag']
    remark = request.form['remark']
    # session.permanent = True
    if not tag:
        return 'error'

    if not remark:
        remark = ''

    template_root = os.path.abspath(os.path.join(app.root_path, '..'))
    logroot = os.path.join(template_root, 'logs')
    logpath = os.path.join(logroot, tag, 'meta.json')
    if not os.path.isfile(logpath):
        return 'error'

    with open(logpath, 'r') as f:
        meta = json.load(f)

    meta[0]['remarks'] = remark

    with open(logpath, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)

    return 'ok'


@views.route('/')
@views.route('/index')
def index():
    url = request.url_root # http://172.20.118.80:8000/
    port = 6006
    pos = url.rfind(':')
    if pos > 6:
        tensorboard_url = url[:pos] + f':{port}/'
    else:
        tensorboard_url = url.rstrip('/') + f':{port}/'

    global tensorboard
    metas = get_meta()  # 

    tensorboard_not_open = tensorboard is None

    template_root = os.path.abspath(os.path.join(app.root_path, '..'))
    return render_template('index.html', metas=metas, tensorboard_not_open=tensorboard_not_open, tensorboard_url=tensorboard_url, val=False, path=template_root)


@views.route('/val')
def validation():
    url = request.url_root # http://172.20.118.80:8000/
    port = 6006
    pos = url.rfind(':')
    if pos > 6:
        tensorboard_url = url[:pos] + f':{port}/'
    else:
        tensorboard_url = url.rstrip('/') + f':{port}/'

    global tensorboard
    metas = get_meta(dir='results')  # 

    tensorboard_not_open = tensorboard is None

    template_root = os.path.abspath(os.path.join(app.root_path, '..'))
    return render_template('index.html', metas=metas, tensorboard_not_open=tensorboard_not_open, tensorboard_url=tensorboard_url, val=True, path=template_root)


@views.route('/html/user/static/<path:path>')
def themes_handler(path):
    filename = safe_join(app.root_path, 'html', 'user', 'static', path)
    if os.path.isfile(filename):
        return send_file(filename)
    else:
        abort(404)
