import datetime
import os
import time

from flask import current_app as app, Blueprint, jsonify, render_template, abort, send_file, session, request
from flask.helpers import safe_join
from app import utils


api = Blueprint('api', __name__)


# @views.route('/')
# @views.route('/index')
# def index():
#     # session.permanent = True
#     return render_template('index.html')
#     # response = {'strangers': []}
#     # s_q = Strangers.query.all()
#     # for row in s_q:
#     #     response['strangers'].append(row.time)
#     # return jsonify(response)


# @views.route('/query')
# def query():
#     d_q = Data.query.all()
#     return render_template('query.html', url='/query', data=d_q)


# @views.route('/new')
# def new_detect():
#     requests = request.args
#     if 'name' not in requests or 'value' not in requests:
#         return jsonify([])

#     name = requests['name']
#     value = requests['value']
#     timestr = utils.get_time_str(utils.get_time_stamp())
#     name_q = Data.query.filter_by(name=name).first()
#     status = 'Add'
#     if name_q:
#         status = 'Update'
#         name_q.value = value
#         name_q.time = timestr
#         db.session.commit()
#     else:
#         new_data = Data(name, value, timestr)
#         db.session.add(new_data)
#         db.session.commit()
#     return jsonify([status, name, value, timestr])


# @views.route('/delete_all', methods=['GET'])
# def delete_all():
#     Data.query.delete()
#     db.session.commit()
#     return 'succeed'


# @views.route('/html/user/static/<path:path>')
# def themes_handler(path):
#     filename = safe_join(app.root_path, 'html', 'user', 'static', path)
#     if os.path.isfile(filename):
#         return send_file(filename)
#     else:
#         abort(404)
