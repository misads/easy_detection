import os
import hashlib
from flask import current_app as app, request, redirect, url_for, session, render_template, abort, jsonify
from .misc_utils import *


def sha512(string):
    return hashlib.sha512(string).hexdigest()


def init_utils(app):
    @app.context_processor
    def inject_user():
        if session:
            return dict(session)
        return dict()

    @app.before_request
    def csrf():
        try:
            func = app.view_functions[request.endpoint]
        except KeyError:
            abort(404)
        if hasattr(func, '_bypass_csrf'):
            return
        if not session.get('nonce'):
            session['nonce'] = sha512(os.urandom(10))
        if request.method == "POST":
            if session['nonce'] != request.form.get('nonce'):
                return jsonify([str(session['nonce'])])

    @app.before_request
    def disable_jinja_cache():
        app.jinja_env.cache = {}
