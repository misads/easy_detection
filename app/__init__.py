import os

from flask import Flask

from app.utils import init_utils
from app.views import views
from app.api import api
from jinja2 import FileSystemLoader


class ThemeLoader(FileSystemLoader):
    """Custom FileSystemLoader that switches themes based on the configuration value"""
    def __init__(self, searchpath, encoding='utf-8', followlinks=False):
        super(ThemeLoader, self).__init__(searchpath, encoding, followlinks)
        self.overriden_templates = {}

    def get_source(self, environment, template):
        # Check if the template has been overriden
        if template in self.overriden_templates:
            return self.overriden_templates[template], template, True

        # Check if the template requested is for the admin panel
        if template.startswith('admin/'):
            template = template[6:]  # Strip out admin/
            template = "/".join(['admin', 'templates', template])
            return super(ThemeLoader, self).get_source(environment, template)

        # Load regular theme data
        template = "/".join(['user', 'templates', template])
        return super(ThemeLoader, self).get_source(environment, template)


def create_app(config='app.config.Config'):
    app = Flask(__name__)
    with app.app_context():
        app.config.from_object(config)

        theme_loader = ThemeLoader(os.path.join(app.root_path, 'html'), followlinks=True)
        app.jinja_loader = theme_loader

        init_utils(app)

        app.register_blueprint(views)
        app.register_blueprint(api)
        return app




