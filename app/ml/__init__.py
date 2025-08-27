from flask import Blueprint

bp = Blueprint('ml', __name__)

from . import routes
