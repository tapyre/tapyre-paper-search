from flask import Blueprint
from abc import ABC, abstractmethod

class BaseBlueprint(ABC):
    def __init__(self, name, import_name, url_prefix=None):
        self.blueprint = Blueprint(name, import_name, url_prefix=url_prefix)