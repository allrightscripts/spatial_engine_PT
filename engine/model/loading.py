"""Load a trusted installed model module; JSON contains no executable formulas."""
import importlib
import re

# Custom models live in engine/model/<name>.py and expose build_model().
# Keep the name stable once it appears in archived provider descriptors.
# Import a simple trusted module name under engine.model and invoke
# build_model(model_parameters). Unknown names produce an explicit model-path error.
def load_model(config):
    name = config.get('model', 'cultural_v01')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name):
        raise ValueError('model must be a simple module name inside engine/model')
    try:
        module = importlib.import_module('engine.model.' + name)
    except ModuleNotFoundError as exc:
        if exc.name == 'engine.model.' + name:
            raise ValueError(f'Unknown model {name!r}; expected engine/model/{name}.py') from exc
        raise
    return module.build_model(config.get('model_parameters', {}))
