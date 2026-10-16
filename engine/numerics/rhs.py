"""Universal, explicit provider dispatch. JSON is data, never eval or source code.

A provider exposes order, parameters, rhs(t,a), source(), archive_extras().
Add another trusted provider type here for a fundamentally new numerical backend.
"""
import json

# Identify JSON provider-descriptor text by its leading non-whitespace character; legacy
# equation text follows the other path.
def is_provider_source(source):
    return source.lstrip().startswith('{')

# Parse a JSON descriptor and restore the trusted Galerkin provider, including model-
# source compatibility checks.
def load_provider(source):
    descriptor=json.loads(source)
    from engine.galerkin.provider import from_descriptor
    return from_descriptor(descriptor)

def evaluate_rhs(compiled,t,state,parameters):
    """Shared adapter for existing Newton/spectrum; evaluate vector RHS only once."""
    if hasattr(compiled,'rhs'):
        if parameters != compiled.parameters:
            raise ValueError('Provider parameter overrides need a new segment descriptor/config')
        return compiled.rhs(t,state)
    import numpy as np
    ns={'np':np,'t':t,'u':state,**parameters}
    return np.array([eval(expr,{'__builtins__':{}},ns) for expr in compiled],float)
