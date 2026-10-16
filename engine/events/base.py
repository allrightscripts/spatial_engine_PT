"""Serializable event API. Event handlers act on a small Simulation interface."""
from dataclasses import dataclass, field, asdict

@dataclass
class Event:
    time: float
    source: str = 'config'
    targets: list = field(default_factory=list)
    geometry: dict = field(default_factory=dict)
    strength: float = 1.0
    parameters: dict = field(default_factory=dict)
    rng: dict = field(default_factory=dict)
    event_id: str = ''
    kind = 'manual'

    # Convert the event dataclass to a serializable dictionary and add its structural
    # event_type tag.
    def record(self):
        data=asdict(self); data['event_type']=self.kind
        return data

class PBirthEvent(Event): kind='p_birth'
class LandscapeEvent(Event): kind='landscape'

class NucleationEvent(Event): kind='nucleation'
class SplitEvent(Event): kind='split'
class MutationEvent(Event): kind='mutation'
class ExtinctionEvent(Event): kind='extinction'
class CatastropheEvent(Event): kind='catastrophe'
class SabotageEvent(Event): kind='sabotage'
class ManualEvent(Event): kind='manual'

TYPES={c.kind:c for c in (PBirthEvent,LandscapeEvent,NucleationEvent,SplitEvent,MutationEvent,ExtinctionEvent,CatastropheEvent,SabotageEvent,ManualEvent)}

# Build the existing event class selected by type, moving a top-level semantic selector
# into parameters. Reject unknown event types.
def from_config(data):
    args=dict(data); kind=args.pop('type')
    if 'target_selector' in args:
        args['parameters']={**args.get('parameters',{}),'target_selector':args.pop('target_selector')}
    if kind not in TYPES: raise ValueError(f'Unknown event type {kind}')
    return TYPES[kind](**args)
