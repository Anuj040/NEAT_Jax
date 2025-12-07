from dataclasses import dataclass

@dataclass
class NeatHyperParams:
    MAX_NODES = 64       # total (inputs + bias + hidden + outputs)
    MAX_CONNS = 256
