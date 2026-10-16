"""Run a JSON-configured model and save via the standard archive contract."""
import argparse
import json
from pathlib import Path
from engine.simulation.solver import Simulation
from engine.archive import write_archive,read_archive,decode

# Run the independent FV solver from JSON, write a standard NPZ archive and reload it to
# report its shape/genealogy.
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sim=Simulation(json.loads(args.config.read_text(encoding='utf-8')))
    data=sim.run(); write_archive(args.output,data); loaded=read_archive(args.output)
    print(f"Saved and reloaded {args.output}: {loaded['coefficients'].shape}; {len(sim.registry)} registered fields; {len(sim.events)} events")
    print('Genealogy:',decode(loaded,'genealogy_json'))
if __name__=='__main__': main()
