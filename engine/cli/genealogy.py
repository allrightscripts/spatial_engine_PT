"""Print separate source and inheritance forests; export CSV and optional DOT."""
import argparse
from pathlib import Path
from engine.analysis.genealogy import analyze_genealogy


# Source provenance answers WHERE birth occurred; inheritance answers FROM
# WHICH P parameters came. Keep these as separate trees even if edges coincide.
# Parse manifest/export destinations, print source and inheritance forests separately,
# then optionally write text, CSV and DOT outputs.
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--csv',type=Path,help='Genealogy CSV output')
    parser.add_argument('--graphviz',type=Path,help='DOT file; no Graphviz installation required')
    parser.add_argument('--output',type=Path,help='Save the human-readable report as UTF-8 text')
    args=parser.parse_args()
    manifest=args.manifest/'manifest.json' if args.manifest.is_dir() else args.manifest
    result=analyze_genealogy(manifest,args.csv,args.graphviz)
    print(result['text'])
    print('\nFIELD TABLE')
    for row in result['rows']:
        print(f"{row['field_id']} | {row['display_label']} | {row['family']} | path={row['lineage_path']} | source={row['source_T_id']} | inherits={row['inheritance_parent_id']} | birth={row['birth_time']} | death={row['death_time']} | segment={row['birth_segment']} | generation={row['generation']}")
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(result['text']+'\n',encoding='utf-8')

if __name__=='__main__':main()
