"""problem.py -- the problem record: what the modelling tool is asked to do, in one structured place.

Everything the agents need to know about the case, and nothing about its answer:
  base_model         the model in the knowledge graph whose balance equations are kept (a Sup id)
  unknown            the model symbol no trusted law supplies yet, and that the search fills (here r)
  unknown_text       how to name it in the prompts
  unknown_dimension  its dimension, used to query the graph for candidate laws ('1 / [time]': rates)
  measured, inputs   which model symbols are measured, which are set by the operator
  roles              how roles the graph gives law variables map onto model symbols (substrate -> x2, ...);
                     used where code maps laws without an agent (survey.py)
  hints              the few domain words in the prompts (mechanism examples, which variables exist, what to look
                     for in the data and the misfits). GENERIC is used for any hint a problem does not give.

Who writes it: in the finished tool, the interpreter, from the user's description and data (and a person can
check it). In the experiments of 3-4 Nov it is written by hand (YEAST below) and stored in each data file
(data.py), so the batch tests the system GIVEN a correct problem record; the interpreter is then scored on
how close its record comes to this one. Data files without a record (made before 4 Nov) get YEAST.
"""
from __future__ import annotations
import copy

GENERIC = {
    'mechanisms': 'the mechanisms known for this kind of term in the\n  literature',
    'variables': '; a law that needs a variable the model does not have\n  cannot be used here.',
    'data': 'how the measured quantities behave across the experiments and\n  their operating conditions',
    'misfit': 'which operating range',
    'modeler_example': 'a concentration -> the model\'s symbol for that concentration',
    'challenger_example': 'another form of the\n  same mechanism when the accepted law has one',
}

YEAST = {
    'name': "baker's yeast, fed-batch (Asprey and Macchietto 2000)",
    'base_model': 'asprey_2000_monod_kinetics',
    'unknown': 'r',
    'unknown_text': 'the specific growth rate r',
    'unknown_dimension': '1 / [time]',
    'measured': ['x1', 'x2'],
    'inputs': ['u1', 'u2'],
    'roles': {'substrate': 'x2', 'biomass': 'x1'},
    'hints': {
        'mechanisms': 'saturation, substrate inhibition,\n  biomass dependence (Contois type), and so on',
        'variables': ' (substrate, biomass); a law that needs, for example, glutamine,\n  ammonia or oxygen cannot '
                     'be used here.',
        'data': 'how biomass and substrate behave at high and low substrate, across\n  experiments with different '
                'starting biomass',
        'misfit': 'high or low substrate',
        'modeler_example': "S (substrate) -> the model's substrate symbol",
        'challenger_example': 'another form of substrate\n  inhibition when the accepted law has one',
    },
    'written_by': 'hand (stands in for the interpreter in the 3-4 Nov experiments)',
}


def of(data: dict) -> dict:
    """The data file's problem record (or YEAST for older files), with GENERIC filling missing hints."""
    p = copy.deepcopy(data.get('problem') or YEAST)
    p['hints'] = {**GENERIC, **p.get('hints', {})}
    return p


def render(template: str, p: dict) -> str:
    """Fill a prompt template: <<UNKNOWN>>, <<UNKNOWN_TEXT>> and <<HINT:name>>."""
    out = template.replace('<<UNKNOWN_TEXT>>', p['unknown_text']).replace('<<UNKNOWN>>', p['unknown'])
    for k, v in p['hints'].items():
        out = out.replace(f'<<HINT:{k}>>', v)
    return out
