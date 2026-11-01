"""daltonx.grounding.scorer
============================
Scoring functions and Match class for evaluating query vs reference strings.
Ported 1-to-1 from gilda-master/gilda/scorer.py.
"""
from .process import replace_dashes, replace_whitespace, get_capitalization_pattern

__all__ = [
    "Match",
    "generate_match",
    "score_string_match",
    "score_status",
    "score",
]


class Match(object):
    """Class representing a match between a query and a reference string."""

    def __init__(self, query, ref, exact=None, space_mismatch=None,
                 dash_mismatches=None, cap_combos=None):
        self.query = query
        self.ref = ref
        self.exact = exact if exact is not None else False
        self.space_mismatch = space_mismatch if space_mismatch is not None \
            else False
        self.dash_mismatches = dash_mismatches if dash_mismatches is not None \
            else set()
        self.cap_combos = cap_combos if cap_combos is not None else []

    def __str__(self):
        return 'Match(%s)' % (','.join(['%s=%s' % (k, v) for k, v in
                                        self.__dict__.items()]))

    def __repr__(self):
        return str(self)

    def to_json(self):
        return {
            'query': self.query,
            'ref': self.ref,
            'exact': self.exact,
            'space_mismatch': self.space_mismatch,
            'dash_mismatches': list(self.dash_mismatches),
            'cap_combos': self.cap_combos
        }

    def _query_cases(self):
        return {c[0] for c in self.cap_combos}

    def _ref_cases(self):
        return {c[1] for c in self.cap_combos}

    def score_short_abbr(self):
        if len(self.ref) <= 3 and \
                (('all_caps', 'all_lower') in self.cap_combos or
                 ('all_lower', 'all_caps') in self.cap_combos):
            return 0
        else:
            return 1

    def score_mixed(self):
        if ('mixed', 'mixed') in self.cap_combos:
            return 0
        elif ('mixed' in self._query_cases()) or \
                ('mixed' in self._ref_cases()):
            return 1
        else:
            return 2

    def score_exact(self):
        return 1 if self.exact is True else 0

    def score_acic(self):
        if self.exact is True and not self.cap_combos:
            return 2
        elif set(self.cap_combos) == {('sentence_initial', 'all_lower')}:
            return 2
        if self.exact is True and set(self.cap_combos) <= \
                {('all_caps', 'sentence_initial_cap'),
                 ('sentence_initial_cap', 'all_caps')}:
            return 1
        else:
            return 0

    def score_combo(self):
        qc = self._query_cases()
        rc = self._ref_cases()
        query_combo = 4 - len(qc)
        ref_combo = 4 - len(rc)
        if 'single_cap_letter' in qc and \
                ('all_caps' in qc or 'initial_cap' in qc):
            query_combo += 1
        if 'single_cap_letter' in rc and \
                ('all_caps' in rc or 'initial_cap' in rc):
            ref_combo += 1
        if 'sentence_initial_cap' in qc and \
                (len(qc) == 1 and
                 ('sentence_initial_cap', 'all_lower') in self.cap_combos) \
                or \
                 {'single_cap_letter', 'initial_cap', 'all_lower'} & qc:
            query_combo += 1
        combo = max(query_combo, ref_combo)
        return combo

    def score_dash(self):
        return 2 - len(self.dash_mismatches)


def generate_match(query, ref, beginning_of_sentence=False):
    """Return a Match object characterizing the relationship between query and ref."""
    query = replace_dashes(replace_whitespace(query))
    ref = replace_whitespace(ref).rstrip(' ')
    ref = replace_dashes(ref)

    if not beginning_of_sentence and query == ref:
        return Match(query, ref, exact=True)
    query_suffix = query
    ref_suffix = ref
    query_pieces = ['']
    ref_pieces = ['']
    dash_mismatches = set()
    while query_suffix and ref_suffix:
        qs = (query_suffix[0] == ' ')
        rs = (ref_suffix[0] == ' ')
        if qs and rs:
            query_suffix = query_suffix[1:]
            ref_suffix = ref_suffix[1:]
            query_pieces.append('')
            ref_pieces.append('')
        elif qs and not rs or rs and not qs:
            return Match(query, ref, space_mismatch=True)

        qd = (query_suffix[0] == '-')
        rd = (ref_suffix[0] == '-')
        if qd and rd:
            query_suffix = query_suffix[1:]
            ref_suffix = ref_suffix[1:]
            query_pieces.append('')
            ref_pieces.append('')
        elif qd and not rd:
            dash_mismatches.add('query')
            query_suffix = query_suffix[1:]
            query_pieces.append('')
            ref_pieces.append('')
        elif not qd and rd:
            dash_mismatches.add('ref')
            ref_suffix = ref_suffix[1:]
            query_pieces.append('')
            ref_pieces.append('')
        else:
            query_pieces[-1] += query_suffix[0]
            ref_pieces[-1] += ref_suffix[0]
            ref_suffix = ref_suffix[1:]
            query_suffix = query_suffix[1:]

    combinations = []
    first = True
    exact = False
    for qp, rp in zip(query_pieces, ref_pieces):
        first_bos = first and beginning_of_sentence
        first = False
        if qp == rp and not first_bos:
            exact = True
        else:
            qcp = get_capitalization_pattern(qp, first_bos)
            rcp = get_capitalization_pattern(rp, False)
            if qcp == rcp and qp == rp:
                exact = True
            else:
                combinations.append((qcp, rcp))
    return Match(query, ref, dash_mismatches=dash_mismatches,
                 exact=exact, cap_combos=combinations)


def score_string_match(match):
    """Return a score between 0 and 1 for the goodness of a string match."""
    terms = [
        (match.score_short_abbr, 2),
        (match.score_mixed, 3),
        (match.score_exact, 2),
        (match.score_acic, 3),
        (match.score_combo, 5),
        (match.score_dash, 3)
    ]
    sc = 0
    norm = 1
    for fun, coeff in terms:
        sc = coeff * sc + fun()
        norm *= coeff
    sc /= (norm - 1)
    return sc


def score_status(term):
    scores = {
        'curated': 4,
        'name': 3,
        'synonym': 2,
        'former_name': 1,
    }
    return scores.get(term.status, 2)


def score(match, term):
    """Compute overall score combining string match and status score."""
    string_match_score = score_string_match(match)
    status_score = score_status(term)
    sc = ((0 * 5 + status_score) * 2 + string_match_score) / 9
    return sc
