"""kg_query: the rate laws of the graph, and laws used in their corrected form."""
import kg_query

KG = kg_query.KG


def test_find_rate_laws_count():
    r = kg_query.find_rate_laws()
    assert len(r) == 42
    assert len({x['iri'] for x in r}) == 42


def test_find_rate_laws_includes_candidates():
    iris = {x['iri'] for x in kg_query.find_rate_laws()}
    for p in ['asprey_2000_monod_kinetics/law/asprey_2000_eq17', 'asprey_2000_contois_kinetics/law/asprey_2000_eq18',
              'zentou_2019_andrews/law/zentou_2019_eq5', 'muloiwa_2020_free_laws/law/muloiwa_2020_eq11']:
        assert KG + p in iris


def test_corrected_laws_drop_removed_symbols():
    corrected = [x for x in kg_query.find_rate_laws() if x['corrected']]
    assert corrected
    for x in corrected:
        L = kg_query.get_law(x['iri'])['laws'][0]
        kept = set(L['uses']) | set(L['parameters'])
        assert not kept & set(L['dropped_by_correction'])
        assert kg_query.symbols_in(L['latex'], list(kept)) == kept


def test_muloiwa_eq6_lowercase_s_dropped():
    L = kg_query.get_law(KG + 'muloiwa_2020_free_laws/law/muloiwa_2020_eq6')['laws'][0]
    assert L['corrected'] and L['dropped_by_correction'] == ['s']
    assert 's' not in L['uses'] and 'S' in L['uses']
    assert 's^{2}' in L['latex_printed'] and 'S^{2}' in L['latex']


def test_muloiwa_eq12_lowercase_s_dropped():
    L = kg_query.get_law(KG + 'muloiwa_2020_free_laws/law/muloiwa_2020_eq12')['laws'][0]
    assert L['dropped_by_correction'] == ['s'] and 's' not in L['uses']


def test_uncorrected_law_keeps_symbols():
    L = kg_query.get_law(KG + 'zentou_2019_andrews/law/zentou_2019_eq5')['laws'][0]
    assert not L['corrected'] and 'dropped_by_correction' not in L
