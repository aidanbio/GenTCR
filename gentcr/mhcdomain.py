import unittest
from collections import OrderedDict
import logging
import numpy as np

from gentcr.common import BaseTest
from gentcr.mhcnc import MHCAlleleName

# Logger
logger = logging.getLogger('gentcr')

class PanMHCIContactDomain(object):
    G_ALPHA1_POS = (24, 89)
    G_ALPHA2_POS = (90, 182)

    DOMAIN_OFFSET_MAP = OrderedDict({ # mhc_gene => offset
        'BoLA-T2c': 21,
        'BoLA-1': 21,
        'BoLA-2': 21,
        'BoLA-3': 21,
        'BoLA-4': 21,
        'BoLA-6': 21,
        'Caja-E': 21,
        'DLA-88': 0,
        'Eqca-1': 21,
        'Eqca-16': 24,
        'Eqca-N': 24,
        'Gaga-BF1': 26,
        'Gaga-BF2': 21,
        'Gogo-B': 24,
        'H2-D': 24,
        'H2-K': 21,
        'H2-L': 24,
        'HLA-A': 24,
        'HLA-B': 24,
        'HLA-C': 24,
        'HLA-E': 21,
        'HLA-G': 24,
        'Mamu-A1': 24,
        'Mamu-A2': 24,
        'Mamu-A7': 24,
        'Mamu-B': 24,
        'Patr-A': 24,
        'Patr-B': 24,
        'Rano-A': 24,
        'Rano-A1': 21,
        'Rano-A2': 0,
        'SLA-1': 21,
        'SLA-2': 24,
        'SLA-3': 21,
    })

    DEFAULT_DOMAIN_OFFSET = G_ALPHA1_POS[0]
    DEFAULT_DOMAIN_MAXLEN = 180

    # IMGT pMHC binding domain and reference contact sites, {Ehrenmann:2009dn}
    IMGT_MHCI_8_CONTACT_SITES = [(0, 58), (0, 61), (0, 62), (0, 65), (0, 162), (0, 166), (0, 170),
                                 (1, 6), (1, 23), (1, 44), (1, 98),
                                 (2, 98), (2, 113), (2, 152), (2, 155), (2, 156), (2, 159),
                                 (4, 6), (4, 8), (4, 21), (4, 69), (4, 73), (4, 96), (4, 98), (4, 113), (4, 115),
                                 (5, 148), (5, 150), (5, 152), (5, 155),
                                 (6, 72), (6, 75), (6, 76),
                                 (7, 76), (7, 79), (7, 80), (7, 83), (7, 94), (7, 115), (7, 122), (7, 123), (7, 144),
                                 (7, 148)]
    IMGT_MHCI_9_CONTACT_SITES = [(0, 4), (0, 58), (0, 61), (0, 62), (0, 65), (0, 162), (0, 166), (0, 170),
                                 (1, 6), (1, 8), (1, 21), (1, 23), (1, 33), (1, 44), (1, 62), (1, 65), (1, 66), (1, 69),
                                 (2, 96), (2, 98), (2, 113), (2, 155), (2, 156), (2, 159),
                                 (3, 64), (3, 65), (3, 155),
                                 (4, 69), (4, 72), (4, 73), (4, 96), (4, 115), (4, 155), (4, 156),
                                 (5, 65), (5, 68), (5, 69), (5, 72), (5, 73), (5, 96), (5, 113), (5, 151), (5, 155),
                                 (6, 96), (6, 113), (6, 148), (6, 150), (6, 152), (6, 155),
                                 (7, 71), (7, 72), (7, 75), (7, 79), (7, 147),
                                 (8, 76), (8, 79), (8, 80), (8, 83), (8, 94), (8, 115), (8, 122), (8, 123), (8, 144),
                                 (8, 148)]

    # NetMHCPan pMHCI contact sites, {Nielsen:2007ga}
    NETMHCPAN_MHCI_9_CONTACT_SITES = [(0, 6), (0, 58), (0, 61), (0, 62), (0, 65), (0, 158), (0, 162), (0, 166),
                                      (0, 170),
                                      (1, 6), (1, 8), (1, 23), (1, 44), (1, 61), (1, 62), (1, 65), (1, 66), (1, 69),
                                      (1, 98), (1, 158),
                                      (2, 69), (2, 96), (2, 98), (2, 113), (2, 155), (2, 158),
                                      (3, 65), (3, 157), (3, 158), (3, 162),
                                      (4, 68), (4, 69), (4, 157),
                                      (5, 68), (5, 69), (5, 72), (5, 73), (5, 96), (5, 155),
                                      (6, 68), (6, 72), (6, 96), (6, 113), (6, 146), (6, 149), (6, 151), (6, 155),
                                      (7, 72), (7, 75), (7, 76), (7, 146),
                                      (8, 73), (8, 76), (8, 79), (8, 80), (8, 83), (8, 94), (8, 96), (8, 115), (8, 117),
                                      (8, 142), (8, 146)]

    # EpicCapo pMHCI contact sites, {Saethang:2012di}
    EPICCAPO_MHCI_9_CONTACT_SITES = [(0, 4), (0, 6), (0, 8), (0, 44), (0, 57), (0, 58), (0, 61), (0, 62), (0, 65),
                                     (0, 66), (0, 162), (0, 163), (0, 166), (0, 170),
                                     (1, 6), (1, 8), (1, 21), (1, 23), (1, 33), (1, 44), (1, 62), (1, 65), (1, 66),
                                     (1, 69), (1, 98), (1, 158),
                                     (2, 8), (2, 65), (2, 66), (2, 69), (2, 96), (2, 98), (2, 151), (2, 154), (2, 155),
                                     (2, 158), (2, 159),
                                     (3, 61), (3, 64), (3, 65), (3, 154), (3, 157),
                                     (4, 64), (4, 68), (4, 69), (4, 71), (4, 72), (4, 73), (4, 96), (4, 113), (4, 115),
                                     (4, 146), (4, 149), (4, 150), (4, 151), (4, 154), (4, 155),
                                     (5, 64), (5, 65), (5, 68), (5, 69), (5, 72), (5, 73), (5, 96), (5, 98), (5, 113),
                                     (5, 146), (5, 150), (5, 151), (5, 154), (5, 155),
                                     (6, 58), (6, 62), (6, 96), (6, 113), (6, 115), (6, 132), (6, 145), (6, 146),
                                     (6, 149), (6, 151), (6, 154),
                                     (7, 71), (7, 72), (7, 75), (7, 76), (7, 79), (7, 145), (7, 146),
                                     (8, 25), (8, 32), (8, 54), (8, 57), (8, 76), (8, 79), (8, 80), (8, 83), (8, 94),
                                     (8, 96), (8, 115), (8, 122), (8, 123), (8, 141), (8, 142), (8, 145), (8, 146)]

    # Chelvanayagam pMHCI contact sites, {Luo:2016iw}
    CHELV_MHCI_9_CONTACT_SITES = [
        (0, 4), (0, 6), (0, 32), (0, 58), (0, 61), (0, 62), (0, 65), (0, 98), (0, 158), (0, 162), (0, 166), (0, 170),
        (1, 6), (1, 8), (1, 23), (1, 24), (1, 25), (1, 33), (1, 34), (1, 35), (1, 44), (1, 61), (1, 62), (1, 65),
        (1, 66), (1, 69), (1, 98), (1, 158), (1, 162), (1, 166),
        (2, 6), (2, 8), (2, 61), (2, 65), (2, 69), (2, 96), (2, 98), (2, 113), (2, 151), (2, 154), (2, 155), (2, 158),
        (2, 162),
        (3, 61), (3, 64), (3, 65), (3, 68), (3, 69), (3, 154), (3, 155), (3, 158),
        (4, 68), (4, 69), (4, 72), (4, 73), (4, 96), (4, 113), (4, 115), (4, 151), (4, 154), (4, 155), (4, 158),
        (5, 6), (5, 8), (5, 21), (5, 23), (5, 65), (5, 68), (5, 69), (5, 72), (5, 73), (5, 96), (5, 98), (5, 113),
        (5, 115), (5, 132), (5, 146), (5, 151), (5, 154), (5, 155),
        (6, 72), (6, 76), (6, 96), (6, 113), (6, 115), (6, 132), (6, 145), (6, 146), (6, 149), (6, 151), (6, 154),
        (6, 155),
        (7, 72), (7, 75), (7, 76), (7, 79), (7, 96), (7, 142), (7, 145), (7, 146),
        (8, 69), (8, 72), (8, 73), (8, 75), (8, 76), (8, 79), (8, 80), (8, 83), (8, 94), (8, 95), (8, 96), (8, 113),
        (8, 115), (8, 122), (8, 123), (8, 141), (8, 142), (8, 145), (8, 146)]

    _DEFAUL_CSS_MAP = {
        'imgt_mhci_8': IMGT_MHCI_8_CONTACT_SITES,
        'imgt_mhci_9': IMGT_MHCI_9_CONTACT_SITES,
        'netmhcpan_mhci_9': NETMHCPAN_MHCI_9_CONTACT_SITES,
        'epiccapo_mhci_9': EPICCAPO_MHCI_9_CONTACT_SITES,
        'chelv_mhci_9': CHELV_MHCI_9_CONTACT_SITES
    }


    def __init__(self, css_list=None):
        self._peplen_cs_map = {} # peplen => css
        self._all_hla_sites = []
        self.add_contact_site_list(css_list)

    def add_contact_site_list(self, css_list=None):
        if css_list is not None:
            for pep_len, css in css_list:
                if isinstance(css, str):
                    self.add_contact_sites(pep_len, self._DEFAUL_CSS_MAP[css])
                else:
                    self.add_contact_sites(pep_len, css)

    def add_contact_sites(self, pep_len, css):
        if pep_len not in self._peplen_cs_map:
            self._peplen_cs_map[pep_len] = []

        target_cs = self._peplen_cs_map[pep_len]
        for cs in css:
            if cs not in target_cs:
                target_cs.append(cs)

        target_cs.sort(key=lambda cs: cs[0])
        self._update_all_hla_sites()

    def set_contact_site_list(self, css_list=None):
        self._peplen_cs_map = {}
        self.add_contact_site_list(css_list=css_list)

    def set_contact_sites(self, pep_len, css):
        self._peplen_cs_map[pep_len] = css
        self._update_all_hla_sites()

    def contact_sites(self, pep_len, p_extend=0, h_extend=0):
        if pep_len in self._peplen_cs_map:
            old_css = self._peplen_cs_map[pep_len]
        else:
            # Consider all AAs of the peptide contact with all HLA contact sites
            old_css = []
            for i in range(pep_len):
                for j in self._all_hla_sites:
                    old_css.append((i, j))

        # Extend contact sites
        return self._extend_css(old_css, pep_len, p_extend, h_extend)

    def contact_site_seq(self, allele, pep_len, h_extend=0, replace_gap=True):
        raise NotImplementedError()

    def all_hla_sites(self):
        return self._all_hla_sites

    def _update_all_hla_sites(self):
        self._all_hla_sites = []
        for pep_len, css in self._peplen_cs_map.items():
            for cs in css:
                if cs[1] not in self._all_hla_sites:
                    self._all_hla_sites.append(cs[1])

        self._all_hla_sites.sort()

    def _extend_css(self, old_css, pep_len, p_extend=0, h_extend=0):
        css = []
        old_h_sites = sorted(np.unique([cs[1] for cs in old_css]))
        old_h_range = (min(old_h_sites), max(old_h_sites) + 1)

        for cs in old_css:
            p_begin = p_end = 0
            h_begin = h_end = 0
            if p_extend >= 0:
                p_begin = max(0, cs[0] - p_extend)
                p_end = min(pep_len, cs[0] + p_extend + 1)
            else:
                p_begin = 0
                p_end = pep_len

            if h_extend >= 0:
                h_begin = max(0, cs[1] - h_extend)
                h_end = min(old_h_range[1] + 100, cs[1] + h_extend + 1)
            else:
                h_begin = old_h_range[0]
                h_end = old_h_range[1]

            for cs_i in range(p_begin, p_end):
                for cs_j in range(h_begin, h_end):
                    new_cs = (cs_i, cs_j)
                    if new_cs not in css:
                        css.append(new_cs)
        return css

    def _get_domain_offset(self, allele):
        mhcgene = MHCAlleleName.sub_name(allele, level=1)
        for k, v in self.DOMAIN_OFFSET_MAP.items():
            if k == mhcgene:
                return v
        return self.DEFAULT_DOMAIN_OFFSET

class PanMHCIContactDomainTest(BaseTest):

    def setUp(self):
        logger.setLevel(logging.DEBUG)
        self.cdomain = PanMHCIContactDomain()

    def test_add_contact_sites(self):
        expected_css = [(0, 9), (1, 10)]
        self.cdomain.add_contact_sites(9, expected_css)

        self.assertTrue(np.array_equal(expected_css, self.cdomain.contact_sites(9)))

        expected_css = [(0, 9), (1, 10), (2, 11)]
        self.cdomain.add_contact_sites(9, expected_css)
        self.assertTrue(np.array_equal(expected_css, self.cdomain.contact_sites(9)))

    def test_all_hla_sites(self):
        self.assertTrue(len(self.cdomain.all_hla_sites()) == 0)

        self.cdomain.add_contact_sites(9, [(0, 9), (1, 10)])

        self.assertTrue(np.array_equal([9, 10], self.cdomain.all_hla_sites()))

        self.cdomain.add_contact_sites(15, [(0, 7), (1, 9), (2, 12)])
        self.assertTrue(np.array_equal([7, 9, 10, 12], self.cdomain.all_hla_sites()))

        self.cdomain.set_contact_sites(9, [(0, 11), (1, 8)])
        self.assertTrue(np.array_equal([7, 8, 9, 11, 12], self.cdomain.all_hla_sites()))

    def test_contact_sites_with_p_extend(self):
        self.cdomain.set_contact_sites(9, [(2, 9)])
        expected_css = [(0, 9), (1, 9), (2, 9), (3, 9), (4, 9)]
        actual_css = self.cdomain.contact_sites(9, p_extend=2)

        self.assertTrue(np.array_equal(expected_css, actual_css))

    def test_contact_sites_with_h_extend(self):
        self.cdomain.set_contact_sites(9, [(2, 9)])
        expected_css = [(2, 7), (2, 8), (2, 9), (2, 10), (2, 11)]

        actual_css = self.cdomain.contact_sites(9, h_extend=2)
        self.assertTrue(np.array_equal(expected_css, actual_css))

    def test_contact_sites_with_no_ref_css(self):
        self.cdomain.set_contact_sites(9, [(2, 9)])
        self.cdomain.set_contact_sites(15, [(0, 8)])
        expected_css = []
        pep_len = 12
        for i in range(pep_len):
            expected_css.append((i, 8))
            expected_css.append((i, 9))

        actual_css = self.cdomain.contact_sites(pep_len)
        self.assertTrue(np.array_equal(expected_css, actual_css))


if __name__ == '__main__':
    unittest.main()
