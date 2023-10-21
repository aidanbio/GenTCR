import logging.config
import unittest
import re
from enum import auto

import numpy as np

from tcredit.common import StrEnum, BaseTest, StrUtils

# Logger
logger = logging.getLogger('epidab')

ALLELE_SEP = ','

class MHCClass(StrEnum):
    I = auto()
    II = auto()

    @classmethod
    def values(cls):
        return [c.value for c in cls]

class MHCAlleleName(object):

    EXP_STATUS = ['N', 'L', 'S', 'C', 'A', 'Q']

    FN_GENE_MAP = '../data/mhcinfo/mhc_gene_map.txt'
    FN_SYNONYM_MAP = '../data/mhcinfo/allele_syn_map.txt'
    FN_ALLELE_MAP = '../data/mhcinfo/mhc_allele_map.txt'

    # Class variables
    _gene_map = None
    _synonym_map = None
    _allele_map = None

    def __init__(self, species=None, clazz=None,
                 gene=None, group=None, protein=None,
                 syn_subst=None, nc_subst=None, exp_status=None):
        self.species = species
        self.clazz = clazz
        self.gene = gene
        self.group = group
        self.protein = protein
        self.syn_subst = syn_subst
        self.nc_subst = nc_subst
        self.exp_status = exp_status

    # level: 0(species), 1(gene), 2(group), 3(protein), 4(syn_subst), 5(nc_subst)
    def format(self, level=3):
        if level > 5:
            raise ValueError('level should be less than 6')

        vals = [self.species, self.gene, self.group, self.protein, self.syn_subst, self.nc_subst]
        s = ''
        for i in range(level + 1):
            if not StrUtils.empty(vals[i]):
                if i == 1:
                    s += '-'
                elif i == 2:
                    s += '*'
                elif i > 2:
                    if self.species != 'H2':
                        s += ':'
                s += vals[i]

        if level >= 3 and not StrUtils.empty(self.exp_status):
            s += self.exp_status
        return s

    @classmethod
    def is_multi(cls, name):
        return ALLELE_SEP in name

    @classmethod
    def filter(cls, name, as_list=False, unique=True, sort=True):
        ans = list(filter(lambda al: cls.is_valid(al), name.split(ALLELE_SEP)))
        if unique:
            ans = np.unique(ans)
        if sort:
            ans = sorted(ans)

        return ans if as_list else ALLELE_SEP.join(ans)

    @classmethod
    def _handle_ma_name(cls, func, as_list=False, **kwargs):
        name = kwargs['name']
        if cls.is_multi(name):
            result = []
            for an in name.split(ALLELE_SEP):
                kwargs['name'] = an
                # # Ignore sub errors
                # try:
                result.append(func(**kwargs))
                # except Exception as e:
                #     logger.error(f'{e} in {an}')

            return result if as_list else ALLELE_SEP.join(result)
        else:
            return func(**kwargs)

    @classmethod
    def level(cls, name):
        return cls._handle_ma_name(func=cls._level, as_list=True, name=name)

    @classmethod
    def _level(cls, name):
        if name in cls.all_species():
            return 0
        allele = cls.parse(name)
        found = [allele.species, allele.gene, allele.group, allele.protein, allele.syn_subst, allele.nc_subst].index(None)
        return found - 1


    @classmethod
    def sub_name(cls, name, level=3):
        return cls._handle_ma_name(func=cls._sub_name, name=name, level=level)

    @classmethod
    def _sub_name(cls, name, level=3):
        allele = cls.parse(name)
        return allele.format(level)

    @classmethod
    def split(cls, name):
        return cls._handle_ma_name(func=cls._split, as_list=True, name=name)

    @classmethod
    def _split(cls, name):
        allele = cls.parse(name)
        tokens = []
        tokens.append(allele.species)
        # tokens.append(allele.clazz)
        tokens.append(allele.gene)
        tokens.append(allele.group if allele.group is not None else '')
        tokens.append(allele.protein if allele.protein is not None else '')
        return tokens

    @classmethod
    def std_name(cls, name):
        return cls._handle_ma_name(func=cls._std_name, name=name)
        # if ALLELE_SEP in name:
        #     return ALLELE_SEP.join(map(lambda an: cls._std_name(an), name.split(ALLELE_SEP)))
        # else:
        #     return cls._std_name(name)

    @classmethod
    def _std_name(cls, name):
        # Get standard name from synonyms if possible
        sname = name
        sm = cls.synonym_map()
        if name in sm:
            sname = sm[name]

        # ex) HLA-A02:01 or HLA-A0201 ==> HLA-A*02:01
        # found = re.search('HLA-[ABCEG]\d', sname)
        # if found is not None:
        #     sname = sname[:5] + '*' + sname[5:]
        #     if len(sname) > 8 and  sname[8] != ':':
        #         sname = sname[:8] + ':' + sname[8:]
        species = cls.species(sname)
        # print('Species for %s: %s' % (species, sname))
        if species is not None:
            genes = cls.classI_genes(species) + cls.classII_genes(species)
            # print('Genes for %s: %s' % (species, genes))
            if len(genes) > 0:
                all_genes = '|'.join(genes)
                pattern = '(%s)[-]?(%s)[\*-]?([\w]{1,3})$' % (species, all_genes)
                found = re.search(pattern, sname)
                if found is not None: # Ends with group name
                    if species == 'H2': # Different patterns in mouse alleles
                        sname = '%s-%s%s' % (found.group(1), found.group(2), found.group(3))
                    else:
                        sname = '%s-%s*%s' % (found.group(1), found.group(2), found.group(3))
                else:
                    pattern = '(%s)[-]?(%s)[\*-]?([\w]{1,3})[:]?([0-9]{2})' % (species, all_genes)
                    sname = re.sub(pattern, '\\1-\\2*\\3:\\4', sname)

                pattern = 'HLA-(%s)\*[\w]{1,3}$' % all_genes # HLA-A*12 => HLA-A*12:01
                found = re.search(pattern, sname)
                if found is not None:
                    sname = sname + ':01'

        return re.sub(r'\s+', '', sname)

    @classmethod
    def parse(cls, name):
        # if name is None or len(name) == 0:
        #     return name
        return cls._handle_ma_name(func=cls._parse, as_list=True, name=name)
        # if ALLELE_SEP in name:
        #     return [cls._parse(an) for an in name.split(ALLELE_SEP)]
        # else:
        #     return cls._parse(name)

    @classmethod
    def _parse(cls, name):
        # gm = cls.gene_map()
        species = cls.species(name)
        if species is None:
            raise ValueError('Unknown species: %s' % name)

        # pattern = None
        genes_c1 = cls.classI_genes(species)
        genes_c2 = cls.classII_genes(species)
        all_genes = genes_c1 + genes_c2

        gene = group = protein = None
        if species == 'H2':  # The allele name pattern for mouse is different from the others
            pattern = '(?P<species>%s)-(?P<gene>%s)(?P<protein>b|d|k|q|wm7)' % (species, all_genes)
            found = re.search(pattern, name)
            if found is None:
                raise ValueError('Invalid allele name:', name)

            gene = found.group('gene')
            protein = found.group('protein')
        else:
            tokens = name.split('-')
            subname = tokens[1]
            if len(tokens) == 2 and subname in all_genes:
                gene = subname
            else:
                tokens = subname.split('*')
                gene = tokens[0]
                subname = tokens[1]
                if (len(tokens) == 2) and (':' not in subname) and (len(subname) < 4):
                    group = subname
                else:
                    tokens = subname.split(':')
                    group = tokens[0]
                    protein = tokens[1]

            # pattern = '(?P<species>%s)-(?P<gene>%s)\*?(?P<group>[\w]{1,3})?$' % (species, all_genes)
            # found = re.search(pattern, name)
            # if found is not None:  # Ends with group name
            #     gene = found.group('gene')
            #     group = found.group('group')
            # else:
            #     pattern = '(?P<species>%s)-(?P<gene>%s)\*(?P<group>[\w]{1,3}):(?P<protein>[0-9]{2})' % (species, all_genes)
            #     found = re.search(pattern, name)
            #     if found is None:
            #         raise ValueError('Invalid allele name:', name)
            #
            #     gene = found.group('gene')
            #     group = found.group('group')
            #     protein = found.group('protein')

        return MHCAlleleName(species=species,
                             clazz=MHCClass.I if gene in genes_c1 else MHCClass.II,
                             gene=gene,
                             group=group,
                             protein=protein)

    @classmethod
    def is_valid(cls, name):
        return cls._handle_ma_name(func=cls._is_valid, as_list=True, name=name)
        # if ALLELE_SEP in name:
        #     return [cls._is_valid(an) for an in name.split(ALLELE_SEP)]
        # else:
        #     return cls._is_valid(name)

    @classmethod
    def _is_valid(cls, name):
        try:
            cls.parse(name)
            return True
        except Exception as e:
            logger.warning(e)
            return False

    @classmethod
    def gene_map(cls):
        if cls._gene_map is None:
            cls._gene_map = cls._load_gene_map()
        return cls._gene_map

    @classmethod
    def all_species(cls):
        return cls.gene_map().keys()

    @classmethod
    def species(cls, name):
        for s in cls.gene_map().keys():
            if name.startswith(s):
                return s
        return None

    @classmethod
    def classI_genes(cls, species):
        gm = cls.gene_map()
        genes_pair = gm[species]
        return (genes_pair[0] if genes_pair[0] is not None else [])

    @classmethod
    def classII_genes(cls, species):
        gm = cls.gene_map()
        genes_pair = gm[species]
        return (genes_pair[1] if genes_pair[1] is not None else [])

    @classmethod
    def synonym_map(cls):
        if cls._synonym_map is None:
            cls._synonym_map = cls._load_synonym_map()
        return cls._synonym_map

    @classmethod
    def _load_gene_map(cls):
        data = None
        with open(cls.FN_GENE_MAP, 'r') as fh:
            data = fh.read()
        return eval(data)

    @classmethod
    def _load_synonym_map(cls):
        data = None
        with open(cls.FN_SYNONYM_MAP, 'r') as fh:
            data = fh.read()
        return eval(data)



# HLA supertypes compiled from {Sidney:2008bn}
# class HLASupertype(object):
#     # class variable
#     _allele_hla_st_map = None
#
#     @classmethod
#     def load_hla_supertype_map(cls, fn='data/mhcinfo/HLA-supertype_revised.csv'):
#         print('Load HLASupertype._allele_hla_st_map...')
#         hla_st_tab = pd.read_csv(fn, na_values='Unclassified')
#         hla_st_tab.allele = hla_st_tab.allele.map(lambda s: 'HLA-%s:%s' % (s[:4], s[4:]))
#         hla_st_tab.index = hla_st_tab.allele
#         cls._allele_hla_st_map = hla_st_tab.supertype.to_dict()
#
#     @classmethod
#     def supertype(cls, allele):
#         if cls._allele_hla_st_map is None:
#             cls.load_hla_supertype_map()
#         return cls._allele_hla_st_map[allele]
#
#     @classmethod
#     def has_supertype(cls, allele):
#         if cls._allele_hla_st_map is None:
#             cls.load_hla_supertype_map()
#         return allele in cls._allele_hla_st_map

######################################################################################
# Tests
class MHCClassTest(BaseTest):
    def test_class_eunm(self):
        self.assertEqual('I', MHCClass.I)
        self.assertEqual('II', MHCClass.II)
        self.assertArrayEqual(['I', 'II'], MHCClass.values())

class MHCAlleleNameTest(BaseTest):
    def test_format(self):
        an = MHCAlleleName(species='HLA', clazz=MHCClass.I, gene='A', group='01', protein='01')
        self.assertEqual('HLA', an.format(level=0))
        self.assertEqual('HLA-A', an.format(level=1))
        self.assertEqual('HLA-A*01', an.format(level=2))
        self.assertEqual('HLA-A*01:01', an.format(level=3))

        an = MHCAlleleName(species='H2', clazz=MHCClass.I, gene='K', group=None, protein='d')
        self.assertEqual('H2', an.format(level=0))
        self.assertEqual('H2-K', an.format(level=1))
        self.assertEqual('H2-K', an.format(level=2))
        self.assertEqual('H2-Kd', an.format(level=3))

        an = MHCAlleleName(species='Rano', clazz=MHCClass.I, gene='Bb', group='n')
        self.assertEqual('Rano', an.format(level=0))
        self.assertEqual('Rano-Bb', an.format(level=1))
        self.assertEqual('Rano-Bb*n', an.format(level=2))
        self.assertEqual('Rano-Bb*n', an.format(level=3))

    def test_level(self):
        self.assertEqual(0, MHCAlleleName.level('HLA'))
        self.assertEqual(1, MHCAlleleName.level('HLA-A'))
        self.assertEqual(2, MHCAlleleName.level('HLA-A*24'))
        self.assertEqual(3, MHCAlleleName.level('HLA-A*24:01'))

        # Multi allele
        ma_name = ALLELE_SEP.join(['HLA', 'HLA-A', 'HLA-A*24', 'HLA-A*24:01'])
        levels = MHCAlleleName.level(ma_name)
        self.assertEqual(0, levels[0])
        self.assertEqual(1, levels[1])
        self.assertEqual(2, levels[2])
        self.assertEqual(3, levels[3])

        # Invalid names
        with self.assertRaises(Exception):
            MHCAlleleName.level('')

        with self.assertRaises(Exception):
            MHCAlleleName.level(None)

        with self.assertRaises(Exception):
            MHCAlleleName.level('HLA Class I')

        with self.assertRaises(Exception):
            MHCAlleleName.level(ALLELE_SEP.join(['HLA', 'HLA-A', 'XXX', 'HLA-A*24:01']))

    def test_std_name(self):
        for synm in self.test_synonyms:
            aname = MHCAlleleName.std_name(synm)
            self.assertIsNotNone(aname)

        bypass = 'XXXX'
        self.assertEqual(bypass, MHCAlleleName.std_name(bypass))
        self.assertEqual('#', MHCAlleleName.std_name('#'))

        with self.assertRaises(Exception):
            MHCAlleleName.std_name(None)

        # HLA-A01:01 or HLA-A0101==>HLA-A*01:01
        self.assertEqual('HLA-A*01:01', MHCAlleleName.std_name('HLA-A01:01'))
        self.assertEqual('HLA-A*01:01', MHCAlleleName.std_name('HLA-A0101'))

        self.assertEqual('HLA-B*01:01', MHCAlleleName.std_name('HLA-B01:01'))
        self.assertEqual('HLA-B*01:01', MHCAlleleName.std_name('HLA-B0101'))
        self.assertEqual('HLA-C*01:01', MHCAlleleName.std_name('HLA-C01:01'))
        self.assertEqual('HLA-C*01:01', MHCAlleleName.std_name('HLA-C0101'))
        self.assertEqual('HLA-E*01:01', MHCAlleleName.std_name('HLA-E01:01'))
        self.assertEqual('HLA-E*01:01', MHCAlleleName.std_name('HLA-E0101'))
        self.assertEqual('HLA-G*01:01', MHCAlleleName.std_name('HLA-G01:01'))
        self.assertEqual('HLA-G*01:01', MHCAlleleName.std_name('HLA-G0101'))

        self.assertEqual('Eqca-N*001:01', MHCAlleleName.std_name('Eqca-N-00101'))
        self.assertEqual('Eqca-1*019:01', MHCAlleleName.std_name('Eqca-1-01901'))
        self.assertEqual('Rano-Bb*n', MHCAlleleName.std_name('Rano-Bb-n'))

        self.assertEqual('H2-Db', MHCAlleleName.std_name('H2-Db'))
        self.assertEqual('H2-Kb', MHCAlleleName.std_name('H2-Kb'))
        self.assertEqual('H2-Lq', MHCAlleleName.std_name('H2-Lq'))
        self.assertEqual('H2-Kwm7', MHCAlleleName.std_name('H2-Kwm7'))

        # When the name ends with group
        self.assertEqual('HLA-A*23:01', MHCAlleleName.std_name('HLA-A23'))
        self.assertEqual('HLA-A*23:01', MHCAlleleName.std_name('HLA-A*23'))
        self.assertEqual('HLA-A*23:01', MHCAlleleName.std_name('HLA-A-23'))
        self.assertEqual('HLA-B*23:01', MHCAlleleName.std_name('HLA-B23'))
        self.assertEqual('HLA-B*23:01', MHCAlleleName.std_name('HLA-B*23'))
        self.assertEqual('HLA-B*23:01', MHCAlleleName.std_name('HLA-B-23'))
        self.assertEqual('HLA-C*23:01', MHCAlleleName.std_name('HLA-C23'))
        self.assertEqual('HLA-C*23:01', MHCAlleleName.std_name('HLA-C*23'))
        self.assertEqual('HLA-C*23:01', MHCAlleleName.std_name('HLA-C-23'))

        # For white space within name
        self.assertEqual('HLAclassI', MHCAlleleName.std_name('HLA class I'))
        self.assertEqual('H2-kclassI', MHCAlleleName.std_name('H2-k classI'))

        # Multi-allele
        expected = ALLELE_SEP.join(['HLA-B*01:01', 'HLA-B*01:01','HLA-C*01:01',
                                    'HLA-C*01:01', 'HLA-E*01:01', 'HLA-G*01:01'])
        actual = MHCAlleleName.std_name(ALLELE_SEP.join(['HLA-B01:01', 'HLA-B0101', 'HLA-C01:01',
                                                         'HLA-C0101', 'HLA-E01:01', 'HLA-G0101']))
        self.assertEqual(expected, actual)

        expected = ALLELE_SEP.join(['HLA-B*01:01', 'HLA-B*01:01','HLA-C*01:01',
                                    'HLA-C*01:01', 'CCCC', 'HLA-G*01:01'])
        actual = MHCAlleleName.std_name(ALLELE_SEP.join(['HLA-B01:01', 'HLA-B0101', 'HLA-C01:01',
                                                         'HLA-C0101', 'CCCC', 'HLA-G0101']))
        self.assertEqual(expected, actual)

    def test_sub_name(self):
        self.assertEqual('BoLA', MHCAlleleName.sub_name('BoLA-1*007:01:01', level=0))
        self.assertEqual('BoLA-1', MHCAlleleName.sub_name('BoLA-1*007:01:01', level=1))
        self.assertEqual('BoLA-1*007', MHCAlleleName.sub_name('BoLA-1*007:01:01', level=2))
        self.assertEqual('BoLA-1*007:01', MHCAlleleName.sub_name('BoLA-1*007:01:01', level=3))

        self.assertEqual('BoLA', MHCAlleleName.sub_name('BoLA-T2c', level=0))
        self.assertEqual('BoLA-T2c', MHCAlleleName.sub_name('BoLA-T2c', level=1))
        self.assertEqual('BoLA-T2c', MHCAlleleName.sub_name('BoLA-T2c', level=2))
        self.assertEqual('BoLA-T2c', MHCAlleleName.sub_name('BoLA-T2c', level=3))

        self.assertEqual('H2', MHCAlleleName.sub_name('H2-Db', level=0))
        self.assertEqual('H2-D', MHCAlleleName.sub_name('H2-Db', level=1))
        self.assertEqual('H2-D', MHCAlleleName.sub_name('H2-Db', level=2))
        self.assertEqual('H2-Db', MHCAlleleName.sub_name('H2-Db', level=3))

        self.assertEqual('H2', MHCAlleleName.sub_name('H2-Kd', level=0))
        self.assertEqual('H2-K', MHCAlleleName.sub_name('H2-Kd', level=1))
        self.assertEqual('H2-K', MHCAlleleName.sub_name('H2-Kd', level=2))
        self.assertEqual('H2-Kd', MHCAlleleName.sub_name('H2-Kd', level=3))

        self.assertEqual('H2', MHCAlleleName.sub_name('H2-Kwm7', level=0))
        self.assertEqual('H2-K', MHCAlleleName.sub_name('H2-Kwm7', level=1))
        self.assertEqual('H2-K', MHCAlleleName.sub_name('H2-Kwm7', level=2))
        self.assertEqual('H2-Kwm7', MHCAlleleName.sub_name('H2-Kwm7', level=3))

        self.assertEqual('H2', MHCAlleleName.sub_name('H2-Lq', level=0))
        self.assertEqual('H2-L', MHCAlleleName.sub_name('H2-Lq', level=1))
        self.assertEqual('H2-L', MHCAlleleName.sub_name('H2-Lq', level=2))
        self.assertEqual('H2-Lq', MHCAlleleName.sub_name('H2-Lq', level=3))

        # Multi allele
        expected = ALLELE_SEP.join(['BoLA-1*007:01', 'HLA-A*01:01', 'H2-Kwm7', 'HLA-B*01:01'])
        actual = MHCAlleleName.sub_name(ALLELE_SEP.join(['BoLA-1*007:01:01', 'HLA-A*01:01:01', 'H2-Kwm7', 'HLA-B*01:01:03']))
        self.assertEqual(expected, actual)

        # Invalid names
        with self.assertRaises(Exception):
            MHCAlleleName.sub_name('')

        with self.assertRaises(Exception):
            MHCAlleleName.sub_name(None)

        with self.assertRaises(Exception):
            MHCAlleleName.sub_name('HLA Class I')

        with self.assertRaises(Exception):
            MHCAlleleName.sub_name(ALLELE_SEP.join(['BoLA-1*007:01:01', 'HLA-A*01:01:01', 'XXXX', 'HLA-B*01:01:03']))

        # # Empy name
        # self.assertTrue(len(MHCAlleleName.parse('')) == 0)
        # self.assertIsNone(MHCAlleleName.parse(None))

    def test_parse(self):
        an = MHCAlleleName.parse('BoLA-1*007:01')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'BoLA')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, '1')
        self.assertEqual(an.group, '007')
        self.assertEqual(an.protein, '01')

        an = MHCAlleleName.parse('BoLA-T2c')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'BoLA')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'T2c')
        self.assertIsNone(an.group)
        self.assertIsNone(an.protein)

        an = MHCAlleleName.parse('Eqca-1*001:01')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'Eqca')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, '1')
        self.assertEqual(an.group, '001')
        self.assertEqual(an.protein, '01')

        an = MHCAlleleName.parse('Gogo-B*01:01')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'Gogo')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'B')
        self.assertEqual(an.group, '01')
        self.assertEqual(an.protein, '01')

        an = MHCAlleleName.parse('H2-Db')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'H2')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'D')
        self.assertEqual(an.protein, 'b')

        an = MHCAlleleName.parse('H2-Kd')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'H2')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'K')
        self.assertEqual(an.protein, 'd')

        an = MHCAlleleName.parse('H2-Kwm7')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'H2')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'K')
        self.assertEqual(an.protein, 'wm7')

        an = MHCAlleleName.parse('H2-Lq')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'H2')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'L')
        self.assertEqual(an.protein, 'q')

        an = MHCAlleleName.parse('HLA-A*02:01')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'HLA')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'A')
        self.assertEqual(an.group, '02')
        self.assertEqual(an.protein, '01')

        an = MHCAlleleName.parse('HLA-A*02:05')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'HLA')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'A')
        self.assertEqual(an.group, '02')
        self.assertEqual(an.protein, '05')

        an = MHCAlleleName.parse('HLA-A*02:05N')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'HLA')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'A')
        self.assertEqual(an.group, '02')
        self.assertEqual(an.protein, '05N')

        an = MHCAlleleName.parse('HLA-A*01:01:01:02N')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'HLA')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'A')
        self.assertEqual(an.group, '01')
        self.assertEqual(an.protein, '01')

        an = MHCAlleleName.parse('HLA-B*45:01')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'HLA')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'B')
        self.assertEqual(an.group, '45')
        self.assertEqual(an.protein, '01')

        an = MHCAlleleName.parse('HLA-C*05:01')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'HLA')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'C')
        self.assertEqual(an.group, '05')
        self.assertEqual(an.protein, '01')

        an = MHCAlleleName.parse('Mamu-A1*001:01')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'Mamu')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'A1')
        self.assertEqual(an.group, '001')
        self.assertEqual(an.protein, '01')

        an = MHCAlleleName.parse('Mamu-B*004:01')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'Mamu')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'B')
        self.assertEqual(an.group, '004')
        self.assertEqual(an.protein, '01')

        an = MHCAlleleName.parse('Patr-A*001:01')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'Patr')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'A')
        self.assertEqual(an.group, '001')
        self.assertEqual(an.protein, '01')

        an = MHCAlleleName.parse('SLA-1*007:01')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'SLA')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, '1')
        self.assertEqual(an.group, '007')
        self.assertEqual(an.protein, '01')

        an = MHCAlleleName.parse('Rano-Bb*n')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'Rano')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'Bb')
        self.assertEqual(an.group, 'n')

        an = MHCAlleleName.parse('Rano-A1*b')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'Rano')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'A1')
        self.assertEqual(an.group, 'b')

        an = MHCAlleleName.parse('Rano-A*f')
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'Rano')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'A')
        self.assertEqual(an.group, 'f')

        # Multi allele
        ma_name = ALLELE_SEP.join(['BoLA-1*007:01', 'BoLA-T2c', 'Eqca-1*001:01', 'Rano-A1*b'])
        ans = MHCAlleleName.parse(ma_name)

        self.assertEqual(4, len(ans))
        an = ans[0]
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'BoLA')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, '1')
        self.assertEqual(an.group, '007')
        self.assertEqual(an.protein, '01')

        an = ans[1]
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'BoLA')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'T2c')
        self.assertIsNone(an.group)
        self.assertIsNone(an.protein)

        an = ans[2]
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'Eqca')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, '1')
        self.assertEqual(an.group, '001')
        self.assertEqual(an.protein, '01')

        an = ans[3]
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'Rano')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'A1')
        self.assertEqual(an.group, 'b')

        ma_name = ALLELE_SEP.join(['BoLA-1*007:01', 'BoLA-T2c', 'Eqca-1*001:01', 'Rano-A1*b'])
        ans = MHCAlleleName.parse(ma_name)

        self.assertEqual(4, len(ans))
        an = ans[0]
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'BoLA')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, '1')
        self.assertEqual(an.group, '007')
        self.assertEqual(an.protein, '01')

        an = ans[1]
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'BoLA')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'T2c')
        self.assertIsNone(an.group)
        self.assertIsNone(an.protein)

        an = ans[2]
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'Eqca')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, '1')
        self.assertEqual(an.group, '001')
        self.assertEqual(an.protein, '01')

        an = ans[3]
        self.assertIsNotNone(an)
        self.assertEqual(an.species, 'Rano')
        self.assertEqual(an.clazz, MHCClass.I)
        self.assertEqual(an.gene, 'A1')
        self.assertEqual(an.group, 'b')

        # With invalid name
        with self.assertRaises(Exception):
            MHCAlleleName.parse('')

        with self.assertRaises(Exception):
            MHCAlleleName.parse(None)

        with self.assertRaises(Exception):
            MHCAlleleName.parse('HLA Class I')

        with self.assertRaises(Exception):
            MHCAlleleName.parse(ALLELE_SEP.join(['BoLA-1*007:01', 'XXXX', 'Eqca-1*001:01', 'Rano-KKK']))

    def test_is_valid(self):
        self.assertTrue(MHCAlleleName.is_valid('Eqca-1*002:01'))
        self.assertTrue(MHCAlleleName.is_valid('BoLA-T2c'))
        self.assertTrue(MHCAlleleName.is_valid('Rano-A*f'))
        self.assertTrue(MHCAlleleName.is_valid('Gaga-BF1'))
        self.assertTrue(MHCAlleleName.is_valid('Gaga-BF2'))

        self.assertFalse(MHCAlleleName.is_valid('RT1-Bl'))
        self.assertFalse(MHCAlleleName.is_valid('Eqca-1*00201'))
        self.assertFalse(MHCAlleleName.is_valid('BoLA-A11'))
        self.assertFalse(MHCAlleleName.is_valid('HLA-Aw24'))
        self.assertFalse(MHCAlleleName.is_valid('HLA-A-01:01'))
        self.assertFalse(MHCAlleleName.is_valid('HLA-A01:01'))
        self.assertFalse(MHCAlleleName.is_valid('HLA-A0101'))
        self.assertFalse(MHCAlleleName.is_valid('XXXX'))

        # Multi-allele
        ma_name = ALLELE_SEP.join(['Eqca-1*002:01', 'BoLA-T2c', 'Rano-A*f',
                                   'RT1-Bl', 'Eqca-1*00201', 'BoLA-A11'])
        result = MHCAlleleName.is_valid(ma_name)

        self.assertEqual(6, len(result))
        self.assertTrue(result[0])
        self.assertTrue(result[1])
        self.assertTrue(result[2])
        self.assertFalse(result[3])
        self.assertFalse(result[4])
        self.assertFalse(result[5])

    def test_filter(self):
        anames = ALLELE_SEP.join(['BoLA-1*007:01', 'A*01:01', 'HLAClassI', 'H2-Kwm7', 'HLA-B*01:01'])
        expected = ['BoLA-1*007:01', 'H2-Kwm7', 'HLA-B*01:01']
        self.assertListEqual(expected, MHCAlleleName.filter(anames, as_list=True))
        self.assertEqual(ALLELE_SEP.join(expected), MHCAlleleName.filter(anames, as_list=False))

        anames = ALLELE_SEP.join(['A*01:01', 'HLAClassI'])
        self.assertTrue(len(MHCAlleleName.filter(anames, as_list=True)) == 0)
        self.assertTrue(len(MHCAlleleName.filter(anames, as_list=False)) == 0)

# class HLASupertypeTest(BaseTest):
#     def test_supertype(self):
#         self.assertEqual('A01', HLASupertype.supertype('HLA-A*01:01'))
#         self.assertEqual('A02', HLASupertype.supertype('HLA-A*02:01'))
#         self.assertEqual('A03', HLASupertype.supertype('HLA-A*03:01'))
#         self.assertEqual('B07', HLASupertype.supertype('HLA-B*07:02'))
#         self.assertEqual('B08', HLASupertype.supertype('HLA-B*08:01'))
#         print(HLASupertype.supertype('HLA-A*01:13'))
#         self.assertTrue(pd.isnull(HLASupertype.supertype('HLA-A*01:13')))
#         self.assertTrue(pd.isnull(HLASupertype.supertype('HLA-B*07:13')))

if __name__ == '__main__':
    unittest.main()
