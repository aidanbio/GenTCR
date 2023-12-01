import subprocess

import unittest
from enum import Enum
from random import random
from urllib import request

import numpy as np
import pandas as pd
import warnings
import logging.config
from io import StringIO
from collections import OrderedDict
import re
from scipy.stats import pearsonr
import tempfile
from gentcr.mhcnc import MHCAlleleName
from gentcr.common import BaseTest, NumpyUtils, FileUtils, RemoteUtils, StrUtils

# Logger
logger = logging.getLogger('gentcr')


class IupacAminoAcid(Enum):
    A = ('A', 'Ala', 'Alanine')
    R = ('R', 'Arg', 'Arginine')
    N = ('N', 'Asn', 'Asparagine')
    D = ('D', 'Asp', 'Aspartic acid')
    C = ('C', 'Cys', 'Cysteine')
    Q = ('Q', 'Gln', 'Glutamine')
    E = ('E', 'Glu', 'Glutamic acid')
    G = ('G', 'Gly', 'Glycine')
    H = ('H', 'His', 'Histidine')
    I = ('I', 'Ile', 'Isoleucine')
    L = ('L', 'Leu', 'Leucine')
    K = ('K', 'Lys', 'Lysine')
    M = ('M', 'Met', 'Methionine')
    F = ('F', 'Phe', 'Phenylalanine')
    P = ('P', 'Pro', 'Proline')
    O = ('O', 'Pyl', 'Pyrrolysine')
    S = ('S', 'Ser', 'Serine')
    U = ('U', 'Sec', 'Selenocysteine')
    T = ('T', 'Thr', 'Threonine')
    W = ('W', 'Trp', 'Tryptophan')
    Y = ('Y', 'Tyr', 'Tyrosine')
    V = ('V', 'Val', 'Valine')
    B = ('B', 'Asx', 'Aspartic acid or Asparagine')
    Z = ('Z', 'Glx', 'Glutamic acid or Glutamine')
    X = ('X', 'Xaa', 'Any amino acid')

    # J = ('J', 'Xle', 'Leucine or Isoleucine')

    @property
    def code(self):
        return self.value[0]

    @property
    def abbr(self):
        return self.value[1]

    @property
    def name(self):
        return self.value[2]

    @property
    def index(self):
        return self.codes().index(self.code)

    @classmethod
    def codes(cls):
        return [c.value[0] for c in cls]

    @classmethod
    def abbrs(cls):
        return [c.value[1] for c in cls]

    @classmethod
    def names(cls):
        return [c.value[2] for c in cls]


AMINO_ACID = IupacAminoAcid
AA_CODES = AMINO_ACID.codes()
GAP = '-'
AASEQ_SEP = '|'


def is_valid_aaseq(seq, allow_gap=False):
    aas = ''.join(AA_CODES)

    if allow_gap:
        aas = aas + GAP
    pattern = '^[%s]+$' % aas
    found = re.match(pattern, seq)
    return found is not None
    # return all([(aa in aas) for aa in seq])


def rand_aaseqs(N=10, seq_len=9, aa_probs=None):
    return [rand_aaseq(seq_len, aa_probs=aa_probs) for i in range(N)]


def rand_aaseq(seq_len=9, aa_probs=None):
    aas = np.asarray(AA_CODES)
    indices = np.random.choice(aas.shape[0], seq_len, p=aa_probs)
    return ''.join(aas[indices])


def split_aaseq(seq=None, window_size=9):
    seqs = []
    seqlen = len(seq)
    if seqlen <= window_size:
        seqs.append(seq)
    else:
        for i in range(seqlen - window_size + 1):
            seqs.append(seq[i:(i + window_size)])
    return seqs


def write_fa(fn, seqs, headers=None):
    with open(fn, 'w') as fh:
        fh.write(format_fa(seqs, headers))


def format_fa(seqs, headers=None):
    return '\n'.join(
        map(lambda h, seq: '>%s\n%s' % (h, seq), range(1, len(seqs) + 1) if headers is None else headers, seqs))


def write_seqs(fn, seqs, sep='\n'):
    with open(fn, 'w') as fh:
        fh.write(sep.join(seqs))


def read_fa_from_file(fn_fasta=None):
    with open(fn_fasta, 'r') as f:
        return read_fa(f)


def read_fa_from_url(url=None):
    with request.urlopen(url, context=RemoteUtils._ssl_context) as response:
        return read_fa(response)


def read_fa(in_stream=None):
    loader = FastaSeqLoader()
    parser = FastaSeqParser()

    parser.add_parse_listener(loader)
    parser.parse(in_stream=in_stream)
    return loader.headers, loader.seqs


class FastaSeqParser(object):
    class Listener(object):
        def on_begin_parse(self):
            pass

        def on_seq_read(self, header=None, seq=None):
            pass

        def on_end_parse(self):
            pass

    def __init__(self):
        self._listeners = []

    def add_parse_listener(self, listener=None):
        self._listeners.append(listener)

    def remove_parse_listener(self, listener=None):
        self._listeners.remove(listener)

    def parse(self, in_stream, decode=None):
        #         Tracer()()
        self._fire_begin_parse()
        header = None
        seq = ''
        for line in in_stream:
            line = line.strip()
            if decode is not None:
                line = decode(line)
            if line.startswith('>'):
                if len(seq) > 0:
                    self._fire_seq_read(header=header, seq=seq)

                header = line[1:]
                seq = ''
            else:
                seq += line

        self._fire_seq_read(header=header, seq=seq)
        self._fire_end_parse()

    def _fire_begin_parse(self):
        for listener in self._listeners:
            listener.on_begin_parse()

    def _fire_seq_read(self, header=None, seq=None):
        for listener in self._listeners:
            listener.on_seq_read(header=header, seq=seq)

    def _fire_end_parse(self):
        for listener in self._listeners:
            listener.on_end_parse()


class FastaSeqLoader(FastaSeqParser.Listener):
    def on_begin_parse(self):
        self.headers = []
        self.seqs = []

    def on_seq_read(self, header=None, seq=None):
        if not is_valid_aaseq(seq, allow_gap=True):
            raise ValueError('Invaild amino acid sequence:' % seq)
        # lseq = list(seq)
        # if len(self.seqs) > 0:
        #     last = self.seqs[-1]
        #     if len(last) != len(lseq):
        #         raise ValueError('Current seq is not the same length: %s != %s' % (len(last), len(lseq)))
        self.headers.append(header)
        self.seqs.append(seq)
    #
    # def load(self, fn_fasta=None):
    #     with open(fn_fasta, 'r') as f:
    #         parser = FastaSeqParser()
    #         parser.add_parse_listener(self)
    #         parser.parse(f)
    #
    #     return self.headers, self.seqs


NEEDLE_PATH = '/home/hym/tools/EMBOSS-6.6.0/emboss/needle'
def needle_aaseq_pair(seq1, seq2, output_identity=True, output_similarity=True):
    with tempfile.NamedTemporaryFile('w') as f1, tempfile.NamedTemporaryFile('w') as f2:
        f1.write(f'>1\n{seq1}')
        f1.flush()
        f2.write(f'>2\n{seq2}')
        f2.flush()
        args = [
            NEEDLE_PATH,
            '-asequence', f1.name,
            '-bsequence', f2.name,
            '-sprotein1', 'Y',
            '-sprotein2', 'Y',
            '-gapopen', '10',
            '-gapextend', '0.5',
            '-outfile', 'stdout'
        ]
        results = OrderedDict()
        output = subprocess.run(args, stdout=subprocess.PIPE).stdout.decode()
        results['output'] = output
        for line in output.split('\n'):
            line = line.strip().upper()
            if output_identity and 'IDENTITY' in line:
                results['identity'] = StrUtils.extract_floats(line.split()[-1])[0]/100.
            if output_similarity and 'SIMILARITY' in line:
                results['similarity'] = StrUtils.extract_floats(line.split()[-1])[0]/100.
        return results
    return None

class PositionSpecificScoringMatrix(object):
    SIMILARITY_SCORER_MAP = {
        'pearsonr': lambda x, y: pearsonr(x, y)[0],
        'kld': lambda x, y: 0.5 * (np.exp2(-np.sum(x * np.log2(x / y))) + np.exp2(-np.sum(y * np.log2(y / x)))),
        'euclidean': lambda x, y: np.exp2(-np.linalg.norm(x - y))
    }

    def __init__(self, row_index=AA_CODES, values=None):
        if values is not None and values.shape[0] != len(row_index):
            raise ValueError('values.shape[0] should be equal to len(row_index): %s!=%s' %
                             (values.shape[0], len(row_index)))

        self.row_index = np.array(row_index)
        self.atoi = OrderedDict(zip(row_index, range(len(row_index))))
        self.values = values
        if self.values is not None:
            self.values = self.values.astype(np.float32)
            if np.isnan(self.values).any():
                raise ValueError('any value is NaN')
            zerocols = np.flatnonzero(~self.values.any(axis=0))
            if len(zerocols) > 0:
                raise ValueError('PositionSpecificScoringMatrix.__init__, all values of %s cols were zeros' % zerocols)
                # val = 1./self.values.shape[0]
                # logger.warning('PositionSpecificScoringMatrix.__init__, all values of %s cols were zeros' % zerocols)
                # logger.warning('They will be filled with %s' % val)
                # self.values[:, zerocols] = val

    def ps_freq_scores(self, pos=None, as_prob=False, prob_range=(0.001, 0.999)):
        scores = self.values[:, pos]
        if as_prob:
            scores = NumpyUtils.to_probs(scores, prob_range=prob_range)
        return scores

    def aa_freq_scores(self, aa=None):
        return self.values[self.atoi[aa], :]

    def conservation_probs(self, prob_range=(0.001, 0.999)):
        sf = self.values.max(axis=0) - self.values.min(axis=0)  # specificity factor
        sd = self.values.std(axis=0)
        scores = sf + sd

        # scores = (2*sf*sd)/(sf+sd) # Harmonic mean
        # if scale:
        #     # Scaling between 0 and 1
        #     if np.all(scores == scores[0]): # all the same values
        #         scores[:] = 0 if np.all(scores == 0) else 1
        #     else:
        #         scores = (scores - scores.min()) / (scores.max() - scores.min())
        return NumpyUtils.to_probs(scores, prob_range=prob_range)

    def variability_probs(self, prob_range=(0.001, 0.999)):
        c_scores = self.conservation_probs(prob_range)
        v_scores = NumpyUtils.align_by_rrank(c_scores)
        return v_scores / v_scores.sum(axis=0)

    def choice_variable_positions(self, n_pos, prob_range=(0.001, 0.999)):
        return self._choice_positions(n_pos, prob_range, mode='variable')

    def choice_conserved_positions(self, n_pos, prob_range=(0.001, 0.999)):
        return self._choice_positions(n_pos, prob_range, mode='conserved')

    def _choice_positions(self, n_pos, prob_range=(0.001, 0.999), mode='variable'):
        seqlen = self.values.shape[1]
        if n_pos > seqlen:
            raise ValueError('Larger than length: %s > %s' % (n_pos, self.values.shape[1]))

        probs = None
        if mode == 'variable':
            probs = self.variability_probs(prob_range=prob_range)
        elif mode == 'conserved':
            probs = self.conservation_probs(prob_range=prob_range)
        else:
            raise ValueError('mode should be \'variable\' or \'conserved\'')

        n_nzeros = np.count_nonzero(probs)
        if n_pos > n_nzeros:
            tmp = np.random.choice(np.where(probs == 0)[0], n_pos - n_nzeros)
            probs[tmp] = 1 / seqlen
            logger.debug('Fewer non-zero probs: n_nzeros(%s) < n_pos(%s)' % (n_nzeros, n_pos))
            probs = probs / probs.sum(axis=0)

        return sorted(np.random.choice(seqlen, n_pos, replace=False, p=probs))

    def subst_aa_at(self, pos, aa=None):
        """
        Return the most frequent aa at the position to substitute the given aa
        """
        scores = self.ps_freq_scores(pos)
        iaa = self.atoi[aa]
        probs = NumpyUtils.to_probs(scores, prob_range=(0.001, 0.999))
        probs[iaa] = 0.
        probs = probs / probs.sum(axis=0)
        new_aa = np.random.choice(self.row_index, 1, p=probs)[0]
        return new_aa

    def extend_length(self, to=None):
        if to > self.values.shape[1]:
            d = to - self.values.shape[1]
            new_values = np.zeros((self.values.shape[0], to))
            denom = np.full(to, d + 1)
            for i in range(d + 1):
                new_values[:, i:(i + self.values.shape[1])] += self.values
                denom[i] -= (d - i)
                denom[-(i + 1)] = denom[i]

            self.values = new_values / denom
        else:
            warnings.warn('target length(%s) <= %s' % (to, self.values.shape[1]))

    def shrink_length(self, to=None):
        if to < self.values.shape[1]:
            d = self.values.shape[1] - to
            new_values = np.zeros((self.values.shape[0], to))
            for i in range(d + 1):
                new_values += self.values[:, i:(i + to)]

            self.values = new_values / (d + 1)
        else:
            warnings.warn('target length(%s) >= %s' % (to, self.values.shape[1]))

    def fit_length(self, to=None):
        if to > self.values.shape[1]:
            self.extend_length(to=to)
        elif to < self.values.shape[1]:
            self.shrink_length(to=to)

    def __len__(self):
        return self.values.shape[1]

    def itoa(self, i):
        return self.row_index[i]

    def mm_scale(self, score_range=(0.001, 0.999), inplace=False):
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=score_range)
        new_vals = scaler.fit_transform(self.values)
        if inplace:
            self.values = new_vals
            return self
        else:
            new_pssm = PositionSpecificScoringMatrix(row_index=self.row_index, values=new_vals)
            return new_pssm

    def seq_score(self, seq):
        if len(seq) != self.values.shape[1]:
            raise ValueError('len(seq) != self.values.shape[1]')
        scores = []
        for pos, aa in enumerate(seq):
            ai = self.atoi[aa]
            scores.append(self.values[ai, pos])
        score = np.mean(scores)
        return score

    def rand_seq(self, reverse_scores=False):
        scaled_pssm = self.mm_scale(inplace=False)
        seq = ''
        for pos in range(len(scaled_pssm)):
            scores = scaled_pssm.ps_freq_scores(pos)
            if reverse_scores:
                scores = 1 - scores
            probs = NumpyUtils.to_probs(scores, prob_range=(0, 1))
            aa = np.random.choice(scaled_pssm.row_index, 1, p=probs)[0]
            seq = seq + aa
        return seq

    def similarity_score(self, other, method='pearsonr'):
        """
        Estimate similarity with other PSSM
        :param other: other PSSM
        :param col_scorer: column-wise similarity scoring func
        :return: the score
        """
        if len(self) != len(other):
            raise ValueError('len(self) should be equal len(other): %s != %s' % (len(self), len(other)))

        col_scorer = self.SIMILARITY_SCORER_MAP[method]
        scores = []
        for pos in range(len(self)):
            score = col_scorer(self.ps_freq_scores(pos, as_prob=True),
                               other.ps_freq_scores(pos, as_prob=True))
            logger.debug('%s column score: %s' % (pos, score))
            scores.append(score)
        return np.mean(scores)
        # return gmean(scores)

    def choice_seqs(self, N=10, source_seqs=None, threshold=0.3, compare_op=np.less):
        seq_len = len(self)
        maxes = np.array([max(0, len(seq) - seq_len + 1) for seq in source_seqs])
        N = min(N, sum(maxes))
        choice_ns = NumpyUtils.rand_assign_numbers(N, maxes)
        result_seqs = []
        for i, source_seq in enumerate(source_seqs):
            for start in range(choice_ns[i]):
                seq = source_seq[start:start + seq_len]
                if (seq not in result_seqs):
                    score = self.seq_score(seq)
                    if compare_op(score, threshold):
                        result_seqs.append(seq)
        return result_seqs


class MultipleSequenceAlignment(object):
    # Constants
    _FN_MHC_MSA = '../data/mhcinfo/prot/{0}/{1}.aln'

    def __init__(self, df=None):
        if df is None:
            raise ValueError('df_enc should be not None')

        aas = AMINO_ACID.codes() + [GAP]
        not_aa = list(filter(lambda aa: aa not in aas, np.ravel(df.values)))
        if len(not_aa) > 0:
            raise ValueError('Unknown AA chars: %s' % not_aa)

        self._df = df
        self._pssm = self._create_pssm()

    def pssm(self, aa_positions=None):
        if aa_positions is not None:
            return PositionSpecificScoringMatrix(values=np.copy(self._pssm.values[:, aa_positions]))
        else:
            return self._pssm

    def _create_pssm(self):
        row_index = AMINO_ACID.codes()
        values = np.zeros((len(row_index), self._df.shape[1]), dtype=np.float32)
        for ci in range(self._df.shape[1]):
            vals = self._df.iloc[:, ci]
            aa_occurs = np.array([np.count_nonzero(vals == aa) for aa in row_index])
            values[:, ci] = aa_occurs
        return PositionSpecificScoringMatrix(row_index=row_index, values=values)

    @property
    def has_gap(self):
        return np.count_nonzero(self._df.values == GAP) > 0

    def seq(self, index_key=None):
        seq = self._df.loc[index_key]
        return seq

    def iseq(self, index=None):
        seq = self._df.iloc[index]
        return seq

    def aas_at(self, pos):
        return self._df.iloc[:, pos]

    def sub_msa(self, positions=None):
        df = self._df.loc[:, positions]
        df.columns = list(range(len(positions)))
        return MultipleSequenceAlignment(df)

    @property
    def names(self):
        return self._df.index.values

    @property
    def positions(self):
        return self._df.columns.values

    def __eq__(self, other):
        return self._df.equals(other._df)

    @classmethod
    def from_fasta(cls, fn_fasta=None):
        msa = None
        headers, seqs = read_fa_from_file(fn_fasta)
        df = pd.DataFrame(list(map(lambda seq: list(seq), seqs)), index=headers)
        return MultipleSequenceAlignment(df)

    @classmethod
    def from_seqs(cls, seqs=None, fill_with_gap=True):
        msa = None

        values = []
        for seq in seqs:
            if not is_valid_aaseq(seq, allow_gap=True):
                raise ValueError('Invaild amino acid sequence:' % seq)
            lseq = list(seq)
            if len(values) > 0:
                last = values[-1]
                if not fill_with_gap and len(last) != len(lseq):
                    raise ValueError('Current seq is not the same length: %s != %s' % (len(last), len(lseq)))
            values.append(lseq)

        df = pd.DataFrame(values)
        if fill_with_gap:
            df = df.fillna(GAP)
        return MultipleSequenceAlignment(df=df)

    @classmethod
    def from_imgt_msa(cls, species=None, gene=None, subst_mat=None):
        fn = cls._FN_MHC_MSA.format(species, gene)
        logger.debug('Loading domain msa for %s-%s from %s' % (species, gene, fn))

        od = OrderedDict()
        with open(fn, 'r') as f:
            for line in f:
                tokens = line.split()
                if len(tokens) > 0:
                    allele = MHCAlleleName.std_name(tokens[0])
                    if MHCAlleleName.is_valid(allele):
                        seq = ''.join(tokens[1:])
                        logger.debug('Current allele, seq: %s, %s' % (allele, seq))
                        if len(seq) > 0:
                            if seq[-1] == 'X':
                                seq = seq[:-1]
                            if allele in od:
                                od[allele].extend(list(seq))
                            else:
                                od[allele] = list(seq)

        # Filter nonsynonymous alleles, replacing '-', '*', and '.' with AA in rep_seq, random AA, and '-', respectively
        rep_seq = None
        new_od = OrderedDict()
        for allele, seq in od.items():
            if rep_seq is None:  # The first allele is the representative seq
                rep_seq = seq.copy()

            ns_allele = MHCAlleleName.sub_name(allele, level=3)

            if ns_allele not in new_od:
                # Replace special chars, such as '-', '*', and '.'
                new_seq = []
                for i, aa in enumerate(seq):
                    if aa == '-':
                        new_seq.append(rep_seq[i])
                    elif aa == '*':  # Unknown AA
                        new_aa = GAP
                        if subst_mat is not None:  # Use AASubstitutionScoreMatrix for substitution of the AA
                            new_aa = subst_mat.subst_aa(rep_seq[i])
                        new_seq.append(new_aa)
                    elif aa == '.' or aa == '?':  # '.' means indel, '?' means...
                        new_seq.append(GAP)
                    elif aa == 'X':  # Stop codon
                        break
                    else:
                        new_seq.append(aa)

                logger.debug('Add allele seq: %s(%s), %s' % (ns_allele, allele, ''.join(new_seq)))
                new_od[ns_allele] = new_seq

        df = pd.DataFrame.from_dict(new_od, orient='index')
        df = df.fillna(GAP)

        # rep_seq에서 GAP에 해당하는 컬럼을 지운다.
        df = df.loc[:, df.iloc[0] != GAP]
        df.columns = list(range(df.shape[1]))
        return MultipleSequenceAlignment(df=df)


class AAPairwiseScoreMatrix(object):
    def __init__(self, df=None):
        if df is not None:
            if not all([(c in AA_CODES) for c in df.index.values]):
                raise ValueError('Unknown AA in df.index != %s' % df.index.values)
            if not all([(c in AA_CODES) for c in df.columns.values]):
                raise ValueError('Unknown AA in df.columns != %s' % df.columns.values)
            if df.shape[0] != df.shape[1]:
                raise ValueError('Non-symetric matrix: %s' % str(df.shape))
        self._df = df

    def __len__(self):
        return self._df.shape[0]

    @property
    def aas(self):
        return self._df.index.values.tolist()

    def scores(self, aa=None):
        return self._df.loc[aa, :].values


class AASubstScoreMatrix(AAPairwiseScoreMatrix):
    df_blosum = None

    def __init__(self, df=None):
        super().__init__(df)

    def subst_aa(self, aa=None, prob_range=(0.001, 0.999)):
        aas = self.aas
        probs = NumpyUtils.to_probs(self.scores(aa), prob_range=prob_range)
        probs[aas.index(aa)] = 0.
        probs = probs / probs.sum(axis=0, keepdims=True)
        # logger.debug('Probs for %s: %s' % (aa, list(zip(aas, probs))))
        st_aa = np.random.choice(aas, 1, p=probs)[0]
        return st_aa

    @classmethod
    def from_blosum(cls, fn_blosum='../data/blosum/blosum62.blast.new.iupac'):
        if cls.df_blosum is None:
            df = pd.read_table(fn_blosum, header=6, index_col=0, sep=' +')
            target_aas = list(filter(lambda c: c in df.index.values, AA_CODES))
            df = df.loc[target_aas, target_aas]
            df = df.transpose()
            # df.index = target_aas
            # df.columns = target_aas
            cls.df_blosum = df

        return cls(df=cls.df_blosum)


class SeqAASubstitutor(object):
    def subst_aa_at(self, seq, pos):
        raise NotImplementedError('Not implemented yet')


class PSSMSeqAASubstitutor(SeqAASubstitutor):
    def __init__(self, pssm=None):
        self.pssm = pssm

    def subst_aa_at(self, seq, pos):
        if len(seq) != len(self.pssm):
            raise ValueError('len(seq) should be equal to len(pssm)' % (len(seq), len(self.pssm)))

        aa = seq[pos]
        return self.pssm.subst_aa_at(pos, aa)


class AASustMatSeqAASubstitutor(SeqAASubstitutor):
    def __init__(self, aasubst_mat=None):
        self.aasubst_mat = aasubst_mat

    def subst_aa_at(self, seq, pos):
        aa = seq[pos]
        return self.aasubst_mat.subst_aa(aa)


class AASeqMutator(object):
    def __init__(self, mut_ratio=0.1, mut_probs=None, aasubstor=None, reverse=False):
        self.mut_ratio = mut_ratio
        self.mut_probs = mut_probs  # [deletion, substitution]
        self.aasubstor = aasubstor
        if self.aasubstor is None:
            self.aasubstor = AASustMatSeqAASubstitutor(aasubst_mat=AASubstScoreMatrix.from_blosum())
        self.reverse = reverse

    def mutate(self, seq):
        muted_seq = list(seq)
        muted_postions = self.choice_mut_positions(seq)

        # if self.reverse:
        #     l_seq = len(seq)
        #     n_muts = round(l_seq * self.mut_ratio)
        #     muted_postions = list(filter(lambda p: p not in muted_postions, range(len(seq))))
        #     if n_muts < len(muted_postions):
        #         muted_postions = np.random.choice(muted_postions, n_muts, replace=False).tolist()

        orig_aas = []
        for pos in muted_postions:
            orig_aas.append(seq[pos])
            r = random()
            if r < self.mut_probs[0]:  # Deletion
                muted_seq[pos] = GAP
            elif r < (self.mut_probs[0] + self.mut_probs[1]):  # Substitution
                muted_seq[pos] = self.aasubstor.subst_aa_at(seq, pos)
        return muted_seq, muted_postions, orig_aas

    def choice_mut_positions(self, seq):
        raise NotImplementedError()


class UniformAASeqMutator(AASeqMutator):
    def __init__(self, mut_ratio=0.1, mut_probs=None, aasubstor=None, reverse=False):
        super().__init__(mut_ratio, mut_probs, aasubstor, reverse)

    def choice_mut_positions(self, seq):
        seqlen = len(seq)
        return sorted(np.random.choice(seqlen, round(seqlen * self.mut_ratio), replace=False))


class NormalAASeqMutator(AASeqMutator):
    def __init__(self, mut_ratio=0.1, mut_probs=None, aasubstor=None, reverse=False):
        super().__init__(mut_ratio, mut_probs, aasubstor, reverse)

    def choice_mut_positions(self, seq):
        seqlen = len(seq)
        mu = sigma = seqlen / 2
        pos = np.around(np.random.normal(mu, sigma, seqlen)).astype(int)
        pos[pos < 0] = 0
        pos[pos >= seqlen] = seqlen - 1
        logger.debug(f'Random choice of mut positions with normal dist.: seqlen={seqlen}, positions={pos}')
        return sorted(np.random.choice(pos, round(seqlen * self.mut_ratio), replace=False))


class CalisImmunogenicAASeqMutator(AASeqMutator):
    _CALIS_IMMUNOGENIC_AASCORE_MAP = {'A': 0.583, 'C': 0.370, 'D': 0.544, 'E': 0.722, 'F': 0.761,
                                      'G': 0.571, 'H': 0.567, 'I': 0.798, 'K': 0.0, 'L': 0.468,
                                      'M': 0.092, 'N': 0.479, 'P': 0.468, 'Q': 0.228, 'R': 0.612,
                                      'S': 0.115, 'T': 0.582, 'V': 0.588, 'W': 1.0, 'Y': 0.485}
    _CALIS_IMMUNOGENIC_POS_WEIGHTS = [0.00, 0.00, 0.10, 0.31, 0.30, 0.29, 0.26, 0.18, 0.00]

    def __init__(self, mut_ratio=0.1, mut_probs=None, aasubstor=None, reverse=False, default_aa_weight=0.05,
                 default_pos_weight=0.3):
        """
        Mutate an AA seq using the observed immunogenic scores of MHC-I presented peptides by {Calis:2013cp}
        Ref: 1. Calis, J. et al, Properties of MHC Class I Presented Peptides That Enhance Immunogenicity.
        PLoS Comput Biol 2013, 9, e1003266–14.
        """
        super().__init__(mut_ratio, mut_probs, aasubstor, reverse)

        self.default_aa_weight = default_aa_weight
        self.default_pos_weight = default_pos_weight

    def choice_mut_positions(self, seq):
        seqlen = len(seq)
        n_pos = round(seqlen * self.mut_ratio)
        position_weights = self._CALIS_IMMUNOGENIC_POS_WEIGHTS
        if seqlen > 9:
            position_weights = position_weights[:5] + \
                               ((seqlen - 9) * [self.default_pos_weight]) + position_weights[5:]
        elif seqlen < 9:
            position_weights = position_weights[(9 - seqlen):]

        # Get position-specific scores
        scores = np.zeros(seqlen)
        for i, aa in enumerate(seq):
            scores[i] = self._CALIS_IMMUNOGENIC_AASCORE_MAP.get(aa, self.default_aa_weight) * position_weights[i]

        if self.reverse:
            scores = NumpyUtils.align_by_rrank(scores)

        probs = NumpyUtils.to_probs(scores)
        return sorted(np.random.choice(seqlen, n_pos, replace=False, p=probs))


### Tests
class IupacAminoAcidTest(BaseTest):
    def test_index(self):
        for i, aa in enumerate(IupacAminoAcid):
            self.assertEqual(i, aa.index)


class FastaSeqParserTest(BaseTest):
    class MyParserListener(FastaSeqParser.Listener):
        def __init__(self):
            self.headers = []
            self.seqs = []

        def on_seq_read(self, header=None, seq=None):
            print('Header:%s, Seq:%s' % (header, seq))
            self.headers.append(header)
            self.seqs.append(seq)

    #     def setUp(self):
    #         self.parser = FastaSeqParser()
    def test_parse(self):
        parser = FastaSeqParser()
        listener = FastaSeqParserTest.MyParserListener()

        parser.add_parse_listener(listener)
        seqs = ['AAA', 'BBB', 'CCC']
        headers = ['HA', 'HB', 'HC']
        fasta = format_fa(seqs=seqs, headers=headers)

        parser.parse(StringIO(fasta))

        self.assertTrue(np.array_equal(headers, listener.headers))
        self.assertTrue(np.array_equal(seqs, listener.seqs))


class PositionSpecificScoringMatrixTest(BaseTest):
    def setUp(self):
        self.values = np.random.randint(100, size=(len(AA_CODES), 3))
        # logger.debug('setUp: values: %s' % self.values)

    def test_error_for_values_first_shape(self):
        with self.assertRaises(ValueError):
            PositionSpecificScoringMatrix(values=np.array([[1, 2], [3, 4]]))

    def test_dtype_of_values_is_float(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        self.assertTrue(pssm.values.dtype == np.float32)

    def test_raise_error_when_zero_cols(self):
        with self.assertRaises(ValueError) as ctx:
            self.values[:, [0, 2]] = 0
            PositionSpecificScoringMatrix(values=self.values)

    def test_raise_error_when_a_nan(self):
        with self.assertRaises(ValueError) as ctx:
            self.values[0, 1] = np.nan
            PositionSpecificScoringMatrix(values=self.values)

    # def test_when_all_of_cols_are_zeros(self):
    #     cols = [1, 2]
    #     self.values[:, cols] = 0.
    #     val = 1./self.values.shape[0]
    #     pssm = PositionSpecificScoringMatrix(values=self.values)
    #     self.assertTrue(np.all(pssm.values[:, cols] == val))

    def test_ps_freq_scores(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        self.assertArrayEqual(self.values[:, 0], pssm.ps_freq_scores(0))
        self.assertArrayEqual(self.values[:, 1], pssm.ps_freq_scores(1))
        self.assertArrayEqual(self.values[:, 2], pssm.ps_freq_scores(2))

        np.testing.assert_almost_equal(np.sum(pssm.ps_freq_scores(0, as_prob=True)), 1)
        np.testing.assert_almost_equal(np.sum(pssm.ps_freq_scores(1, as_prob=True)), 1)
        np.testing.assert_almost_equal(np.sum(pssm.ps_freq_scores(2, as_prob=True)), 1)

    def test_aa_freq_scores(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        for i, aa in enumerate(AMINO_ACID.codes()):
            self.assertArrayEqual(self.values[i], pssm.aa_freq_scores(aa=aa))

    def test_conservation_probs(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)

        probs = pssm.conservation_probs(prob_range=(0.001, 0.999))
        print(pssm.values)
        print(probs)
        self.assertTrue(np.all(probs > 0))
        np.testing.assert_almost_equal(probs.sum(), 1)

    def test_variability_probs(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        c_scores = pssm.conservation_probs()
        actual = pssm.variability_probs()
        print('c_scores: %s' % c_scores)
        print('v_scores: %s' % actual)

        self.assertEqual(np.argmax(c_scores), np.argmin(actual))
        self.assertEqual(np.argmin(c_scores), np.argmax(actual))
        self.assertTrue(np.all(actual > 0))
        self.assertTrue(actual.sum() == 1)

    def test_conservation_probs_highly_variable(self):
        self.values[:, :] = 1
        self.values[10:12, 2] = 2

        pssm = PositionSpecificScoringMatrix(values=self.values)
        actual = pssm.conservation_probs()
        print(pssm.values)
        print(actual)
        self.assertEqual(actual[0], actual[1])
        self.assertTrue(actual[1] < actual[2])
        self.assertTrue(actual.sum() == 1)

    def test_variability_scores_highly_variable(self):
        self.values[:, :] = 1
        self.values[10:12, 2] = 2

        pssm = PositionSpecificScoringMatrix(values=self.values)
        actual = pssm.variability_probs()
        print(pssm.values)
        print(actual)
        self.assertEqual(actual[0], actual[1])
        self.assertTrue(actual[1] > actual[2])
        self.assertTrue(actual.sum() == 1)

    def test_conservation_probs_with_highly_conserved(self):
        self.values[:, 0] = 0
        self.values[:, 1] = 0
        self.values[:, 2] = 1

        diff = 10
        self.values[10, 0] = diff
        self.values[10, 1] = diff

        pssm = PositionSpecificScoringMatrix(values=self.values)
        actual = pssm.conservation_probs()
        print(pssm.values)
        print(actual)
        self.assertEqual(actual[0], actual[1])
        self.assertTrue(actual[1] > actual[2])
        self.assertTrue(actual.sum() == 1)

    def test_variability_probs_with_highly_conserved(self):
        self.values[:, 0] = 0
        self.values[:, 1] = 0
        self.values[:, 2] = 1

        diff = 10
        self.values[10, 0] = diff
        self.values[10, 1] = diff

        pssm = PositionSpecificScoringMatrix(values=self.values)
        actual = pssm.variability_probs()
        print(pssm.values)
        print(actual)
        self.assertEqual(actual[0], actual[1])
        self.assertTrue(actual[1] < actual[2])
        self.assertTrue(actual.sum() == 1)

    def test_choice_variable_positions(self):
        values = np.random.randint(100, size=(len(AA_CODES), 12))
        pssm = PositionSpecificScoringMatrix(values=values)
        v_probs = pssm.variability_probs()
        positions = np.argsort(v_probs)[::-1]
        n_pos = 6
        expected = positions[:n_pos]
        n_try = 100
        n_inters = []
        for i in range(n_try):
            actual = pssm.choice_variable_positions(n_pos)
            self.assertEqual(len(expected), len(actual))
            inter = np.intersect1d(expected, actual)
            print('%s try: expected: %s, actural: %s, inter: %s(%s)' % (i, expected, actual, inter, len(inter)))
            n_inters.append(len(inter))

        m_inter = np.mean(n_inters)
        print('Mean of n_inters: %s' % m_inter)
        self.assertTrue(m_inter > (n_pos / 2))

    def test_choice_variable_positions_highly_conserved(self):
        values = np.random.randint(10, size=(len(AA_CODES), 12))
        values[:, :10] = 0
        values[8, :10] = 100
        pssm = PositionSpecificScoringMatrix(values=values)
        v_probs = pssm.variability_probs()
        print(pssm.values)
        print(v_probs)

        self.assertTrue(np.all(v_probs[:10] == v_probs[0]))
        self.assertTrue(v_probs[10] > v_probs[0])
        self.assertTrue(v_probs[11] > v_probs[0])

        positions = np.argsort(v_probs)[::-1]
        n_pos = 6
        expected = positions[:n_pos]
        n_try = 100
        n_inters = []
        for i in range(n_try):
            actual = pssm.choice_variable_positions(n_pos)
            self.assertEqual(len(expected), len(actual))
            inter = np.intersect1d(expected, actual)
            print('%s try: expected: %s, actural: %s, inter: %s(%s)' % (i, expected, actual, inter, len(inter)))
            n_inters.append(len(inter))

        m_inter = np.mean(n_inters)
        print('Mean of n_inters: %s' % m_inter)
        self.assertTrue(m_inter > (n_pos / 2))

    def test_subst_aa_at(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        seq = rand_aaseq(len(pssm))
        for i, aa in enumerate(seq):
            new_aa = pssm.subst_aa_at(i, aa)
            print('%s: %s=>%s' % (i, aa, new_aa))
            self.assertNotEqual(aa, new_aa)

        pssm.values[:, 0] = 0
        pssm.values[0, 0] = 10
        pssm.values[:, 1] = 0
        pssm.values[2, 1] = 20
        pssm.values[5, 1] = 30
        pssm.values[:, 2] = 0
        pssm.values[4, 2] = 30

        seq = rand_aaseq(len(pssm))

        print(pssm.row_index)
        print(pssm.values)
        print('seq:', seq)

        for i, aa in enumerate(seq):
            new_aa = pssm.subst_aa_at(i, aa)
            print('%s: %s=>%s' % (i, aa, new_aa))
            self.assertNotEqual(aa, new_aa)

    def test_subst_aa_at_when_all_the_same_freq(self):
        self.values[:, 0] = 1
        self.values[:, 1] = 2
        self.values[:, 2] = 3
        pssm = PositionSpecificScoringMatrix(values=self.values)
        seq = rand_aaseq(len(pssm))
        for i, aa in enumerate(seq):
            new_aa = pssm.subst_aa_at(i, aa)
            print('%s: %s=>%s' % (i, aa, new_aa))
            self.assertNotEqual(aa, new_aa)

    def test_extend_length(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        expected_len = self.values.shape[1]
        expected_len += 2

        # print('Before:', pssm.values)
        pssm.extend_length(expected_len)
        # print('After:', pssm.values)
        self.assertEqual(expected_len, len(pssm))

    def test_shrink_length(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        expected_len = self.values.shape[1]
        expected_len -= 1

        print('Before:', pssm.values)
        pssm.shrink_length(expected_len)
        print('After:', pssm.values)
        self.assertEqual(expected_len, len(pssm))

    def test_mm_scale(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        min = 0.001
        max = 0.999

        pssm.mm_scale(score_range=(min, max), inplace=True)

        for pos in range(len(pssm)):
            scores = pssm.ps_freq_scores(pos)
            self.assertTrue(np.alltrue((np.around(scores, decimals=3) >= min) &
                                       (np.around(scores, decimals=3) <= max)))

        # Always updated to the same values even if multiple mm_scale is called multiple times
        old_values = np.copy(pssm.values)
        pssm.mm_scale(score_range=(min, max), inplace=True)
        np.testing.assert_equal(np.around(old_values, decimals=3),
                                np.around(pssm.values, decimals=3))

    def test_seq_score(self):
        seq = 'AMN'
        pssm = PositionSpecificScoringMatrix(values=self.values)
        min = 0
        max = 1
        pssm.mm_scale(score_range=(min, max), inplace=True)

        expected = 0
        for pos, aa in enumerate(seq):
            ai = pssm.atoi[aa]
            expected += pssm.values[ai, pos]
        expected = expected / len(seq)

        self.assertAlmostEquals(expected, pssm.seq_score(seq))

    def test_rand_seq(self):
        pssm = PositionSpecificScoringMatrix(values=self.values)
        min = 0.001
        max = 0.999
        pssm.mm_scale(score_range=(min, max), inplace=True)

        success = 0
        for i in range(100):
            seq = pssm.rand_seq()
            score = pssm.seq_score(seq)
            print('rand_seq: %s, score: %s' % (seq, score))
            if score > (max * 0.5):
                success += 1
        print('>>>Success: %s' % success)
        self.assertTrue(success > 80)

        print('>>>When scores are reversed')
        success = 0
        for i in range(100):
            seq = pssm.rand_seq(reverse_scores=True)
            score = pssm.seq_score(seq)
            print('rand_seq: %s, score: %s' % (seq, score))
            if score < (max * 0.5):
                success += 1
        print('>>>Success: %s' % success)
        self.assertTrue(success > 60)

    def test_similarity_score(self):
        pm1 = PositionSpecificScoringMatrix(values=self.values)
        pm2 = PositionSpecificScoringMatrix(values=self.values)

        method = 'pearsonr'
        score = pm1.similarity_score(pm2, method=method)
        print(score)
        np.testing.assert_almost_equal(score, 1)

        pm2.values[:, 0] = pm2.values[:, 0][::-1]
        score = pm1.similarity_score(pm2, method=method)
        print(score)
        self.assertTrue(score < 1)

        method = 'kld'
        pm2.values[:, 0] = pm2.values[:, 0][::-1]
        score = pm1.similarity_score(pm2, method=method)
        print(score)
        np.testing.assert_almost_equal(score, 1)

        pm2.values[:, 0] = pm2.values[:, 0][::-1]
        score = pm1.similarity_score(pm2, method=method)
        print(score)
        self.assertTrue(score < 1)

        method = 'euclidean'
        pm2.values[:, 0] = pm2.values[:, 0][::-1]
        score = pm1.similarity_score(pm2, method=method)
        print(score)
        np.testing.assert_almost_equal(score, 1)

        pm2.values[:, 0] = pm2.values[:, 0][::-1]
        score = pm1.similarity_score(pm2, method=method)
        print(score)
        self.assertTrue(score < 1)

    def test_choice_seqs(self):
        epitopes = FileUtils.pkl_load('../data/bglib/sars2_epitope_9mer.pkl')
        source_seqs = read_fa_from_file('../data/bglib/ref_proteome_human.fa')[1]
        source_seqs += read_fa_from_file('../data/bglib/ref_proteome_sars2.fa')[1]
        pssm = MultipleSequenceAlignment.from_seqs(epitopes).pssm()
        pssm.mm_scale(inplace=True)

        scores = []
        for epitope in epitopes:
            score = pssm.seq_score(epitope)
            scores.append(score)
            print('Epitope %s score: %s' % (epitope, score))
            # self.assertTrue(score > 0.5, epitope)
        print('Mean of scores: %s' % np.mean(scores))

        N = 50000
        threshold = 0.47
        compare_op = np.less
        result_seqs = pssm.choice_seqs(N=N, source_seqs=source_seqs, threshold=threshold, compare_op=compare_op)
        print(len(result_seqs), result_seqs)

        for seq in result_seqs:
            self.assertTrue(seq not in epitopes, seq)
            score = pssm.seq_score(seq)
            self.assertTrue(compare_op(score, threshold))


class MultipleSequenceAlignmentTest(BaseTest):
    def setUp(self):
        super().setUp()
        self.valid_fasta_df = pd.DataFrame(data=[
            ['A', 'M', 'N', 'Q', 'P'],
            ['A', 'K', 'D', 'L', '-'],
            ['A', 'M', 'D', '-', 'P']
        ], index=['seq1', 'seq2', 'seq3'])

        self.invalid_fasta_df = pd.DataFrame(data=[
            ['B', 'M', '.', 'Q', 'P'],
            ['A', '*', 'D', 'L', '-'],
            ['A', 'M', 'D', '-', 'P']
        ], index=['seq1', 'seq2', 'seq3'])

    def write_fasta_df(self, df, fn):
        indices = df.index.values
        seqs = [''.join(df.loc[index, :]) for index in indices]
        write_fa(fn, seqs=seqs, headers=indices)

    def test_init_msa(self):
        msa = MultipleSequenceAlignment(self.valid_fasta_df)
        self.assertIsNotNone(msa)
        self.assertTrue(msa.has_gap)
        with self.assertRaises(ValueError):
            MultipleSequenceAlignment(self.invalid_fasta_df)

    def test_from_fasta(self):
        fn = '../tmp/test.fa'
        self.write_fasta_df(self.valid_fasta_df, fn)
        self.assertEqual(MultipleSequenceAlignment(self.valid_fasta_df), MultipleSequenceAlignment.from_fasta(fn))

    def test_from_seqs(self):
        seqs = ['AMNQP', 'AKDLM', 'ANDLV']
        msa = MultipleSequenceAlignment.from_seqs(seqs)

        self.assertTrue(msa is not None)
        self.assertArrayEqual(list(range(3)), msa.names)
        self.assertArrayEqual(list(range(5)), msa.positions)
        self.assertArrayEqual(list('AMNQP'), msa.seq(0))
        self.assertArrayEqual(list('AKDLM'), msa.seq(1))
        self.assertArrayEqual(list('ANDLV'), msa.seq(2))

        seqs = ['AMNQP', 'AKD', 'ANDL']
        msa = MultipleSequenceAlignment.from_seqs(seqs)

        self.assertTrue(msa is not None)
        self.assertArrayEqual(list(range(3)), msa.names)
        self.assertArrayEqual(list(range(5)), msa.positions)
        self.assertArrayEqual(list('AMNQP'), msa.seq(0))
        self.assertArrayEqual(list('AKD') + [GAP] * 2, msa.seq(1))
        self.assertArrayEqual(list('ANDL') + [GAP], msa.seq(2))

        with self.assertRaises(ValueError) as ctx:
            seqs = ['AMNQP', 'AKD', 'ANDL']
            msa = MultipleSequenceAlignment.from_seqs(seqs, fill_with_gap=False)

    def test_pssm(self):
        msa = MultipleSequenceAlignment(self.valid_fasta_df)
        expected_ri = AMINO_ACID.codes()
        pssm = msa.pssm()
        self.assertArrayEqual(expected_ri, pssm.row_index)
        expected_vals = np.zeros((len(expected_ri), self.valid_fasta_df.shape[1]))
        expected_vals[expected_ri.index('A'), 0] = 3
        expected_vals[expected_ri.index('M'), 1] = 2
        expected_vals[expected_ri.index('K'), 1] = 1
        expected_vals[expected_ri.index('N'), 2] = 1
        expected_vals[expected_ri.index('D'), 2] = 2
        expected_vals[expected_ri.index('Q'), 3] = 1
        expected_vals[expected_ri.index('L'), 3] = 1
        expected_vals[expected_ri.index('P'), 4] = 2
        self.assertTrue(np.array_equal(expected_vals, pssm.values))

        aa_positions = [0, 2, 3]
        pssm = msa.pssm(aa_positions=aa_positions)
        expected_vals = expected_vals[:, aa_positions]
        self.assertTrue(np.array_equal(expected_vals, pssm.values))

    def test_pssm_for_sub_msa(self):
        ['A', 'M', 'N', 'Q', 'P'],
        ['A', 'K', 'D', 'L', '-'],
        ['A', 'M', 'D', '-', 'P']

        msa = MultipleSequenceAlignment(self.valid_fasta_df)

        positions = [1, 3, 4]
        sub = msa.sub_msa(positions=positions)
        self.assertArrayEqual(msa.names, sub.names)
        self.assertArrayEqual(list(range(3)), sub.positions)

        pssm = sub.pssm()
        self.assertEqual(len(positions), len(pssm))
        expected_vals = np.zeros(pssm.values.shape)
        expected_vals[IupacAminoAcid.M.index, 0] = 2
        expected_vals[IupacAminoAcid.K.index, 0] = 1
        expected_vals[IupacAminoAcid.Q.index, 1] = 1
        expected_vals[IupacAminoAcid.L.index, 1] = 1
        expected_vals[IupacAminoAcid.P.index, 2] = 1

    def test_aas_at(self):
        msa = MultipleSequenceAlignment(self.valid_fasta_df)

        self.assertArrayEqual(['A', 'A', 'A'], msa.aas_at(0))
        self.assertArrayEqual(['M', 'K', 'M'], msa.aas_at(1))
        self.assertArrayEqual(['N', 'D', 'D'], msa.aas_at(2))
        self.assertArrayEqual(['Q', 'L', '-'], msa.aas_at(3))
        self.assertArrayEqual(['P', '-', 'P'], msa.aas_at(4))

    def test_seq(self):
        msa = MultipleSequenceAlignment(self.valid_fasta_df)

        self.assertArrayEqual(['A', 'M', 'N', 'Q', 'P'], msa.seq('seq1'))
        self.assertArrayEqual(['A', 'K', 'D', 'L', '-'], msa.seq('seq2'))
        self.assertArrayEqual(['A', 'M', 'D', '-', 'P'], msa.seq('seq3'))

        self.assertArrayEqual(['A', 'M', 'N', 'Q', 'P'], msa.iseq(0))
        self.assertArrayEqual(['A', 'K', 'D', 'L', '-'], msa.iseq(1))
        self.assertArrayEqual(['A', 'M', 'D', '-', 'P'], msa.iseq(2))

        #
        # self.assertEqual(len(expected_ri), pssm.values.shape[0])
        # self.assertEqual(pssm.values.shape[1], msa.values.shape[1])
        #
        # conv_scores = pssm.conservation_scores()
        # conv_scores /= conv_scores.sum(axis=0)
        # print(conv_scores)

    def test_sub_msa(self):
        msa = MultipleSequenceAlignment(self.valid_fasta_df)

        positions = [1, 3, 4]
        sub = msa.sub_msa(positions=positions)
        self.assertArrayEqual(msa.names, sub.names)
        self.assertArrayEqual(list(range(3)), sub.positions)

        self.assertArrayEqual(['M', 'Q', 'P'], sub.seq('seq1'))
        self.assertArrayEqual(['K', 'L', '-'], sub.seq('seq2'))
        self.assertArrayEqual(['M', '-', 'P'], sub.seq('seq3'))

        self.assertArrayEqual(['M', 'Q', 'P'], sub.iseq(0))
        self.assertArrayEqual(['K', 'L', '-'], sub.iseq(1))
        self.assertArrayEqual(['M', '-', 'P'], sub.iseq(2))

    def test_from_imgt_aln(self):
        MultipleSequenceAlignment._FN_MHC_MSA = '../data/mhcinfo/prot/{0}/{1}.sample.aln'

        msa = MultipleSequenceAlignment.from_imgt_msa('HLA', 'A')

        expected_names = ['HLA-A*01:01', 'HLA-A*01:02', 'HLA-A*01:03', 'HLA-A*80:01', 'HLA-A*80:02', 'HLA-A*80:04']

        self.assertArrayEqual(expected_names, msa.names)
        expected_seq = list('MAVMAPRTLLLLLSDQETRNMKAHSQTDRANL')
        self.assertArrayEqual(expected_seq, msa.seq('HLA-A*01:01'))

        expected_seq = list('MAVM---TLLLLLSDQETRNMKAHSQTDRANL')
        self.assertArrayEqual(expected_seq, msa.seq('HLA-A*01:02'))

        expected_seq = list('MAVMAPRTLLLLLSD-ETRNMKAHSQTDRANL')
        self.assertArrayEqual(expected_seq, msa.seq('HLA-A*01:03'))

        expected_seq = list('MAVMPPRTLLLLLSDEETRNVKAHSQTNRANL')
        self.assertArrayEqual(expected_seq, msa.seq('HLA-A*80:01'))

        expected_seq = list('MAVMAPRTLLLLLSDEETRNVKAHSQTDRVDL')
        self.assertArrayEqual(expected_seq, msa.seq('HLA-A*80:02'))

        expected_seq = list('MAVMAPRTLLLLLSDEETRNVKAHSQTNRENL')
        actual_seq = msa.seq('HLA-A*80:04')
        self.assertArrayEqual(expected_seq[:3], actual_seq[:3])
        self.assertNotEqual(expected_seq[3], actual_seq[3])
        self.assertNotEqual(expected_seq[4], actual_seq[4])
        self.assertArrayEqual(expected_seq[5:], actual_seq[5:])

    def test_from_imgt_aln_real(self):
        MultipleSequenceAlignment._FN_MHC_MSA = '../data/mhcinfo/prot/{0}/{1}.aln'

        target_genes = []
        for allele_name in self.target_classI_alleles:
            allele = MHCAlleleName.parse(allele_name)
            if (allele.species, allele.gene) not in target_genes:
                target_genes.append((allele.species, allele.gene))

        for species, gene in target_genes:
            print('>>>Loading MSA for %s-%s' % (species, gene))
            msa = MultipleSequenceAlignment.from_imgt_msa(species, gene)
            self.assertIsNotNone(msa)
            print('msa.names for %s-%s: %s' % (species, gene, msa.names))
            print('msa.positions for %s-%s: %s' % (species, gene, msa.positions))


class AASubstScoreMatrixTest(BaseTest):
    def setUp(self):
        super().setUp()

        self.subst_mat = AASubstScoreMatrix.from_blosum()

    def test_blosum_scores(self):
        for aa in AA_CODES:
            scores = self.subst_mat.scores(aa)
            self.assertIsNotNone(scores)
            self.assertEquals(len(self.subst_mat), len(scores))
            logger.debug('%s: %s' % (aa, scores))

    def test_subst_aa(self):
        aas = AA_CODES
        for i, aa in enumerate(aas):
            print('%s: %s' % (aa, list(zip(aas, self.subst_mat.scores(aa)))))
            st_aa = self.subst_mat.subst_aa(aa)
            self.assertIn(st_aa, aas)
            self.assertNotEqual(aa, st_aa)
            logger.debug('%s==>%s' % (aa, st_aa))

    def test_same_df_when_from_blosum(self):
        other = AASubstScoreMatrix.from_blosum()
        self.assertEqual(id(self.subst_mat._df), id(other._df))


class SeqAASustitutorTest(BaseTest):

    def test_pssm_subst_aa_at(self):
        pssm = PositionSpecificScoringMatrix(values=np.random.randint(100, size=(len(AA_CODES), 3)))
        seq = rand_aaseq(len(pssm))
        aasubstor = PSSMSeqAASubstitutor(pssm=pssm)
        for i, aa in enumerate(seq):
            new_aa = aasubstor.subst_aa_at(seq, i)
            logger.debug('%s: %s=>%s' % (i, aa, new_aa))
            self.assertNotEqual(aa, new_aa)

        pssm.values[:, 0] = 0
        pssm.values[pssm.atoi['A'], 0] = 10
        pssm.values[pssm.atoi['Q'], 0] = 5
        pssm.values[pssm.atoi['C'], 0] = 5

        pssm.values[:, 1] = 0
        pssm.values[pssm.atoi['F'], 1] = 20
        pssm.values[pssm.atoi['M'], 1] = 30

        pssm.values[:, 2] = 0
        pssm.values[pssm.atoi['I'], 2] = 30
        pssm.values[pssm.atoi['D'], 0] = 5
        pssm.values[pssm.atoi['B'], 0] = 10

        for i, aa in enumerate(seq):
            new_aa = aasubstor.subst_aa_at(seq, i)
            logger.debug('%s: %s=>%s' % (i, aa, new_aa))
            self.assertNotEqual(aa, new_aa)

    def test_aasubstmat_subst_aa_at(self):
        aasubst_mat = AASubstScoreMatrix.from_blosum()
        logger.debug(aasubst_mat._df)

        seq = rand_aaseq(3)
        aasubstor = AASustMatSeqAASubstitutor(aasubst_mat=aasubst_mat)
        for i, aa in enumerate(seq):
            new_aa = aasubstor.subst_aa_at(seq, i)
            logger.debug('%s: %s=>%s' % (i, aa, new_aa))
            self.assertNotEqual(aa, new_aa)


class AASeqMutatorTest(BaseTest):
    def setUp(self):
        self.seq = 'KVMYNKMLY'

    def test_mutate(self):
        mutators = [
            UniformAASeqMutator(mut_ratio=0.2, mut_probs=(0.7, 0.3)),
            CalisImmunogenicAASeqMutator(mut_ratio=0.2, mut_probs=(0.7, 0.3)),
        ]
        for mutator in mutators:
            muted_seq, muted_positions, orig_aas, = mutator.mutate(self.seq)
            self.assert_mutated(mutator.mut_ratio, muted_positions, muted_seq, orig_aas)

    def test_mutate_with_reverse(self):
        mut_ratio = 0.2
        mutator = CalisImmunogenicAASeqMutator(mut_ratio=mut_ratio, mut_probs=(0.7, 0.3))
        muted_seq, muted_positions, orig_aas, = mutator.mutate(self.seq)

        self.assert_mutated(mutator.mut_ratio, muted_positions, muted_seq, orig_aas)

        mutator = CalisImmunogenicAASeqMutator(mut_ratio=mut_ratio, mut_probs=(0.7, 0.3), reverse=True)
        r_muted_seq, r_muted_positions, r_orig_aas, = mutator.mutate(self.seq)
        self.assert_mutated(mutator.mut_ratio, r_muted_positions, r_muted_seq, r_orig_aas)

        self.assertTrue(all([p not in r_muted_positions for p in muted_positions]))

    def assert_mutated(self, mut_ratio, muted_positions, muted_seq, orig_aas):
        print('seq:      %s' % list(self.seq))
        print('muted_seq:%s' % muted_seq)
        print('orig_aas: %s' % orig_aas)
        print('muted_positions: %s' % muted_positions)
        self.assertEqual(len(muted_positions), round(len(self.seq) * mut_ratio))
        self.assertTrue(all([self.seq[pos] != muted_seq[pos] for pos in muted_positions]))
        for i, pos in enumerate(muted_positions):
            muted_seq[pos] = orig_aas[i]
        self.assertEqual(self.seq, ''.join(muted_seq))

class BioSeqFuncTest(BaseTest):
    def test_split_aaseq(self):
        self.assertListEqual(['ADCYLVMK'], split_aaseq('ADCYLVMK', window_size=9))
        self.assertListEqual(['ADCYLVMKV'], split_aaseq('ADCYLVMKV', window_size=9))
        self.assertListEqual(['ADCYLVMKV', 'DCYLVMKVL', 'CYLVMKVLN', 'YLVMKVLND'],
                             split_aaseq('ADCYLVMKVLND', window_size=9))
        self.assertListEqual(['ADCYLVMKV', 'DCYLVMKVL', 'CYLVMKVLN', 'YLVMKVLND', 'LVMKVLNDV'],
                             split_aaseq('ADCYLVMKVLNDV', window_size=9))

    def test_needle_aaseq_pair(self):
        results = needle_aaseq_pair('ADCKKKMKY', 'ADCKKRMKY',
                                    output_identity=True,
                                    output_similarity=True)
        print(results)
        self.assertTrue(results['similarity'] > 0.9)
        results = needle_aaseq_pair('AAADDCFLAFCASSSLADYRYEQYF', 'MDFYLCFLAFCALLLLLDLLLKQYL')
        print(results)
        self.assertTrue(results['similarity'] < 0.5)

if __name__ == '__main__':
    unittest.main()
