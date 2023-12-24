from itertools import repeat

import copy
import os
import unittest
from enum import auto, IntEnum
from multiprocessing import Pool
import argparse
import numpy as np
import pandas as pd
import logging.config
from collections import OrderedDict
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer
import datasets as hf_dataset
from tqdm import tqdm

from gentcr.bioseq import IupacAminoAcid, is_valid_aaseq, GAP, UniformAASeqMutator, CalisImmunogenicAASeqMutator, \
    FixedPosAASeqMutator
from gentcr.mhcdomain import PanMHCIContactDomain
from gentcr.common import StrEnum, FileUtils, StrUtils, basename, BaseTest
from gentcr.mhcnc import MHCAlleleName, ALLELE_SEP

# Logger
logger = logging.getLogger('gentcr')


class EpitopeTargetComponent(IntEnum):
    UNK = 0
    EPITOPE = 1
    MHC = 2
    TCR = 3

    def __str__(self):
        return self.name

    @classmethod
    def from_str(cls, s):
        s = s.upper()
        for m in cls:
            if m.name == s:
                return m
        return None

    @classmethod
    def values(cls):
        return [c.value for c in cls]


IC50_THRESHOLD = 500
DEFAULT_IC50_THRESHOLDS = (IC50_THRESHOLD * 0.2, IC50_THRESHOLD, IC50_THRESHOLD * 10)


class BindLevel(IntEnum):
    NEGATIVE = 0
    POSITIVE_LOW = 1
    POSITIVE = 2
    POSITIVE_HIGH = 3

    def __str__(self):
        return self.name

    @classmethod
    def is_binder(cls, level):
        return level > 0

    @classmethod
    def from_str(cls, s):
        s = s.upper()
        for m in cls:
            if m.name == s:
                return m
        return None

    @classmethod
    def from_ic50(cls, x, thresholds=DEFAULT_IC50_THRESHOLDS):
        if x < thresholds[0]:
            return cls.POSITIVE_HIGH
        elif x < thresholds[1]:
            return cls.POSITIVE
        elif x < thresholds[2]:
            return cls.POSITIVE_LOW
        else:
            return cls.NEGATIVE

    @classmethod
    def values(cls):
        return [c.value for c in cls]


class AssayType(StrEnum):
    MHC = 'mhc'
    TCELL = 'tcell'
    BCELL = 'bcell'
    TCR = 'tcr'

    @classmethod
    def values(cls):
        return [c.value for c in cls]


class EpitopeTargetDataset(Dataset):
    class ColumnName(StrEnum):
        epitope_species = auto()
        epitope_gene = auto()
        epitope_seq = auto()
        orig_epitope_seq = auto()
        epitope_start = auto()
        epitope_end = auto()
        epitope_len = auto()
        mhc_allele = auto()
        mhc_seq = auto()
        orig_mhc_seq = auto()
        cdr3b_seq = auto()
        orig_cdr3b_seq = auto()
        cdr3b_len = auto()
        ref_id = auto()
        source = auto()
        bind_level = auto()

        @classmethod
        def values(cls):
            return [c.value for c in cls]

    # Filters
    class Filter(object):
        def filter(self, df):
            raise NotImplementedError()

    class NotDuplicateFilter(Filter):
        def filter(self, df):
            logger.info('Drop duplicates with the same index')
            df = df[~df.index.duplicated()]
            logger.info('Current df.shape: %s' % str(df.shape))
            return df

    class MoreThanCDR3bNumberFilter(Filter):
        def __init__(self, cutoff=None):
            self.cutoff = cutoff

        def filter(self, df):
            if self.cutoff and self.cutoff > 0:
                logger.info('Select all epitope with at least %s CDR3B sequences' % self.cutoff)
                tmp = df[CN.epitope_seq].value_counts()
                tmp = tmp[tmp >= self.cutoff]
                df = df[df[CN.epitope_seq].map(lambda x: x in tmp.index)]
                logger.info('Current df.shape: %s' % str(df.shape))
            return df

    class QueryFilter(Filter):
        def __init__(self, query=None):
            self.query = query

        def filter(self, df):
            if self.query is not None:
                logger.info("Selecting all epitopes by query: %s" % self.query)
                df = df.query(self.query, engine='python')
                logger.info('Current df.shape: %s' % str(df.shape))
            return df

    class ValidMHCAlleleFilter(Filter):
        def filter(self, df):
            logger.info('Filtering valid MHC allele names')
            df = df[
                df[CN.mhc_allele].map(lambda an: pd.notnull(an)
                                                 and len(an) > 0
                                                 and (all(MHCAlleleName.is_valid(an)) if MHCAlleleName.is_multi(an)
                                                      else MHCAlleleName.is_valid(an)))
            ]
            logger.info('Current df.shape: %s' % str(df.shape))

    class EpitopeTargetSeqMutator(object):
        def __init__(self,
                     bind_target=EpitopeTargetComponent.TCR,
                     epitope_seq_mutator=None,
                     target_seq_mutator=None,
                     n_mutations=1):
            self.bind_target = bind_target
            self.epitope_seq_mutator = epitope_seq_mutator
            self.target_seq_mutator = target_seq_mutator
            self.n_mutations = n_mutations

        def mutate(self, df=None, args=None):
            logger.info(f'Start to mutate epitope and target sequences in MAIN proc, df_source.shape: {df.shape}')
            n_workers = args.n_workers if args else 1
            if n_workers > 1:
                with Pool(processes=n_workers) as pool:
                    params = zip(np.array_split(df, n_workers))
                    results = pool.starmap(self._mutate, params)
                df = pd.concat(results)
            else:
                df = self._mutate(df)
            logger.info(f'Done to mutate epitope and target sequences in MAIN proc, df.shape: {df.shape}')
            return df

        def _mutate(self, df):
            logger.info(f'Start to mutate epitope and target sequences in SUB proc, df.shape: {df.shape}')
            new_df = df.copy()
            for index, row in tqdm(df.iterrows(), total=len(df)):
                for mi in range(self.n_mutations):
                    new_row = row.copy()
                    if self.epitope_seq_mutator:
                        epitope_seq = new_row[CN.epitope_seq]
                        new_row[CN.orig_epitope_seq] = epitope_seq
                        new_row[CN.epitope_seq] = ''.join(self.epitope_seq_mutator.mutate(epitope_seq)[0])
                    if self.target_seq_mutator:
                        if self.bind_target == EpitopeTargetComponent.MHC:
                            target_seq = new_row[CN.mhc_seq]
                            new_row[CN.orig_mhc_seq] = target_seq
                            new_row[CN.mhc_seq] = ''.join(self.target_seq_mutator.mutate(target_seq)[0])
                        else:
                            target_seq = new_row[CN.cdr3b_seq]
                            new_row[CN.orig_cdr3b_seq] = target_seq
                            new_row[CN.cdr3b_seq] = ''.join(self.target_seq_mutator.mutate(target_seq)[0])
                    if mi == 0:
                        new_df.loc[index] = new_row
                        logger.debug(f'Update mutated row: {new_row}')
                    else:
                        # new_df = pd.concat([new_df, new_row.to_frame().T], ignore_index=True)
                        new_df.loc[len(new_df)] = new_row
                        logger.debug(f'Append mutated row: {new_row}')

            new_df.index = new_df.apply(lambda r: EpitopeTargetDataset._get_index(r, bind_target=self.bind_target), axis=1)
            logger.info(f'Done to mutate epitope and target sequences in SUB proc, new_df.shape: {new_df.shape}')
            return new_df

    #############
    FN_DATA_CONFIG = '../config/data.json'
    _configs = None

    def __init__(self, config):
        self.config = config
        self._filters = None
        self._seq_mutator = None
        self._mhc_domain = None
        self.df = None

    @property
    def name(self):
        return self.config['name']

    @property
    def description(self):
        return self.config.get('description')

    @property
    def bind_target(self):
        raise EpitopeTargetComponent.MHC

    @property
    def fn_output_csv(self):
        return self.get_fn_output_csv(self.name)

    @property
    def fn_summary_csv(self):
        return self.get_fn_summary_csv(self.name)

    @property
    def filters(self):
        if self._filters is None:
            self._filters = self._create_filters()
        return self._filters

    @property
    def seq_mutator(self):
        if self._seq_mutator is None:
            self._seq_mutator = self._create_seq_mutator()
        return self._seq_mutator

    @property
    def is_mutated(self):
        return self.seq_mutator is not None

    @property
    def mhc_domain(self):
        if self._mhc_domain is None and self.bind_target == EpitopeTargetComponent.MHC:
            self._mhc_domain = self._create_mhc_domain()
        return self._mhc_domain

    @property
    def max_epitope_len(self):
        return self.df[CN.epitope_len].max()

    @property
    def max_target_len(self):
        if self.bind_target == EpitopeTargetComponent.MHC:
            return len(self.mhc_domain.all_hla_sites())
        else:
            return self.df[CN.cdr3b_len].max()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self._get_row(index)
        epitope_seq = row[CN.epitope_seq]
        orig_epitope_seq = row[CN.orig_epitope_seq]
        if self.bind_target == EpitopeTargetComponent.MHC:
            target_seq = row[CN.mhc_seq]
            orig_target_seq = row[CN.orig_mhc_seq]
        else:
            target_seq = row[CN.cdr3b_seq]
            orig_target_seq = row[CN.orig_cdr3b_seq]

        binder = 1 if row[CN.bind_level] > 0 else 0
        return (epitope_seq, target_seq), (orig_epitope_seq, orig_target_seq), binder

    def _get_row(self, index):
        return self.df.iloc[index, :]

    def get_index(self, row, sep='_'):
        return self._get_index(row, self.bind_target, sep)

    def load_df(self, args=None):
        if (not os.path.exists(self.fn_output_csv)) or self.config.get('overwrite', False):
            logger.info(f'Loading data frame with config: {self.config}, args: {args}')
            df = self._load_df(args)

            # Mutate epitope and target sequences
            if self.seq_mutator:
                logger.info(f'Mutating epitope and target sequences')
                df = self.seq_mutator.mutate(df, args)
                logger.info(f'After mutation, df.shape: {df.shape}')

            # Apply filters if necessary
            if self.filters:
                for cur in self.filters:
                    logger.info(f'Applying filter: {cur}')
                    df = cur.filter(df)
                    logger.info(f'After filtering, df.shape: {df.shape}')

            # Shuffle data
            if self.config.get('shuffle', False):
                logger.info(f'Shuffling data')
                df = df.sample(frac=1)
                logger.info(f'After shuffling, df.shape: {df.shape}')

            logger.info(f'Done to load data frame. Saving to {self.fn_output_csv}, df.shape: {df.shape}')
            self.df = df
            self.save_csv()
        else:
            self.df = self._read_csv()
            logger.info(f'Loaded {self.name} data from {self.fn_output_csv}, df.shape: {self.df.shape}')

    def _read_csv(self):
        return pd.read_csv(self.fn_output_csv, index_col=0)

    def save_csv(self):
        self.df.to_csv(self.fn_output_csv)
        logger.info(f'The dataset saved to {self.fn_output_csv}')

        if self.config.get('summary', True):
            df_summary = self._summary_df(self.df)
            summary_csv = self.fn_summary_csv
            df_summary.to_csv(summary_csv)
            logger.info(f'Summary of the dataset saved to {summary_csv}')

    def delete_csv(self):
        os.unlink(self.fn_output_csv)
        os.unlink(self.fn_summary_csv)

    def train_test_split(self, test_size=0.2, shuffle=False, name_format='{name}.{split}', save_csv=False):
        train_df, test_df = train_test_split(self.df,
                                             test_size=test_size,
                                             shuffle=shuffle,
                                             stratify=(self.df[CN.bind_level].values if shuffle else None))
        train_ds = eval(self.config['type'])(copy.deepcopy(self.config))
        train_ds.config['name'] = name_format.format(name=self.name, split='train')
        train_ds.df = train_df.copy()
        test_ds = eval(self.config['type'])(copy.deepcopy(self.config))
        test_ds.config['name'] = name_format.format(name=self.name, split='test')
        test_ds.df = test_df.copy()
        if save_csv:
            train_ds.save_csv()
            test_ds.save_csv()
        return train_ds, test_ds

    def copy(self, save_csv=False):
        c = eval(self.config['type'])(copy.deepcopy(self.config))
        c.config['name'] = f'{self.name}.copy'
        c.df = self.df.copy()
        if save_csv:
            c.save_csv()
        return c

    def split_by_epitope(self, name_format='{name}.{epitope}', save_csv=False, as_dict=False):
        dss = OrderedDict() if as_dict else []
        for epitope, subtab in self.df.groupby(CN.epitope_seq):
            ds = eval(self.config['type'])(copy.deepcopy(self.config))
            ds.config['name'] = name_format.format(name=self.name, epitope=epitope)
            ds.df = subtab.copy()
            if save_csv:
                ds.save_csv()
            if as_dict:
                dss[epitope] = ds
            else:
                dss.append(ds)
        return dss

    def exclude_by(self, exclude_ds, target_cols=None, inplace=False):
        if inplace:
            result_ds = self
        else:
            result_ds = eval(self.config['type'])(copy.deepcopy(self.config))
            result_ds.df = self.df.copy()

        for col in target_cols:
            logger.info(f'Excluding {exclude_ds.name} data by column: {col} from {self.name}')
            if "index" == col:
                result_ds.df = result_ds.df[~np.isin(result_ds.df.index.values, exclude_ds.df.index.values)]

            else:
                result_ds.df = result_ds.df[~np.isin(result_ds.df[col].values, exclude_ds.df[col].values)]
            logger.info(f'Current {result_ds.name} data.shape: {result_ds.df.shape}')
        return result_ds

    def summary(self):
        if os.path.exists(self.fn_summary_csv):
            return pd.read_csv(self.fn_summary_csv, index_col=0)
        else:
            return self._summary_df(self.df)

    def to_hf(self, select_cols=None):
        df = self.df[select_cols] if select_cols else self.df
        return hf_dataset.Dataset.from_pandas(df, preserve_index=False)

    def _summary_df(self, df):
        rows = []
        if EpitopeTargetComponent.MHC == self.bind_target:
            colnames = [CN.source, CN.mhc_allele, CN.epitope_len, 'peptide', 'positive_high', 'positive',
                        'positive_low', 'negative']
            for (allele, pep_len), subtab in df.groupby([CN.mhc_allele, CN.epitope_len]):
                row = OrderedDict()
                row[CN.source] = ';'.join(subtab[CN.source].unique())
                row[CN.mhc_allele] = allele
                row[CN.epitope_len] = pep_len
                row['peptide'] = subtab.shape[0]
                row['positive_high'] = np.count_nonzero(subtab[CN.bind_level] == BindLevel.POSITIVE_HIGH)
                row['positive'] = np.count_nonzero(subtab[CN.bind_level] == BindLevel.POSITIVE)
                row['positive_low'] = np.count_nonzero(subtab[CN.bind_level] == BindLevel.POSITIVE_LOW)
                row['negative'] = np.count_nonzero(subtab[CN.bind_level] == BindLevel.NEGATIVE)
                rows.append(row)
        else:
            colnames = [CN.source, CN.epitope_species, CN.epitope_gene, CN.epitope_seq, 'cdr3_beta',
                        'positive', 'negative']
            df = df.fillna({CN.epitope_species: '', CN.epitope_gene: '', CN.epitope_start: -1, CN.epitope_end: -1})
            for (epitope,), subtab in df.groupby([CN.epitope_seq]):
                row = OrderedDict()
                row[CN.source] = ';'.join(subtab[CN.source].unique())
                row[CN.epitope_species] = ';'.join(subtab[CN.epitope_species].unique())
                row[CN.epitope_gene] = ';'.join(subtab[CN.epitope_gene].unique())
                row[CN.epitope_seq] = epitope
                row['cdr3_beta'] = subtab.shape[0]
                row['positive'] = np.count_nonzero(subtab[CN.bind_level] > 0)
                row['negative'] = np.count_nonzero(subtab[CN.bind_level] == BindLevel.NEGATIVE)
                rows.append(row)
        return pd.DataFrame(rows, columns=colnames)

    def _load_df(self, args=None):
        raise NotImplementedError()

    def _create_filters(self):
        filters = []
        if self.config.get('nodup', False):
            filters.append(self.NotDuplicateFilter())
        if 'query' in self.config:
            filters.append(self.QueryFilter(query=self.config['query']))
        if 'n_cdr3b_cutoff' in self.config:
            filters.append(self.MoreThanCDR3bNumberFilter(cutoff=self.config['n_cdr3b_cutoff']))
        return filters

    def _create_seq_mutator(self):
        if 'seq_mutator' in self.config:
            mut_config = copy.deepcopy(self.config['seq_mutator'].get('epitope'))
            epitope_seq_mutator = eval(mut_config.pop('type'))(**mut_config) if mut_config else None
            mut_config = copy.deepcopy(self.config['seq_mutator'].get('target'))
            target_seq_mutator = eval(mut_config.pop('type'))(**mut_config) if mut_config else None
            return self.EpitopeTargetSeqMutator(epitope_seq_mutator=epitope_seq_mutator,
                                                target_seq_mutator=target_seq_mutator,
                                                bind_target=self.bind_target,
                                                n_mutations=self.config['seq_mutator'].get('n_mutations', 1))
        else:
            return None

    def _create_mhc_domain(self):
        mhc_css = self.config.get('mhc_css', [(9, 'netmhcpan_mhci_9')])
        return PanMHCIContactDomain(css_list=mhc_css)

    @classmethod
    def get_fn_output_csv(cls, name):
        cls.load_configs()
        fn = cls._configs['output_csv_pattern'].replace('{name}', name)
        return f"{cls._configs['basedir']}/{fn}"

    @classmethod
    def get_fn_summary_csv(cls, name):
        fn_out_csv = cls.get_fn_output_csv(name)
        return f'{os.path.splitext(fn_out_csv)[0]}.summary.csv'

    @classmethod
    def load_configs(cls, reload=False):
        if cls._configs is None or reload:
            cls._configs = OrderedDict(FileUtils.json_load(cls.FN_DATA_CONFIG))

    @classmethod
    def dump_configs(cls):
        FileUtils.json_dump(cls._configs, cls.FN_DATA_CONFIG)

    @classmethod
    def get_config(cls, key):
        cls.load_configs()
        config = copy.deepcopy(cls._configs['ds_map'][key])
        if 'name' not in config:
            config['name'] = key
        return config

    @classmethod
    def from_key(cls, key, args=None):
        config = cls.get_config(key)
        the = eval(config.get('type', 'DefaultEpitopeTargetDataset'))(config=config)
        the.load_df(args)
        return the

    @classmethod
    def _get_index(cls, row, bind_target=None, sep='_'):
        epitope = row[CN.epitope_seq]
        target = row[CN.mhc_allele] if EpitopeTargetComponent.MHC == bind_target else row[CN.cdr3b_seq]
        return f'{epitope}{sep}{target}'


CN = EpitopeTargetDataset.ColumnName


class DefaultEpitopeTargetDataset(EpitopeTargetDataset):
    def __init__(self, config):
        super().__init__(config)

    @property
    def bind_target(self):
        return EpitopeTargetComponent.from_str(self.config['bind_target'])

    def _load_df(self, args=None):
        fn = self.config['fn']
        df = pd.read_csv(fn, index_col=0)
        logger.info(f'Loaded source df from {fn}, df.shape: {df.shape}')

        logger.info('Select valid beta CDR3 and epitope sequences')
        df = df.dropna(subset=[CN.epitope_seq, CN.cdr3b_seq])
        df = df[
            (df[CN.epitope_seq].map(is_valid_aaseq)) &
            (df[CN.cdr3b_seq].map(is_valid_aaseq))
            ]
        logger.info('Final data frame df.shape: %s' % str(df.shape))

        # df.index = df.apply(lambda row: self.get_index(row), axis=1)
        # df = df.loc[:, CN.values()]
        return df


class IEDBEpitopeMHCDataset(EpitopeTargetDataset):
    def __init__(self, config):
        super().__init__(config)

    IEDB_BIND_LEVEL_MAP = {
        'NEGATIVE': BindLevel.NEGATIVE.value,
        'POSITIVE-LOW': BindLevel.POSITIVE_LOW.value,
        'POSITIVE-INTERMEDIATE': BindLevel.POSITIVE.value,
        'POSITIVE': BindLevel.POSITIVE.value,
        'POSITIVE-HIGH': BindLevel.POSITIVE_HIGH.value
    }

    @property
    def bind_target(self):
        return EpitopeTargetComponent.MHC

    def _load_df(self, args=None):
        def get_bind_level(row, assay_type):
            assay_group = StrUtils.default_str(row['Assay Group']).upper()
            qual_meas = StrUtils.default_str(row['Qualitative Measure']).upper()
            bind_level = self.IEDB_BIND_LEVEL_MAP[qual_meas]
            if BindLevel.is_binder(bind_level):
                if AssayType.TCELL == assay_type or 'LIGAND PRESENTATION' in assay_group:
                    bind_level = BindLevel.POSITIVE_HIGH
            return bind_level

        fn = self.config['fn']
        assay_type = self.config['assay_type']
        valid_allele = self.config['valid_allele']
        logger.debug('self.config: %s' % self.config)

        logger.info('Loading iedb epitope-MHC data from %s' % (fn))
        df = pd.read_csv(fn, skiprows=1, low_memory=False)
        logger.info('Done to load iedb epitope-MHC data from %s: %s' % (fn, str(df.shape)))

        logger.debug('Selecting only class I')
        tmpcn = 'MHC allele class' if 'MHC allele class' in df.columns else 'Class'
        if tmpcn in df.columns:
            df = df[
                df[tmpcn].str.strip().str.upper() == "I"
                ]
        logger.debug('Current df.shape: %s' % str(df.shape))

        logger.debug('Dropping alleles with N/A')
        df = df[df['Allele Name'].map(lambda x: pd.notnull(x) and len(x) > 0)]
        logger.debug('Current df.shape: %s' % str(df.shape))

        logger.debug('Converting to standard allele name')
        df[CN.mhc_allele] = df['Allele Name'].str.strip().map(MHCAlleleName.std_name)
        df[CN.mhc_seq] = None
        df[CN.orig_mhc_seq] = df[CN.mhc_seq]
        logger.debug('Current df.shape: %s' % str(df.shape))

        logger.debug('Dropping mutant alleles')
        df = df[
            (~df[CN.mhc_allele].str.contains('mutant')) &
            (~df[CN.mhc_allele].str.contains('CD1'))
            ]
        logger.debug('Current df.shape: %s' % str(df.shape))

        # invalid_alleles = np.unique(iedb_df.allele[~iedb_df['allele'].map(MHCAlleleName.is_valid)])
        # print('Invalid allele names:', invalid_alleles)
        if valid_allele:
            logger.debug("Dropping alleles with invalid names")
            df = df[df[CN.mhc_allele].map(MHCAlleleName.is_valid)]
            # Take sub allele name: HLA-A*02:01:01 => HLA-A*02:01
            df[CN.mhc_allele] = df[CN.mhc_allele].map(MHCAlleleName.sub_name)
            logger.debug('Current df.shape: %s' % str(df.shape))

        # Select valid peptide sequences
        logger.debug('Selecting valid peptide sequences')
        df[CN.epitope_seq] = df['Description'].str.strip()
        df = df[
            df[CN.epitope_seq].map(lambda x: is_valid_aaseq(x))
        ]
        df[CN.orig_epitope_seq] = df[CN.epitope_seq]
        df[CN.epitope_len] = df[CN.epitope_seq].map(lambda x: len(x))
        logger.debug('Current df.shape: %s' % str(df.shape))

        df.index = df.apply(lambda row: self.get_index(row), axis=1)

        df[CN.epitope_species] = df['Organism Name']
        df[CN.epitope_gene] = df['Antigen Name']
        df[CN.epitope_start] = df['Starting Position']
        df[CN.epitope_end] = df['Ending Position']
        df[CN.cdr3b_seq] = None
        df[CN.orig_cdr3b_seq] = df[CN.cdr3b_seq]
        df[CN.ref_id] = df['PubMed ID'].map(lambda x: ('PMID:%s' % x) if x else None)
        df[CN.source] = 'IEDB'
        df[CN.bind_level] = df.apply(lambda row: get_bind_level(row, assay_type), axis=1)
        df = df.loc[:, CN.values()]

        logger.debug('Value count per mhc allele: %s' % df[CN.mhc_allele].value_counts())
        logger.debug('Value count per epitope_len: %s' % df[CN.epitope_len].value_counts())
        logger.debug('Value count per bind_level: %s' % df[CN.bind_level].value_counts())
        logger.info('Final IEDB dataset, df.shape: %s' % str(df.shape))
        return df


######
# For Epitope-TCR
#####
class DashEpitopeTCRDataset(EpitopeTargetDataset):
    GENE_INFO_MAP = OrderedDict({
        'BMLF': ('EBV', 'GLCTLVAML', 'HLA-A*02:01'),
        'pp65': ('CMV', 'NLVPMVATV', 'HLA-A*02:01'),
        'M1': ('IAV', 'GILGFVFTL', 'HLA-A*02:01'),
        'F2': ('IAV', 'LSLRNPILV', 'H2-Db'),
        'NP': ('IAV', 'ASNENMETM', 'H2-Db'),
        'PA': ('IAV', 'SSLENFRAYV', 'H2-Db'),
        'PB1': ('IAV', 'SSYRRPVGI', 'H2-Kb'),
        'm139': ('mCMV', 'TVYGFCLL', 'H2-Kb'),
        'M38': ('mCMV', 'SSPPMFRV', 'H2-Kb'),
        'M45': ('mCMV', 'HGIRNASFI', 'H2-Db'),
    })

    @property
    def bind_target(self):
        return EpitopeTargetComponent.TCR

    def _load_df(self, args=None):
        fn = self.config['fn']

        logger.info('Loading Dash data from %s' % fn)
        df = pd.read_table(fn, sep='\t')
        logger.debug('Current df.shape: %s' % str(df.shape))

        df[CN.epitope_gene] = df['epitope']
        df[CN.epitope_species] = df[CN.epitope_gene].map(lambda x: self.GENE_INFO_MAP[x][0])
        df[CN.epitope_seq] = df[CN.epitope_gene].map(lambda x: self.GENE_INFO_MAP[x][1])
        df[CN.orig_epitope_seq] = df[CN.epitope_seq]
        df[CN.epitope_len] = df[CN.epitope_seq].map(lambda x: len(x))
        df[CN.epitope_start] = None
        df[CN.epitope_end] = None
        df[CN.mhc_allele] = df[CN.epitope_gene].map(lambda x: self.GENE_INFO_MAP[x][2])
        df[CN.mhc_seq] = None
        df[CN.orig_mhc_seq] = df[CN.mhc_seq]
        df[CN.cdr3b_seq] = df['cdr3b'].str.strip().str.upper()
        df[CN.orig_cdr3b_seq] = df[CN.cdr3b_seq]
        df[CN.cdr3b_len] = df[CN.cdr3b_seq].map(len)
        df[CN.source] = 'Dash'
        df[CN.ref_id] = 'PMID:28636592'
        df[CN.bind_level] = BindLevel.POSITIVE

        logger.debug('Select valid beta CDR3 and epitope sequences')
        df = df.dropna(subset=[CN.epitope_seq, CN.cdr3b_seq])
        df = df[
            (df[CN.epitope_seq].map(is_valid_aaseq)) &
            (df[CN.cdr3b_seq].map(is_valid_aaseq))
            ]
        logger.debug('Current df.shape: %s' % str(df.shape))

        df.index = df.apply(lambda row: self.get_index(row), axis=1)
        df = df.loc[:, CN.values()]
        return df


class VDJDbEpitopeTCRDataset(EpitopeTargetDataset):
    @property
    def bind_target(self):
        return EpitopeTargetComponent.TCR

    def _load_df(self, args=None):
        fn = self.config['fn']
        logger.info('Loading VDJdb data from %s' % fn)
        df = pd.read_table(fn, sep='\t', header=0)
        logger.debug('Current df.shape: %s' % str(df.shape))

        # Select beta CDR3 sequence
        logger.debug('Select beta CDR3 sequences and MHC-I restricted epitopes')
        df = df[(df['gene'] == 'TRB') & (df['mhc.class'] == 'MHCI')]
        logger.debug('Current df.shape: %s' % str(df.shape))

        # Select valid CDR3 and peptide sequences
        logger.debug('Select valid CDR3 and epitope sequences')
        df = df.dropna(subset=['antigen.epitope', 'cdr3'])
        df = df[
            (df['antigen.epitope'].map(is_valid_aaseq)) &
            (df['cdr3'].map(is_valid_aaseq))
            ]
        logger.debug('Current df.shape: %s' % str(df.shape))

        logger.debug('Select confidence score > 0')
        df = df[df['vdjdb.score'].map(lambda score: score > 0)]
        logger.debug('Current df.shape: %s' % str(df.shape))

        df[CN.epitope_seq] = df['antigen.epitope'].str.strip().str.upper()
        df[CN.orig_epitope_seq] = df[CN.epitope_seq]
        df[CN.epitope_species] = df['antigen.species']
        df[CN.epitope_gene] = df['antigen.gene']
        df[CN.epitope_len] = df[CN.epitope_seq].map(lambda x: len(x))
        df[CN.epitope_start] = None
        df[CN.epitope_end] = None
        df[CN.cdr3b_seq] = df['cdr3'].str.strip().str.upper()
        df[CN.orig_cdr3b_seq] = df[CN.cdr3b_seq]
        df[CN.cdr3b_len] = df[CN.cdr3b_seq].map(len)
        df[CN.mhc_allele] = df['mhc.a']
        df[CN.mhc_seq] = None
        df[CN.orig_mhc_seq] = df[CN.mhc_seq]
        df[CN.source] = 'VDJdb'
        df[CN.ref_id] = df['reference.id']
        df[CN.bind_level] = BindLevel.POSITIVE

        df.index = df.apply(lambda row: self.get_index(row), axis=1)
        df = df.loc[:, CN.values()]
        return df


class McPASEpitopeTCRDataset(EpitopeTargetDataset):
    EPITOPE_SEP = '/'

    @property
    def bind_target(self):
        return EpitopeTargetComponent.TCR

    def _load_df(self, args=None):
        fn = self.config['fn']
        logger.info('Loading McPAS-TCR data from %s' % fn)
        df = pd.read_csv(fn)
        logger.debug('Current df.shape: %s' % str(df.shape))

        logger.debug('Select valid beta CDR3 and epitope sequences')
        df = df.dropna(subset=['CDR3.beta.aa', 'Epitope.peptide'])
        df = df[
            (df['CDR3.beta.aa'].map(is_valid_aaseq)) &
            (df['Epitope.peptide'].map(is_valid_aaseq))
            ]
        logger.debug('Current df.shape: %s' % str(df.shape))

        df[CN.epitope_seq] = df['Epitope.peptide'].str.strip().str.upper()
        df[CN.orig_epitope_seq] = df[CN.epitope_seq]

        # Handle multiple epitope
        logger.debug('Extend by multi-epitopes')
        tmpdf = df[df[CN.epitope_seq].str.contains(self.EPITOPE_SEP)].copy()
        for multi_epitope, subdf in tmpdf.groupby([CN.epitope_seq]):
            logger.debug('Multi epitope: %s' % multi_epitope)
            tokens = multi_epitope.split(self.EPITOPE_SEP)
            logger.debug('Convert epitope: %s to %s' % (multi_epitope, tokens[0]))
            df[CN.epitope_seq][df[CN.epitope_seq] == multi_epitope] = tokens[0]

            for epitope in tokens[1:]:
                logger.debug('Extend by epitope: %s' % epitope)
                subdf[CN.epitope_seq] = epitope
                subdf[CN.orig_epitope_seq] = subdf[CN.epitope_seq]
                df = df.append(subdf)
        logger.debug('Current df.shape: %s' % (str(df.shape)))

        df[CN.epitope_gene] = None
        df[CN.epitope_species] = df['Pathology']
        df[CN.epitope_len] = df[CN.epitope_seq].map(lambda x: len(x))
        df[CN.epitope_start] = None
        df[CN.epitope_end] = None
        df[CN.cdr3b_seq] = df['CDR3.beta.aa'].str.strip().str.upper()
        df[CN.orig_cdr3b_seq] = df[CN.cdr3b_seq]
        df[CN.cdr3b_len] = df[CN.cdr3b_seq].map(len)
        df[CN.mhc_allele] = df['MHC'].str.strip()
        df[CN.mhc_seq] = None
        df[CN.orig_mhc_seq] = df[CN.mhc_seq]
        df[CN.source] = 'McPAS'
        df[CN.ref_id] = df['PubMed.ID'].map(lambda x: '%s:%s' % ('PMID', x))
        df[CN.bind_level] = BindLevel.POSITIVE

        df.index = df.apply(lambda row: self.get_index(row), axis=1)

        logger.debug('Select MHC-I restricted entries')
        df = df[
            (df[CN.mhc_allele].notnull()) &
            (np.logical_not(df[CN.mhc_allele].str.contains('DR|DP|DQ')))
            ]
        logger.debug('Current df.shape: %s' % str(df.shape))
        df = df.loc[:, CN.values()]
        return df


class ShomuradovaEpitopeTCRDataset(EpitopeTargetDataset):
    @property
    def bind_target(self):
        return EpitopeTargetComponent.TCR

    def _load_df(self, args=None):
        fn = self.config['fn']
        logger.info('Loading Shomuradova data from %s' % fn)
        df = pd.read_csv(fn, sep='\t')
        logger.debug('Current df.shape: %s' % str(df.shape))

        logger.debug('Select TRB Gene')
        df = df[df['Gene'] == 'TRB']
        logger.debug('Current df.shape: %s' % str(df.shape))

        df[CN.epitope_seq] = df['Epitope'].str.strip().str.upper()
        df[CN.orig_epitope_seq] = df[CN.epitope_seq]
        df[CN.epitope_gene] = df['Epitope gene']
        df[CN.epitope_species] = df['Epitope species']
        df[CN.epitope_len] = df[CN.epitope_seq].map(lambda x: len(x))
        df[CN.epitope_start] = None
        df[CN.epitope_end] = None
        df[CN.mhc_allele] = df['MHC A']
        df[CN.mhc_seq] = None
        df[CN.orig_mhc_seq] = df[CN.mhc_seq]
        df[CN.cdr3b_seq] = df['CDR3'].str.strip().str.upper()
        df[CN.orig_cdr3b_seq] = df[CN.cdr3b_seq]
        df[CN.cdr3b_len] = df[CN.cdr3b_seq].map(len)
        df[CN.source] = 'Shomuradova'
        df[CN.ref_id] = 'PMID:33326767'
        df[CN.bind_level] = BindLevel.POSITIVE

        logger.debug('Select valid beta CDR3 and epitope sequences')
        df = df.dropna(subset=[CN.epitope_seq, CN.cdr3b_seq])
        df = df[
            (df[CN.epitope_seq].map(is_valid_aaseq)) &
            (df[CN.cdr3b_seq].map(is_valid_aaseq))
            ]
        logger.debug('Current df_enc.shape: %s' % str(df.shape))

        df.index = df.apply(lambda row: self.get_index(row), axis=1)
        df = df.loc[:, CN.values()]
        return df


class ImmuneCODEEpitopeTargetDataset(EpitopeTargetDataset):

    @property
    def bind_target(self):
        return EpitopeTargetComponent.from_str(self.config.get('bind_target', 'tcr'))

    def _load_df(self, args=None):
        fn = self.config['fn']

        logger.info('Loading ImmuneCODE data from %s' % fn)
        df = pd.read_csv(fn)
        logger.debug('Current df.shape: %s' % str(df.shape))

        fn_subj = self.config['fn_subject']
        df_subj = pd.read_csv(fn_subj, na_values=['N/A'])
        df_subj.index = df_subj['Experiment']

        new_rows = []
        for i, row in df.iterrows():
            cdr3b = row['TCR BioIdentity'].split('+')[0]
            epitopes = row['Amino Acids']
            orfs = row['ORF Coverage']
            exp = row['Experiment']

            alleles = df_subj.loc[exp, df_subj.columns.str.contains('HLA-')].values
            alleles = list(filter(lambda an: pd.notnull(an) and len(an) > 0
                                             and MHCAlleleName.is_valid(f'HLA-{an.strip()}'), alleles))
            alleles = sorted(np.unique(
                list(map(lambda an: MHCAlleleName.sub_name(MHCAlleleName.std_name(f'HLA-{an.strip()}')), alleles))))

            for epitope in epitopes.split(','):
                new_row = OrderedDict()
                new_row[CN.epitope_species] = 'SARS-CoV-2'
                new_row[CN.epitope_gene] = orfs
                new_row[CN.epitope_seq] = epitope
                new_row[CN.orig_epitope_seq] = epitope
                new_row[CN.epitope_len] = len(epitope)
                new_row[CN.mhc_allele] = ALLELE_SEP.join(alleles)
                new_row[CN.cdr3b_seq] = cdr3b
                new_row[CN.orig_cdr3b_seq] = cdr3b
                new_row[CN.cdr3b_len] = len(cdr3b)
                new_row[CN.ref_id] = 'PMC:7418738'
                new_row[CN.source] = 'ImmuneCODE_002.1'
                new_row[CN.bind_level] = BindLevel.POSITIVE
                new_rows.append(new_row)
                logger.debug('Appended new row: %s' % new_rows[-1])

        df = pd.DataFrame(new_rows, columns=CN.values())
        logger.debug('Done to append rows, df.shape: %s' % str(df.shape))

        logger.debug('Dropping invalid epitope sequences')
        df = df.dropna(subset=[CN.epitope_seq])
        df = df[
            df[CN.epitope_seq].map(is_valid_aaseq)
        ]
        logger.debug('Current df.shape: %s' % str(df.shape))

        if self.bind_target == EpitopeTargetComponent.MHC:
            logger.debug('Dropping invalid data with invalid allele name')
            df = df[
                df[CN.mhc_allele].map(lambda an: pd.notnull(an) and len(an) > 0)
            ]
            logger.debug('Current df.shape: %s' % str(df.shape))
        elif self.bind_target == EpitopeTargetComponent.TCR:
            logger.debug('Dropping invalid CDR3-beta sequences')
            df = df.dropna(subset=[CN.cdr3b_seq])
            df = df[
                df[CN.cdr3b_seq].map(is_valid_aaseq)
            ]
            logger.debug('Current df.shape: %s' % str(df.shape))

        df.index = df.apply(lambda row: self.get_index(row), axis=1)
        return df


class IEDBEpitopeTCRDataset(EpitopeTargetDataset):
    @property
    def bind_target(self):
        return EpitopeTargetComponent.TCR

    def _load_df(self, args=None):
        fn = self.config['fn']
        logger.info('Loading IEDB epitope-TCR data from %s' % fn)
        df = pd.read_csv(fn)
        logger.debug('Current df.shape: %s' % str(df.shape))

        df[CN.epitope_seq] = df['Description'].str.strip().str.upper()
        df[CN.orig_epitope_seq] = df[CN.epitope_seq]
        df[CN.epitope_gene] = df['Antigen']
        df[CN.epitope_species] = df['Organism']
        df[CN.epitope_len] = df[CN.epitope_seq].map(lambda x: len(x))
        df[CN.epitope_start] = None
        df[CN.epitope_end] = None
        df[CN.mhc_allele] = df['MHC Allele Names']
        df[CN.mhc_seq] = None
        df[CN.orig_mhc_seq]= df[CN.mhc_seq]
        df[CN.cdr3b_seq] = df['Chain 2 CDR3 Curated'].str.strip().str.upper()
        df[CN.orig_cdr3b_seq] = df[CN.cdr3b_seq]
        df[CN.cdr3b_len] = df[CN.cdr3b_seq].map(len)
        df[CN.source] = 'IEDB'
        df[CN.bind_level] = BindLevel.POSITIVE

        if 'Reference ID' in df.columns:
            df[CN.ref_id] = df['Reference ID'].map(lambda x: 'IEDB:%s' % x)
        elif 'Reference IRI' in df.columns:
            df[CN.ref_id] = df['Reference IRI'].map(
                lambda x: 'IEDB:%s' % basename(x if x is not None else '', ext=False))

        logger.debug('Select valid beta CDR3 and epitope sequences')
        df = df.dropna(subset=[CN.epitope_seq, CN.cdr3b_seq])
        df = df[
            (df[CN.epitope_seq].map(is_valid_aaseq)) &
            (df[CN.cdr3b_seq].map(is_valid_aaseq))
            ]
        logger.debug('Current df.shape: %s' % str(df.shape))

        df.index = df.apply(lambda row: self.get_index(row), axis=1)
        df = df.loc[:, CN.values()]
        return df


class pMTnetEpitopeTCRDataset(EpitopeTargetDataset):
    @property
    def bind_target(self):
        return EpitopeTargetComponent.TCR

    def _load_df(self, args=None):
        fn = self.config['fn']
        logger.info('Loading pMTnet data from %s' % fn)
        df = pd.read_csv(fn)
        logger.debug('Current df.shape: %s' % str(df.shape))

        df[CN.epitope_seq] = df['Antigen'].str.strip().str.upper()
        df[CN.orig_epitope_seq] = df[CN.epitope_seq]
        df[CN.epitope_gene] = None
        df[CN.epitope_species] = None
        df[CN.epitope_len] = df[CN.epitope_seq].map(lambda x: len(x))
        df[CN.epitope_start] = None
        df[CN.epitope_end] = None
        df[CN.mhc_allele] = df['HLA'].str.strip().str.upper()
        df[CN.mhc_seq] = None
        df[CN.orig_mhc_seq] = df[CN.mhc_seq]
        df[CN.cdr3b_seq] = df['CDR3'].str.strip().str.upper()
        df[CN.orig_cdr3b_seq] = df[CN.cdr3b_seq]
        df[CN.cdr3b_len] = df[CN.cdr3b_seq].map(len)
        df[CN.source] = 'pMTnet'
        df[CN.ref_id] = 'lu2021deep'
        df[CN.bind_level] = BindLevel.POSITIVE

        logger.debug('Select valid beta CDR3 and epitope sequences')
        df = df.dropna(subset=[CN.epitope_seq, CN.cdr3b_seq])
        df = df[
            (df[CN.epitope_seq].map(is_valid_aaseq)) &
            (df[CN.cdr3b_seq].map(is_valid_aaseq))
            ]
        logger.debug('Current df.shape: %s' % str(df.shape))

        df.index = df.apply(lambda row: self.get_index(row), axis=1)
        df = df.loc[:, CN.values()]
        logger.debug('Loaded pMTnet data. Current df.shape: %s' % str(df.shape))
        return df


class GfellerEpitopeTCRDataset(EpitopeTargetDataset):
    @property
    def bind_target(self):
        return EpitopeTargetComponent.TCR

    def _load_df(self, args=None):
        fn = self.config['fn']
        logger.info('Loading Gfeller data from %s' % fn)
        df = pd.read_excel(fn, header=1)
        logger.debug('Current df.shape: %s' % str(df.shape))
        df = df[
            df['Chain'] == 'B'
            ]
        df[CN.epitope_seq] = df['PeptideSequence'].str.strip().str.upper()
        df[CN.orig_epitope_seq] = df[CN.epitope_seq]
        df[CN.epitope_gene] = None
        df[CN.epitope_species] = None
        df[CN.epitope_len] = df[CN.epitope_seq].map(lambda x: len(x))
        df[CN.epitope_start] = None
        df[CN.epitope_end] = None
        df[CN.mhc_allele] = df['Info'].str.strip().str.upper()
        df[CN.mhc_seq] = None
        df[CN.orig_mhc_seq] = df[CN.mhc_seq]
        df[CN.cdr3b_seq] = df['CDR3'].str.strip().str.upper()
        df[CN.orig_cdr3b_seq] = df[CN.cdr3b_seq]
        df[CN.cdr3b_len] = df[CN.cdr3b_seq].map(len)
        df[CN.source] = 'Gfeller'
        df[CN.ref_id] = '{Gfeller,2023}}'
        df[CN.bind_level] = BindLevel.POSITIVE

        logger.debug('Select valid beta CDR3 and epitope sequences')
        df = df.dropna(subset=[CN.epitope_seq, CN.cdr3b_seq])
        df = df[
            (df[CN.epitope_seq].map(is_valid_aaseq)) &
            (df[CN.cdr3b_seq].map(is_valid_aaseq))
            ]
        logger.debug('Current df.shape: %s' % str(df.shape))

        df.index = df.apply(lambda row: self.get_index(row), axis=1)
        df = df.loc[:, CN.values()]
        logger.debug('Loaded Gfeller data. Current df.shape: %s' % str(df.shape))
        return df


class TCRdbEpitopeTCRDataset(EpitopeTargetDataset):
    @property
    def bind_target(self):
        return EpitopeTargetComponent.TCR

    def _load_df(self, args=None):
        logger.info(f'Loading TCRdb data with config: {self.config}')
        project_ids = self.config['project_ids']

        n_workers = args.n_workers if args else 1
        if n_workers > 1:
            with Pool(processes=n_workers) as pool:
                results = pool.starmap(self._subproc_load, zip(np.array_split(project_ids, n_workers)))
            df = pd.concat(results)
        else:
            df = self._subproc_load(project_ids)

        df.index = df.apply(lambda row: self.get_index(row), axis=1)
        logger.info(f'Loaded TCRdb data. Current df.shape: {df.shape}')
        return df

    def _subproc_load(self, project_ids):
        baseurl = self.config['baseurl']
        clonality_cutoff = self.config['clonality_cutoff']
        pseudo_epitope = self.config['pseudo_epitope']
        logger.info(f'Starting sub process of loading TCRdb data with project ids: {project_ids}')

        df = pd.DataFrame(columns=CN.values())
        success_pids = []
        failed_pids = []
        for i, pid in enumerate(project_ids):
            url = f'{baseurl}/{pid}.tsv'
            try:
                logger.info(f'Loading TCRdb data from url: {url}')
                df_tmp = pd.read_csv(url, sep='\t')
                logger.info(f'Done to load TCRdb data from url: {url}, df_tmp.shape: {df_tmp.shape}')
                logger.debug('Select valid CDR3 beta sequences')
                df_tmp = df_tmp[
                    df_tmp['AASeq'].map(lambda x: is_valid_aaseq(x))
                ]
                df_tmp[CN.epitope_seq] = pseudo_epitope
                df_tmp[CN.orig_epitope_seq] = df_tmp[CN.epitope_seq]
                df_tmp[CN.epitope_len] = len(pseudo_epitope)
                df_tmp[CN.mhc_seq] = None
                df_tmp[CN.orig_mhc_seq] = df_tmp[CN.mhc_seq]
                df_tmp[CN.cdr3b_seq] = df_tmp['AASeq']
                df_tmp[CN.orig_cdr3b_seq] = df_tmp[CN.cdr3b_seq]
                df_tmp[CN.cdr3b_len] = df_tmp[CN.cdr3b_seq].map(len)
                df_tmp[CN.ref_id] = 'PMC:7418738'
                df_tmp[CN.source] = 'TCRdb'
                logger.debug(f'Current df_tmp.shape: {df_tmp.shape}')

                low_clonality = df_tmp['cloneFraction'].quantile(clonality_cutoff)
                high_clonality = df_tmp['cloneFraction'].quantile(1 - clonality_cutoff)

                logger.info(
                    f'Select positives, negatives with clonality >= {high_clonality}, < {low_clonality}, cutoff: {clonality_cutoff}')
                df_pos = df_tmp[
                    df_tmp['cloneFraction'] >= high_clonality
                    ].copy()
                df_pos[CN.bind_level] = BindLevel.POSITIVE
                df_neg = df_tmp[
                    df_tmp['cloneFraction'] < low_clonality
                    ].copy()
                df_neg[CN.bind_level] = BindLevel.NEGATIVE
                logger.debug(f'df_pos.shape: {df_pos.shape}, df_neg.shape: {df_neg.shape}')

                col_names = CN.values()
                df_pos = df_pos.loc[:, df_pos.columns.map(lambda c: c in col_names)]
                df_neg = df_neg.loc[:, df_neg.columns.map(lambda c: c in col_names)]
                df = df.append(df_pos, ignore_index=True)
                df = df.append(df_neg, ignore_index=True)

                logger.info(f'Success to load from {url}, df_pos.shape: {df_pos.shape}, df_neg.shape: {df_neg.shape}')
                success_pids.append(pid)
            except Exception as e:
                logger.error(f'Failed to load TCRdb data from url: {url}, {e}')
                failed_pids.append(pid)
        logger.info(
            f'Done to sub process load TCRdb data, success: {success_pids}, failed: {failed_pids}, df.shape: {df.shape}')
        return df


class ConcatEpitopeTargetDataset(EpitopeTargetDataset):
    def __init__(self, config=None):
        super().__init__(config=config)

    @property
    def bind_target(self):
        return EpitopeTargetComponent.from_str(self.config['bind_target'])

    def _load_df(self, args=None):
        dfs = []
        for key in self.config['datasets']:
            ds = EpitopeTargetDataset.from_key(key, args)
            dfs.append(ds.df)
        return pd.concat(dfs)


class EpitopeTargetMaskedLMCollator:
    def __init__(self,
                 tokenizer=None,
                 max_epitope_len=None,
                 max_target_len=None,
                 has_sep=False):
        """
        Epitope and target sequence collator
        :param tokenizer: transformers.PreTrainedTokenizer
        :param max_epitope_len: max length of epitope sequence
        :param max_target_len: max length of target sequence
        :param has_sep: is whether has separator between epitope and target sequence
        """
        self.tokenizer = tokenizer
        self.has_sep = has_sep
        self.max_len = self._get_max_len(max_epitope_len, max_target_len)

    @property
    def sep_token(self):
        return self.tokenizer.eos_token

    def __call__(self, batch):
        seqs = []
        orig_seqs = []
        for (epitope_seq, target_seq), (orig_epitope_seq, orig_target_seq), _ in batch:
            seqs.append(self.format_seqs(epitope_seq, target_seq))
            orig_seqs.append(self.format_seqs(orig_epitope_seq if pd.notnull(orig_epitope_seq) else epitope_seq,
                                              orig_target_seq) if pd.notnull(orig_target_seq) else target_seq)

        inputs = self.encode(seqs)
        targets = self.encode(orig_seqs)

        input_ids = inputs['input_ids']
        target_ids = targets['input_ids']
        inputs['labels'] = torch.where(input_ids == self.tokenizer.mask_token_id, target_ids, -100)
        return inputs

    def format_seqs(self, epitope_seq, target_seq):
        return f"{self.gap2mask(epitope_seq)}{self.sep_token if self.has_sep else ''}{self.gap2mask(target_seq)}"

    def gap2mask(self, seq):
        return seq.replace(GAP, self.tokenizer.mask_token)

    def encode(self, seqs):
        return self.tokenizer(seqs,
                              padding="max_length",
                              truncation=False,
                              max_length=self.max_len,
                              return_overflowing_tokens=False,
                              return_tensors="pt")

    # def _get_mutated_seq(self, seq, seq_mutator=None):
    #     muted = seq_mutator.mutate(seq)
    #     muted_seq = ''.join(muted[0]).replace(GAP, self.tokenizer.mask_token)
    #     return muted_seq

    def _get_max_len(self, max_epitope_len, max_target_len):
        max_len = max_epitope_len + max_target_len + 2  # +2 for [CLS] and [EOS]
        return max_len + 1 if self.has_sep else max_len


class BaseDatasetTest(BaseTest):
    def setUp(self) -> None:
        super().setUp()

        logger.setLevel(logging.DEBUG)
        pd.set_option('display.max.rows', 999)
        pd.set_option('display.max.columns', 999)

        EpitopeTargetDataset.FN_DATA_CONFIG = '../config/data-test.json'
        self.data_config = FileUtils.json_load(EpitopeTargetDataset.FN_DATA_CONFIG)
        logger.debug(
            f"data_config['ds_map'].keys: {self.data_config['ds_map'].keys()} from {EpitopeTargetDataset.FN_DATA_CONFIG}")

    def check_allele(self, allele, valid_allele=True):
        if valid_allele:
            return MHCAlleleName.is_valid(allele)
        else:
            return pd.notnull(allele) and len(allele) > 0

    def is_valid_index(self, index, bind_target, sep='_', valid_allele=True):
        epitope, target = index.split(sep)[:2]
        return is_valid_aaseq(epitope) and \
            (self.check_allele(target,
                               valid_allele=valid_allele) if EpitopeTargetComponent.MHC == bind_target else is_valid_aaseq(
                target))

    def assert_df(self, df, bind_target, valid_allele=True):
        self.assertIsNotNone(df)
        self.assertTrue(df.shape[0] > 0)
        self.assertTrue(all(df.index.map(lambda x: self.is_valid_index(x, bind_target, valid_allele=valid_allele))))
        self.assertTrue(all(df[CN.epitope_seq].map(lambda x: is_valid_aaseq(x))))
        self.assertTrue(all(df[CN.bind_level].map(lambda x: x in BindLevel.values())))
        if EpitopeTargetComponent.MHC == bind_target:
            self.assertTrue(all(df[CN.mhc_allele].map(lambda x: self.check_allele(x, valid_allele=valid_allele))))
        else:
            self.assertTrue(all(df[CN.cdr3b_seq].map(lambda x: is_valid_aaseq(x))))


class EpitopeTargetDatasetTest(BaseDatasetTest):
    def setUp(self) -> None:
        super().setUp()
        logger.setLevel(logging.DEBUG)

    def _test_from_key(self, key, args):
        ds = EpitopeTargetDataset.from_key(key, args)
        config = ds.config
        bind_target = ds.bind_target
        df = ds.df
        self.assertIsNotNone(ds)
        self.assertTrue(len(ds) > 0)
        self.assert_df(df, bind_target=bind_target, valid_allele=config.get('valid_allele', False))

    def _test_get_item(self, key):
        ds = EpitopeTargetDataset.from_key(key)
        config = ds.config
        bind_target = ds.bind_target
        df = ds.df

        self.assertIsNotNone(ds)
        self.assertTrue(len(ds) > 0)
        self.assert_df(df, bind_target=bind_target, valid_allele=config.get('valid_allele', False))
        max_epitope_len = 0
        max_target_len = 0

        for i, (idx, row) in enumerate(df.iterrows()):
            epitope_seq = row[CN.epitope_seq]
            epitope_len = row[CN.epitope_len]
            if ds.bind_target == EpitopeTargetComponent.MHC:
                mhc_allele = row[CN.mhc_allele]
                target_seq = ds.mhc_domain.contact_site_seq(mhc_allele, pep_len=epitope_len)
            else:
                target_seq = row[CN.cdr3b_seq]
            if len(epitope_seq) > max_epitope_len:
                max_epitope_len = len(epitope_seq)
            if len(target_seq) > max_target_len:
                max_target_len = len(target_seq)
            cur_item = ds[i]
            self.assertEqual(epitope_seq, cur_item[0])
            self.assertEqual(target_seq, cur_item[1])
            self.assertTrue(cur_item[2] in (torch.tensor(0), torch.tensor(1)))

        self.assertEqual(max_epitope_len, ds.max_epitope_len)
        self.assertEqual(max_target_len, ds.max_target_len)

    def _delete_output_csv(self, key):
        output_csv = EpitopeTargetDataset.get_fn_output_csv(key)
        summary_csv = EpitopeTargetDataset.get_fn_summary_csv(key)

        if os.path.exists(output_csv):
            os.unlink(output_csv)
        if os.path.exists(summary_csv):
            os.unlink(summary_csv)

    def test_from_key(self):
        args = argparse.Namespace()
        args.n_workers = 1

        keys = ['immunecode']
        for key in keys:
            self._delete_output_csv(key)
            self._test_from_key(key, args)

    def test_get_item(self):
        self._test_get_item('immunecode')

    def test_exclude_by(self):
        source_ds = EpitopeTargetDataset.from_key('tcrdb_pos')

        exclude_ds = eval(source_ds.config['type'])(config=copy.deepcopy(source_ds.config))
        exclude_ds.config['name'] = 'sample'
        # exclude_ds.encodeder = source_ds.encoder
        exclude_ds.df = source_ds.df.sample(frac=0.1)

        result_ds = source_ds.exclude_by(exclude_ds=exclude_ds, target_cols=['index', CN.epitope_seq, CN.cdr3b_seq],
                                         inplace=False)

        self.assertTrue(len(source_ds) > len(result_ds))
        self.assertFalse(source_ds.df.equals(result_ds.df))
        self.assertTrue(all(exclude_ds.df.index.map(
            lambda x: (x in source_ds.df.index.values) and x not in result_ds.df.index.values)))
        self.assertTrue(all(exclude_ds.df[CN.epitope_seq].map(
            lambda x: (x in source_ds.df[CN.epitope_seq].values) and x not in result_ds.df[CN.epitope_seq].values)))
        self.assertTrue(all(exclude_ds.df[CN.cdr3b_seq].map(
            lambda x: (x in source_ds.df[CN.cdr3b_seq].values) and x not in result_ds.df[CN.cdr3b_seq].values)))

        result_ds = source_ds.exclude_by(exclude_ds=exclude_ds, target_cols=['index', CN.epitope_seq, CN.cdr3b_seq],
                                         inplace=True)
        self.assertTrue(result_ds.df.equals(source_ds.df))
        self.assertTrue(all(exclude_ds.df.index.map(lambda x: x not in source_ds.df.index.values)))
        self.assertTrue(all(exclude_ds.df[CN.epitope_seq].map(lambda x: x not in source_ds.df[CN.epitope_seq].values)))
        self.assertTrue(all(exclude_ds.df[CN.cdr3b_seq].map(lambda x: x not in source_ds.df[CN.cdr3b_seq].values)))

    def test_split_by_epitope(self):
        ds = EpitopeTargetDataset.from_key('immunecode')
        name_format = '{name}.{epitope}'
        dss = ds.split_by_epitope(name_format=name_format)
        self.assertEqual(len(ds), sum(map(lambda x: len(x), dss)))
        epitopes = []
        for cur_ds in dss:
            self.assert_df(cur_ds.df, bind_target=ds.bind_target)
            self.assertTrue(len(cur_ds.df[CN.epitope_seq].unique()) == 1)
            epitope = cur_ds.df[CN.epitope_seq].unique()[0]
            expected_name = name_format.format(name=ds.name, epitope=epitope)
            self.assertEqual(cur_ds.name, expected_name)
            epitopes.append(epitope)

        expected_epitopes = ds.df[CN.epitope_seq].unique()
        self.assertSetEqual(set(epitopes), set(expected_epitopes))

    def test_split_by_epitope_as_dict(self):
        ds = EpitopeTargetDataset.from_key('immunecode')
        name_format = '{name}.{epitope}'
        dss = ds.split_by_epitope(name_format=name_format, as_dict=True)
        self.assertEqual(len(ds), sum(map(lambda x: len(x), dss.values())))
        self.assertSetEqual(set(dss.keys()), set(ds.df[CN.epitope_seq].unique()))

        for epitope, cur_ds in dss.items():
            self.assert_df(cur_ds.df, bind_target=ds.bind_target)
            self.assertTrue(len(cur_ds.df[CN.epitope_seq].unique()) == 1)
            self.assertEqual(epitope, cur_ds.df[CN.epitope_seq].unique()[0])
            expected_name = name_format.format(name=ds.name, epitope=epitope)
            self.assertEqual(cur_ds.name, expected_name)

    def test_train_test_split(self):
        test_size = 0.2
        ds = EpitopeTargetDataset.from_key('immunecode')
        name_format = '{name}.{split}'
        train_ds, test_ds = ds.train_test_split(test_size=test_size, shuffle=False, name_format=name_format)
        self.assertEqual(name_format.format(name=ds.name, split='train'), train_ds.name)
        self.assert_df(train_ds.df, bind_target=ds.bind_target)
        self.assertEqual(name_format.format(name=ds.name, split='test'), test_ds.name)
        self.assert_df(test_ds.df, bind_target=ds.bind_target)
        self.assertEqual(len(ds), len(train_ds) + len(test_ds))
        self.assertTrue(np.array_equal(ds.df.index.values, np.append(train_ds.df.index.values, test_ds.df.index.values)))


class DatasetTestFixture:
    @classmethod
    def create_iris_dataset(cls):
        df = pd.read_csv('../data/iris.csv', dtype={'sepal_length': np.float32,
                                                    'sepal_width': np.float32,
                                                    'petal_length': np.float32,
                                                    'petal_width': np.float32})

        # transform species to numerics
        df.loc[df.species == 'Iris-setosa', 'species'] = 0
        df.loc[df.species == 'Iris-versicolor', 'species'] = 1
        df.loc[df.species == 'Iris-virginica', 'species'] = 2
        df['species'] = df['species'].astype(np.int32)
        X = df.values[:, :-1]
        y = df.values[:, -1]
        return TensorDataset(torch.tensor(X, dtype=torch.float32),
                             torch.tensor(y, dtype=torch.long))

    @classmethod
    def create_data_loader(cls, key, batch_size=64, shuffle=False, n_workers=None, val_size=None):
        if key == 'iris':
            ds = cls.create_iris_dataset()
            if val_size:
                n_data = len(ds)
                train_ds, val_ds = random_split(ds, [int(n_data * (1 - val_size)), int(n_data * val_size)])
                return (DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers),
                        DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers))
            else:
                return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)
        elif key == 'epitope_target':
            ds = EpitopeTargetDataset.from_key('immunecode_mutated')
            plm_name_or_path = '../output/peft_esm2_t33_650M_UR50D'
            tokenizer = AutoTokenizer.from_pretrained(plm_name_or_path)
            collator = EpitopeTargetMaskedLMCollator(tokenizer=tokenizer,
                                                     max_epitope_len=ds.max_epitope_len,
                                                     max_target_len=ds.max_target_len)
            if val_size:
                train_ds, val_ds = ds.train_test_split(test_size=val_size)
                return (DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers,
                                   collate_fn=collator),
                        DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers,
                                   collate_fn=collator))
            else:
                return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers,
                                  collate_fn=collator)


class EpitopeTargetMaskedLMCollatorTest(BaseTest):
    def setUp(self):
        super().setUp()

        EpitopeTargetDataset.FN_DATA_CONFIG = '../config/data-test.json'
        self.ds = EpitopeTargetDataset.from_key('immunecode_mutated')
        self.plm_name_or_path = '../output/peft_esm2_t33_650M_UR50D'
        self.tokenizer = AutoTokenizer.from_pretrained(self.plm_name_or_path)
        self.collator = EpitopeTargetMaskedLMCollator(tokenizer=self.tokenizer,
                                                      max_epitope_len=self.ds.max_epitope_len,
                                                      max_target_len=self.ds.max_target_len)

    def test_call_for_mutated(self):
        self.assertTrue(self.ds.is_mutated)
        inputs = [self.ds[i] for i in range(len(self.ds))]
        outputs = self.collator(inputs)
        self.assert_outputs_for_inputs(inputs=inputs, outputs=outputs)

    def assert_outputs_for_inputs(self, inputs=None, outputs=None):
        seqs = []
        orig_seqs = []
        for (epitope_seq, target_seq), (orig_epitope_seq, orig_target_seq), _ in inputs:
            seqs.append(self.collator.format_seqs(epitope_seq, target_seq))
            orig_seqs.append(self.collator.format_seqs(orig_epitope_seq, orig_target_seq))

        target_ids = self.collator.encode(orig_seqs)['input_ids']

        input_ids = outputs['input_ids']
        attention_mask = outputs['attention_mask']
        labels = outputs['labels']
        expected = (len(inputs), self.collator.max_len)
        self.assertEqual(expected, input_ids.shape)
        self.assertEqual(expected, attention_mask.shape)
        self.assertEqual(expected, labels.shape)
        self.assertEqual(expected, target_ids.shape)

        for i in range(len(inputs)):
            for j in range(self.collator.max_len):
                input_id = input_ids[i, j]
                target_id = target_ids[i, j]
                if labels[i, j] == -100:
                    self.assertEqual(input_id, target_id)
                else:
                    self.assertNotEqual(input_id, target_id)

        self.assertArrayEqual([s.replace(self.tokenizer.mask_token, '') for s in seqs], self.get_decoded_seqs(input_ids))
        self.assertArrayEqual(orig_seqs, self.get_decoded_seqs(target_ids))

    def get_decoded_seqs(self, input_ids):
        decoded_seqs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        decoded_seqs = list(map(lambda seq: StrUtils.rm_nonwords(seq), decoded_seqs))
        return decoded_seqs


class EpitopeTargetDSDataLoaderTest(EpitopeTargetMaskedLMCollatorTest):
    def setUp(self):
        super().setUp()

        self.batch_size = 64
        self.data_loader = DataLoader(self.ds, batch_size=self.batch_size, shuffle=False, collate_fn=self.collator)
        self.real_batch_sizes = [self.batch_size] * (len(self.ds) // self.batch_size) + [len(self.ds) % self.batch_size]

    def first_batch(self):
        return next(iter(self.data_loader))

    def assert_a_batch(self, batch_idx, batch):
        begin = batch_idx * self.batch_size
        end = (batch_idx + 1) * self.batch_size
        end = min(end, len(self.ds))
        inputs = [self.ds[i] for i in range(begin, end)]
        self.assert_outputs_for_inputs(inputs=inputs, outputs=batch)

    def test_first_batch(self):
        batch = self.first_batch()
        self.assert_a_batch(0, batch)

    def test_all_batches(self):
        for i, batch in enumerate(self.data_loader):
            print(f'>>>batch: {i}, batch_size: {self.real_batch_sizes[i]}')
            self.assert_a_batch(i, batch)


if __name__ == '__main__':
    unittest.main()
