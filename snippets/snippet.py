def _estimate_max_length(self):
    input_format = self.config.get('input_seq_format', '{epitope_seq}{target_seq}')
    max_len = 0
    if '{epitope_seq}' in input_format:
        max_len += self.df[CN.epitope_len].max()
    if '{target_seq}' in input_format:
        if self.bind_target == EpitopeComplexComponent.MHC:
            max_len += len(self.mhc_domain.all_hla_sites())
        else:
            max_len += self.df[CN.cdr3b_len].max()
    max_len += len(input_format.replace('{epitope_seq}', '').replace('{target_seq}', ''))
    return max_len


self.seq_mutator = UniformAASeqMutator(mut_ratio=0.2, mut_probs=(0.7, 0.3))
self.collator = EpitopeTargetSeqCollator(tokenizer=self.tokenizer,
                                         epitope_seq_mutator=self.seq_mutator,
                                         target_seq_mutator=self.seq_mutator,
                                         max_epitope_len=self.train_ds.max_epitope_len,
                                         max_target_len=self.train_ds.max_target_len,
                                         seq_format='{epitope_seq}{target_seq}')
