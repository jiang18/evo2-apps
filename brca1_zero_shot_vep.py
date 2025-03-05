# Required imports
from Bio import SeqIO
import gzip
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.metrics import roc_auc_score

brca1_df = pd.read_excel(
    os.path.join('notebooks', 'brca1', '41586_2018_461_MOESM3_ESM.xlsx'),
    header=2,
)
brca1_df = brca1_df[[
    'chromosome', 'position (hg19)', 'reference', 'alt', 'function.score.mean', 'func.class',
]]

brca1_df.head(10)

# Rename columns
brca1_df.rename(columns={
    'chromosome': 'chrom',
    'position (hg19)': 'pos',
    'reference': 'ref',
    'alt': 'alt',
    'function.score.mean': 'score',
    'func.class': 'class',
}, inplace=True)

# Convert to two-class system
brca1_df['class'] = brca1_df['class'].replace(['FUNC', 'INT'], 'FUNC/INT')

brca1_df.head(10)

WINDOW_SIZE = 8192

# Read the reference genome sequence of chromosome 17
with gzip.open(os.path.join('notebooks', 'brca1', 'GRCh37.p13_chr17.fna.gz'), "rt") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        seq_chr17 = str(record.seq)
        break

def parse_sequences(pos, ref, alt):
    """
    Parse reference and variant sequences from the reference genome sequence.
    """
    p = pos - 1 # Convert to 0-indexed position
    full_seq = seq_chr17
    
    ref_seq_start = max(0, p - WINDOW_SIZE//2)
    ref_seq_end = min(len(full_seq), p + WINDOW_SIZE//2)
    ref_seq = seq_chr17[ref_seq_start:ref_seq_end]
    snv_pos_in_ref = min(WINDOW_SIZE//2, p)
    var_seq = ref_seq[:snv_pos_in_ref] + alt + ref_seq[snv_pos_in_ref+1:]
    
    # Sanity checks
    assert len(var_seq) == len(ref_seq)
    assert ref_seq[snv_pos_in_ref] == ref
    assert var_seq[snv_pos_in_ref] == alt
    
    return ref_seq, var_seq

# Parse sequences for the first variant
row = brca1_df.iloc[0]
ref_seq, var_seq = parse_sequences(row['pos'], row['ref'], row['alt'])

print(row)
print('--')
print(f'Reference, SNV 0: ...{ref_seq[4082:4112]}...')
print(f'Variant, SNV 0:   ...{var_seq[4082:4112]}...')


from evo2.models import Evo2

# Load model
model = Evo2('evo2_7b')


# Build mappings of unique reference sequences
ref_seqs = []
ref_seq_to_index = {}

# Parse sequences and store indexes
ref_seq_indexes = []
var_seqs = []

for _, row in brca1_df.iterrows():
    ref_seq, var_seq = parse_sequences(row['pos'], row['ref'], row['alt'])
    
    # Get or create index for reference sequence
    if ref_seq not in ref_seq_to_index:
        ref_seq_to_index[ref_seq] = len(ref_seqs)
        ref_seqs.append(ref_seq)
    
    ref_seq_indexes.append(ref_seq_to_index[ref_seq])
    var_seqs.append(var_seq)

ref_seq_indexes = np.array(ref_seq_indexes)

print(f'Scoring likelihoods of {len(ref_seqs)} reference sequences with Evo 2...')
ref_scores = model.score_sequences(ref_seqs)

print(f'Scoring likelihoods of {len(var_seqs)} variant sequences with Evo 2...')
var_scores = model.score_sequences(var_seqs)

# Subtract score of corresponding reference sequences from scores of variant sequences
delta_scores = np.array(var_scores) - np.array(ref_scores)[ref_seq_indexes]

# Add delta scores to dataframe
brca1_df[f'evo2_delta_score'] = delta_scores

brca1_df.head(10)


# Calculate AUROC of zero-shot predictions
y_true = (brca1_df['class'] == 'LOF')
auroc = roc_auc_score(y_true, -brca1_df['evo2_delta_score'])

print(f'Zero-shot prediction AUROC: {auroc:.2}')

from sklearn.metrics import auc, average_precision_score, precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_true, -brca1_df['evo2_delta_score'])
auprc = auc(recall, precision)
print(f'Zero-shot prediction AUPRC: {auprc:.2}')

ap = average_precision_score(y_true, -brca1_df['evo2_delta_score'])
print(f'Zero-shot prediction AP: {ap:.2}')

# 1b_base: AUPRC = 0.52
# 7b: AUPRC = 0.68