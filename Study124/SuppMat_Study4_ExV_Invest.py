import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from SuppMat_delay_discount import rescale

# Finalized: 9/27/2022

def load_Study3_processed():
    fp_in = r'UG_data\processed_RoleChange_Study3.csv'
    df = pd.read_csv(fp_in)
    return df


def prepare_data():
    df = load_Study3_processed()
    prev_sn = None
    prev_d = defaultdict(lambda: np.nan)
    df_as_l = []
    for sn_block, df_sn_b in tqdm(df.groupby(['id', 'block_number']), desc='prepare_data'):
        df_sn_b['ExV_r'] = df['proposerTake'] - df_sn_b['E_r']
        df_sn_b['ExV_r_abs'] = abs(df_sn_b['ExV_r'])

        df_sn_b['ExV_p'] = df['proposerTake'] - df_sn_b['E_p']
        df_sn_b['ExV_p_abs'] = abs(df_sn_b['ExV_p'])

        sn = sn_block[0]
        if sn != prev_sn:
            prev_d = defaultdict(lambda: np.nan)
            prev_sn = sn

        cols_kept = ['invest', 'proposerTake', 'subjectTake', 'ExV_r', 'ExV_r_abs', 'ExV_p', 'ExV_p_abs',
                     'E_p', 'E_r']
        d = {'condition': df_sn_b['condition'].iloc[0][:], 'sn': int(sn)}

        for col in cols_kept:
            d[col] = df_sn_b[col].mean()

        d['prev_condition'] = prev_d['condition']
        d['prev_proposerTake'] = prev_d['proposerTake'] # used for regress_invest_on_prev_partner(...)

        df_as_l.append(d)
        prev_d = d
    df_out = pd.DataFrame(df_as_l)
    df_out['sn'] = df_out['sn'].astype(int)
    df_out['proposerTake'] = -df_out['proposerTake'] # proposerTake originally
    # represented the amount the computer proposer took. This is inverted to
    # represented the amount the participant received, which may be more intuitive
    return df_out


def regress_invest_on_prev_partner(only_3_computers=True):
    # This function was used for a response to reviewer comments but these results
    #   were not included in the final manuscript.
    # The results showed that the partner from the previous block does not
    #   impact invest in the following block.

    from pymer4.models import Lmer  # pymer4 import is rather slow. it's placed here
                                    # in case the present file is used as a module
    df = prepare_data()
    df.dropna(subset=['proposerTake', 'invest', 'subjectTake'], inplace=True)

    if only_3_computers:
        df = df[df['condition'].isin(['generous', 'reciprocity', 'selfish'])]
        df = df[df['prev_condition'].isin(['generous', 'reciprocity', 'selfish'])]

    cols = ['ExV_p_abs', 'ExV_r_abs', 'proposerTake', 'invest']
    for col in cols:
        df[col] = rescale(df[col])

    if only_3_computers:
        # no random slopes for the 3-computer regression as it would saturate the model
        formula = 'invest ~ 1 + proposerTake + prev_proposerTake +' \
                           '(1 | sn)'
    else:
        formula = 'invest ~ 1 + ' \
                           'proposerTake + prev_proposerTake + ' \
                           '(1 | sn)'

    mod = Lmer(formula, data=df)
    summary = mod.fit(REML=False)
    print(summary)


def regress_invest_on_ExV(only_3_computers=False):
    from pymer4.models import Lmer # pymer4 import is rather slow. it's placed here
                                   # in case the present file is used as a module
    df = prepare_data()
    df.dropna(subset=['proposerTake', 'invest', 'subjectTake'], inplace=True)

    if only_3_computers:
        # ExV_p_abs remains significance even when one or both of these next
        #   two lines are commented out
        df = df[df['condition'].isin(['generous', 'reciprocity', 'selfish'])]
        df = df[df['prev_condition'].isin(['generous', 'reciprocity', 'selfish'])]

    cols = ['ExV_p_abs', 'ExV_r_abs', 'proposerTake', 'invest']
    for col in cols:
        df[col] = rescale(df[col])

    if only_3_computers:
        # no random slopes for the 3-computer regression as it would saturate the model
        formula = 'invest ~ 1 + ' \
                           'proposerTake + ExV_p_abs + ExV_r_abs + condition + ' \
                           '(1 | sn)'
    else:
        formula = 'invest ~ 1 + ' \
                           'proposerTake + ExV_p_abs + ExV_r_abs + condition + ' \
                           '(1 | sn)'

    mod = Lmer(formula, data=df)
    summary = mod.fit(REML=False)
    print(summary)

if __name__ == '__main__':
    regress_invest_on_ExV(only_3_computers=True)
    # regress_invest_on_prev_partner()
