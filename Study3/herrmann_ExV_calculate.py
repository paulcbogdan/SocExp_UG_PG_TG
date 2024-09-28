from itertools import cycle
import pandas as pd
from collections import defaultdict
from Study124.Agent import get_exponentially_weighted_mean
import numpy as np

from tqdm import tqdm


def M(l):
    if len(l):
        return sum(l) / len(l)
    else:
        return np.nan

class RowProcessor:
    '''
    Although this feature was not used, class can use exponential temporal
        weighing to vary the contribution of previous trials to present
        expectations (E). The decay and depth (depth = how far back to look)
        parameters are related to the temporal weighing. If decay = 1, there is
        no temporal weighing.
    '''
    def __init__(self, decay=1.0, depth=20):
        self.decay = decay
        self.depth = depth
        self.subj_prev_p = defaultdict(list)
        self.subj_prev_r = defaultdict(list)
        self.subj_prev_punished = defaultdict(list)
        self.subj_prev_punish = defaultdict(list)
        self.subj_prev_r_specific = defaultdict(lambda: defaultdict(list))
        self.kept_cols = ['groupid', 'sn', 'city', 'period',
                          's_contrib', 'female', 'age']


    def update_punishment(self, row):
        if len(self.subj_prev_punished[row.sn]):
            prev_punished = self.subj_prev_punished[row.sn][-1]
            E_punish = M(self.subj_prev_punish[row.sn])
            E_punished = M(self.subj_prev_punished[row.sn])
        else:
            prev_punished = np.nan
            E_punish = np.nan
            E_punished = np.nan

        self.subj_prev_punished[row.sn].append(row.recpun)
        self.subj_prev_punish[row.sn].append(M((row.punish_0, row.punish_1, row.punish_2)))
        return E_punish, E_punished, prev_punished

    def update_get_E_p(self, row):
        '''
        Calculates E[proposed] as E_p.
            E_p for trial[n] includes the participant's contribution in trial[n].
            e.g., for trial[1], E_p = (s_contrib[0] + s_contrib[1]) / 2,
            where s_contrib[n] is the participant's contribution in trial[n].
        :param row:
        :return:
        '''
        sn = row.sn
        self.subj_prev_p[sn].append(row.s_contrib)
        E_p = get_exponentially_weighted_mean(self.subj_prev_p[sn], self.decay, self.depth)
        return E_p

    def update_get_E_r(self, row):
        '''
        Calculates 3 versions of E[received]
        E_r_person: E[received][i] calculated based player i's past contributions.
            Excludes the current trial[n] because player i's contribution in
            trial[n] is what is being evaluated.
        E_r_sans_person: E[received][i] calculated based on players != i's past contributions
            Includes the current trial[n] as the evaluation specifically concerns
            player i's contribution and participants can reflect on others'
            contributions in that trial.
        E_r_sans_trial: E[received][i] calculated based on all contributions thus
            far except the current player i contribution in current trial[n].
        :param row:
        :return:
        '''
        sn = row.sn

        E_r_person = {} # E_r_person is based on trials before the current one
        for i in range(3):
            E_r_person[i] = get_exponentially_weighted_mean(self.subj_prev_r_specific[sn][i],
                                                              self.decay, self.depth)

        if len(self.subj_prev_r[sn]):
            r_avg_prev = self.subj_prev_r[sn][-1]
        else:
            r_avg_prev = np.nan


        # update to include others' contributions for the current trials
        self.subj_prev_r[sn].append(row.r_avg)
        for i in range(3):
            self.subj_prev_r_specific[sn][i].append(row[f'r_{i}'])


        E_r_sans_trial = {} # calculates expectation about i based on current + previous data only excluding
                            # person i's current offer
        E_r_sans_person = {}
        for i in range(3):
            E_r_sans_trial_i = np.empty(3)
            E_r_sans_person_i = 0
            weights = np.empty(3)
            for j in range(3):
                if i == j:
                    E_r_sans_trial_i[j] = E_r_person[j]
                    weights[j] = self.decay * \
                                 (len(self.subj_prev_r_specific[sn][i])-1)/len(self.subj_prev_r_specific[sn][i])
                                 # this weighs less person j if their current trial is ignored, they have contributed
                                 # 1 fewer trials to the mean
                else:
                    E_r_sans_trial_i[j] = get_exponentially_weighted_mean(
                        self.subj_prev_r_specific[sn][j], self.decay, self.depth)
                    weights[j] = 1.0
                    E_r_sans_person_i += E_r_sans_trial_i[j]
            E_r_sans_trial[i] = np.average(E_r_sans_trial_i, weights=weights)
            E_r_sans_person[i] = E_r_sans_person_i / 2

        return r_avg_prev, E_r_person, E_r_sans_trial, E_r_sans_person

    def process_row(self, row):
        E_p = self.update_get_E_p(row)
        r_avg_prev, E_r_person, E_r_sans_trial, E_r_sans_person = \
            self.update_get_E_r(row)
        E_punish, E_punished, prev_punished = self.update_punishment(row)

        out_l = []
        for i in range(3):
            punish_pkg = {'E_punished': E_punished,
                          'E_punish': E_punish,
                          'prev_punished': prev_punished}

            E_pkg = {'E_p': E_p,
                     'r_avg_prev': r_avg_prev,
                     'E_r_specific': E_r_person[i],
                     'E_r_sans_trial': E_r_sans_trial[i],
                     'E_r_sans_person': E_r_sans_person[i]}

            rest_specific = {'r': row[f'r_{i}'],
                             'punish': row[f'punish_{i}'],
                             'target': i
                             }
            kept_vals = {}
            for col in self.kept_cols:
                kept_vals[col] = row[col]
            E_pkg.update(rest_specific) # Note, in python 3.9, this can be done with z = x | y
            E_pkg.update(kept_vals)
            E_pkg.update(punish_pkg)
            out_l.append(E_pkg)

        return out_l

def combine_row_triplets(df):
    '''
    The Herrmann et al. (2008) data is organized such that for each trial of
        each player, there are three rows, corresponding to the three other
        players. These represent what each other player contributed and how much
        the participant punished that player. There is some duplication and
        redundancy.
    This function combines the three rows into one row with seperate columns
        for each player's contribution and punishment.
    '''
    num_cycle = cycle([0, 1, 2])
    df['partner_relative'] = [next(num_cycle) for _ in range(len(df))]
    df = df[df['p'] == 'P-experiment']

    df0 = df[df['partner_relative'] == 0].reset_index()
    df1 = df[df['partner_relative'] == 1].reset_index()
    df2 = df[df['partner_relative'] == 2].reset_index()
    df0.rename(columns={'otherscontribution': 'r_0', 'punishment': 'punish_0',
                        'senderscontribution': 's_contrib', 'subjectid': 'sn'},
               inplace=True)
    df0['r_1'] = df1['otherscontribution']
    df0['r_2'] = df2['otherscontribution']
    df0['punish_1'] = df1['punishment']
    df0['punish_2'] = df2['punishment']
    df = df0

    df['r_avg'] = df[['r_0', 'r_1', 'r_2']].mean(axis=1)
    return df

def process_PGG_data(decay=1.0):
    fp_in = r'PGG_data\Herrmann_Data.csv'
    df = pd.read_csv(fp_in)
    df = combine_row_triplets(df)

    RP = RowProcessor() # Much like the Agent class for the UG studies (1, 2 & 4)
                        # the Herrmann data is processed by essentially
                        # simulating agents that process the data row by row.

    tqdm.pandas(desc='apply progress...')
    l_of_l = df.progress_apply(RP.process_row, axis=1)
    l = [item for sublist in l_of_l for item in sublist]
    df_out = pd.DataFrame(l)

    decay_str = '' if decay == 1.0 else f'_decay_{decay}'
    fp_out = fr'PGG_data\Herrmann_Data_Processed{decay_str}.csv'

    df_out.to_csv(fp_out, index=False)

if __name__ == '__main__':
    process_PGG_data(decay=1.0)