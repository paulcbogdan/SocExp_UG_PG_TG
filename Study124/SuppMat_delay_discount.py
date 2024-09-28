import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Main_process_data_expectations import proc_data

'''
Although not explicitly imported, running the delay discount analyses requires
    rpy2, as it uses pymer4. pymer4 is imported in do_lmer(...) because it is
    slow to load. I believe using rpy2 and pymer4 requires you to have R 
    installed with the lme4 and lmerTest packages. If you do not have this set
    up, I think the easiest way to set this up (or at least to install rpy2)
    is via Anaconda. pip sometimes mishandles complex packages like this.
    (tensorflow and CUDA is another example of software that pip struggles with)  
'''

def delay_discount_analysis(study=1, reset_on_block=False,
                            both_E=True):
    '''
    This code runes the lmer for every level of exponential temporal decay.
        Although not reported in the paper (for brevity), preliminary analyses
        were performed examining whether the delay discounting led to greater
        model fit. It did. There was also an interesting pattern associated
        with part two where the relationship between the delay weighing
        and model fit led to two local maxima, suggesting both shorter
        and longer term expectations. None of the analyses on model fit were
        reported in the main text or Supplemental Materials.
    '''

    pd.options.mode.chained_assignment = None
    # This suppresses SettingWithCopyWarning, which otherwise would pop up in
    #   every loop of th delay_discount analysis. This warning isn't anything to be
    #   concerned about because, here, the df and the df copy aren't used after
    #   the warning arises.

    fits = []
    for delay in tqdm(np.linspace(.01, .99, 99), desc='delay_loop'):
        df = proc_data(study, reset_on_block, do_exclusion=True,
                       delay_discount=delay, save=False)
        fit, coefs = do_lmer(df, both_E=both_E)
        plot_delay_stats(delay, fit, coefs)
        fits.append(fit)
    min_fit = min(fits)
    max_fit = max(fits)
    if max_fit - min_fit < 14:
        plt.yticks(range(int(min_fit), int(max_fit) + 2, 2))
    else:
        plt.yticks(range(int(min_fit), int(max_fit) + 5, 5))

    plt.title(f'Study {study} | {both_E=} | {reset_on_block=}')
    plt.ylabel('Model fit (log-likelihood)')
    plt.xlabel('Delay (λ)')
    plt.show()


def do_lmer(df, both_E=True):
    df.dropna(subset=['proposerTake', 'E_p', 'E_r', 'subject_response_bool',
                      'prev_response_bool'], inplace=True)
    df = df[df['proposerTake'] > 5]

    formula_cols = ['E_p', 'E_r', 'proposerTake', 'prev_response_bool']
    df.dropna(subset=formula_cols + ['subject_response_bool'], inplace=True)

    for col in formula_cols:
        if col == 'prev_response_bool':
            continue
        df[col] = rescale(df[col])

    if both_E:
        fx = 'proposerTake + E_p + E_r + prev_response_bool'
    else:
        fx = 'proposerTake + E_r + prev_response_bool'


    formula = fr'subject_response_bool ~ 1 + {fx} + ' \
              fr'(1 + {fx} | id)'

    print('Lmering...')
    from pymer4.models import Lmer  # this package is slow to load, so it's
                                    # imported within this function.
    mod = Lmer(formula, data=df, family='binomial')
    summary = mod.fit(REML=False, control="optimizer='optimx', "
                                          "optCtrl = list(method='nlminb', kkt=FALSE)")
    # The optimx optimizer helps with achieving convergence.
    #   Its use requires the optimx R package to be installed.

    print(summary)
    fit = mod.logLike
    coefs = mod.coefs
    return fit, coefs

def rescale(col):
    return (col - col.mean()) / col.std()

def plot_delay_stats(delay, fit, coefs, annotate_t=False):
    E_p_t = coefs['Z-stat'].loc['E_p']
    E_r_t = coefs['Z-stat'].loc['E_r'] if 'E_r' in coefs['Z-stat'] else 0

    if E_p_t > 100 or E_r_t > 100:
        return
    if E_p_t > 2.0 and E_r_t > 2.0:
        color = 'purple'
    elif E_p_t > 2.0:
        color = 'b'
    elif E_r_t > 2.0:
        color = 'r'
    else:
        color = 'k'

    if annotate_t: plt.annotate(f'{E_p_t:.3f}', xy=(delay, fit+.3), color=color)
    plt.scatter([delay], [fit], color=color)


if __name__ == '__main__':
    delay_discount_analysis(study=2, reset_on_block=False)
