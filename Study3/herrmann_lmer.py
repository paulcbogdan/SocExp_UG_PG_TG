import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm


def rescale(col):
    return (col - col.mean()) / col.std()


def add_ExV(df, col):
    df[col.replace('E', 'ExV')] = df['r'] - df[col]
    df[col.replace('E', 'ExV') + '_abs'] = abs(df['r'] - df[col])


def do_lmer_by_city(key_p = 'ExV_p_w_curr_abs', key_r = 'ExV_r_sans_trial_abs',
                    random_grps='sn', do_rfx=True):
    '''
    Plots model fit by study for a pair of expectation variables
    '''
    df = load_and_basic_preprocess()
    cities = df['city'].unique()
    fits_p = []
    fits_r = []
    p_higher = []

    for city in tqdm(cities, desc='looping cities'):
        df_city = df[df['city'] == city]
        N = len(df_city.sn.unique())
        fixed_ef_base = '1 + r + E_punished'
        rand_ef_base = fixed_ef_base if do_rfx else '1'
        formula_base = f'punish ~ {fixed_ef_base} + ({rand_ef_base} | {random_grps})'
        fit_base, coefs_base = do_city_lmer(df_city, formula_base)

        fixed_ef_p = f'1 + r + {key_p} + E_punished'
        rand_ef_p = fixed_ef_p if do_rfx else '1'
        formula_p = f'punish ~ {fixed_ef_p} + ({rand_ef_p} | {random_grps})'
        fit_p, coefs_p = do_city_lmer(df_city, formula_p)
        fit_p -= fit_base

        fixed_ef_r = f'1 + r + {key_r} + E_punished'
        rand_ef_r = fixed_ef_r if do_rfx else '1'
        formula_r = f'punish ~ {fixed_ef_r} + ({rand_ef_r} | {random_grps})'
        fit_r, coefs_r = do_city_lmer(df_city, formula_r)
        fit_r -= fit_base
        if coefs_p.loc[key_p]['P-val'] < 0.05 and coefs_r.loc[key_r]['P-val'] < 0.05:
            c = 'purple'
        elif coefs_p.loc[key_p]['P-val'] < 0.05:
            c = 'b'
        elif coefs_r.loc[key_r]['P-val'] < 0.05:
            c = 'r'
        else:
            c = 'k'

        fit_p /= N
        fit_r /= N

        if fit_p > fit_r:
            p_higher.append(1)
        else:
            p_higher.append(0)

        plt.scatter([fit_r], [fit_p], color=c)
        plt.text(fit_r + 0.01, fit_p, f'{city}\nN={N}')
        fits_p.append(fit_p)
        fits_r.append(fit_r)
    for i in range(5):
        print()
    print('--------------------------')
    print()
    for fit_p, fit_r, city in zip(fits_p, fits_r, cities):
        print(f'{city}: {fit_p:.3f} [{key_p}] | {fit_r:.3f} [{key_r}]')

    fit_max = max(fits_r + fits_p)
    plt.plot([0, fit_max*1.1], [0, fit_max*1.1], 'k--')
    plt.xlabel(f'{key_r} fit quality')
    plt.ylabel(f'{key_p} fit quality')
    plt.title(f'{do_rfx=} | {sum(p_higher)=:.4f}')
    plt.show()


def do_city_lmer(df_city, formula):
    from pymer4.models import Lmer
    mod = Lmer(formula, data=df_city)
    mod.fit(REML=False, summary=False)
    fit = mod.logLike
    print(mod.coefs)
    return fit, mod.coefs


def load_and_basic_preprocess():
    in_fp = fr'PGG_data\Herrmann_Data_Processed.csv'
    df = pd.read_csv(in_fp)

    df.dropna(subset=['E_p', 'E_r_sans_trial', 'E_punished'], inplace=True)

    add_ExV(df, 'E_p')
    add_ExV(df, 'E_r_sans_trial') # Alternative formulations for E[received]
    add_ExV(df, 'E_r_sans_person') # were calculated. We used E_r_sans_trial as
    add_ExV(df, 'E_r_specific') # it yielded the strongest fit and thus serves
                                # as the strongest comparison to E_p.

    df['r'] = df['r'].astype(np.float64)
    df['s_contrib'] = df['s_contrib'].astype(np.float64)

    for col in df.columns: # standardize variables to get standardized coefficients
        if df[col].dtype == np.float64: # this also helps avoid convergence errors
            df[col] = rescale(df[col])
    return df


def do_E_ExV_lmer(do_ExV=True, do_REML=False, do_rfx=True, do_p=True, do_r=True):
    # Note that do_REML should be False for significance testing of coefficients
    #   but True when looking at model fit (see, Meteyard & Davies, 2020).
    from pymer4.models import Lmer
    df = load_and_basic_preprocess()

    # The patterns of significance do not change if additional random-levels
    #   are added (e.g., adding a city-level or group-level).
    if do_ExV:
        fixed_ef = '1 + r + E_punished'
        fixed_ef += ' + ExV_p_abs' if do_p else ''
        fixed_ef += ' + ExV_r_sans_trial_abs' if do_r else ''
    else:
        fixed_ef = '1 + r + E_punished'
        fixed_ef += ' + E_p' if do_p else ''
        fixed_ef += ' + E_r_sans_trial' if do_r else ''

    rand_ef = '1' if do_rfx else fixed_ef
    formula = fr'punish ~ {fixed_ef} + ' \
                     fr'( {rand_ef} | sn)'

    print('Lmering...')
    # The optimx optimizer helps with achieving convergence.
    #   Requires the optimx R package
    mod = Lmer(formula, data=df)
    summary = mod.fit(REML=do_REML, control="optimizer='optimx', "
                                            "optCtrl = list(method='nlminb',"
                                            "kkt=FALSE)")
    print(summary)

    #  Log likelihood results (REML = True) (Table 1 in manuscript)
    #              LL         ΔLL
    # Base LL	37629.47
    # E P	    37498.33	 131.14   (do_p = True & do_ExV = False)
    # E R	    37505.51	 123.96   (do_r = True & do_ExV = False)
    # ExV P	    36984.37	 645.10   (do_p = True & do_ExV = True)
    # ExV R	    37373.78	 255.69   (do_r = True & do_ExV = True)

def do_trial_mediation():
    from pymer4.models import Lmer
    df = load_and_basic_preprocess()

    print('---------------------')
    print(f' Reciprocity effect ')
    print('---------------------')
    formula = r's_contrib ~ 1 + r_avg_prev + prev_punished + ' \
              r'(1 +  r_avg_prev + prev_punished | sn)'
    mod = Lmer(formula, data=df)
    summary = mod.fit(REML=False, control = "optimizer='optimx', "
                                            "optCtrl = list(method='nlminb', "
                                            "kkt=FALSE)")
    print(summary)
    print()

    print('---------------------')
    print(f' Expectation effect ')
    print('---------------------')

    formula = r'punish ~ 1 + r + s_contrib + r_avg_prev + prev_punished + ' \
              r'(1 + r + s_contrib + r_avg_prev + prev_punished | sn)'
    mod = Lmer(formula, data=df)
    summary = mod.fit(REML=False, control ="optimizer='optimx', "
                                           "optCtrl = list(method='bobyqa', "
                                           "kkt=FALSE)")
    print(summary)


if __name__ == '__main__':
    # change settings to generate all the results
    do_E_ExV_lmer(do_ExV=False, do_REML=True, do_p=True, do_r=False)

    # do_lmer_by_city(key_p='ExV_p_abs', key_r='ExV_r_sans_trial_abs', do_rfx=False)
    # do_lmer_by_city(key_p='E_p', key_r='E_r_sans_trial', do_rfx=False)
    # do_trial_mediation()
