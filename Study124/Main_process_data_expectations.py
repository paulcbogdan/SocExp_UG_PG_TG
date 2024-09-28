import pandas as pd
from Agent import DelayDiscountAgent



def get_df_with_E_p_E_r(df, reset_on_block=True, delay_discount=1):
    '''
    Adds E_p and E_r columns to df. This works by essentially "simulating"
        an Agent, which processes each row of the dataframe one by one.
    Can handle exponential weighing of previous trials by temporal distance,
        which was used for the Supplemental Materials. That analysis involved
        not resetting each block the pool of trials contributing to expectations
    '''
    print('---------------------------')
    print(f'\t{delay_discount=:.3f}')
    print(f'\t{reset_on_block=}')
    DDA = DelayDiscountAgent(delay_discount, reset_on_block=reset_on_block)
    df[['E_p', 'E_r']] = df.apply(lambda row: DDA.process_row(row), axis=1,
                                  result_type="expand")
    return df

def proc_data(study, reset_on_block=True, delay_discount=1, save=False,
              do_exclusion=True):
    '''
    Loads data .csv and adds expectation (E[proposed] & E[received]) columns.
    If save == true, then this function saves a new .csv, otherwise it returns
        the processed Pandas dataframe.
    '''
    fp_in = fr'UG_data\RoleChange_Study{study}_anonymized.csv'
    df = pd.read_csv(fp_in)

    df = get_df_with_E_p_E_r(df, reset_on_block=reset_on_block,
                             delay_discount=delay_discount)

    if do_exclusion: df = df[~df['excluded'].astype(bool)]   # remove excluded participants
    df['received'] = 10 - df['proposerTake'] # proposerTake represents amount computer proposed to subject
                                             # this is converted to the amount received by the subject
                                             # e.g., for a recived $3:$7 offer: proposeTake = 7, M2 = 3
    if save:
        fp_out = fr'UG_data\processed_RoleChange_Study{study}.csv'
        df.to_csv(fp_out, index=False)
    else:
        return df



if __name__ == '__main__':
    for STUDY in [4]:
        proc_data(STUDY, save=True, do_exclusion=False)