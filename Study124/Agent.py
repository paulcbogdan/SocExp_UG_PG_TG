import numpy as np


def get_exponentially_weighted_mean(l, decay=0.5, depth=100):
    if len(l) == 0:
        return np.nan
    depth = min([len(l), depth])
    weights = [1]
    for _ in range(1, depth):
        weights.append(weights[-1]*decay)
    weights = np.array(weights[::-1])
    l_end = np.array(l[-depth:])
    try:
        E = np.average(l_end[~np.isnan(l_end)], weights=weights[~np.isnan(l_end)])
    except ZeroDivisionError:
        return np.nan
    return E

class DelayDiscountAgent:
    def __init__(self, decay, depth=400, reset_on_block=False,
                 ):
        self.decay = decay
        self.depth = depth
        self.reset_on_block = reset_on_block
        self.prev_p = []
        self.prev_r = []
        self.prev_subject_response_bool = []
        self.prev_block = -1

    def process_row(self, row):
        if row.block_number != self.prev_block and self.reset_on_block:
            self.prev_p = []
            self.prev_r = []
            self.prev_subject_response_bool = []
            self.prev_block = row.block_number

        E_p = get_exponentially_weighted_mean(self.prev_p, self.decay,
                                              self.depth)
        E_r = get_exponentially_weighted_mean(self.prev_r, self.decay,
                                              self.depth)

        if row.role == 'p':
            self.prev_p.append(row.subjectTake)
        elif row.role == 'r':
            self.prev_r.append(row.proposerTake)
            self.prev_subject_response_bool.append(row.subject_response_bool)
        return E_p, E_r#, E_resp
