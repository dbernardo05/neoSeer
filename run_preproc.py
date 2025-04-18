#!/usr/bin/env python3

import preproc

if __name__ == '__main__':
    kfolds_dir = 'kfolds_all_cork_huh'
    regen_kf_data = True
    allowed_augments = ['none', 'vertflip', 'horizflip', 'horizvertflip']

    preproc.preproc_df_noTD(allowed_augments, kfolds_dir, num_kfolds=10, regen_kf_data=regen_kf_data)
