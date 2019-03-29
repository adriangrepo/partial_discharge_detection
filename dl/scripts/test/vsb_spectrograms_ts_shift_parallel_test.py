import unittest
import logging
import pandas as pd
from ..vsb_spectrograms_ts_shift_parallel import recombine

logger = logging.getLogger(__name__)


class VSBSpectrogramsTSShiftParallelTest(unittest.TestCase):

    def test_recombine(self):
        lf = {'lf': [38, 20, 26, 10, 12, 17, 25, 15, 52, 31, 34,67,78,34,79,21]}
        data = {'y': [38, 20, 26, 10, 12, 17, 25, 15, 52, 31]}
        df = pd.DataFrame(data)
        lf_df = pd.DataFrame(data)
        recombine(lf_df, y, flip=False, index)

            '''recombine the HF component with a similar phase LF component
            print(f'len(list(lf_df)): {len(list(lf_df))}')
            # print(lf_df.head())
            if flip:
                v = (y.values) * -1
            else:
                v = y.values
            alt_sig = similar_lf_phased_sig(index)
            # lf_df.loc[lf_df[?column_name?] == some_value]

            y_lf = lf_df.loc[[str(alt_sig)]]
            y_lf = y_lf.T.values.flatten()
            assert len(v) == samples
            assert len(y_lf) == samples
            v = y_lf + v
            return v
            '''
        pass


if __name__ == '__main__':
    unittest.main()