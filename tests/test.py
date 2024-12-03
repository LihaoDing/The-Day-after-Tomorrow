import pytest
from speed import load_json_data, df_transformer, init_dataset, \
    CustomDataset, load_img_data
import pandas


def is_float(element):
    return isinstance(element, float)


class TestSpeedFunctions(object):
    '''Test the all functions in package speed'''

    @pytest.mark.parametrize('folder_path', [
        ('./test_data/')
    ])
    def test_JSON_loaded(self, folder_path):
        '''Test the output of load_json_data function'''
        df, file_name_dict = load_json_data(folder_path)
        assert isinstance(df, pandas.core.frame.DataFrame)
        assert df.shape == (3, 4)
        assert isinstance(file_name_dict, dict)

    def test_df_transformer(self):
        '''Test the output of function df_transformer'''
        input_data = {'storm_id': ['AAA', 'BBB', 'CCC'],
                      'relative_time': [0, 123, 423],
                      'ocean': ['1', '2', '3'],
                      'wind_speed': [12, 23, 34]}
        input_data = pandas.DataFrame(input_data)
        df, _ = df_transformer(input_data)
        assert df.columns[-1] == "id"
        assert df.shape[1] == 4
        is_float_df = df.applymap(is_float)
        assert is_float_df.all().all()

    @pytest.mark.parametrize('folder_path', [
        ('./test_data/')
    ])
    def test_load_img(self, folder_path):
        df, file_name_dict = load_json_data(folder_path)
        img_list = load_img_data(file_name_dict, folder_path)
        assert len(img_list) == 3

    @pytest.mark.parametrize('folder_path', [
        ('./test_data/')
    ])
    def test_init_dataset(self, folder_path):
        df, file_name_dict = load_json_data(folder_path)
        df, _ = df_transformer(df)
        img_list = load_img_data(file_name_dict, folder_path)
        dataset = init_dataset(img_list, df)
        assert isinstance(dataset, CustomDataset)
