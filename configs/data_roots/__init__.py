import os
import importlib
import misc_utils as utils

data_names = os.listdir('configs/data_roots')

datasets = {}

# 解析configs/data_roots中的所有py文件
for name in data_names:
    if os.path.isfile(f'configs/data_roots/{name}'):
        if name in ['__pycache__', '__init__.py', '.DS_Store']:
            continue

        name = utils.get_file_name(name)
        datasets[name] = importlib.import_module(f'.{name}', 'configs.data_roots').Data

def get_one_dataset(dataset: str):
    if dataset in datasets:
        return datasets[dataset]
    else:
        raise Exception('No such dataset: "%s", available: {%s}.' % (dataset, '|'.join(datasets.keys())))

