import pytest
import os, sys
import numpy as np
import pandas as pd

from melbourne_model import MelbourneModel

class TestMelbourneModel:

    @pytest.fixture
    def sample_size(self):
        return 5000

    @pytest.fixture
    def data(self, sample_size):
        data = pd.read_hdf(f'{os.getenv("TEST_DIR")}/testdata.h5')
        data = data.sample(n=sample_size, random_state=0)
        assert data is not None
        return data
    
    @pytest.fixture
    def target(self, data):
        target = 'Price'
        assert target in data.columns
        return target
    
    @pytest.fixture
    def features(self, data, target):
        return data.columns.drop(target)
    
    @pytest.fixture
    def features_numeric(self, data, target):
        features_numeric = data.columns[(data.dtypes == 'int') | (data.dtypes == 'float')]
        return features_numeric.drop(target)
    
    @pytest.fixture
    def params(self):
        return [
            { 
                'max_depth': max_depth, 
                # 'min_samples_split': min_samples_split, 
                'min_samples_leaf': min_samples_leaf,
                'random_state': 0
            }
            for max_depth         in [ 2,  3]
            # for min_samples_split in [1000, 2000]
            for min_samples_leaf in [500, 1000]
        ]
    
    # @pytest.mark.skip
    def test_constructor(self, mocker, params, data, target, features, sample_size):
        melbourne = MelbourneModel(df=data, features=features)
        assert melbourne.min_price == data.Price.min()
        assert melbourne.max_price == data.Price.max()
        assert melbourne.mean_price == data.Price.mean()
        assert len(melbourne.train) == int(.75 * len(data))

        melbourne = MelbourneModel(df=data, features=features, random_state=0, train_test_ratio=.6)
        assert len(melbourne.train) == int(.6 * len(data))
        # Test if train and test set 
        idxes_train_test = pd.concat([melbourne.train, melbourne.test]).sort_index()
        idxes_data = data.sort_index()
        assert idxes_train_test.equals(idxes_data)
    
    # @pytest.mark.skip
    def test_integration(self, mocker, params, data, target, features, features_numeric, sample_size):
        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
        estimator = DecisionTreeClassifier()
        melbourne = MelbourneModel(df=data, features=features_numeric, random_state=0, train_test_ratio=.75)
        cv = 4
        report = melbourne.grid_cross_val_score(estimator, params=params, cv=cv, n_jobs=8)
        assert (report.avgs < 1).all()
        assert (report.devs > 0).all()

        min_avg, max_dev = 0, 1
        optimal_param = report.compute_optimal_param(min_avg=min_avg, max_dev=max_dev)
        assert len(optimal_param) > 0

        melbourne.train_test(estimator, optimal_param)

        assert melbourne.score < 1
        assert melbourne.error > 0