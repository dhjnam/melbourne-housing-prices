import pytest
import os, sys
import numpy as np
import pandas as pd

from cross_val import TrainTest

test_dir = os.getenv("TEST_DIR")

class TestCrossVal:

    @pytest.fixture
    def train_test_ratio(self):
        return .8
    
    @pytest.fixture
    def sample_size(self):
        return 5000

    @pytest.fixture
    def data(self, sample_size):
        data = pd.read_hdf(f'{test_dir}/testdata.h5')
        data = data.sample(n=sample_size, random_state=0)
        # Only use numeric features in data
        numeric_features = data.columns[(data.dtypes == 'int') | (data.dtypes == 'float')]
        data = data.loc[:, numeric_features]
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
    def train_test(self, data, target, features, train_test_ratio, sample_size):
        train = data.sample(int(train_test_ratio * sample_size), random_state=0)
        test = data[ ~data.index.isin(train.index) ]
        return TrainTest(train=train, test=test, features=features, target=target)
    
    @pytest.fixture
    def param(self):
        return dict(
            max_depth=5,
            min_samples_leaf=10,
            random_state=0
        )

    # @pytest.mark.skip
    def test_train_test(self, mocker, train_test, param):
        """
        Test ``CrossVal.grid_cross_val_score`` on a ``DecisionTreeRegressor``.
        Test the returned ``Report`` object.
        """
        from sklearn.tree import DecisionTreeRegressor
        estimator = DecisionTreeRegressor()
        
        spy_fit = mocker.spy(estimator, 'fit')
        spy_predict = mocker.spy(estimator, 'predict')

        train_test.train_test(estimator=estimator, param=param)

        assert spy_fit.call_count == 1
        assert spy_predict.call_count == 1

        assert train_test.error > 0
        assert 0 < train_test.score and train_test.score < 1