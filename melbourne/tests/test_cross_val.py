import pytest
import os, sys
import numpy as np
import pandas as pd

from cross_val import CrossVal

test_dir = os.getenv("TEST_DIR")

class TestCrossVal:

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
    def cross_val(self, data, target, features):
        return CrossVal(df=data, features=features, target=target)
    
    @pytest.fixture
    def params(self):
        return [
            { 
                'max_depth': max_depth, 
                'min_samples_leaf': min_samples_leaf,
                'random_state': 0,
            }
            for max_depth         in [  2,   3,   4,    5]
            for min_samples_leaf  in [400, 600, 800, 1000]
        ]


    # @pytest.mark.skip
    def test_grid_cross_val_score(self, mocker, cross_val, params):
        """
        Test ``CrossVal.grid_cross_val_score`` on a ``DecisionTreeRegressor``.
        Test the returned ``Report`` object.
        """
        from sklearn import model_selection
        from sklearn.tree import DecisionTreeRegressor
        estimator = DecisionTreeRegressor()
        L = len(params)
        cv = 3
        
        spy_cross_val_score = mocker.spy(model_selection, 'cross_val_score')

        report = cross_val.grid_cross_val_score(estimator, params, cv=cv)

        # For each param in params there must have been one call to ``cross_val_score``
        assert spy_cross_val_score.call_count == L

        assert report.scores.shape == (L, cv)
        assert np.all(report.scores <= 1)
        assert np.all(report.avgs <= 1)
        assert len(report.avgs) == L
        assert len(report.devs) == L
        assert report.params == params
