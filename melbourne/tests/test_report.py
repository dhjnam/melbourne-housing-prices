import pytest
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from util.cross_val import Report, NoOptimalParamsError

test_dir = os.getenv("TEST_DIR")

class TestReport:

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
    def params(self):
        return [
            { 'a': a, 'b': b }
            for a in range(10)
            for b in range(10)
        ]
    
    @pytest.fixture
    def report_mu_sigma(self, params):
        rng = np.random.default_rng(seed=0)
        L, cv = len(params), 10
        mu, sigma = .3, .1
        scores = np.minimum(rng.normal(mu, sigma, size=(L, cv)), 1)
        report = Report(scores=scores, params=params)
        return report, mu, sigma
    
    # @pytest.mark.skip
    def test_compupte_optimal_params(self, report_mu_sigma):
        report, mu, sigma = report_mu_sigma
        
        min_avg, max_dev = 2 * mu, sigma / 2
        
        with pytest.raises(NoOptimalParamsError):
            optimal_param = report.compute_optimal_param(min_avg=min_avg, max_dev=max_dev, objective='avg')
            assert report.optimal_params.dtypes[0] == 'int64'
            assert report.optimal_params.dtypes[1] == object

        min_avg, max_dev = mu / 2, 2 * sigma
        
        optimal_param = report.compute_optimal_param(min_avg=min_avg, max_dev=max_dev, objective='avg')
        assert report.optimal_params.dtypes[0] == 'int64'
        assert report.optimal_params.dtypes[1] == object

        idx, optimal_parameter = report.optimal_params.loc[(min_avg, max_dev)]

        constrained_avgs = report.avgs[ (report.avgs > min_avg) & (report.devs < max_dev) ]
        assert report.avgs[idx] > min_avg
        assert report.devs[idx] < max_dev
        assert np.max(constrained_avgs) == report.avgs[idx]
        assert optimal_parameter == report.optimal_params.loc[(min_avg, max_dev), 'param']
        assert optimal_param == optimal_parameter

        optimal_param = report.compute_optimal_param(min_avg=min_avg, max_dev=max_dev, objective='dev')
        assert report.optimal_params.dtypes[0] == 'int64'
        assert report.optimal_params.dtypes[1] == object

        idx, optimal_parameter = report.optimal_params.loc[(min_avg, max_dev)]

        constrained_devs = report.devs[ (report.avgs > min_avg) & (report.devs < max_dev) ]
        assert report.avgs[idx] > min_avg
        assert report.devs[idx] < max_dev
        assert np.min(constrained_devs) == report.devs[idx]
        assert optimal_parameter == report.optimal_params.loc[(min_avg, max_dev), 'param']
        assert optimal_param == optimal_parameter

    # pytest.mark.skip
    def test_report_plot_hist(self, mocker, report_mu_sigma):

        report, mu, sigma = report_mu_sigma

        facet_grid = report.plot_hist()
        assert type(facet_grid) == sns.FacetGrid
        
        facet_grid = report.plot_hist(level='figure')
        assert type(facet_grid) == sns.FacetGrid
        
        ax = report.plot_hist(level='axis')
        assert type(ax) == plt.Axes

    # pytest.mark.skip
    def test_report_plot_hist2d(self, mocker, report_mu_sigma):
        report, mu, sigma = report_mu_sigma

        facet_grid = report.plot_hist2d()
        assert type(facet_grid) == sns.FacetGrid
        
        facet_grid = report.plot_hist2d(level='figure')
        assert type(facet_grid) == sns.FacetGrid
        
        ax = report.plot_hist2d(level='axis')
        assert type(ax) == plt.Axes