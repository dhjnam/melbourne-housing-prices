import pytest
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from cross_val import Report, NoOptimalParamsError


class TestReport:

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
            assert report.optimal_params.dtypes.iloc[0] == 'int64'
            assert report.optimal_params.dtypes.iloc[1] == object

        min_avg, max_dev = mu / 2, 2 * sigma
        
        optimal_param = report.compute_optimal_param(min_avg=min_avg, max_dev=max_dev, objective='avg')
        assert report.optimal_params.dtypes.iloc[0] == 'int64'
        assert report.optimal_params.dtypes.iloc[1] == object

        idx, optimal_parameter = report.optimal_params.loc[(min_avg, max_dev)]

        constrained_avgs = report.avgs[ (report.avgs > min_avg) & (report.devs < max_dev) ]
        assert report.avgs[idx] > min_avg
        assert report.devs[idx] < max_dev
        assert np.max(constrained_avgs) == report.avgs[idx]
        assert optimal_parameter == report.optimal_params.loc[(min_avg, max_dev), 'param']
        assert optimal_param == optimal_parameter

        optimal_param = report.compute_optimal_param(min_avg=min_avg, max_dev=max_dev, objective='dev')
        assert report.optimal_params.dtypes.iloc[0] == 'int64'
        assert report.optimal_params.dtypes.iloc[1] == object

        idx, optimal_parameter = report.optimal_params.loc[(min_avg, max_dev)]

        constrained_devs = report.devs[ (report.avgs > min_avg) & (report.devs < max_dev) ]
        assert report.avgs[idx] > min_avg
        assert report.devs[idx] < max_dev
        assert np.min(constrained_devs) == report.devs[idx]
        assert optimal_parameter == report.optimal_params.loc[(min_avg, max_dev), 'param']
        assert optimal_param == optimal_parameter

    # @pytest.mark.skip
    def test_report_plot_hist(self, report_mu_sigma):

        report, mu, sigma = report_mu_sigma

        facet_grid = report.plot_hist()
        assert isinstance(facet_grid, sns.FacetGrid)
        
        facet_grid = report.plot_hist(level='figure')
        assert isinstance(facet_grid, sns.FacetGrid)
        
        ax = report.plot_hist(level='axis')
        assert isinstance(ax, plt.Axes)

    # @pytest.mark.skip
    def test_report_plot_hist2d(self, report_mu_sigma):
        report, mu, sigma = report_mu_sigma

        facet_grid = report.plot_hist2d()
        assert isinstance(facet_grid, sns.FacetGrid)
        
        facet_grid = report.plot_hist2d(level='figure')
        assert isinstance(facet_grid, sns.FacetGrid)
        
        ax = report.plot_hist2d(level='axis')
        assert isinstance(ax, plt.Axes)