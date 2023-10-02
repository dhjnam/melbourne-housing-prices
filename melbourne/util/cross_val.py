import numpy as np
import pandas as pd
import seaborn as sns

class NoOptimalParamsError(Exception):
    def __init__(self, min_avg, max_dev, params):
        self.min_avg = min_avg
        self.max_dev = max_dev
        self.params = params
        self.message = f'No parameter dict has avg score > {min_avg} and standard deviation < {max_dev}'
        super().__init__(self.message)

class Report:
    """
    Report object containing information on ``cross_val_score``s over parameter
    grid.

    Indices and their optimal parameters are stored in the attribute 
    ``optimal_params``, a DataFrame indexed by the ``min_avg`` and ``max_dev`` 
    constraining those parameters.
    """

    def __init__(self, scores, params):
        self.scores = scores
        self.avgs = np.mean(scores, axis=1)
        self.devs = np.std(scores, axis=1)
        self.params = params
        
        optimal_params = pd.DataFrame(
            index=pd.MultiIndex(levels=[[], []], codes=[[], []], name=['min_avg', 'max_dev']),
            columns=['idx', 'param'],
        )
        self.optimal_params = optimal_params.astype({'idx': 'int64'})

    def compute_optimal_param(
        self, 
        min_avg: float=0, 
        max_dev: float=1,
        objective: str='avg' or 'dev'
    ) -> dict | NoOptimalParamsError:
        """
        Compute parameters from the parameter grid ``self.params`` with best
        score, meaning that the ``cv`` many cross validation scores from a 
        cross validation in ``CrossVal.grid_cv`` scored on average at least
        as good as ``avg`` and deviated no more than ``dev`` from that 
        ``mean``.

        Parameters
        ----------
        min_avg : float, default=0
            Score must have been on average at least as good as ``avg``.
        max_dev : float, default=1
            Score must not deviate more than ``dev`` from ``avg``. This 
            indicates that the parameters yield a stable estimation.
        objective : {"avg", "dev"}, default "avg"
            If ``"avg"`` optimize 
        Returns
        -------
        dict or NoOptimalParamsError
            The parameter dictionary with maximal average score 
            (if ``objective="avg"``) or minimal standard deviation 
            (if ``objective="dev"``) among the parameter dictionaries with 
            average score at least ``min_avg`` and standard deviation at most 
            ``max_dev``.
            If no such parrameter dictionary exists, raises 
            ``NoOptimalParamsError``.
        """
        idxes = np.argwhere((self.avgs > min_avg) & (self.devs < max_dev))
        if len(idxes) > 0:
            if objective == 'avg':
                argmax_avgs_in_idxes = np.argmax(self.avgs[idxes])
                idx = idxes[argmax_avgs_in_idxes][0]
            elif objective == 'dev':
                argmin_devs_in_idxes = np.argmin(self.devs[idxes])
                idx = idxes[argmin_devs_in_idxes][0]
            self.optimal_params.loc[(min_avg, max_dev), :] = [idx, self.params[idx]]
            # For some strange reason, after insertion, pandas infers the dtype
            # in ``idx`` as ``float64`` 
            self.optimal_params = self.optimal_params.astype({'idx': 'int64'})
            return self.params[idx]
        raise NoOptimalParamsError(min_avg, max_dev, self.params)

    def plot_hist(self, level: str='figure' or 'axis'):
        """
        Plot histogram of average scores ``avgs``.
        
        Params
        ------
        level : {"figure", "axis"}, default="figure"
            Whether seaborn should make a figure-level or axis-level plot
        """
        if level == 'figure':
            return sns.displot(x=self.avgs)
        elif level == 'axis':
            return sns.histplot(x=self.avgs)

    def plot_hist2d(self, level: str='figure' or 'axis'):
        """
        Plot 2D histogram with average scores ``avgs`` and standard deviations 
        ``devs``
        """
        if level == 'figure':
            return sns.displot(x=self.avgs, y=self.devs)
        elif level == 'axis':
            return sns.histplot(x=self.avgs, y=self.devs)

class CrossVal:
    """
    Generic class to perform cross validation on a DataFrame.
    Notice that cross validation should be performed on the training data.
    The user is responsible himself to hold back a test data set for final 
    testing purposes.

    See also https://scikit-learn.org/stable/modules/cross_validation.html

    Parameters
    ----------
    df : pd.DataFrame
        The ``DataFrame`` cross validation is performed on.
        It is assumed that ``df`` comes in complete.
        Any subclass should therefore deal with dropping or imputing missing entries in the data.
    features : list[str]
        List of strings containing the training features.
    target: str
        Prediction target.
    """


    def __init__(
        self, 
        df: pd.DataFrame, 
        features: list[str], 
        target: str, 
    ):
        self.df = df
        self.X = self.df.loc[:, features].to_numpy()
        self.y = self.df[target].to_numpy()


    def grid_cross_val_score(
        self, 
        estimator, 
        params: list[dict], 
        cv=4, 
        n_jobs=6
    ) -> Report:
        """
        Cross validation for estimator with parameters from a parameter grid

        Parameters
        ----------
        estimator : 
            The estimator (regressor or classifier) for which to perform the 
            cross validation.
        params : list of dicts
            The parameters passed to ``model_class``
            ``params[i] = { <key 1>: <value 1>, ..., <key n>: <value n> }``

        Returns
        -------
        report : Report
            The ``Report`` object contains all relevant information on the 
            cross validation scores.
        """
        from sklearn.model_selection import cross_val_score
        # scores is `np.array` holding the scores for each of the `cv` many validations
        scores = np.zeros(shape=(len(params), cv))
        for i, param in enumerate(params):
            estimator.set_params(**param)

            scores[i, :] = cross_val_score(estimator, self.X, self.y, cv=cv, n_jobs=n_jobs)
        
        report = Report(scores=scores, params=params)
        return report
