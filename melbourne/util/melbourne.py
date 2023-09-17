class TrainTest:

    def __init__(self, df, features, target):
        """
        It is assumed that df comes in complete.
        Any subclass should therefore deal with dropping or imputing missing entries in the data.
        """
        self.df = df
        self.X = self.df.loc[:, features]
        self.y = self.df[target]

    def train_test_split(self, random_state=0, train_size=None, test_size=None):
        from sklearn.model_selection import train_test_split
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.X, self.y, random_state=random_state, train_size=train_size, test_size=test_size)

    def over_under_fit(self, algorithm, metric, params, random_state=0, train_size=None, test_size=None):
        """ 
        `params` must be an array of kwargs. They are passed when `model.fit` is called.
        param[i] = { 'key1': value1, 'key2': value2, ..., 'keyN': valueN }
        """
        # The r2_score ranges from 0 to 1 and measures the quality of a regressor.
        # The higher the score, the lower the error metric!
        # In essence, the r2_score tells us how advanced our prediction is compared to 
        # the naive way of always using the mean to predict y.
        from sklearn.metrics import r2_score
        self.train_test_split(random_state=random_state, train_size=train_size, test_size=test_size)
        self.fits = []
        self.errors = []
        self.r2_scores = []
        for param in params:
            model = algorithm(**param)
            self.fits.append(model.fit(self.train_X, self.train_y))
            pred_y = model.predict(self.test_X)
            self.errors.append(metric(self.test_y, pred_y))
            self.r2_scores.append(r2_score(self.test_y, pred_y))


# TODO: find a way to deal with uncomplete data.
class Melbourne(TrainTest):

    def __init__(self, df, frac=1):
        
        self.target = 'Price'
        self.numerical_features = list(filter(lambda feature: feature in df.columns, [
            'Rooms', 
            'Distance', 
            'Bedroom2', 
            'Bathroom', 
            'Car', 
            'Landsize', 
            'BuildingArea', 
            'YearBuilt', 
            'Latitude', 
            'Longitude', 
            'Propertycount',
        ]))
        self.time_features = list(filter(lambda feature: feature in df.columns, [
            'Date',
            'Month',
        ]))
        self.nominal_features = list(filter(lambda feature: feature in df.columns, [
            'Suburb',
            'Address',
            'Type',
            'Method',
            'SellerG',
            'Postcode',
            'CouncilArea',
            'Regionname',
       ]))
        self.numericals = self.numerical_features + [self.target]

        self.raw_df = df.sample(frac=frac)
        
        # self.long_lat = self.raw_df[['Longitude', 'Latitude']]
        
        self.cbd_lat = -37.814
        self.cbd_long = 144.963
        self.prices = self.raw_df.Price
        self.min_price, self.max_price = self.prices.min(axis=0), self.prices.max(axis=0)
        self.mean_price = self.prices.mean(axis=0)
        self.prices_01 = self.prices.apply(lambda price: (price - self.min_price) / (self.max_price - self.min_price))

        self.numerical_data = self.raw_df[self.numericals]
        self.complete_data = self.numerical_data.dropna()

        super(Melbourne, self).__init__(self.complete_data, self.numerical_features, self.target)


    # TODO: Name reminds too munch on train_test_SPLIT, though no connection.
    # Rather name this separate_q
    # Change all usages
    def split_q(self, q, frac=1):
        self.q_separator = self.prices.quantile(q=q)
        return Melbourne(self.raw_df.loc[self.prices <= self.q_separator], frac=frac), \
            Melbourne(self.raw_df.loc[self.prices > self.q_separator], frac=frac)
    
    def linear_regression(self):
        from sklearn.linear_model import LinearRegression
        self.reg = LinearRegression(fit_intercept=True).fit( X=self.X, y=self.y )

    def pca(self, n=3):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n)
        self.pca_X = pca.fit(self.X)

