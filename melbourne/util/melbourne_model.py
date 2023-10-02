import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from cross_val import CrossVal

class MelbourneModel(CrossVal):
    """
    Proxy acting as a wrapper for Melbourne housing data.

    Assume that DataFrame is complete enough to perform training and validation.

    Parameters
    ----------
    df : pd.DataFrame
        This data frame contains the melbourne housing data.
    features : list[str]
        List of features to be incoporated to pass to parent class ``CrossVal``
    target : {"Price", "PriceCategory"}, default="Price"
        The target to predict. If ``"Price"``, use a regressor. If ``"PriceCategory"``, use a classifier.
    """

    def __init__(
        self, 
        df: pd.DataFrame, 
        features: list[str],
        random_state=0, 
        train_test_ratio: float=1.0, 
    ):

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

        self.train = df.sample(
            n=int(len(df) * train_test_ratio),
            random_state=random_state
        )
        self.test = df[ ~df.index.isin(self.train.index) ]

        self.min_price, self.max_price = df.Price.min(axis=0), df.Price.max(axis=0)
        self.mean_price = df.Price.mean(axis=0)
        self.prices_01 = df.Price.apply(lambda price: (price - self.min_price) / (self.max_price - self.min_price))

        super(MelbourneModel, self).__init__(df, features=features, target='Price')