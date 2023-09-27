Feature addition:

`data_preparation.ipynb`:
    * read `melb_data.cvs`
    * Add `Month`
    * write `melb_data.h5`

`price_category.ipynb`:
    * read `melb_data.h5`
    * Add `PriceCategory`
    * write `melb_data_price_categories.h5`

`outliers.ipynb`:
    * read `melb_data_price_categories.h5`
    * Remove outliers from dataset
    * write `melb_data_outliers_removed.h5`

# Observations from `plots.ipynb`

Feature combinations that seem to cluster or patternize the target `PriceCategory`. 
We therefore try to train the following models 
    * decision trees / random forests
    * Support Vector Regressors (SVR) with suitable kernels
on the following feature combinations
    * `BuildingArea`, `Landsize`, `Longitude`, `Latitude`, `Distance`
    * `Rooms`, `Bathroom`, `Bedroom2`, `Car`, `Distance`

