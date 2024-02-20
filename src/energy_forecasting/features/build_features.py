import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.entities.feature_params import FeatureParams


def parse_cat_params(params: FeatureParams) -> tuple:

    cat_col_params = [col.split(';') for col in params.categorical_features]
    cat_features_name_list = []
    cat_features_len_list = []

    for i in cat_col_params:
        cat_features_name_list.append(i[0])
        cat_features_len_list.append(i[1])

    return cat_col_params, cat_features_name_list, cat_features_len_list


def build_numerical_pipeline_w_scaler() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("custom_scaler", StandardScaler()),
        ]
    )
    return num_pipeline


def build_numerical_pipeline_wo_scaler() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean"))
        ]
    )
    return num_pipeline


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder(sparse=False)),
        ]
    )
    return categorical_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:

    _, cat_features, _ = parse_cat_params(params)

    if params.normalize_numerical:
        transformer = ColumnTransformer(
            [
                (
                    "categorical_pipeline",
                    build_categorical_pipeline(),
                    cat_features,
                ),
                (
                    "numerical_pipeline",
                    build_numerical_pipeline_w_scaler(),
                    params.numerical_features,
                ),
            ]
        )
    else:
        transformer = ColumnTransformer(
            [
                (
                    "categorical_pipeline",
                    build_categorical_pipeline(),
                    cat_features,
                ),
                (
                    "numerical_pipeline",
                    build_numerical_pipeline_wo_scaler(),
                    params.numerical_features,
                ),
            ]
        )

    return transformer


def featurize(df_train):
    url1 = df_train['url'].str.split('//', n=-1, expand=True)[1]
    url2 = url1.str.split('www.', n=-1, expand=True)[1]
    webname = url2.str.split('.', n=-1, expand=True)[0]
    url3 = url2.str.split('.', n=-1, expand=True)[1]
    domain = url3.str.split('/', n=-1, expand=True)[0]
    website_type = url3.str.split('/', n=-1, expand=True)[1]
    url4 = url3.str.split('/', n=-1, expand=True)[2]
    website_type2 = url4.str.split('/', n=-1, expand=True)[0]

    df_train["website"] = webname
    df_train["website_type"] = website_type
    df_train["website_type2"] = website_type2
    df_train["domain"] = domain

    df_train['alchemy_category_score'] = pd.to_numeric(df_train['alchemy_category_score'], errors='coerce')
    df_train["is_news"] = pd.to_numeric(df_train["is_news"], errors='coerce')
    df_train["news_front_page"] = pd.to_numeric(df_train["news_front_page"], errors='coerce')

    return df_train


def prepare_dataset(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:

    df_data = featurize(df)
    df_data['website_type'] = df_data['website_type'].replace({'2007': 'YEAR', '2008': 'YEAR',
                                                                 '2009': 'YEAR', '2010': 'YEAR',
                                                                 '2011': 'YEAR', '2012': 'YEAR',
                                                                 '2013': 'YEAR'})

    col_params, _, _ = parse_cat_params(params)

    for col in col_params:
        df_data = get_most_freq_cat(df_data, col)

    return df_data


def get_most_freq_cat(df: pd.DataFrame, col_params: list) -> pd.DataFrame:

    col_name = col_params[0]
    cat_nums = int(col_params[1])
    # берем cat_nums самых популярных категорий
    most_freq_val = list(df[col_name].value_counts().index[:cat_nums])
    df[col_name].where(df[col_name].isin(most_freq_val), 'other', inplace=True)

    return df


def get_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_col]