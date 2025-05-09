#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:07:50 2025

@author: malihafarahmand
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class AutoOutlierFlagger(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=10.0):
        self.threshold = threshold
        self.outlier_columns_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.outlier_columns_ = []
        for col in X_df.select_dtypes(include=[np.number]).columns:
            q1 = X_df[col].quantile(0.25)
            q3 = X_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_ratio = ((X_df[col] < lower) | (X_df[col] > upper)).mean() * 100
            if outlier_ratio > self.threshold:
                self.outlier_columns_.append(col)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in self.outlier_columns_:
            X_df[col + "_outlier_flag"] = 0
            q1 = X_df[col].quantile(0.25)
            q3 = X_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            X_df[col + "_outlier_flag"] = ((X_df[col] < lower) | (X_df[col] > upper)).astype(int)
        return X_df

class DropLowVariance(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.to_drop_ = [col for col in X_df.columns if X_df[col].std() < self.threshold]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        return X_df.drop(columns=self.to_drop_, errors='ignore')

class DropHighlyCorrelated(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        corr_matrix = X_df.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.threshold)]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        return X_df.drop(columns=self.to_drop_, errors='ignore')
