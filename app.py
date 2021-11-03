from flask import Flask, render_template, request

import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pandas_profiling import ProfileReport
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from joblib import Parallel, delayed
import time
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,RidgeCV,LassoCV, ElasticNet , ElasticNetCV,LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

app = Flask(__name__)


# class Linearregression():
#     def __init__(self, file_path, predicted_col):
#         self.file_path = file_path
#         self.predicted_col = predicted_col
#         logging.basicConfig(filename='linear_regression.log', level=logging.DEBUG,
#                             format='%(asctime)s:%(levelname)s:%(message)s')
#         logging.info('Linearregression class object is created.')
#
#     def load_data(self):
#         """
#         Load csv file as pandas dataframe.
#
#
#         Parameters
#         ----------
#         None
#
#         Returns:
#         ----------
#         None
#         """
#         logging.info('Dataset is getting loaded as pandas dataframe.')
#         try:
#             self.original_df = pd.read_csv(self.file_path)
#         except FileNotFoundError:
#             logging.error("File not found: exception occured while loading csv as pandas dataframe.")
#         except pd.errors.EmptyDataError:
#             logging.error("No data: exception occured while loading csv as pandas dataframe.")
#         except pd.errors.ParserError:
#             logging.errornt("Parse error: exception occured while loading csv as pandas dataframe.")
#         except Exception as e:
#             logging.error("{} occured while loading csv as pandas dataframe.".format(str(e)))
#
#     def pandas_profiling(self, output_html):
#         """
#         Create pandas profiling report for the loaded dataset and
#         save it as a html file.
#
#         Parameters
#         ----------
#         output_html: Output htmla file named to be saved.
#
#         Returns:
#         ----------
#         None
#         """
#         logging.info('Pandas profiling report is started.')
#         pf = ProfileReport(self.original_df)
#         pf.to_widgets()
#         pf.to_file(output_html)
#         logging.info('Pandas profiling report is finished ans saved inside {}.'.format(output_html))
#
#     def check_NaN(self):
#         """
#         Calculate the number NaN values present in the dataset.
#
#         Parameters
#         ----------
#         None.
#
#         Returns:
#         ----------
#         None.
#         """
#         try:
#             logging.info('Total number of NaN inside dataset is getting calculated.')
#             return self.original_df.isna().sum().sum()
#         except Exception as e:
#             logging.error("{} occured while calculating total number of NaN inside dataset.".format(str(e)))
#             return None
#
#     def view_multicolinearity_by_vif(self):
#         """
#         This functions helps to judge the mulicolinearity among independent feature by calculating their
#         VIF (Variable Inflation Factors).
#
#         Parameters
#         ----------
#         None.
#
#         Returns:
#         ----------
#         None.
#         """
#         logging.info('VIF values for all features inside dataset will be calculated.')
#         try:
#             result = self.original_df.copy()
#             ignore_columns = ["UDI", "Product ID", "Type"]
#             X_variables_col = []
#             for feature_name in result.columns:
#                 if feature_name not in ignore_columns:
#                     X_variables_col.append(feature_name)
#             self.X_variables = result[X_variables_col]
#             self.vif_data = pd.DataFrame()
#             self.vif_data["feature"] = self.X_variables.columns
#             self.vif_data["VIF"] = [variance_inflation_factor(self.X_variables.values, i) for i in
#                                     range(len(self.X_variables.columns))]
#             print(self.vif_data)
#         except Exception as e:
#             logging.error("{} occured while calculating VIF values for all features inside dataset.".format(str(e)))
#
#     def drop_multicolinearity_by_vif(self, vif_thresh):
#         """
#         This functions drops tyhose columns whose values are more than threshold VIF passed as parameter.
#
#         Parameters
#         ----------
#         vif_thresh: This is the threshold VIF value above which dataset column will be dropped.
#
#         Returns:
#         ----------
#         None.
#         """
#         logging.info('All features with VIF more than {} will be dropped from the dataset.'.format(vif_thresh))
#         try:
#             X = self.X_variables
#             variables = [X.columns[i] for i in range(X.shape[1])]
#             dropped = True
#             while dropped:
#                 dropped = False
#                 vif = Parallel(n_jobs=-1, verbose=5)(
#                     delayed(variance_inflation_factor)(X[variables].values, ix) for ix in range(len(variables)))
#
#                 maxloc = vif.index(max(vif))
#                 if max(vif) > vif_thresh:
#                     if X[variables].columns[maxloc] is not self.predicted_col:
#                         logging.info(
#                             time.ctime() + ' dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(
#                                 maxloc))
#                         variables.pop(maxloc)
#                         dropped = True
#
#             logging.info('Remaining variables:')
#             logging.info([variables])
#             self.final_df = X[[i for i in variables]]
#         except Exception as e:
#             logging.error(
#                 "{} occured while droping some of the feature from dataset based on vif threshold.".format(str(e)))
#
#     def create_X_Y(self):
#         """
#         Create and reshuffle dataset based on Independent and dependent feature name.
#
#         Parameters
#         ----------
#         None.
#
#         Returns:
#         ----------
#         None.
#         """
#         logging.info('New dataset is created after reschuffle based on dependent feature')
#         try:
#             self.Y = self.original_df[[self.predicted_col]]
#             feature_name = self.final_df.columns.tolist()
#             self.X = self.final_df[feature_name]
#         except Exception as e:
#             logging.error("{} occured while dataset reschuffle based on dependent feature.".format(str(e)))
#
#     def build_model(self):
#         """
#         Build linear regression model.
#
#         Parameters
#         ----------
#         None.
#
#         Returns:
#         ----------
#         None.
#         """
#         logging.info('Linear regression model will be built now.')
#         try:
#             self.linear = LinearRegression()
#             self.model = self.linear.fit(self.x_train, self.y_train)
#         except Exception as e:
#             logging.error("{} occured while dbuilding linear regression model.".format(str(e)))
#
#     def save_model(self, file_name):
#         """
#         Save the linear regresion model based on the input file name.
#
#         Parameters
#         ----------
#         file_name: linear regression model will be saved with this file name.
#
#         Returns:
#         ----------
#         None.
#         """
#         logging.info('Save the linear regression model into file: {}.'.format(file_name))
#         try:
#             pickle.dump(self.model, open(file_name, 'wb'))
#         except Exception as e:
#             logging.error("{} occured while saving linear regression model.".format(str(e)))
#
#     def calc_accuracy(self):
#         """
#         Calculate the accuracy of the linear regression model.
#
#         Parameters
#         ----------
#         None.
#
#         Returns:
#         ----------
#         Returns the accuracy of the model.
#         """
#         logging.info('Accuracy of the model will be calculated here.')
#         try:
#             return self.linear.score(self.x_test, self.y_test)
#         except Exception as e:
#             logging.error("{} occured while calculating accuracy linear regression model.".format(str(e)))
#             return None
#
#     def predict(self, test_case):
#         """
#         Predict the dependent feature based on the input test case.
#
#         Parameters
#         ----------
#         test_case: It is the independent variable list value.
#
#         Returns:
#         ----------
#         Returns the predicted feature.
#         """
#         logging.info('Prediction will be done for the testcase {}.'.format(test_case))
#         try:
#             return self.linear.predict(test_case)
#         except Exception as e:
#             logging.error("{} occured while predicting dependent feature.".format(str(e)))
#             return None
#
#     def train_test_split(self, test_size, random_state):
#         self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.independent_scaled, self.Y,
#                                                                                 test_size=test_size,
#                                                                                 random_state=random_state)
#
#     def adj_r2(self, x, y):
#         r2 = self.linear.score(x, y)
#         n = x.shape[0]
#         p = x.shape[1]
#         adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
#         return adjusted_r2
#
#     def build_lasso_model(self, cv, max_iter):
#         self.lassocv = LassoCV(alphas=None, cv=cv, max_iter=max_iter, normalize=True)
#         self.lassocv.fit(self.x_train, self.y_train)
#         self.lasso_lr = Lasso(alpha=self.lassocv.alpha_)
#         self.lasso_model = self.lasso_lr.fit(self.x_train, self.y_train)
#
#     def save_lasso_model(self, file_name):
#         """
#         Save the linear regresion model based on the input file name.
#
#         Parameters
#         ----------
#         file_name: linear regression model will be saved with this file name.
#
#         Returns:
#         ----------
#         None.
#         """
#         logging.info('Save lasso regularized linear regression model into file: {}.'.format(file_name))
#         try:
#             pickle.dump(self.lasso_model, open(file_name, 'wb'))
#         except Exception as e:
#             logging.error("{} occured while saving lasso regularized linear regression model.".format(str(e)))
#
#     def calc_lasso_accuracy(self):
#         """
#         Calculate the accuracy of the linear regression model.
#
#         Parameters
#         ----------
#         None.
#
#         Returns:
#         ----------
#         Returns the accuracy of the model.
#         """
#         logging.info('Accuracy of the lasso regularizd model will be calculated here.')
#         try:
#             return self.lasso_lr.score(self.x_test, self.y_test)
#         except Exception as e:
#             logging.error(
#                 "{} occured while calculating accuracy lasso regularized linear regression model.".format(str(e)))
#             return None
#
#     def build_ridge_model(self, cv):
#         self.ridgecv = RidgeCV(alphas=np.random.uniform(0, 10, 50), cv=cv, normalize=True)
#         self.ridgecv.fit(self.x_train, self.y_train)
#         self.ridge_lr = Ridge(alpha=self.ridgecv.alpha_)
#         self.ridge_model = self.ridge_lr.fit(self.x_train, self.y_train)
#
#     def save_ridge_model(self, file_name):
#         """
#         Save the linear regresion model based on the input file name.
#
#         Parameters
#         ----------
#         file_name: linear regression model will be saved with this file name.
#
#         Returns:
#         ----------
#         None.
#         """
#         logging.info('Save ridge regularized linear regression model into file: {}.'.format(file_name))
#         try:
#             pickle.dump(self.ridge_model, open(file_name, 'wb'))
#         except Exception as e:
#             logging.error("{} occured while saving ridge regularized linear regression model.".format(str(e)))
#
#     def calc_ridge_accuracy(self):
#         """
#         Calculate the accuracy of the linear regression model.
#
#         Parameters
#         ----------
#         None.
#
#         Returns:
#         ----------
#         Returns the accuracy of the model.
#         """
#         logging.info('Accuracy of the ridge regularizd model will be calculated here.')
#         try:
#             return self.ridge_lr.score(self.x_test, self.y_test)
#         except Exception as e:
#             logging.error(
#                 "{} occured while calculating accuracy of ridge regularizd linear regression model.".format(str(e)))
#             return None
#
#     def build_elasticnet_model(self, cv):
#         self.elastic = ElasticNetCV(alphas=None, cv=cv)
#         self.elastic.fit(self.x_train, self.y_train)
#         self.elastic_lr = ElasticNet(alpha=self.elastic.alpha_, l1_ratio=self.elastic.l1_ratio_)
#         self.elastic_model = self.elastic_lr.fit(self.x_train, self.y_train)
#
#     def save_elasticnet_model(self, file_name):
#         """
#         Save the linear regresion model based on the input file name.
#
#         Parameters
#         ----------
#         file_name: linear regression model will be saved with this file name.
#
#         Returns:
#         ----------
#         None.
#         """
#         logging.info('Save elastic regularized linear regression model into file: {}.'.format(file_name))
#         try:
#             pickle.dump(self.elastic_model, open(file_name, 'wb'))
#         except Exception as e:
#             logging.error("{} occured while saving elastic regularized linear regression model.".format(str(e)))
#
#     def calc_elasticnet_accuracy(self):
#         """
#         Calculate the accuracy of the linear regression model.
#
#         Parameters
#         ----------
#         None.
#
#         Returns:
#         ----------
#         Returns the accuracy of the model.
#         """
#         logging.info('Accuracy of the elasticnet regularizd model will be calculated here.')
#         try:
#             return self.elastic_lr.score(self.x_test, self.y_test)
#         except Exception as e:
#             logging.error(
#                 "{} occured while calculating accuracy of elasticnet regularizd linear regression model.".format(
#                     str(e)))
#             return None
#
#     def standardize_train(self):
#         self.scaler = StandardScaler()
#         self.independent_scaled = self.scaler.fit_transform(self.X)
#
#     def scale_test(self, test_data):
#         scaled_data = self.scaler.transform(test_data)
#         return scaled_data

@app.route('/')
def form():
    return render_template('form.html')


@app.route('/data/', methods=['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"

    if request.method == "POST":
        frontal_axis_reading = float(request.form['frontal_axis_reading(g)'])
        vertical_axis_reading = float(request.form['vertical_axis_reading(g)'])
        lateral_axis_reading = float(request.form['lateral_axis_reading(g)'])
        if request.form.get("Predict_Activity_using_SVC"):
            file = './notebook/svc_model'
        elif request.form.get("Predict_Activity_using_KNN"):
            file = './notebook/knn_model'
        elif request.form.get("Predict_Activity_using_Decision_Tree"):
            file = './notebook/dt_model'
        elif request.form.get("Predict_Activity_using_Random_Forest"):
            file = './notebook/rf_model'
        elif request.form.get("Predict_Activity_using_Stacking"):
            file = './notebook/stack_model'
        saved_model = pickle.load(open(file, 'rb'))
        prediction = saved_model.predict([[frontal_axis_reading,vertical_axis_reading,lateral_axis_reading]])
        print('prediction is', prediction)
        return render_template('results.html', prediction=prediction)


@app.route('/profie_report/', methods=['POST', 'GET'])
def profie_report():
    return render_template('har_profiling.html')


if __name__ == '__main__':
    # linear_regr = Linearregression('challange_dataset.csv', 'Air temperature [K]')
    #
    # # load_data()
    # linear_regr.load_data()
    #
    # # profiling_data()
    # linear_regr.pandas_profiling('ori_df_profiling.html')
    #
    # # fillna()
    # nan_count = linear_regr.check_NaN()
    # print(nan_count)
    #
    # # handle_multicolinearity()
    # linear_regr.view_multicolinearity_by_vif()
    # linear_regr.drop_multicolinearity_by_vif(vif_thresh=10)
    #
    # # create independent feature and dependent feature
    # linear_regr.create_X_Y()
    #
    # # Standardization
    # linear_regr.standardize_train()
    #
    # # Split dataset
    # linear_regr.train_test_split(test_size=0.15, random_state=100)
    #
    # # build_model()
    # linear_regr.build_model()
    #
    # # save_model()
    # linear_regr.save_model('linear_reg.sav')
    #
    # # model_accuracy()
    # accuracy = linear_regr.calc_accuracy()
    # print(accuracy)
    #
    # # build_lasso_model()
    # linear_regr.build_lasso_model(cv=10, max_iter=20000)
    #
    # # save_lasso_model()
    # linear_regr.save_lasso_model('lasso_linear_reg.sav')
    #
    # # lasso_model_accuracy()
    # lasso_accuracy = linear_regr.calc_lasso_accuracy()
    # print(accuracy)
    #
    # # build_ridge_model()
    # linear_regr.build_ridge_model(cv=10)
    #
    # # save_ridge_model()
    # linear_regr.save_ridge_model('ridge_linear_reg.sav')
    #
    # # ridge_model_accuracy()
    # ridge_accuracy = linear_regr.calc_ridge_accuracy()
    # print(ridge_accuracy)
    #
    # # build_elasticnet_model()
    # linear_regr.build_elasticnet_model(cv=10)
    #
    # # save_elasticnet_model()
    # linear_regr.save_elasticnet_model('elastic_linear_reg.sav')
    #
    # # elasticnet_model_accuracy()
    # elasticnet_accuracy = linear_regr.calc_elasticnet_accuracy()
    # print(elasticnet_accuracy)

    app.run(host='localhost', port=5000)