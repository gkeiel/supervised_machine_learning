import time_series_prediction_functions as tspr


def main():
    # generate time series data
    data = tspr.time_series()    

    # prediction from AR model
    y_pred_ar = tspr.arx_model(data)

    # prediction from decision tree
    y_pred_tree, y_test, split_idx = tspr.decision_tree(data)

    # metrics
    mse_ar, mse_tree = tspr.metrics(y_pred_ar, y_pred_tree, data['y'], y_test)

    # plot
    tspr.plot_res(data['y'], data['t'], y_pred_ar, y_pred_tree, mse_ar, mse_tree, split_idx)


if __name__ == "__main__":
    main()