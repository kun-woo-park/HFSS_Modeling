import numpy as np
from custom_modules import hfss_data_load, analytic_data_load, remove_outliers, model_train, mse_cardinality_compare, \
    plot_mse_card, plot_train_result, plot_hfss_analytic_prediction

if __name__ == "__main__":
    data_path = "./table.xlsx"                  # data path
    x_train_hfss, y_train_hfss = hfss_data_load(data_path)     # load data for hfss

    gamma_vals = np.logspace(-3, 1, 100)       # set gamma(11 regularize term)
    nb_epochs = 1000                            # set training epochs
    # compare mse and cardinality
    mse, card = mse_cardinality_compare(x_train_hfss, y_train_hfss, gamma_vals, nb_epochs)
    plot_mse_card(mse, card, gamma_vals)

    # train hfss model with gamma = 0.001
    model_hfss, loss = model_train(x_train_hfss, y_train_hfss, nb_epochs=nb_epochs)
    predicted_hfss = np.squeeze(model_hfss(x_train_hfss).detach().numpy())
    label_hfss = y_train_hfss.detach().numpy()
    plot_train_result(predicted_hfss, label_hfss)

    x_train_analytic, y_train_analytic, data = analytic_data_load(data_path)     # load data for analytic

    # train analytic model with gamma = 0.001
    model_analytic, loss = model_train(x_train_analytic, y_train_analytic, nb_epochs=nb_epochs)
    predicted_analytic = np.squeeze(model_analytic(x_train_analytic).detach().numpy())
    label_analytic = y_train_analytic.detach().numpy()
    plot_train_result(predicted_analytic, label_analytic)

    # remove outliers
    out_lier_index = np.nonzero(np.square(predicted_analytic-label_analytic) > 0.5)
    x_train_analytic, y_train_analytic, data = remove_outliers(data, out_lier_index)

    # re-train analytic model with gamma = 0.001
    model_analytic, loss = model_train(x_train_analytic, y_train_analytic, nb_epochs=nb_epochs)
    predicted_analytic = np.squeeze(model_analytic(x_train_analytic).detach().numpy())
    label_analytic = y_train_analytic.detach().numpy()
    plot_train_result(predicted_analytic, label_analytic)

    # plot compare with hfss, analytic, prediction results
    predicted_analytic = np.exp(predicted_analytic)
    hfss = data[:, 6]
    analytic = data[:, 7]
    plot_hfss_analytic_prediction(hfss, analytic, predicted_analytic)

    # print model performance
    print('MSE_Predicted:', np.linalg.norm(predicted_analytic-analytic)**2/len(analytic))
    print('RMSE_Predicted:', np.sqrt(
        np.linalg.norm(predicted_analytic-analytic)**2/len(analytic)))
    print('Percentage error_Predicted:', np.sum(
        np.abs(predicted_analytic-analytic)/analytic)/len(analytic)*100, '%')
    print('MSE_HFSS:', np.linalg.norm(hfss-analytic)**2/len(analytic))
    print('RMSE_HFSS:', np.sqrt(np.linalg.norm(hfss-analytic)**2/len(analytic)))
    print('Percentage error_HFSS:', np.sum(
        np.abs(hfss-analytic)/analytic)/len(analytic)*100, '%')


