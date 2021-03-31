import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

matplotlib.use("TKAgg")


# function for restore nan values
def data_restore(dat):
    for col_name in dat.columns[:5]:
        temp = 0
        for i in range(len(dat[col_name])):
            if dat.loc[i, col_name] != 0:
                temp = dat.loc[i, col_name]
            if dat.loc[i, col_name] == 0:
                dat.loc[i, col_name] = temp
    return dat


# HFSS data load function
def hfss_data_load(data_path):
    # Restore data
    xls_data = pd.read_excel(data_path)
    table2_dat = xls_data.loc[:, "# of turns":"HFSS"]
    table3_dat = xls_data.loc[:, "# of turns.1":"HFSS.1"]
    table4_dat = xls_data.loc[:, "# of turns.2":"HFSS.2"]
    table2_dat = table2_dat.replace(np.nan, 0)
    table3_dat = table3_dat.replace(np.nan, 0)
    table4_dat = table4_dat.replace(np.nan, 0)
    table2_dat = data_restore(table2_dat)
    table3_dat = data_restore(table3_dat)
    table4_dat = data_restore(table4_dat)

    # Feature engineering
    total_dat_temp = np.vstack(
        [table2_dat.values, table3_dat.values, table4_dat.values])
    del_index = np.where(total_dat_temp[:, 6] == 0)[0]
    total_dat_temp = pd.DataFrame(total_dat_temp, columns=table2_dat.columns)
    total_dat = total_dat_temp.drop(del_index)
    data = total_dat.values
    data = np.delete(data, 100, axis=0)
    log_dat_x = np.log(data[:, :6])
    log_dat_y = np.log(data[:, 6])
    x_train = log_dat_x
    max_x = np.max(x_train, axis=0)

    # normalize by max value
    x_train = x_train / max_x
    y_train = log_dat_y
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)

    return x_train, y_train


# analytic data load function
def analytic_data_load(data_path):
    # Restore data
    xls_data = pd.read_excel(data_path)
    table2_dat = xls_data.loc[:, "# of turns":"Analytic"]
    table3_dat = xls_data.loc[:, "# of turns.1":"Analytic.1"]
    table4_dat = xls_data.loc[:, "# of turns.2":"Analytic.2"]
    table2_dat = table2_dat.replace(np.nan, 0)
    table3_dat = table3_dat.replace(np.nan, 0)
    table4_dat = table4_dat.replace(np.nan, 0)
    table2_dat = data_restore(table2_dat)
    table3_dat = data_restore(table3_dat)
    table4_dat = data_restore(table4_dat)

    # Feature engineering
    total_dat_temp = np.vstack(
        [table2_dat.values, table3_dat.values, table4_dat.values])
    del_index = np.where(total_dat_temp[:, 6] == 0)[0]
    total_dat_temp = pd.DataFrame(total_dat_temp, columns=table2_dat.columns)
    total_dat = total_dat_temp.drop(del_index)
    data = total_dat.values
    data = np.delete(data, 100, axis=0)
    log_dat_x = np.log(data[:, :6])
    log_dat_y = np.log(data[:, 7])
    x_train = log_dat_x
    max_x = np.max(x_train, axis=0)

    # normalize by max value
    x_train = x_train / max_x
    y_train = log_dat_y
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)

    return x_train, y_train, data


# function for remove outliers
def remove_outliers(data, out_lier_index):
    data = np.delete(data, out_lier_index, axis=0)
    log_dat_x = np.log(data[:, :6])
    log_dat_y = np.log(data[:, 7])
    x_train = log_dat_x
    max_x = np.max(x_train, axis=0)
    x_train = x_train / max_x
    y_train = log_dat_y
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    return x_train, y_train, data


# training model
def model_train(x_train, y_train, gamma=0.001, nb_epochs=1000):
    model = nn.Linear(6, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(nb_epochs + 1):
        prediction = torch.squeeze(model(x_train))
        regularity = torch.norm(model.weight, p=1)
        loss = F.mse_loss(prediction, y_train)
        cost = loss + gamma * regularity
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, cost.item()
            ))
    return model, loss


# training model for comparing mse and cardinality
def mse_cardinality_compare(x_train, y_train, gamma_vals, nb_epochs):
    mse = []  # list for save mse
    card = []  # list for save cardinality
    for gamma in gamma_vals:
        model, loss = model_train(x_train, y_train, gamma, nb_epochs)
        print('MSE: {:.6f} Cardinality: {:.6f}, Gamma{:.6f}'.format(
            loss.item(), np.sum(model.weight.detach().numpy() < 1e-6), gamma
        ))
        mse.append(loss.item())
        card.append(np.sum(model.weight.detach().numpy() < 1e-6))
    return mse, card


def plot_train_result(predicted, label):
    plt.plot(label, label)
    plt.scatter(predicted, label)
    plt.xlabel("Predicted value")
    plt.ylabel("True label")
    plt.show()


# plot result of training mse_cardinality_compare
def plot_mse_card(mse, card, gamma_vals):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel(r'gamma')
    ax1.set_ylabel('MSE', color=color)
    ax1.semilogx(gamma_vals, mse, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid()

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('cardinality', color=color)
    ax2.semilogx(gamma_vals, card, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()


# plot compare with hfss, analytic, prediction results
def plot_hfss_analytic_prediction(hfss, analytic, predicted_analytic):
    plt.plot(analytic, analytic)
    plt.scatter(predicted_analytic, analytic, label='Predicted')
    plt.scatter(hfss, analytic, label='HFSS')
    plt.legend()
    plt.show()
