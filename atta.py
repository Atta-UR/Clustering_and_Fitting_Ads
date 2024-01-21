import sklearn.cluster as cluster
import cluster_tools as ct
import errors as err
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def extract_file_data(address):
    """
    Reads a CSV file and returns a pandas DataFrame.

    """
    dataframe = pd.read_csv(address, skiprows=4)
    dataframe = dataframe.drop(
        columns=[
            'Country Code',
            'Indicator Name',
            'Indicator Code',
            'Unnamed: 67'])
    return dataframe


def fit_exponential_and_predict(df, Country, Feature, tit, tit_fore):
    """
    Funtion for fitting and predicting.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    Country : TYPE
        DESCRIPTION.
    Feature : TYPE
        DESCRIPTION.
    tit : TYPE
        DESCRIPTION.
    tit_fore : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # fit exponential growth
    popt, pcorr = opt.curve_fit(exp_growth, df.index, df[Country],
                                p0=[4e3, 0.001])
    # much better
    df["pop_exp"] = exp_growth(df.index, *popt)
    plt.figure()
    plt.plot(df.index, df[Country], label="data")
    plt.plot(df.index, df["pop_exp"], label="fit", color="lightcoral")
    plt.legend()
    plt.xlabel('Years')
    plt.ylabel(Feature)
    plt.title(tit)
    plt.savefig(Country + '.png', dpi=300)
    years = np.linspace(1970, 2030)
    pop_exp = exp_growth(years, *popt)
    sigma = err.error_prop(years, exp_growth, popt, pcorr)
    low = pop_exp - sigma
    up = pop_exp + sigma
    plt.figure()
    plt.title(tit_fore)
    plt.plot(df.index, df[Country], label="data")
    plt.plot(years, pop_exp, label="Forecast", color="lightcoral")
    # plot error ranges with transparency
    plt.fill_between(years, low, up, alpha=0.5, color="lightcoral")
    plt.legend(loc="upper left")
    plt.xlabel('Years')
    plt.ylabel(Feature)
    plt.savefig(Country + '_forecast.png', dpi=300)
    plt.show()


def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    f = scale * np.exp(growth * (t - 1960))
    return f


def Con_data(df, country_name, start_year, end_year):
    """
    Funtion to Extract data based on Countries.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    country_name : TYPE
        DESCRIPTION.
    start_year : TYPE
        DESCRIPTION.
    end_year : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    df = df.T
    df.columns = df.iloc[0]
    df = df.drop(['Country Name'])
    df = df[[country_name]]
    df.index = df.index.astype(int)
    df = df[(df.index > start_year) & (df.index <= end_year)]
    df[country_name] = df[country_name].astype(float)
    return df


def data_clustering(
        df,
        ind1,
        ind2,
        xlabel,
        ylabel,
        tit,
        n_clu_cen,
        df_fit,
        df_min,
        df_max):
    """
    Function for the clustering of Data.
    """
    nc = n_clu_cen  # number of cluster centres
    kmeans = cluster.KMeans(n_clusters=nc, n_init=10, random_state=0)
    kmeans.fit(df_fit)
    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    plt.figure(figsize=(8, 8))
    # scatter plot with colours selected using the cluster numbers
    # now using the original dataframe
    scatter = plt.scatter(df[ind1], df[ind2], c=labels, cmap="tab10")
    # colour map Accent selected to increase contrast between colours
    # rescale and show cluster centres
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:, 0]
    yc = scen[:, 1]
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(tit)
    plt.savefig('Clustering_plot.png', dpi=300)
    plt.show()


def Find_n_Elbow(df_fit):
    """
    Find Number of Clusters Using Elbow Method.

    Parameters
    ----------
    df_fit : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sse1 = []
    for i in range(1, 11):
        kmeans = cluster.KMeans(n_clusters=i, init='k-means++',
                                max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df_fit)
        sse1.append(kmeans.inertia_)
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.plot(range(1, 11), sse1, color='red')
    axs.set_title('Elbow Method')
    axs.set_xlabel('Number of clusters')
    axs.set_ylabel('SSE')


GDP_per_capita_current_US = extract_file_data('GDP_per_capita_current_US$.csv')
Urban_population_of_total_population = extract_file_data(
    'Urban_population _%_of_total_population.csv')
country = 'India'
df_GD = Con_data(GDP_per_capita_current_US, 'India', 1960, 2020)
df_UP = Con_data(Urban_population_of_total_population, 'India', 1960, 2020)

# df_fit, df_min, df_max = ct.scaler(df_cluster)

df = pd.merge(df_GD, df_UP, left_index=True, right_index=True)
df = df.rename(
    columns={
        country +
        "_x": 'GDP_per_capita_current_US',
        country +
        "_y": 'Urban Population'})
df_fit, df_min, df_max = ct.scaler(df)
Find_n_Elbow(df_fit)

data_clustering(
    df,
    'GDP_per_capita_current_US',
    'Urban Population',
    'GDP Per Capita Current US$',
    'Urban population %',
    'GDP Per Capita Current US$ vs Urban Population % In India',
    3,
    df_fit,
    df_min,
    df_max)
fit_exponential_and_predict(
    df_UP,
    'India',
    'Urban Population Percentage',
    "Urban Population In India 1960-2020",
    "Urban Population In India Forecast Untill 2030")
fit_exponential_and_predict(
    df_GD,
    'India',
    'GDP per Capita US$',
    "GDP Per Capita In India 1960-2020",
    "GDP Per Capita In India Forecast Untill 2030")
