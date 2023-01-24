import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
from matplotlib.axis import Axis
import matplotlib.patches as mpatches
from tkinter import messagebox
from tkinter.filedialog import askopenfilename  # Dateidialog
from tkinter import filedialog,simpledialog,messagebox,colorchooser
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
from pandas import DataFrame

import pandas as pd
import numpy as np
import plotly
import datetime
from datetime import date

# Seaborn style
plt.style.use('seaborn')
# plt.style.use('seaborn-colorblind') #alternative
# plt.rcParams['figure.figsize'] = [16, 9] # Plot Size
plt.rcParams['figure.dpi'] = 100 # Plot resolution
warnings.simplefilter(action='ignore', category=FutureWarning)

import seaborn as sns
import scipy.stats as scs

### Statsmodels is a Python module that provides classes and functions for the estimation of many different
# statistical models, as well as for conducting statistical tests, and statistical data exploration.
# An extensive list of result statistics are available for each estimator. The results are tested against
# existing statistical packages to ensure that they are correct. The package is released under the open source
# Modified BSD (3-clause) license. The online documentation is hosted at statsmodels.org.

import statsmodels.api as sm  # conda install -c conda-forge statsmodels
import statsmodels.tsa.api as smt

"""########## Autoregressive Conditional Heteroskedasticity (ARCH) and other tools for financial econometrics,
############# written in Python (with Cython and/or Numba used to improve performance)
"""

from arch import arch_model   # To install this package with conda run one of the following:
# conda install -c conda-forge arch-py
# or as below :
# conda install -c bashtage arch
###################################### Yahoo Finance API ###########################################################

import yfinance as yf   # It’s completely free and super easy to install the library:  pip install yfinance --upgrade --no-cache-dir

"""##################### Using the yfinance Library to Extract Stock Data###########################################

Using the `Ticker` module we can create an object that will allow us to access functions to extract data. 
To do this we need to provide the ticker symbol for the stock, here the company is IBM and the ticker 
symbol is `IBM`.

It’s completely free and super easy to setup- a single line to install the library:

    pip install yfinance --upgrade --no-cache-dir
    
###############################################################################################################

#  /F0010/ Börsenticker Eingabe :  Eröﬀnung eines Dialogfenster um eienen gewünschte
#  Börsenticker eingeben. Der eingegebene Ticker wird als globale Variable deklariert und zur Abfrage via API 
#  mittels Callback funktion verwendet  (s. Pflichtenheft, Seite 6)
"""

def dialog():

    # Globale Variable deklaration
    global mystr

    # show Input Dialog for insert Ticker
    title = "Ticker"
    prompt = "Enter Ticker"

    # Dialog Window
    while True:

        mystr = simpledialog.askstring(title, prompt)
        print(type(mystr))

        try:
            if type(mystr) == str:
            # show Info Messagebox
                res = messagebox.askyesno(title=f'Do you want Ticker {mystr}?')  # immer modal
                print(res, type(res))

                b5.config(state="normal")
                b5a.config(state="normal")
                b5b.config(state="normal")
                b5c.config(state="normal")
                b5e.config(state="normal")
                b5d.config(state="normal")
                b5f.config(state="normal")
                b5j.config(state="normal")
                b5k.config(state="normal")
                b5l.config(state="normal")

        except ValueError:
            res = messagebox.askyesno(title="Please try again?")  # immer modal
            print(res)

            if res:
                print("Select file")
            else:
                b6.invoke()
        else:
            res = messagebox.showinfo(f'API for Ticker {mystr} is ready', "You can start Analyze.")
            print(res)

            if res:
                print("Select file")
            else:
                b6.invoke()

            break


# Activation Menu buttons  after entering the sticker number
    #b5.config(state="normal")
    # b5a.config(state="normal")
    # b5b.config(state="normal")
    # b5c.config(state="normal")
    # b5e.config(state="normal")
    # b5d.config(state="normal")
    # b5f.config(state="normal")
    # b5j.config(state="normal")
    # b5k.config(state="normal")
    # b5l.config(state="normal")


"""
###########   Now we can access functions and variables to extract the type of data we need.###############
"""


"""
#  /F0020/ History Max :  bei Klick auf die "History Max" Taste (zweite Position oben)
#  wird eine Abfrage mittels Callback funktion an API erstellt, um den Höchstwert einer Aktie (Open) 
#  in einem bestimmten Zeitraum (1960 - 2021) abzurufen. Das Ergebnis wird in einem Plotter dargestellt und 
#  in einem visuellen Graph präsentiert. Der Graph kann manuell oder automatischem Modus im Stammverzeichnis 
#  gespeichert werden.(s. Pflichtenheft, Seite 7)
"""

def max_history_stock(): ###### Max Stock price from 1960 - 2021 #################################

    global mystr

    # Download the adjusted prices from Yahoo Finance:

    tic = yf.Ticker(mystr)
    df1 = tic.history( period="max")
    print(df1)
    ax = df1.plot(title=f' {mystr} Historical Chart ')
    plt.show()

    # Dialogfenster
    res = messagebox.askyesno(title="Would you like to continue?")  # immer modal
    print(res)
    if res:
        print("Select file")
    else:
        b6.invoke()
##################### END:  Max Exchange rate from 1960 - 2021 #################################


"""
#  /F0030/ Open Last Month :  bei Klick auf die Open Last Month Taste
#  wird eine Abfrage mittels Callback funktion an API erstellt, um den Wert einer Aktie (Open) 
#  im letzten Monat abzurufen. Das Ergebnis wird in einem Plotter dargestellt und in einem visuellen 
#  Graph präsentiert. Der Graph kann manuell oder automatischem Modus im Stammverzeichnis gespeichert werden.
#  (s. Pflichtenheft, Seite 6)
"""
def last_month(): ########################## Open last Month #####################################################


    global mystr

    # Download the adjusted prices from Yahoo Finance:

    tic = yf.Ticker(mystr)

    # Last Month History
    # ^ returns a named tuple of Ticker objects

    df3= tic.history(titel={mystr}, period="1mo")
    print("Last Month High Stock History",df3)

    ax3 = df3.plot(title=f' {mystr} 1-Month High Stock ')
    plt.show()

    # Dialogfenster
    res = messagebox.askyesno(title="Would you like to continue??")  # immer modal
    print(res)
    if res:
        print("select file")
    else:
        b6.invoke()

################################# END: Open last Month ###################################################################

"""
#  /F0040/  Cash Flow :  bei Klick auf die Cash Flow Taste 
#  wird eine Abfrage mittels Callback funktion an API erstellt, um das Cashflow abzurufen. Das Ergebnis wird in einem 
#  Plotter dargestellt und in einem visuellen Graph präsentiert. Der Graph kann manuell oder automatischem Modus im 
#  Stammverzeichnis gespeichert werden (s. Pflichtenheft, Seite 6)
"""
def cash_flow(): ################ Cashflow ###################################################

    global mystr

    # Download the adjusted prices from Yahoo Finance:
    cf = yf.Ticker(mystr)
    df2 = cf.cashflow
    info = cf.info
    #print(cf.info['country'])
    #print(cf.info['sector'])

    print("Cash Flow", df2)
    ax2 = df2.plot(title=f' {mystr} Cash Flow')

    plt.show()

    # Dialogfenster
    res = messagebox.askyesno(title="Would you like to continue?")  # immer modal
    print(res)
    if res:
        print("select file")
    else:
        b6.invoke()

################ END: Cashflow ###################################################

"""
# /F0050/  Volatility:  bei Klick auf die Volatility Taste wird eine Abfrage mittels Callback funktion an API erstellt,
# um die Monthly realized Volatility vs Log Returns(%) der Aktie von 2000 bis 2021  abzurufen. Das Ergebnis wird in 
# einem Plotter dargestellt und in einem visuellen Graph präsentiert. Der Graph kann manuell oder automatischem Modus 
# im Stammverzeichnis gespeichert werden
"""
def volat_retn(): ################## Log returns vs Monthly realized volatility #########################################

    global mystr

    # Download the adjusted prices from Yahoo Finance:

    df = yf.download({mystr},
                     start='2000-01-01',
                     end='2021-12-31',
                     auto_adjust=False,
                     progress=False)

    # keep only the adjusted close price

    df = df.loc[:, ['Adj Close']]
    df.rename(columns={'Adj Close': 'adj_close'}, inplace=True)

    # calculate simple returns
    df['log_rtn'] = np.log(df.adj_close / df.adj_close.shift(1))

    # remove redundant data
    df.drop('adj_close', axis=1, inplace=True)
    df.dropna(axis=0, inplace=True)

    print(df.head)

    # calculate realized_volatility
    def realized_volatility(x):

        return np.sqrt(np.sum(x ** 2))

    df_rv = df.groupby(pd.Grouper(freq='M')).apply(realized_volatility)
    df_rv.rename(columns={'log_rtn': 'rv'}, inplace=True)

    df_rv.rv = df_rv.rv * np.sqrt(12)

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].plot(df)
    ax[0].set(title=f' {mystr} Log returns', ylabel='Log returns (%)')

    ax[1].plot(df_rv)
    ax[1].set(title=f' {mystr} Monthly realized volatility', ylabel='Monthly volatility')

    # Plotting
    plt.show()

    # Dialogfenster
    res = messagebox.askyesno(title="Would you like to continue?")  # immer modal
    print(res)
    if res:
        print("select file")
    else:
        b6.invoke()

############################## Log returns vs Monthly realized volatility #########################################

"""
#  /F0060/  Dividens:  bei Anklick auf die Dividens Taste wird eine Abfrage an mittels Callback funktion an API erstellt, 
#  um die Dividendenrendite seit dem Datum der Notierung (Dividens) bis 2021  abzurufen. Das Ergebnis wird in einem 
#  Plotter dargestellt und in einem visuellen Graph präsentiert. Der Graph kann manuell oder automatischem Modus im 
#  Stammverzeichnis gespeichert werden (s. Pflichtenheft, Seite 7)
"""
def dividens(): #################### Dividens #############################################################

    global mystr

# Download the adjusted prices from Yahoo Finance:

    d = yf.Ticker(mystr)
    df2 = d.dividends
    print(df2)

    ax2 = df2.plot(title=f' {mystr} Dividens ')  # Dividense

    plt.show()

    res = messagebox.askyesno(title="Would you like to continue?")  # immer modal
    print(res)
    if res:
        print("select file")
    else:
        b6.invoke()

#################### End Dividens #############################################################

"""
#  /F0070/  Simulating stock price (SDE):bei Klick auf die SDE Taste wird eine Abfrage mittels Callback funktion an API
#  erstellt,um den Aktienwert  in einem bestimmten Zeitraum abzurufen.  Der nächste Schritt ist die Simulation des 
#  Aktienkurses der für den kommenden Monat mit Hilfe eines Vorhersagealgorithmus die Simulation erstellt, der auf der 
#  Methode der stochastischen Differentialgleichungen (SDE) basiert. Das Ergebnis wird in einem Plotter dargestellt und 
#  in einem visuellen Graph präsentiert. Der Graph kann manuell oder automatischem Modus im Stammverzeichnis gespeichert 
#  werden (s. Pflichtenheft, Seite 7)
"""

# Module  von Alexander Zvetkov

def sde_calc():

    global mystr

# The Simulating stock price dynamics using Geometric Brownian Motion
# Define parameters for downloading data:

    RISKY_ASSET = [mystr]
    START_DATE = '2021-01-01'
    END_DATE = '2021-09-25'

# Download data from Yahoo Finance:

    df = yf.download(RISKY_ASSET, start=START_DATE, end=END_DATE, adjusted=True)
    print(f'Downloaded {df.shape[0]} rows of data.')
    # print(df.shape[0])
# Calculate daily returns:
    # print(returns)

    adj_close = df['Adj Close']
    returns = adj_close.pct_change().dropna()

    ax = returns.plot()
    ax.set_title(f'{RISKY_ASSET} returns: {START_DATE} - {END_DATE}', fontsize=16)

    plt.tight_layout()
    plt.show()

    # average return: about 0.21..0,24
    print(f'Average return: {100 * returns.mean():.2f}%')

    # Split data into the training and test sets
    # We will try to simulate and to get stock price 10  day ahead. Today is 20.09.2021, IBM Stock : 114,90 EUR

    train = returns['2021-01-01':'2021-09-15']
    test = returns['2021-09-16':'2021-09-25']

    # Specify the parameters of the simulation:
    T = len(test)
    N = len(test)
    S_0 = adj_close[train.index[-1]]
    N_SIM = 100
    mu = train.mean()
    sigma = train.std()

    # Define the function used for simulations:

    def simulate_gbm(s_0, mu, sigma, n_sims, T, N, random_seed=48):
        '''
        Function used for simulating stock returns using Geometric Brownian Motion.

        Parameters
        ------------
        s_0 : float
            Initial stock price
        mu : float
            Drift coefficient
        sigma : float
            Diffusion coefficient
        n_sims : int
            Number of simulations paths
        dt : float
            Time increment, most commonly a day
        T : float
            Length of the forecast horizon, same unit as dt
        N : int
            Number of time increments in the forecast horizon
        random_seed : int
            Random seed for reproducibility

        Returns
        -----------
        S_t : np.ndarray
            Matrix (size: n_sims x (T+1)) containing the simulation results.
            Rows respresent sample paths, while columns point of time.
        '''
        np.random.seed(random_seed)

        dt = T / N
        dW = np.random.normal(scale=np.sqrt(dt), size=(n_sims, N))
        W = np.cumsum(dW, axis=1)

        time_step = np.linspace(dt, T, N)
        time_steps = np.broadcast_to(time_step, (n_sims, N))

        S_t = s_0 * np.exp((mu - 0.5 * sigma ** 2) * time_steps + sigma * W)
        S_t = np.insert(S_t, 0, s_0, axis=1)

        return S_t

    # Run the simulations:

    gbm_simulations = simulate_gbm(S_0, mu, sigma, N_SIM, T, N)

    #Plot simulation results:

    # prepare objects for plotting
    last_train_date = train.index[-1].date()
    first_test_date = test.index[0].date()
    last_test_date = test.index[-1].date()
    plot_title = (f'{RISKY_ASSET} Simulation '
                  f'({first_test_date}:{last_test_date})')

    selected_indices = adj_close[last_train_date:last_test_date].index
    index = [date.date() for date in selected_indices]

    gbm_simulations_df = pd.DataFrame(np.transpose(gbm_simulations), index=index)

    # plotting
    ax = gbm_simulations_df.plot(alpha=0.2, legend=False)
    line_1, = ax.plot(index, gbm_simulations_df.mean(axis=1), color='red')
    line_2, = ax.plot(index, adj_close[last_train_date:last_test_date], color='blue')
    ax.set_title(plot_title, fontsize=16)
    ax.legend((line_1, line_2), ('mean', 'actual'))

    plt.tight_layout()
    #plt.savefig('SDE_saved.png')
    plt.show()

"""
#  /F0080/ Forecast Volatility:  bei Klick auf die Forecast Volatility Taste wird eine Abfrage mittels Callback funktion 
#  an API erstellt um die Multivariate Vorhersage der Volatilität basierend auf dem GARCH-Modele (Risky Asset)abzurufen. 
#  Das Ergebnis wird in einem Plotter dargestellt und in einem visuellen Graph präsentiert. Der Graph kann manuell oder 
#  automatischem Modus im Stammverzeichnis gespeichert werden. 

# ** ARCH-Modelle (ARCH, Akronym für: AutoRegressive Conditional Heteroscedasticity) bzw. autoregressive bedingt 
#  heteroskedastische Zeitreihenmodelle sind stochastische Modelle zur Zeitreihenanalyse, mit deren Hilfe 
#  insbesondere finanzmathematische Zeitreihen mit nicht konstanter Volatilität beschrieben werden können. Sie 
#  gehen von der Annahme aus, dass die bedingte Varianz der zufälligen Modellfehler abhängig ist vom realisierten 
#  Zufallsfehler der Vorperiode, so dass große und kleine Fehler dazu tendieren,in Gruppen 
#  aufzutreten. (s. Pflichtenheft, Seite 7)
"""

def risky_assets(): #################### Risky Assets #############################################################

    global mystr

    RISKY_ASSETS = ['GOOG', 'MSFT', 'AAPL', mystr]
    N = len(RISKY_ASSETS)
    START_DATE = '2000-01-01'
    END_DATE = '2021-12-30'

    # Download the adjusted prices from Yahoo Finance:

    df4 = yf.download(RISKY_ASSETS, start=START_DATE, end=END_DATE, adjusted=True)
    returns = 500 * df4['Adj Close'].pct_change().dropna()
    returns.plot(subplots=True, title=f'CCC-GARCH model for multivariate volatility forecasting: {START_DATE} - {END_DATE} ');

    coeffs = []
    cond_vol = []
    std_resids = []
    models = []

    for asset in returns.columns:
        model = arch_model(returns[asset], mean='Constant',vol='GARCH', p=1, o=0,q=1).fit(update_freq=0, disp='off')
        coeffs.append(model.params)
        cond_vol.append(model.conditional_volatility)
        std_resids.append(model.resid / model.conditional_volatility)
        models.append(model)

    coeffs_df = pd.DataFrame(coeffs, index=returns.columns)
    cond_vol_df = pd.DataFrame(cond_vol).transpose().set_axis(returns.columns, axis='columns',inplace=False)
    std_resids_df = pd.DataFrame(std_resids).transpose().set_axis(returns.columns,axis='columns',inplace=False)

    coeffs_df

    R = std_resids_df.transpose().dot(std_resids_df).div(len(std_resids_df))

    # define objects
    diag = []
    D = np.zeros((N, N))

    # populate the list with conditional variances
    for model in models:
        diag.append(model.forecast(horizon=1).variance.values[-1][0])
    # take the square root to obtain volatility from variance
    diag = np.sqrt(np.array(diag))
    # fill the diagonal of D with values from diag
    np.fill_diagonal(D, diag)

    # calculate the conditional covariance matrix
    H = np.matmul(np.matmul(D, R.values), D)

    print(H)

    plt.show()

    # Dialogfenster
    res = messagebox.askyesno(title="Would you like to continue?")  # immer modal
    print(res)
    if res:
        print("select file")
    else:
        b6.invoke()

################################# End : Risky Assets  ##########################

"""
# /F0090/ Volatilität vs CBOE Volatility Index :bei Klick auf die VIX Index Taste wird eine Abfrage mittels Callback 
# funktion an API erstellt, um die Volatilität vs CBOE Volatility Index (VIX Index)  1980 - 2021 abzurufen.Das Ergebnis 
# wird in einem Plotter dargestellt und in einem visuellen Graph präsentiert. Der Graph kann manuell oder automatischem 
# Modus im Stammverzeichnis  gespeichert werden. 

# * * * CBOE Volatility Index (VIX) drückt die erwartete Schwankungsbreite des US-amerikanischen 
# Aktienindex S&P 500 aus. Der VIX wird von der Terminbörse Chicago Board Options Exchange (CBOE) in Echtzeit berechnet
"""

def vix_index(): #####################################################################################

# Download and preprocess the prices of Stock and VIX
# CBOE Volatility Index (^VIX) vs Stock

    df = yf.download([mystr, '^VIX'],
                 start='1980-01-01',
                 end='2021-12-31',
                 progress=False)

    df = df[['Adj Close']]
    df.columns = df.columns.droplevel(0)
    df = df.rename(columns={mystr: 'stock', '^VIX': 'vix'})

    #Calculate log returns:
    df['log_rtn'] = np.log(df.stock / df.stock.shift(1))
    df['vol_rtn'] = np.log(df.vix / df.vix.shift(1))
    df.dropna(how='any', axis=0, inplace=True)

    # Plot a scatterplot with the returns on the axes and fit a regression line to identify trend:
    corr_coeff = df.log_rtn.corr(df.vol_rtn)

    ax = sns.regplot(x='log_rtn', y='vol_rtn', data=df, line_kws={'color': 'red'})
    ax.set(title=f' {mystr} Volatility vs. CBOE Volatility Index (VIX) 1980 - 2021 ($\\rho$ = {corr_coeff:.2f})',
        ylabel='VIX log returns',
        xlabel=f' {mystr} log returns')

    plt.tight_layout()
    plt.savefig('VIX_vs_Stock.png')
    plt.show()

    # Dialogfenster
    res = messagebox.askyesno(title="Would you like to continue?")  # immer modal
    print(res)
    if res:
        print("select file")
    else:
        b6.invoke()

################# END CBOE Volatility Index (^VIX) vs Stock ###########################################################

"""
# /F0100/ Simple Return vs. Log Return : bei Klick auf die Stock Sim Log Taste wird eine Abfrage mittels Callback 
# funktion an API erstellt, um die Korrelation von Aktienkurs und logarithmische / 
# Simple Renditen (Stock Log Sim) abzurufen. Das Ergebnis wird in einem Plotter dargestellt und in einem visuellen 
# Graph präsentiert. Der Graph kann manuell oder automatischem Modus im Stammverzeichnis gespeichert werden 
# (s. Pflichtenheft, Seite 8)
"""
def stock_simple_log(): ###### Stocks time series 1960 - 2021 :  Stock price - Simple returns - Log returns######

    # Download the adjusted prices from Yahoo Finance
    # To initialize multiple Ticker objects, use :

    df = yf.download( mystr, auto_adjust = False, progress=False)
    df = df.loc[:, ['Adj Close']]
    df.rename(columns={'Adj Close': 'adj_close'}, inplace=True)

# create simple and log returns

    df['simple_rtn'] = df.adj_close.pct_change()
    df['log_rtn'] = np.log(df.adj_close / df.adj_close.shift(1))

# dropping NA's in the first row

    df.dropna(how = 'any', inplace = True)

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# add prices

    df.adj_close.plot(ax=ax[0])
    ax[0].set(title= f' {mystr} time series 1960 - 2021   Stock price - Simple returns - Log returns',
          ylabel='Stock price ($)')

# add simple returns

    df.simple_rtn.plot(ax=ax[1])
    ax[1].set(ylabel='Simple returns (%)')

# add log returns

    df.log_rtn.plot(ax=ax[2])
    ax[2].set(xlabel='Date', ylabel='Log returns (%)')
    ax[2].tick_params(axis='x',which='major',labelsize=12)

    plt.tight_layout()
    plt.savefig(f'{mystr}_Stock_Returns.png')
    plt.show()

    # Dialogfenster
    res = messagebox.askyesno(title="Would you like to continue?")  # immer modal
    print(res)
    if res:
        print("select file")
    else:
        b6.invoke()

################# END:  Stocks time series 1960 - 2021 :  Stock price - Simple returns - Log returns######

"""
# /F0110/ Holders: bei Klick auf die Holders Taste wird eine Abfrage mittels Callback funktion an API erstellt, 
# um die Holders Liste abzurufen (Liste der 10 Hauptaktionäre (Holders).Das Ergebnis wird in einem Plotter dargestellt 
# und in einem visuellen Graph präsentiert. Der Graph kann manuell oder automatischem Modus im Stammverzeichnis gespeichert
# werden (s. Pflichtenheft, Seite 8)
"""
def holders(): ########## Holders ###################################################################

    global mystr

# Download the adjusted prices from Yahoo Finance:

    ibm = yf.Ticker(mystr)

    #figure1 = plt.Figure(figsize=(11, 6))
    fig = plt.Figure()

    df5 = ibm.institutional_holders.head(10)
    #bar_data = df5.institutional_holders.groupby(['Holder'])['Value'].sum().reset_index()
    bar_data = df5[['Holder', 'Value']].groupby('Holder').sum().reset_index()
    print(bar_data)
    # ax.set_titel('Total number of main institutional Holders')
    ax = bar_data.plot()
    bar_data.plot(kind='bar', legend=True, ax=ax)

    fig.tight_layout()
    fig = px.bar(bar_data, x="Holder", y="Value", title='Total number of main institutional Holders')
    fig.show()

####################### END : Holders ##############################################################################

root = tk.Tk()
#mystr = tk.StringVar()
mystr = tk.Variable()


l1 = tk.Label(root, text="Stock Analysis Toolkit v 1.0")
l1.grid(row=0, column=0, sticky=tk.E + tk.W, ipadx=40)  # Ohne ipadx = 40 wird ein 2-spaltiges Gitter erzeugt.

b4 = tk.Button(root, text='Ticker', command=dialog, state='normal')  # command = enthält den auszuführenden Befehl
b4.grid(row=3, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b5 = tk.Button(root, text='Historical Chart', command=max_history_stock, state='disabled')  # command = enthält den auszuführenden Befehl
b5.grid(row=4, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b5e = tk.Button(root, text='1-Month High Stock', command=last_month, state='disabled')  # command = enthält den auszuführenden Befehl
b5e.grid(row=5, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b5d = tk.Button(root, text='Cash Flow', command=cash_flow, state='disabled')  # command = enthält den auszuführenden Befehl
b5d.grid(row=6, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b5c = tk.Button(root, text='Volatility', command=volat_retn, state='disabled')  # command = enthält den auszuführenden Befehl
b5c.grid(row=7, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b5b = tk.Button(root, text='Dividens', command=dividens, state='disabled')  # command = enthält den auszuführenden Befehl
b5b.grid(row=8, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b5a = tk.Button(root, text='Forecast volatility', command=risky_assets, state='disabled')  # command = enthält den auszuführenden Befehl
b5a.grid(row=9, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b5f = tk.Button(root, text='Stock Volatility vs Volatility Index', command=vix_index, state='disabled')  # command = enthält den auszuführenden Befehl
b5f.grid(row=10, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b5j = tk.Button(root, text='Simple Return vs Log Return', command=stock_simple_log, state='disabled')  # command = enthält den auszuführenden Befehl
b5j.grid(row=11, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b5k = tk.Button(root, text='Holders', command=holders, state='disabled')  # command = enthält den auszuführenden Befehl
b5k.grid(row=12, column=0, sticky=tk.E + tk.W)  # Geometriemanager starten

b5l = tk.Button(root, text='Simulating stock price', command=sde_calc, state='disabled' )  # command = enthält den auszuführenden Befehl
b5l.grid(row=13, column=0, sticky=tk.E + tk.W)  # Geomecandlestickriemanager starten

b6 = tk.Button(root, text="Quit", command=root.quit)
b6.grid(row=14, column=0, sticky=tk.E + tk.W)

root.mainloop()

