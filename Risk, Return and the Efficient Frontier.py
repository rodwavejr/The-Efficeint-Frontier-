import numpy as np 
import pandas as pd
import scipy.optimize as sc 
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
import plotly.graph_objects as go

#### Implementation of Efficient Frontier ####

### Import Data ###
def getData(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

### Calculating the Portfolio Variance ###
def portfolioPerformance(weights,meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*252
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights )) ) *np.sqrt(252)
    return returns, std

### So Now we are going to Optimize to Calculate the Sharpe ratio, "High Risk High return", Using the Inverse "Negative Sharpe Ratio" ####

def negSharpe(weights, meanReturns, covMatrix, riskFreeRate = 0):
    pReturns, pStd = portfolioPerformance( weights,meanReturns,covMatrix)
    return - (pReturns - riskFreeRate)/pStd

### Sharpe Ratio is the Returns Divided by the Standard Deviation ###
## Now we need a function that's going to Maximize our Sharpe Ratio ##
# We do this by minimizing the Negative Sharpe Ratio by altering the weights of the Portfolio#

def maxSharpe (meanReturns, covMatrix, riskFreeRate = 0, constraintSet=(0,1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type':'eq', 'fun': lambda x: np.sum(x) - 1})        ## All the Summation of the weights in the portfolio have to add up to 1 ##
    bounds = constraintSet
    bounds = tuple(bounds for asset in range(numAssets))                 # For every single asset we want to make the bound #
    result = sc.minimize (negSharpe, numAssets*[1./numAssets],args=args,
                           method = 'SLSQP',bounds=bounds,constraints= constraints) # If we had 10 assets this would just be a list of 1/10 assets and args lists everything back into this function#
    return result 

### Minimizing Portfolio variance ###

def portfolioVariance(weights,meanReturns, covMatrix):
    return portfolioPerformance(weights,meanReturns, covMatrix)[1]

def minVariance(meanReturns, covMatrix, constraintSet=(0,1)):
    ## We want to Minimie the portfolio Varance by altering the weights/Allocation of assets in prortfolio ##
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type':'eq', 'fun': lambda x: np.sum(x) - 1})                  ## All the Summation of the weights in the portfolio have to add up to 1 ##
    bounds = constraintSet                                                         # For each asset the Allocation can be from 0% - 100% 
    bounds = tuple(bounds for asset in range(numAssets))                           # For every single asset we want to make the bound #
    result = sc.minimize (portfolioVariance, numAssets*[1./numAssets],args=args,   # Even portfolio weighting for initial guess #
                           method = 'SLSQP',bounds=bounds,constraints= constraints)# If we had 10 assets this would just be a list of 1/10 assets and args lists everything back into this function#
    return result 
### The List of Stocks ###
stockList = ['CBA', 'BHP', 'TLS','QAN','WBC']
stocks = [stock+'.AX' for stock in stockList]

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)

meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)


## Allocation For Maximum Sharp Ration - 0% for CBA - Highest Risk/Volitility,  Highest Reward ##
#                                        0% for BHP
#                                        100% for TLS
## Allocation OFr Minimimum Variance -   11% for CBA - Minimal Risk/Volitility, Highest Reward ##
#                                        22% for BHP 
#                                        65% for TLS

def portfolioReturn(weights,meanReturns, covMatrix):
    return portfolioPerformance(weights,meanReturns, covMatrix)[0]

def efficientOpt(meanReturns, covMatrix, returnTarget,constraintSet=(0,1)):
    ## For Each return target we need to optimize the protfolio for min variance ##
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    
    constraints = ({'type':'eq', 'fun': lambda x: portfolioReturn(x,meanReturns, covMatrix)- returnTarget},
                   {'type':'eq', 'fun': lambda x: np.sum(x) - 1}) 
    bounds = constraintSet                                                        
    bounds = tuple(bounds for asset in range(numAssets))                          
    effOpt = sc.minimize(portfolioVariance, numAssets*[1./numAssets],args=args,   
                        method = 'SLSQP',bounds=bounds,constraints= constraints)
    return effOpt


# Next Our application would be to to take the weights and put them back into the porfolio performance function 
# So we can wark out what those porftolio returns and variance are 

def calculatedResults(meanReturns, covMatrix, riskFreeRate = 0, constraintSet=(0,1)):
    ## Read in the Mean, covariance Matrix and other Financial Information ##
    ## Output the Max Sharpe Ratio, Minimum Volitility/Risk, Efficient frontier ##

                            # Max Sharpe Ratio Portfolio # 
    maxSharpePortfolio = maxSharpe(meanReturns,covMatrix)
    maxSharpeReturns,maxSharpeStd = portfolioPerformance(maxSharpePortfolio['x'],meanReturns, covMatrix)
    
    maxSharpeAllocation = pd.DataFrame(maxSharpePortfolio['x'], index=meanReturns.index, columns=['allocation'])
    maxSharpeAllocation.allocation = [round(i*100,0) for i in maxSharpeAllocation.allocation]

                            # Min Volatility/Risk Portfolio # 
    minVolPortfolio = minVariance(meanReturns,covMatrix)
    minVolReturns,minVolStd = portfolioPerformance(minVolPortfolio['x'],meanReturns, covMatrix)
    
    minVolAllocation = pd.DataFrame(minVolPortfolio['x'], index=meanReturns.index, columns=['allocation'])
    minVolAllocation.allocation = [round(i*100,0) for i in minVolAllocation.allocation]

    # Efficient Frontier 
    efficientList = []
    targetReturns = np.linspace(minVolReturns, maxSharpeReturns,20)
    for target in targetReturns:
        efficientList.append(efficientOpt(meanReturns,covMatrix, target)['fun'])

    maxSharpeReturns,maxSharpeStd = round(maxSharpeReturns*100,2),round(maxSharpeStd*100,2)
    minVolReturns,minVolStd = round(minVolReturns*100,2),round(minVolStd*100,2)

    return maxSharpeReturns,maxSharpeStd,maxSharpeAllocation,minVolReturns,minVolStd,minVolAllocation,efficientList,targetReturns

print(calculatedResults(meanReturns,covMatrix))

#print(efficientOpt(meanReturns,covMatrix,0.06))

import plotly.graph_objects as go

def TheEfficentFrontier(meanReturns, covMatrix, riskFreeRate = 0, constraintSet=(0,1)):
    ## The efficient frontier graph that produces the Minimum Volatility, Maximum Sharpe Ratio, and Efficient Frontier ##
    maxSharpeReturns, maxSharpeStd, maxSharpeAllocation, minVolReturns, minVolStd, minVolAllocation, efficientList, targetReturns = calculatedResults(meanReturns, covMatrix, riskFreeRate, constraintSet)

    # Max Sharpe Ratio 
    MaxSharpeRatio = go.Scatter(
        name='Maximum Sharpe Ratio',
        mode='markers',
        x=[maxSharpeStd],
        y=[maxSharpeReturns],
        marker=dict(color='red',size=14,line=dict(width=3,color='black'))
    )

    # Min Volatility Ratio 
    MinVol = go.Scatter(
        name='Min Volatility',
        mode='markers',
        x=[minVolStd],
        y=[minVolReturns],
        marker=dict(color='green',size=14,line=dict(width=3,color='black'))
    )

    # Efficient Frontier
    EfCurve = go.Scatter(
        name='Efficient Frontier',
        mode='lines',
        x=[round(efStd*100, 2) for efStd in efficientList],
        y=[round(target*100, 2) for target in targetReturns],
        line=dict(color='black', width=4, dash='dashdot')
    )

    data = [MaxSharpeRatio, MinVol, EfCurve]

    layout = go.Layout(
        title='Portfolio Optimization with the Efficient Frontier',
        yaxis=dict(title='Annualized Return (%)'),
        xaxis=dict(title='Annualized Volatility(%)'),
        showlegend=True,
        legend=dict(
            x=0.75, y=0, traceorder='normal',
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2
        ),
        width=800,
        height=600
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()

TheEfficentFrontier(meanReturns, covMatrix)

