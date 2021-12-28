import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    
    y=-3*x**2 + 0.75*(np.exp(1.88*x)) #- 9.4*np.exp(pow(z,2)-16)
    return y

x=np.linspace(0,1,num=50)
# z=np.linspace(-2,5,num=100)
# p=np.linspace(-1,2,num=100)

y=f(x)
print(y)
dfx =pd.DataFrame(x)
dfx.to_csv("xdata.csv",index=False,index_label=False)

# dfz =pd.DataFrame(z)
# dfz.to_csv("zdata.csv",index=False)

# dfp =pd.DataFrame(p)
# dfp.to_csv("pdata.csv",index=False)

dfy =pd.DataFrame(y)
dfy.to_csv("y_measured.csv",index=False,index_label=False)
