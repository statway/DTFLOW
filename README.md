# DTFLOW

DTFLOW: inferring and visualizing dynamic cellular trajectories from single-cell data.

The method only can be used for technology research, not for commercial purposes.


### How to use DTFLOW

Before you run the algorithm, you should have installed the platform Anaconda and you can open the example "guo2010.ipynb" by jupyter notebook.

1. import DTFLOW and single-cell data 

```
from DTFLOW import *
z = DTFLOW(raw_df,time_stages) 
```
Here `raw_df` is a single cell dataset which contains N cells * D genes, time_stages is the time tags of single cells.

2. get one Markov transition_matrix.  

```
M= z.get_transition_matrix(k)
```
Here k is number of each node's neighbors.

3. get one diffusion distribution matrix by the RWR processes.

```
S = z.get_diffusionRWR(p)
```
Here 1-p is the restart probability of RWR processes.

4. dimension reduction with SVD, and plot the result of dimension reduction
```
Y = z.get_Y(slover='svd',dim)
z.plotY(size=(10,10),color ='stages')
```

5. get the pseudo time of each cell.

```
dftimes = z.get_dtflow_times(root_list)
z.plotY(size=(10,10),color ='dftime')
```
here the default of root_list is none.

6. get the result of branches detection
```
branch = z.get_branches(Num) # 1-11 12-112 
z.plotY(size=(10,10),color ='branches')
```
here Num is the minimum number of cells required for forming one sub-branch.