import numpy as np
import scipy as sp
import pandas as pd
import sys

from scipy import spatial,linalg,sparse,stats
from sklearn import neighbors,decomposition,manifold,preprocessing
from sklearn.metrics.pairwise import pairwise_distances

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
# import time

plt.style.use(["seaborn-darkgrid", "seaborn-colorblind", "seaborn-notebook"])

def run_pca(data,pca_break):

    L = data - np.mean(data, 0)
    L = L.values
    P = L @ L.transpose()
    U, s, V = linalg.svd(P)

    sorted_s = sorted(s,reverse=True)
    sum_s = np.sum(s)
    s_i = 0

    for i,ele in enumerate(sorted_s):
        s_i += ele
        if s_i/sum_s > pca_break:
            break

    print("we select top %d which contain %f information of raw data." % (i,s_i/sum_s))
    U, s, V = sparse.linalg.svds(P, k=i)
    S = s * np.identity(i)
    Y = U @ np.sqrt(S)
    
    return Y



class DTFLOW():
    """
    The pseudo-time ordering of single cells.
    """
    def __init__(self, data, stages=None, pca_break = False):
        """
        N cells * D genes.
        :param stages: time stages of single cells.
        """
        if pca_break == False:
            self.data = data
        else:
            self.data = run_pca(data,pca_break)

        self.stages = stages
        self.shape = data.shape

        self.indices = None
        self.M = None
        self.mevals = None
        self.mevecs = None
        self.S = None
        self.H = None
        self.Y = None
        self.ordering = None
        self.df_times = None
        self.branches = None
        self.nf = None
        self.bp = None

    def get_transition_matrix(self, k=10, ann = False):
        """
        implementation of transition matrix.
        :param k: number of each node's neighbors.
        """
        
        # kNN
        if ann == False:
            nbrs = neighbors.NearestNeighbors(n_neighbors=k, metric='euclidean',n_jobs =-1).fit(self.data)
            distances,indices = nbrs.kneighbors(self.data)
        else: # need to imporve
            lshf = neighbors.LSHForest(n_neighbors=3*k).fit(self.data) 
            distances, indices = lshf.kneighbors(self.data, n_neighbors=k)

        N = self.shape[0]
        sqdistances = np.square(distances)
        sigmas = distances[:,-1]  
        self.sigmas = sigmas

        # kernel matrix
        sigs_mul = np.multiply.outer(sigmas,sigmas)
        kernel_matrix = np.zeros((N,k))
        for i in range(N):
            kernel_matrix[i,:] = np.exp(-np.divide(sqdistances[i,:],2*(sigs_mul[i,indices[i,:]])))
        
        weights = kernel_matrix
        indptr = range(0,(N+1)*k,k)
        weight_matrix = sparse.csr_matrix((weights.flatten(),indices.flatten(),indptr),shape=(N,N)).toarray()

        # symmetric
        for i, col in enumerate(indices):
            for j in col:
                if i not in set(indices[j]):
                    weight_matrix[j,i] = weight_matrix[i,j]

        weight_sum = np.power(weight_matrix.sum(axis=0)[:,None],-1/2).flatten()
        weight_sum = np.diag(weight_sum)
        weight_matrix = weight_sum @ weight_matrix @ weight_sum
        M = np.divide(weight_matrix, weight_matrix.sum(axis=1)[:,None])

        mevals, mevecs =  sp.linalg.eigh(M)

        self.M = M
        self.mevals = mevals
        self.mevecs = mevecs
        self.indices = indices

        return M

    def get_transition_matrix2(self, k=10):
        """
        implemantation of transition matrix of DPT.
        :param k: number of each node's neighbors.
        """
        
        # kNN
        N = self.shape[0]
        nbrs = neighbors.NearestNeighbors(n_neighbors=k, metric='euclidean').fit(self.data)
        distances,indices = nbrs.kneighbors(self.data)
        sqdistances = np.square(distances)
        sigmas =  distances[:,-1]/2  

        # kernel matrix
        sigs_sum = np.add.outer(sigmas**2,sigmas**2)
        sig_mul = np.multiply.outer(sigmas,sigmas)
        kernel_matrix = np.zeros((N,k))
        for i in range(N):
            para = np.sqrt(np.divide(2*sig_mul[i,indices[i,:]],sigs_sum[i,indices[i,:]]))
            kern = np.exp(-np.divide(sqdistances[i,:],(sigs_sum[i,indices[i,:]])))  # not *2
            kernel_matrix[i,:] = np.multiply(para,kern)

        weights = kernel_matrix
        indptr = range(0,(N+1)*k,k)
        weight_matrix = sparse.csr_matrix((weights.flatten(),indices.flatten(),indptr),shape=(N,N)).toarray()
        
        # symmetric
        for i, row in enumerate(indices):
            for j in row:
                if i not in set(indices[j]):
                    weight_matrix[j,i] = weight_matrix[i,j]
        
        # normalization
        weight_sum = np.power(weight_matrix.sum(axis=0)[:,None],-1/2).flatten()
        weight_sum = np.diag(weight_sum)
        M = weight_sum @ weight_matrix @ weight_sum
        mevals, mevecs =  sp.linalg.eigh(M)

        self.M = M
        self.mevals = mevals
        self.mevecs = mevecs
        self.indices = indices

        return M

    def get_diffusionRWR(self, p=0.8):

        """
        RWR processes.
        :param p: the restart probability 1-p of RWR. 
        """
        
        # diffusion distribution matrix

        N = self.shape[0]
        I = np.identity(N)
        S = (1-p) * np.linalg.inv(I-p*self.M)
        self.S = S
        return S

    def get_Y(self, slover ='svd', dim = 2, matrix = None):
        """
        dimension reduction.
        :param slover: 'svd'
        :param dim: preserved dimension
        """
        if matrix is not None:
            print("diffusion maps.")
            U, s, V = sparse.linalg.svds(matrix @ matrix.transpose(), k=dim+1)
            S = s[0:-1] * np.identity(dim)
            Y = U[:,0:-1] @ np.sqrt(S)

        else:
            # Parameter matrix
            S = self.S
            A = np.sqrt(S)
            G = A @ A.T
            H = np.log(G)
            np.fill_diagonal(H,0)
            self.H = H

            if slover == 'svd' or slover == 'SVD':
                U, s, V = sparse.linalg.svds(H, k=dim)
                S = s * np.identity(dim)
                Y = U @ np.sqrt(S)  

            elif slover == 'mds' or slover == 'MDS':
                N = self.shape[0]
                J = np.eye(N) - np.ones((N, N))/N
                B = J * H * J
                U, s, V = sparse.linalg.svds(B, k=dim)
                Y = U * np.sqrt(s)
            
            elif slover == 'lap' or slover == 'LAP':
                Dg = np.diag(np.sum(G,1))
                Lg = Dg - G
                eval, evec  = linalg.eigh(Lg,Dg)
                Y = evec[:,1:dim+1]
                pass

        self.Y = Y
        return Y
    
    def get_dtflow_times(self,id_lists):
        """
        get pseudo times.
        """

        root_id = None
        if len(id_lists)>1:
            max_diffusion_dists = 0
            for id in id_lists:
                diffusion_dists = -2*self.H[id,:]
                sum_diffusion_dists = np.sum(diffusion_dists)
                if sum_diffusion_dists > max_diffusion_dists:
                    max_diffusion_dists = sum_diffusion_dists
                    root_id = id
        
        elif len(id_lists)==1:
            root_id = id_lists[0] 
        df_times =  np.sqrt(-2*self.H[root_id,:])
        self.nf =  np.max(df_times) 
        self.df_times = df_times/self.nf
        self.ordering = sorted(range(len(df_times)), key=lambda k: df_times[k])
        
        return self.df_times

    def get_branches(self,Num = 15, delta = 0.9):
        """
        """
        prop_groups = []
        sub_branches = []
        spec_lists = [] 
        N = self.shape[0]
        
        reverse_seq = reversed(self.ordering)
        min_dftime = 1 # store  pseudo-time of bifurcation point
        
        for i,ind in enumerate(reverse_seq):
            ind_neighbor = self.indices[ind,:] 
            ind_neighbor_list = ind_neighbor.tolist()
            ind_neighbor_set = set(ind_neighbor_list)
            
            if i < N-1: # time interval of neighbors
                sct_diff = self.sigmas[ind]/np.max(self.sigmas)
                if sct_diff > delta: # remove some abnormal points
                    spec_lists.append(ind)
                    continue
            
            if not prop_groups:
                #     0: count cells    1:pseudo_time 
                prop_groups.append([[0,1],ind_neighbor_list])
            else:
                share_branch = False
                ind_b_list = []
                ind_pop_list = []
                ind_share_list = []
                
                for j,gp in enumerate(prop_groups):
                    if ind_neighbor_set.intersection(gp[1]):
                        
                        if sub_branches:
                            # remove elements in sub_branches
                            for ind_b,gpb in enumerate(sub_branches):
                                if ind_neighbor_set.intersection(gpb):
                                    share_branch = True
                                    ind_b_list.append(ind_b)
                                    ind_neighbor_set = ind_neighbor_set - set(gpb)
                                    
                        if share_branch: # intersect with prop_groups and sub_branches
                            ind_pop_list.append(j)
                            ind_neighbor_set = ind_neighbor_set-set(gp[1]) # remove replicate elements
                        else: # only intersect with prop_groups
                            ind_neighbor_set = ind_neighbor_set-set(gp[1]) 
                            if ind_neighbor_set:
                                ind_share_list.append(j)
                                prop_groups[j][1].extend(list(ind_neighbor_set)) # extend elements
                            
                if share_branch:
                    new_large_gp = []
                    new_small_gp = []
                    
                    for ind_p in ind_pop_list:
                        pop_list_set = set(prop_groups[ind_p][1]) # get indices
                        
                        for gpb in sub_branches:
                            if pop_list_set.intersection(gpb):
                                pop_list_set = pop_list_set - set(gpb)  # remove replicate elements
                                
                        df_large_gp = [ind for ind in pop_list_set if self.df_times[ind]>min_dftime]
                        
                        if len(df_large_gp) > (Num-1):
                            gp_diff = max(self.df_times[df_large_gp])-min(self.df_times[df_large_gp])
                            if gp_diff > 0.05:
                                sub_branches.append(df_large_gp)
                            else:  # time interval is too small
                                for ind_gg in df_large_gp:
                                    ind_gg_neighbor_list = self.indices[ind_gg].tolist()
                                    ind_gg_dftime = self.df_times[ind_gg_neighbor_list]
                                    sort_indices = np.argsort(np.abs(ind_gg_dftime[1:]-ind_gg_dftime[0]))+1
                                    sort_indices = np.array(ind_gg_neighbor_list)[sort_indices]
                                    elem = False
                                    ind_min = 0
                                    for ind_si in sort_indices:
                                        if elem:
                                            break
                                        for ig,gp in enumerate(sub_branches):
                                            if ind_si in gp:
                                                sub_branches[ig].append(ind_gg)
                                                ind_min = ig
                                                elem = True
                            df_small_gp = [ind for ind in pop_list_set if ind not in df_large_gp]
                            new_small_gp.extend(df_small_gp)
                        else:
                            new_large_gp.extend(df_large_gp)
                            df_small_gp = [ind for ind in pop_list_set if ind not in df_large_gp]
                            new_small_gp.extend(df_small_gp)
                    
                    if ind_neighbor_set: # neighbor set of indices
                        df_large_gp = [ind for ind in ind_neighbor_set if self.df_times[ind] > min_dftime]
                        df_small_gp = [ind for ind in ind_neighbor_set if ind not in df_large_gp]
                        new_large_gp.extend(df_large_gp)
                        new_small_gp.extend(df_small_gp)
                    
                    if new_large_gp: # do something
                        new_large_list = list(set(new_large_gp))
                        new_small_list = list(set(new_small_gp))
                        pre_count = 0
                        pre_ind_list = []
                        
                        for ind_s in list(new_large_list):
                            ind_s_neighbor_list = self.indices[ind_s].tolist()
                            ind_s_dftime = self.df_times[ind_s_neighbor_list]
                            sort_indices = np.argsort(np.abs(ind_s_dftime[1:]-ind_s_dftime[0]))+1
                            sort_indices = np.array(ind_s_neighbor_list)[sort_indices]
                            ele = False
                            ind_min = 0
                            for ind_si in sort_indices:
                                if ele:
                                    break
                                for ig,gp in enumerate(sub_branches):
                                    if ind_si in gp:
                                        sub_branches[ig].append(ind_s)
                                        ind_min = ig
                                        ele = True
                            pre_ind_list.append(ind_min)
                        
                        for ind_p in set(pre_ind_list):
                            pre_count += len(sub_branches[ind_p])
                        if new_small_gp:
                            prop_groups.append([[pre_count,min_dftime],new_small_list])
                    
                    else:
                        ind_b_list = list(set(ind_b_list))
                        pre_count = 0
                        for ind_b in ind_b_list:
                            pre_count = pre_count + len(sub_branches[ind_b])
                        new_small_list = list(set(new_small_gp))
                        if new_small_gp:
                            prop_groups.append([[pre_count,min_dftime],new_small_list])
                    
                    if ind_pop_list:
                        for ids in sorted(ind_pop_list,reverse=True):
                            prop_groups.pop(ids)
                            
                else: 
                    if len(ind_share_list) == 0:
                        if ind_neighbor_set:
                            prop_groups.append([[0,1],list(ind_neighbor_set)])
                            
                    elif len(ind_share_list) == 1:
                        continue
                            
                    elif len(ind_share_list) == 2 or len(ind_share_list) > 2:
        
                        setlist = []
                        new_grouplists = []
        
                        for ids in ind_share_list:
                            setlist.append(set(prop_groups[ids][1]))
                                
                        intersect = set.intersection(*setlist)
                        intersect = list(intersect)
                            
                        # remove elements of intersect in prop_groups 
                        for ids in ind_share_list:
                            new_group = [item for item in prop_groups[ids][1] if item not in intersect]
                            new_grouplists.append(new_group)
                                
                        id_min_dftime = intersect[0]
                        min_dftime = self.df_times[intersect[0]]
                        
        
                        for id_itst in intersect[1:]:
                            if min_dftime > self.df_times[id_itst]:
                                id_min_dftime = id_itst
                                min_dftime = self.df_times[id_itst]
                        # new group list
                        new_gp_list = [id_min_dftime] 
        
                        # add elements less than min_dftime to new_gp_list
                        for gp in new_grouplists:
                            bgp = [id_gp for id_gp in gp if self.df_times[id_gp] < min_dftime]
                            new_gp_list.extend(bgp)
                                
                        # remove elements of new_gp_list in new_grouplists
                        new2_grouplists = []
                        for gp in new_grouplists:
                            ngp = [id_gp for id_gp in gp if id_gp not in new_gp_list]
                            new2_grouplists.append(ngp)
                                
                        # remove empty list
                        new_grouplists =  [gp for gp in new2_grouplists if gp]
        
                        nb_list = []
                        for ind_n,gp in enumerate(new_grouplists):
                            if len(gp) >(Num-1):
                                nb_list.append(ind_n)
                                
                        intersect = [id for id in intersect if id != id_min_dftime]
        
                        if len(nb_list) == 2 or len(nb_list) > 2:
        
                            new_branches = []
                            for ind_nb in nb_list:
                                new_branches.append(new_grouplists[ind_nb])
        
                            new_grouplists_set = set().union(*new_grouplists)
                            new_branches_set = set().union(*new_branches)
                            intersect_set = new_grouplists_set.difference(new_branches_set)
                            intersect_set = intersect_set | set(intersect)
        
                            for ind_s in list(intersect_set):
                                ind_s_neighbor_list = self.indices[ind_s].tolist()
                                ind_s_dftime = self.df_times[ind_s_neighbor_list]
                                sort_indices = np.argsort(np.abs(ind_s_dftime[1:]-ind_s_dftime[0]))+1
                                sort_indices = np.array(ind_s_neighbor_list)[sort_indices]
                                ele = False
                                for ind_si in sort_indices:
                                    if ele:
                                        break
                                    for ib,gp in enumerate(new_branches):
                                        if ind_si in gp:
                                            new_branches[ib].append(ind_s)
                                    
                            for nbc in new_branches:
                                nbc = list(set(nbc))
                                sub_branches.append(nbc)
                                    
                            pre_count = 0
        
                            for ids in sorted(ind_share_list,reverse=True):
                                pre_count += prop_groups[ids][0][0]
                                prop_groups.pop(ids)
                                    
                            prop_groups.append([[pre_count,min_dftime],new_gp_list])
                                
                        else:
        
                            merge_gp_list = []
                            for gp in new_grouplists:
                                merge_gp_list.extend(gp)
                            merge_gp_list.extend(intersect)
                            merge_gp_list.extend(new_gp_list)
        
                            pre_count = 0
        
                            for ids in sorted(ind_share_list,reverse=True):
                                pre_count += prop_groups[ids][0][0]
                                prop_groups.pop(ids)
                                    
                            prop_groups.append([[pre_count,min_dftime],merge_gp_list])
        
        for gid,gp in enumerate(prop_groups):
            df_time_gp_min = np.min([self.df_times[ind] for ind in gp[1]]) 
            if df_time_gp_min == 0:
                min_id = gid
                break
        
        first_branch = prop_groups[min_id][1]
        prop_groups.pop(min_id)
        sub_branches.append(first_branch)
        
        for gp in prop_groups:
            g_min = float("inf")
            j_min = 0
            for j,gb in enumerate(sub_branches):
<<<<<<< HEAD
                gb_min = np.min(-self.H[gp[1][-1],gb]) 
=======
                gb_min = np.min(self.T[gp[1][-1],gb])
>>>>>>> 747bca77d3e7b043defbff12d712e09b7f827be3
        
                if g_min > gb_min:
                    g_min = gb_min
                    j_min = j
                    
            sub_branches[j_min].extend(gp[1])
        
        gp_set = set([id for gp in sub_branches for id in gp])
        spec_list = [i for i in range(N) if i not in gp_set]
        
        while spec_list:
            for ind_s in spec_list:
                ind_neighbor_list = self.indices[ind_s].tolist()
                ind_s_dftime = self.df_times[ind_neighbor_list]
                sort_indices = np.argsort(np.abs(ind_s_dftime[1:]-ind_s_dftime[0]))+1
                sort_indices = np.array(ind_neighbor_list)[sort_indices]
                ele = False
                for ind in sort_indices:
                    if ele:
                        break
                    for i,gp in enumerate(sub_branches):
                        if ind in gp:
                            sub_branches[i].append(ind_s)
                            ele = True
            gp_set = set([id for gp in sub_branches for id in gp])
            spec_list = [i for i in range(N) if i not in gp_set]
        
        sub_branches = sub_branches[::-1]
        
        nbranch = len(sub_branches)
        branch = [None] * N
        for i in range(nbranch):
            for ind in range(N):
                if ind in sub_branches[i]:
                    branch[ind] = i+1
                
        self.branches = branch  
        return branch                 

    def get_results(self, retn = None):
        """
        """
        tn = None
        if retn == 'dftime':
            tn = self.df_times
        elif retn == 'branches':
            tn = self.branches
        elif retn == 'Y':
            tn = self.Y
        elif retn == 'ordering':
            tn = self.ordering
        elif retn == 'H':
            tn = self.H
        elif retn == 'sigmas':
            tn = self.sigmas
        elif retn == 'indices':
            tn = self.indices
        return tn
        
    def plotY(self, size = (20,16),dims=[0,1],color = None,annotation = False):
        """
        Plot the embedding structure of the data.
        :param color: 'stages','pseudotime','groups'
        """  
        Y = self.Y
        d = Y.shape[1] 
        print(size)

        fig = plt.figure(figsize=size)

        if d < len(dims):
            raise Exception('The number of dimensions for plots should be no greater than dimensions for embedding space.')
        
        if len(dims) < 2 or len(dims) > 3:
            print("The number of dimensions for plots should be 2 or 3.")

        if len(dims) == 2 :

            if color == None:
                plt.scatter(Y[:,dims[0]],Y[:,dims[1]])

            elif color == 'stages' or color == 'types':
                time_stage = sorted(set(self.stages))
                color_stage = cm.jet(np.linspace(0, 1, len(time_stage)))
                color_dict = dict(zip(time_stage, color_stage[:len(time_stage)]))
                sc_colors = [color_dict[cst] for cst in self.stages]
                markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
                
                plt.scatter(Y[:,dims[0]],Y[:,dims[1]], c=sc_colors)
                plt.legend(markers, color_dict.keys(), loc=2, numpoints=3)
            
            elif color == 'dftime':
                if color == 'dftime':
                    p = plt.scatter(Y[:,dims[0]],Y[:,dims[1]], c=self.df_times, cmap='jet')
                plt.colorbar(p)

            elif color == 'branches':
                nums_branch = sorted(set(self.branches)) 
                color_branch = cm.jet(np.linspace(0, 1, len(nums_branch)))
                color_dict = dict(zip(nums_branch,color_branch[:len(nums_branch)]))
                sc_colors = [color_dict[cst] for cst in self.branches]
                markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
                
                plt.scatter(Y[:,dims[0]],Y[:,dims[1]], c=sc_colors)
                plt.legend(markers, color_dict.keys(), loc=2, numpoints=3)
            
            if annotation == True:
                for ind,ord in enumerate(self.ordering):
                    plt.annotate(ind,xy=(Y[:,dims[0]][ord],Y[:,dims[1]][ord]),fontsize=15)
        
        elif len(dims) == 3:

            ax = fig.add_subplot(111, projection='3d')

            if color == None:
                plt.scatter(Y[:,dims[0]],Y[:,dims[1]],Y[:,dims[2]])
            
            elif color == 'stages' or color == 'types':
                
                time_stage = sorted(set(self.stages))
                color_stage = cm.jet(np.linspace(0, 1, len(time_stage)))
                color_dict = dict(zip(time_stage, color_stage[:len(time_stage)]))
                sc_colors = [color_dict[cst] for cst in self.stages]
                markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
                
                ax.scatter(Y[:,dims[0]],Y[:,dims[1]],Y[:,dims[2]], c=sc_colors)

                ax.grid(True)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.set_xlabel('D'+str(dims[0]),fontsize=15,fontweight='bold')
                ax.set_ylabel('D'+str(dims[1]),fontsize=15,fontweight='bold')
                ax.set_zlabel('D'+str(dims[2]),fontsize=15,fontweight='bold')
                ax.legend(markers, color_dict.keys(), loc=2, numpoints=3)
            
            elif color == 'dftime':
                if color == 'dftime':
                    p = ax.scatter(Y[:,dims[0]],Y[:,dims[1]],Y[:,dims[2]], c=self.df_times, cmap='jet')
                plt.colorbar(p)
            
            elif color == 'branches':
                nums_branch = sorted(set(self.branches)) 
                color_branch = cm.jet(np.linspace(0, 1, len(nums_branch)))
                color_dict = dict(zip(nums_branch,color_branch[:len(nums_branch)]))
                sc_colors = [color_dict[cst] for cst in self.branches]
                markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_dict.values()]
                ax.scatter(Y[:,dims[0]],Y[:,dims[1]],Y[:,dims[2]], c=sc_colors)
                ax.grid(True)
                ax.set_xlabel('D'+str(dims[0]),fontsize=15,fontweight='bold')
                ax.set_ylabel('D'+str(dims[1]),fontsize=15,fontweight='bold')
                ax.set_zlabel('D'+str(dims[2]),fontsize=15,fontweight='bold')
                ax.legend(markers, color_dict.keys(), loc=2, numpoints=3)               
        
        plt.show()

if __name__ == "__main__":
    pass