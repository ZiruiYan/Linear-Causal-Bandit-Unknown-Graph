import numpy as np
import matplotlib.pyplot as plt
import time

import sys

from sklearn.linear_model import Lasso

import argparse

seed=2024

#d->d->d->d->1

d=2
L=4

b_true = {}
for y in range(2,L+1):
    for x in range(0, d):
        b_true[(y,x)] = {1:[0.5]*(d),0:[1]*(d)}

bN_true = {1: [0.5]*(d),0:[1]*(d)}
N = L*d+1

from collections import deque, defaultdict

def topological_sort(N, pa):
    # Initialize the graph
    graph = defaultdict(list)
    in_degree = [0] * N

    # Build the graph and compute in-degrees
    for child in range(N):
        for parent in pa[child]:
            graph[parent].append(child)
            in_degree[child] += 1

    # Collect nodes with no incoming edges
    queue = deque([node for node in range(N) if in_degree[node] == 0])
    topo_order = []

    while queue:
        node = queue.popleft()
        topo_order.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(topo_order) == N:
        return topo_order
    else:
        raise ValueError("The graph has a cycle and hence a topological ordering is not possible.")


X=np.zeros(N)
for i in range(d):
    X[i]=0.5

#L2 to L4
for j in range(2,L+1):
    for i in range(0,d):
        X[d*(j-1)+i] = sum(b_true[(j,i)][0]*X[d*(j-2):d*(j-1)])+0.5

#reward node
best = sum(bN_true[0]*X[(L-1)*d:L*d])+ 0.5



params = {'mathtext.default': 'regular' }  
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update(params)
    

def A_vec_norm(x,A):
    return np.sqrt(np.matmul(x,np.matmul(A,x)))

def generate_X(b,bN):
    """
    N: number of nodes
    b: parameters for the first layer
    bN: parameters for the reward node
    """
    epsilon = np.random.uniform(low=0.0, high=1.0, size=N)

    #generate X according to Linear SEM
    #L1
    X=np.zeros(N)
    for i in range(d):
        X[i]=epsilon[i]

    #L2 to L4
    for j in range(2,L+1):
        for i in range(0,d):
            X[d*(j-1)+i] = sum(b[(j,i)]*X[d*(j-2):d*(j-1)])+epsilon[d*(j-1)+i]

    #reward node
    X[N-1] = sum(bN*X[(L-1)*d:L*d])+ epsilon[N-1]
    return X


def calculate_m(N):
    m_vec     = np.zeros(N)
    m_pa_vec  = np.zeros(N)
    for i in range(d):
        m_vec[i] = 1
        m_pa_vec[i] = 0
    for j in range(2,L+1):
        for i in range(d):
            m_vec[(j-1)*d+i]=d*m_vec[(j-2)*d+i]+1
            m_pa_vec[(j-1)*d+i] = np.sqrt(d)*m_vec[(j-2)*d+i]
    m_vec[-1]=d*m_vec[-1-d]+1
    m_pa_vec[-1] = np.sqrt(d) * m_vec[-1-d]
    return m_vec, m_pa_vec

def calculate_UCB_GA(hat_V, hat_b, hat_VN, hat_bN,int_set, pa,pa_idx, m_vec,m_pa_vec, Ni):
    mean = np.full(N,0.5)
    UCB = np.full((2**((L-1)*d+1)), -np.inf)
    width = np.full((2**((L-1)*d+1)), -np.inf)
    alpha=0.1
    for action,indi in enumerate(int_set):
        # still in the possible set of optimal intervention
        if indi == 1:
            hat_mu = np.zeros(N)
            w = np.zeros(N)
            # get binary rep of action
            action_binary = [int(s) for s in "{0:b}".format(action)]
            action_binary[:0]=[0]*(((L-1)*d+1)-len(action_binary))
            #construct B_a
            B_a = np.zeros((N,N))
            test=1
            for i in range(N-d):
                if Ni[action_binary[i]][i]==0:
                    test=0
            if test==1:
                for (idx,ai) in enumerate(action_binary):
                    true_idx=idx+d
                    try:
                        B_a[pa_idx[(true_idx//d+1,true_idx%d)],d+idx] = hat_b[(true_idx//d+1,true_idx%d)][ai].flatten()
                    except:
                        B_a[pa_idx[d+idx],d+idx] = hat_bN[ai].flatten()
                #cmg:TODO change to 
                f_B = np.zeros((N,N))
                for l in range(L+1):
                    f_B += np.linalg.matrix_power(B_a,l)
                for i in range(N):
                    hat_mu[i] = np.dot(f_B[:,i], mean) 
                    w[i] = 0
                    if i<N-1:
                        for j in pa_idx[(i//d+1,i%d)]:
                            w[i] += w[j]
                    else:
                        for j in pa_idx[i]:
                            w[i] += w[j]
                    if i>=d:
                        if i<N-1:
                            w[i] += alpha * A_vec_norm(hat_mu[pa_idx[(i//d+1,i%d)]],np.linalg.inv(hat_V[(i//d+1,i%d)][action_binary[i-d]]))
                            #add minimum eigenvalue
                            w[i] += m_pa_vec[i]* alpha/np.sqrt(np.linalg.eigvalsh(hat_V[(i//d+1,i%d)][action_binary[i-d]])[0])
                        else:
                            w[i] += alpha * A_vec_norm(hat_mu[pa_idx[i]],np.linalg.inv(hat_VN[action_binary[i-d]]))
                            #add minimum eigenvalue
                            w[i] += m_pa_vec[i]* alpha/np.sqrt(np.linalg.eigvalsh(hat_VN[action_binary[i-d]])[0])
                UCB[action] = hat_mu[-1] + w[-1]
                width[action] = w[-1]
            else:
                UCB[action] = np.inf
                width[action] = np.inf
    return UCB, width

def GALCB_unknown(N,T,m_vec,m_pa_vec, T_1, T_2):
    #initial the estimators
    hat_V = {}
    for j in range(2, L+1):
        for i in range(d):
            hat_V[(j,i)] = np.array((np.eye(d),np.eye(d)))
    hat_VN = np.array((np.eye(d),np.eye(d)))
    hat_b={}
    g = {}
    for j in range(2,L+1):
        for i in range(d):
            hat_b[(j,i)] = np.zeros((2,d,1))
            g[(j,i)] = np.zeros((2,d,1))
    hat_bN = np.zeros((2,d,1))
    gN = np.zeros((2,d,1))

    #TODO
    Ni = np.zeros((2,(L-1)*d+1))

    pa = {}
    pa_idx = {}
    j=1
    for i in range(d):
        pa[(j,i)]     = []
        pa_idx[(j,i)] = []
    for j in range(2,L+1):
        for i in range(d):
            pa[(j,i)]      =  [(j-1,i) for i in range(0, d )]
            pa_idx[(j,i)]  =  [(j-2)*d+i for i in range(0, d )]

    pa[N-1] = [(j,i) for i in range(d)]
    pa_idx[N-1] = [(j-1)*d+i for i in range(d)]

    X_all = []

    int_set = np.ones(2**((L-1)*d+1),dtype=int)

    de = [[] for i in range(N)] 
    pa = [[] for i in range(N)]
    mean_est=np.zeros((N,N+1))

    #save regret
    regret = np.zeros(T)
    actions = np.zeros(T)


    def create_mapping(N):
        return {i: 2**(N-d-i) if i != 0 else 0 for i in range(N-d+1)}

    # mapping = {0: 0, 1: 8, 2: 4, 3: 2, 4: 1}
    mapping = create_mapping(N)

    tag=0
    for t in range(T):
        s=1
        At = None

        if t<= (N+1)*T_1:
            At = mapping[max(0,t%(N+1)-d)]

        if t == (N+1)*T_1+1:
            #get the ancestor
            for i in range(N-1):
                de[i] = [j for j in range(N) if abs(mean_est[j,0] - mean_est[j,i+1])>0.5/2 and i!=j]

            for i in range(N):
                if len(de[i])>0 or i==N-1:
                    pa[i] = [j for j in range(N-1) if (len(de[j])==0  or i in de[j]) and i!=j]
            # print(pa)
            #check cycle
            for i in range(N):
                for j in de[i]:
                    if i in de[j]:
                        print("ERROR, CIRCLE DETECTED")
                        break

            #get topological order
            pi = topological_sort(N,pa)


        if t<= (N+1)*T_1 + max(T_2-T_1,0) and At == None:
            #in force exploration stage two: Lasso regression
            At=0
        if t == (N+1)*T_1 + max(T_2-T_1,0) + 1:
            X_all = np.array(X_all)
            #do the Lasso regression
            for i in range(N):
                if len(pa[i])>0:
                    lasso = Lasso(alpha=0.1, fit_intercept=False)  # alpha is the regularization parameter
                    lasso.fit(X_all[:,pa[i]],X_all[:,i]-0.5)
                    pa[i]= [item for item, flag in zip(pa[i], lasso.coef_!=0) if flag]
            # print(pa)
            tag = 0


        if sum(int_set)==1:
            At = np.where(int_set==1)[0][0]
        if tag==0 and At == None:
            # s=1
            # store UCB for all arms
            UCB, width = calculate_UCB_GA(hat_V, hat_b, hat_VN, hat_bN,int_set, pa,pa_idx, m_vec,m_pa_vec, Ni)
            if max(width[int_set==1])<m_vec[-1]*np.sqrt(1/T):
                location = np.where(int_set==1)[0]
                At  = location[np.argmax(UCB[location])]
                tag = 1
            while max(width[int_set==1]) < m_vec[-1] * 2**(-s):
                maxUCB = np.max(UCB)
                condition =  (UCB < maxUCB - m_vec[-1] * 2**(1-s))
                indices = np.where(condition)[0]
                int_set[indices] = 0
                # print(t,sum(condition), int_set)
                s += 1
            if not At:
                under = np.where(width >= m_vec[-1] * 2**(-s))[0]
                # choose action that maxmize the UCB also in int_set
                # At = under[np.argmax(UCB[under])]
                At = np.random.choice(under)
        elif tag==1:
            # print("else")
            while max(width[int_set==1])<m_vec[-1]* 2**(-s):
                maxUCB = np.max(UCB)
                condition =  (UCB < maxUCB - m_vec[-1] * 2**(1-s))
                indices = np.where(condition)[0]
                int_set[indices] = 0
                # print("else", t, sum(condition), int_set)
                s += 1
            UCB, width = calculate_UCB_GA(hat_V, hat_b, hat_VN, hat_bN, int_set, pa,pa_idx, m_vec,m_pa_vec, Ni)
            location = np.where(int_set==1)[0]
            At  = location[np.argmax(UCB[location])]

        # get binary representation of action
        At_binary = [int(s) for s in "{0:b}".format(At)]
        At_binary[:0]=[0]*(((L-1)*d+1)-len(At_binary))
        
        # get B matrix of action At B_At
        bAt={}
        for j in range(2,L+1):
            for i in range(d):
                bAt[(j,i)] = b_true[(j,i)][At_binary[(j-2)*d+i]].copy()
        bNAt = bN_true[At_binary[-1]].copy()

        NAt = [Ni[At_binary[s]][s] for s in range((L-1)*d+1)] 

        # calculate regret of action At
        #TODO
        #calculate the coose
        X=np.zeros(N)
        for i in range(d):
            X[i]=0.5

        #L2 to L4
        for j in range(2,L+1):
            for i in range(0,d):
                X[d*(j-1)+i] = sum(bAt[(j,i)][At_binary[(j-2)*d+i]]*X[d*(j-2):d*(j-1)])+0.5

        #reward node
        choose = sum(bNAt[At_binary[-1]]*X[(L-1)*d:L*d])+ 0.5
        regret[t] = (best-choose)
        actions[t] = At



        X=generate_X(bAt,bNAt)

        
        if sum(int_set)>1:
            # update estimator according to At
            for j in range(2,L+1):
                for i in range(d):
                    weights = 1
                    hat_V[(j,i)][At_binary[(j-2)*d+i]] += weights *  np.outer(np.array(X[d*(j-2):d*(j-1)]),np.array(X[d*(j-2):d*(j-1)]))
                    g[(j,i)][At_binary[(j-2)*d+i]] += weights * np.array([X[d*(j-2):d*(j-1)]]).transpose() * (X[d*(j-1)+i]-1)

            weights = 1
            hat_VN[At_binary[-1]]  += weights *  np.outer(np.array(X[d*(L-1):d*L]),np.array(X[d*(L-1):d*L]))
            gN[At_binary[-1]]  += weights * np.array([X[d*(L-1):d*L]]).transpose() * (X[-1]-1)

            for j in range(2,L+1):
                for i in range(d):
                    hat_b[(j,i)][At_binary[(j-2)*d+i]] =  np.matmul(np.linalg.inv(hat_V[(j,i)][At_binary[(j-2)*d+i]]), g[(j,i)][At_binary[(j-2)*d+i]])
            hat_bN[At_binary[-1]]  = np.matmul(np.linalg.inv(hat_VN[At_binary[-1]]), gN[At_binary[-1]])
            
            for i in range((L-1)*d+1):
                Ni[At_binary[i]][i]+=1

        if t<= (N+1)*T_1:
            mean_est[:,t%(N+1)] = (mean_est[:,t%(N+1)] * (t//(N+1)) + X)/ (t//(N+1)+1)

        if t <= (N+1)*T_1 + max(T_2-T_1,0):
            X_all.append(list(X))
            
    return regret, hat_b, hat_bN, actions, UCB



def GALCB_known(N,T,m_vec,m_pa_vec, T_1, T_2):
    #initial the estimators
    hat_V = {}
    for j in range(2, L+1):
        for i in range(d):
            hat_V[(j,i)] = np.array((np.eye(d),np.eye(d)))
    hat_VN = np.array((np.eye(d),np.eye(d)))
    hat_b={}
    g = {}
    for j in range(2,L+1):
        for i in range(d):
            hat_b[(j,i)] = np.zeros((2,d,1))
            g[(j,i)] = np.zeros((2,d,1))
    hat_bN = np.zeros((2,d,1))
    gN = np.zeros((2,d,1))

    #TODO
    Ni = np.zeros((2,(L-1)*d+1))

    pa = {}
    pa_idx = {}
    j=1
    for i in range(d):
        pa[(j,i)]     = []
        pa_idx[(j,i)] = []
    for j in range(2,L+1):
        for i in range(d):
            pa[(j,i)]      =  [(j-1,i) for i in range(0, d )]
            pa_idx[(j,i)]  =  [(j-2)*d+i for i in range(0, d )]

    pa[N-1] = [(j,i) for i in range(d)]
    pa_idx[N-1] = [(j-1)*d+i for i in range(d)]

    X_all = []

    int_set = np.ones(2**((L-1)*d+1),dtype=int)


    #save regret
    regret = np.zeros(T)
    actions = np.zeros(T)


    tag=0
    for t in range(T):
        s=1
        # if t%1000==0:
        #     print(t)    
        #     print(int_set)
        At = None
        if sum(int_set)==1:
            At = np.where(int_set==1)[0][0]
        if tag==0 and At == None:
            # s=1
            # store UCB for all arms
            UCB, width = calculate_UCB_GA(hat_V, hat_b, hat_VN, hat_bN,int_set, pa,pa_idx, m_vec,m_pa_vec, Ni)
            if max(width[int_set==1])<m_vec[-1]*np.sqrt(1/T):
                location = np.where(int_set==1)[0]
                At  = location[np.argmax(UCB[location])]
                tag = 1
            while max(width[int_set==1]) < m_vec[-1] * 2**(-s):
                maxUCB = np.max(UCB)
                condition =  (UCB < maxUCB - m_vec[-1] * 2**(1-s))
                indices = np.where(condition)[0]
                int_set[indices] = 0
                # print(t,sum(condition), int_set)
                s += 1
            if not At:
                under = np.where(width >= m_vec[-1] * 2**(-s))[0]
                # choose action that maxmize the UCB also in int_set
                # At = under[np.argmax(UCB[under])]
                At = np.random.choice(under)
        elif tag==1:
            # print("else")
            while max(width[int_set==1])<m_vec[-1]* 2**(-s):
                maxUCB = np.max(UCB)
                condition =  (UCB < maxUCB - m_vec[-1] * 2**(1-s))
                indices = np.where(condition)[0]
                int_set[indices] = 0
                # print("else", t, sum(condition), int_set)
                s += 1
            UCB, width = calculate_UCB_GA(hat_V, hat_b, hat_VN, hat_bN, int_set, pa,pa_idx, m_vec,m_pa_vec, Ni)
            location = np.where(int_set==1)[0]
            At  = location[np.argmax(UCB[location])]

        # get binary representation of action
        At_binary = [int(s) for s in "{0:b}".format(At)]
        At_binary[:0]=[0]*(((L-1)*d+1)-len(At_binary))
        
        # get B matrix of action At B_At
        bAt={}
        for j in range(2,L+1):
            for i in range(d):
                bAt[(j,i)] = b_true[(j,i)][At_binary[(j-2)*d+i]].copy()
        bNAt = bN_true[At_binary[-1]].copy()

        NAt = [Ni[At_binary[s]][s] for s in range((L-1)*d+1)] 

        # calculate regret of action At
        X=np.zeros(N)
        for i in range(d):
            X[i]=0.5

        #L2 to L4
        for j in range(2,L+1):
            for i in range(0,d):
                X[d*(j-1)+i] = sum(bAt[(j,i)][At_binary[(j-2)*d+i]]*X[d*(j-2):d*(j-1)])+0.5

        #reward node
        choose = sum(bNAt[At_binary[-1]]*X[(L-1)*d:L*d])+ 0.5
        regret[t] = (best-choose)
        actions[t] = At



        X=generate_X(bAt,bNAt)

        
        if sum(int_set)>1:
            # update estimator according to At
            for j in range(2,L+1):
                for i in range(d):
                    weights = 1
                    hat_V[(j,i)][At_binary[(j-2)*d+i]] += weights *  np.outer(np.array(X[d*(j-2):d*(j-1)]),np.array(X[d*(j-2):d*(j-1)]))
                    g[(j,i)][At_binary[(j-2)*d+i]] += weights * np.array([X[d*(j-2):d*(j-1)]]).transpose() * (X[d*(j-1)+i]-1)

            weights = 1
            hat_VN[At_binary[-1]]  += weights *  np.outer(np.array(X[d*(L-1):d*L]),np.array(X[d*(L-1):d*L]))
            gN[At_binary[-1]]  += weights * np.array([X[d*(L-1):d*L]]).transpose() * (X[-1]-1)

            for j in range(2,L+1):
                for i in range(d):
                    hat_b[(j,i)][At_binary[(j-2)*d+i]] =  np.matmul(np.linalg.inv(hat_V[(j,i)][At_binary[(j-2)*d+i]]), g[(j,i)][At_binary[(j-2)*d+i]])
            hat_bN[At_binary[-1]]  = np.matmul(np.linalg.inv(hat_VN[At_binary[-1]]), gN[At_binary[-1]])
            
            for i in range((L-1)*d+1):
                Ni[At_binary[i]][i]+=1
            
    return regret, hat_b, hat_bN, actions, UCB

np.random.seed(seed)
T = 40000
print("d",d)
print("L",L)

m_vec, m_pa_vec = calculate_m(N)

T_1=T_2=1000

avgRegret_GALCB_unknown=np.zeros(T)
avgRegret_GALCB_known=np.zeros(T)

avgtime_GALCB = 0
avgtime_GALCB_known = 0

iteration=20
for i in range(iteration):
    t1=time.time()
    print(i)
    regret_GALCB_unknown, hat_b_GALCB_unknown, hat_bN_GALCB_unknown, actions_GALCB_unknown, UCBGALCB_unknown= GALCB_unknown(N, T, m_vec,m_pa_vec,T_1,T_2)
    t2=time.time()
    regret_GALCB_known, hat_b_GALCB_known, hat_bN_GALCB_known, actions_GALCB_known, UCB_GALCB_known= GALCB_known(N, T, m_vec,m_pa_vec,T_1,T_2)
    t3=time.time()
    avgRegret_GALCB_unknown += regret_GALCB_unknown / iteration
    avgRegret_GALCB_known += regret_GALCB_known / iteration
    print(t2-t1,t3-t2)

    avgtime_GALCB += (t2-t1)/ iteration
    avgtime_GALCB_known += (t3-t2)/ iteration
    plt.figure()
    plt.plot(np.cumsum(avgRegret_GALCB_unknown)*iteration/(i+1),linewidth=2)
    plt.plot(np.cumsum(avgRegret_GALCB_known)*iteration/(i+1),linewidth=2)
    plt.grid()
    plt.ylabel('Cumulative Regret')
    plt.xlabel('Number of Iterations')
    plt.legend(["GA-LCB","GA-LCB(known)"],borderpad=0.2)
    plt.savefig('./result/plot/regret_hierarchical_%d_%d.pdf'%(d,L),bbox_inches='tight',dpi=200)

    np.save('./result/data/regret_hierarchical_avgRegret_GALCB_%d_%d.npy'%(d,L), avgRegret_GALCB_unknown)
    np.save('./result/data/regret_hierarchical_avgRegret_GALCB_known_%d_%d.npy'%(d,L), avgRegret_GALCB_known)

    

plt.rc('font',size=17)
plt.rc('axes',titlesize=17)
plt.rc('axes',labelsize=17)
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('legend',fontsize=17)
plt.rc('figure',titlesize=17)

plt.figure()
plt.plot(np.cumsum(avgRegret_GALCB_unknown),linewidth=2)
plt.plot(np.cumsum(avgRegret_GALCB_known),linewidth=2)
plt.grid()
plt.ylabel('Cumulated Regret')
plt.xlabel('Number of Iterations')
plt.legend(["GA-LCB","GA-LCB(known)"],borderpad=0.2)
plt.savefig('./result/plot/regret_hierarchical_%d_%d.pdf'%(d,L),bbox_inches='tight',dpi=200)

np.save('./result/data/regret_hierarchical_avgRegret_GALCB_%d_%d.npy'%(d,L), avgRegret_GALCB_unknown)
np.save('./result/data/regret_hierarchical_avgRegret_GALCB_known_%d_%d.npy'%(d,L), avgRegret_GALCB_known)


print(
    avgtime_GALCB,
    avgtime_GALCB_known
)