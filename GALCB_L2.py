import numpy as np
import matplotlib.pyplot as plt
import time

# For Lasso estimators
from sklearn.linear_model import Lasso

#d->d->1
d = 3
N = 2*d+1

import random



params = {'mathtext.default': 'regular' }  
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update(params)


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

def A_vec_norm(x,A):
    return np.sqrt(np.matmul(x,np.matmul(A,x)))


#cauclaate the variable bonnd used for Lin-SEM-UCB
def calculate_m(N):
    m_vec=np.zeros(N)
    for i in range(d):
        m_vec[i]=1
    for i in range(d,2*d):
        m_vec[i]=d+1
    m_vec[2*d]=d*(d+1)+1
    return np.sqrt(sum(m_vec*m_vec))

#cauclaate the variable bonnd used for GCB-UCB
def calculate_m_GCB(N):
    m_vec=np.zeros(N)
    for i in range(d):
        m_vec[i]=1
    for i in range(d,2*d):
        m_vec[i]=d+1
    m_vec[2*d]=d*(d+1)+1
    # return np.sqrt(sum(m_vec*m_vec))
    return m_vec[-2], m_vec[-1]


#cauclaate the variable bonnd used for GA-LCB
def calculate_m_GALCB(N):
    m_vec     = np.zeros(N)
    m_pa_vec  = np.zeros(N)
    for i in range(d):
        m_vec[i] = 1
        m_pa_vec[i] = 0
    for i in range(d,2*d):
        m_vec[i]=d+1
        m_pa_vec[i] = np.sqrt(d)
    m_vec[2*d]=d*(d+1)+1
    m_pa_vec[2*d] = np.sqrt(d) * (d+1)
    return m_vec, m_pa_vec
    

def generate_X(N,b,bN):
    """
    N: number of nodes
    b N*1 vector of parameters
    """
    epsilon = np.random.uniform(low=0.0, high=1.0, size=N)

    #generate X according to Linear SEM
    #L1
    X=np.zeros(N)
    for i in range(d):
        X[i]=epsilon[i]
    #L2
    for i in range(d,2*d):
        X[i]= sum(b[i-d]*X[:d])+epsilon[i]
    #reward node
    X[2*d] = sum(bN*X[d:2*d])+epsilon[2*d]
    return X




def calculate_UCB(hat_V, hat_b, hat_VN, hat_bN, beta_T):
    UCB = np.full((2**(d+1)), 0.)
    # for each arm calculate the UCB
    for action in range(2**(d+1)):
        # get binary rep of action
        action_binary = [int(s) for s in "{0:b}".format(action)]
        action_binary[:0]=[0]*((d+1)-len(action_binary))
        bbb = []
        VVV = []
        for i in range(d):
            bbb.append(hat_b[i][action_binary[i]].copy())
            VVV.append(hat_V[i][action_binary[i]].copy())
        bN  = hat_bN[action_binary[-1]].copy()
        VN = hat_VN[action_binary[-1]].copy()

        bbb_s = []
        tmp = []
        for i in range(d):
            bbb_s.append(bbb[i])
            tmp.append(beta_T * np.matmul(np.linalg.inv(VVV[i]),np.ones((d,1)))/np.sqrt(np.matmul(np.ones((1,d)),np.matmul(np.linalg.inv(VVV[i]),np.ones((d,1))))))
            for j in range(d):
                bbb_s[i][j] += tmp[i][j]
        c = np.zeros((d,1))
        for i in range(d):
            c[i] = sum(bbb_s[i])
        tmpN = beta_T * np.matmul(np.linalg.inv(VN),c)/np.sqrt(np.matmul(c.T,np.matmul(np.linalg.inv(VN),c)))
        bN_s =bN
        for i in range(d):
            bN_s[i] += tmpN[i]
            UCB[action] += bN_s[i]*(c[i]+1)
    return UCB

def calculate_UCB_GCB(hat_V, hat_b, hat_VN, hat_bN, beta_T_1,beta_T_2):
    UCB = np.full((2**(d+1)), 0.)
    # for each arm calculate the UCB
    for action in range(2**(d+1)):
        # get binary rep of action
        action_binary = [int(s) for s in "{0:b}".format(action)]
        action_binary[:0]=[0]*((d+1)-len(action_binary))
        bbb = []
        VVV = []
        for i in range(d):
            bbb.append(hat_b[i][action_binary[i]].copy())
            VVV.append(hat_V[i][action_binary[i]].copy())
        bN  = hat_bN[action_binary[-1]].copy()
        VN = hat_VN[action_binary[-1]].copy()

        bbb_s = []
        tmp = []
        for i in range(d):
            bbb_s.append(bbb[i])
            tmp.append(beta_T_1 * np.matmul(np.linalg.inv(VVV[i]),np.ones((d,1)))/np.sqrt(np.matmul(np.ones((1,d)),np.matmul(np.linalg.inv(VVV[i]),np.ones((d,1))))))
            for j in range(d):
                bbb_s[i][j] += tmp[i][j]
        c = np.zeros((d,1))
        for i in range(d):
            c[i] = sum(bbb_s[i])
        tmpN = beta_T_2 * np.matmul(np.linalg.inv(VN),c)/np.sqrt(np.matmul(c.T,np.matmul(np.linalg.inv(VN),c)))
        bN_s =bN
        for i in range(d):
            bN_s[i] += tmpN[i]
            UCB[action] += bN_s[i]*(c[i]+1)
    return UCB


def calculate_UCB_GA(hat_V, hat_b, hat_VN, hat_bN,int_set, pa, m_vec,m_pa_vec):
    mean = np.full(N,0.5)
    UCB = np.full((2**(d+1)), -np.inf)
    LCB = np.full((2**(d+1)), -np.inf)
    width = np.full((2**(d+1)), -np.inf)
    # alpha=np.sqrt(1/2* np.log(2*N*T)/(0.01)) +np.sqrt(3)
    alpha=0.1
    for action,indi in enumerate(int_set):
        # still in the possible set of optimal intervention
        if indi == 1:
            hat_mu = np.zeros(N)
            w = np.zeros(N)
            # get binary rep of action
            action_binary = [int(s) for s in "{0:b}".format(action)]
            action_binary[:0]=[0]*((d+1)-len(action_binary))
            #construct B_a
            B_a = np.zeros((N,N))
            for (idx,ai) in enumerate(action_binary):
                try:
                    B_a[pa[d+idx],d+idx] = hat_b[idx][ai].flatten()
                except:
                    B_a[pa[d+idx],d+idx] = hat_bN[ai].flatten()
            f_B = np.zeros((N,N))
            for l in range(3):
                f_B += np.linalg.matrix_power(B_a,l)
            for i in range(N):
                hat_mu[i] = np.dot(f_B[:,i], mean) 
                w[i] = 0
                for j in pa[i]:
                    w[i] += w[j]
                if i>=d:
                    if i<N-1:
                        w[i] += alpha * A_vec_norm(hat_mu[pa[i]],np.linalg.inv(hat_V[i-d][action_binary[i-d]]))
                        #add minimum eigenvalue
                        w[i] += m_pa_vec[i]* alpha/np.sqrt(np.linalg.eigvalsh(hat_V[i-d][action_binary[i-d]])[0])
                    else:
                        w[i] += alpha * A_vec_norm(hat_mu[pa[i]],np.linalg.inv(hat_VN[action_binary[i-d]]))
                        #add minimum eigenvalue
                        w[i] += m_pa_vec[i]* alpha/np.sqrt(np.linalg.eigvalsh(hat_VN[action_binary[i-d]])[0])
            UCB[action] = hat_mu[-1] + w[-1]
            width[action] = w[-1]
            LCB[action] = hat_mu[-1] - w[-1]
    return UCB, width


def GALCB_unknown(N,T,m_vec,m_pa_vec, T_1, T_2):
    #initial the estimators
    # parameters for other variables
    hat_V=[]
    for i in range(d):
        hat_V.append(np.array((np.eye(d)*m_vec[-1]**2,np.eye(d)*m_vec[-1]**2)))
    hat_VN = np.array((np.eye(d)*m_vec[-1]**2,np.eye(d)*m_vec[-1]**2))
    hat_b=[]
    g = []
    for i in range(d):
        hat_b.append(np.zeros((2,d,1)))
        g.append(np.zeros((2,d,1)))
    hat_bN = np.zeros((2,d,1))
    gN = np.zeros((2,d,1))
    Ni = np.zeros((2,d+1))

    de = [[] for i in range(N)] 
    pa = [[] for i in range(N)]

    # for i in range(d,2*d):
    #     pa[i] =  [i for i in range(0, d )]

    # pa[N-1] = [i for i in range(d,2*d)]

    X_all = []

    int_set = np.ones(2**(d+1),dtype=int)


    #save regret
    regret = np.zeros(T)
    actions = np.zeros(T)
    
    # the first one is under null intervention
    # N+1 is the null intervention
    mean_est=np.zeros((N,N+1))
    mapping = {0: 0, 1: 8, 2: 4, 3: 2, 4: 1}

    s=1
    tag=-1
    for t in range(T):
        At = None
        if t<= (N+1)*T_1:
            #in force exploration stage one
            # id = max(0,i%(N+1)-3)
            At = mapping[max(0,t%(N+1)-d)]
            # print(At)
            #update mu later

        if t == (N+1)*T_1+1:
            #get the ancestor
            for i in range(N-1):
                de[i] = [j for j in range(N) if abs(mean_est[j,0] - mean_est[j,i+1])>0.5/2 and i!=j]

            for i in range(N):
                if len(de[i])>0 or i==N-1:
                    pa[i] = [j for j in range(N-1) if (len(de[j])==0  or i in de[j]) and i!=j]
            print(pa)
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
            print(pa)
            tag = 0


        if sum(int_set)==1:
            At = np.where(int_set==1)[0][0]
        if tag==0 and At == None:
            # s=1
            # store UCB for all arms
            UCB, width = calculate_UCB_GA(hat_V, hat_b, hat_VN, hat_bN,int_set, pa, m_vec,m_pa_vec)
            if max(width[int_set==1])<m_vec[-1]*np.sqrt(1/T):
                location = np.where(int_set==1)[0]
                At  = location[np.argmax(UCB[location])]
                tag = 1
            while max(width[int_set==1]) < m_vec[-1] * 2**(-s):
                maxUCB = np.max(UCB)
                # Calculate the condition
                condition =  (UCB < maxUCB - m_vec[-1] * 2**(1-s))
                # print(t)
                # print(condition)
                # Get the indices where the condition is True
                indices = np.where(condition)[0]
                # print(indices)
                # Perform the assignment
                int_set[indices] = 0
                print(t,int_set)
                s += 1
            if not At:
                under = np.where(width >= m_vec[-1] * 2**(-s))[0]
                # choose action that maxmize the UCB also in int_set
                # At = under[np.argmax(UCB[under])]
                At = np.random.choice(under)
        elif tag==1:
            while max(width[int_set==1])<m_vec[-1]* 2**(-s):
                maxUCB = np.max(UCB)
                # print(t)
                # print("else")
                # print(condition)
                condition =  (UCB < maxUCB - m_vec[-1] * 2**(1-s))

                # Get the indices where the condition is True
                indices = np.where(condition)[0]
                # print(indices)

                # Perform the assignment
                int_set[indices] = 0
                print(t, int_set)

                s += 1
            UCB, width = calculate_UCB_GA(hat_V, hat_b, hat_VN, hat_bN, int_set, pa, m_vec,m_pa_vec)
            location = np.where(int_set==1)[0]
            At  = location[np.argmax(UCB[location])]

        At_binary = [int(s) for s in "{0:b}".format(At)]
        At_binary[:0]=[0]*(d+1-len(At_binary))

        # the B matrix of action At B_At
        bAt=[]
        for i in range(d):
            bAt.append(b_true[i][At_binary[i]].copy())
        bNAt = bN_true[At_binary[-1]].copy()

        NAt = [Ni[At_binary[s]][s] for s in range(d+1)] 
        
        #calculate regret
        choose = 0
        for i in range(d):
            choose += bNAt[i]* (sum(bAt[i])+0.5)
        regret[t] = (best-choose)
        actions[t] = At


        X=generate_X(N,bAt,bNAt)

        if sum(int_set)>1:
            for i in range(d):
                weights = 1
                hat_V[i][At_binary[i]] += weights *  np.outer(np.array(X[:d]),np.array(X[:d]))
                g[i][At_binary[i]] += weights * np.array([X[:d]]).transpose() * (X[i+d]-0.5)

                
            weights = 1
            hat_VN[At_binary[-1]]  += weights *  np.outer(np.array(X[d:2*d]),np.array(X[d:2*d]))
            gN[At_binary[-1]]  += weights * np.array([X[d:2*d]]).transpose() * (X[-1]-0.5)
            
            for i in range(d):
                hat_b[i][At_binary[i]] =  np.matmul(np.linalg.inv(hat_V[i][At_binary[i]]), g[i][At_binary[i]])
            hat_bN[At_binary[-1]]  = np.matmul(np.linalg.inv(hat_VN[At_binary[-1]]), gN[At_binary[-1]])
            
            for i in range(d+1):
                Ni[At_binary[i]][i]+=1

        if t<= (N+1)*T_1:
            mean_est[:,t%(N+1)] = (mean_est[:,t%(N+1)] * (t//(N+1)) + X)/ (t//(N+1)+1)

        if t <= (N+1)*T_1 + max(T_2-T_1,0):
            X_all.append(list(X))

            
    # return regret, hat_b21, hat_b22, hat_b23, hat_bN, actions, UCB
    return regret, hat_b, hat_bN, actions, UCB


def GALCB_known(N,T,m_vec,m_pa_vec, T_1, T_2):
    #initial the estimators
    # parameters for other variables
    hat_V=[]
    for i in range(d):
        hat_V.append(np.array((np.eye(d)*m_vec[-1]**2,np.eye(d)*m_vec[-1]**2)))
    hat_VN = np.array((np.eye(d)*m_vec[-1]**2,np.eye(d)*m_vec[-1]**2))
    hat_b=[]
    g = []
    for i in range(d):
        hat_b.append(np.zeros((2,d,1)))
        g.append(np.zeros((2,d,1)))
    hat_bN = np.zeros((2,d,1))
    gN = np.zeros((2,d,1))
    Ni = np.zeros((2,d+1))

    pa = [[] for i in range(N)]

    for i in range(d,2*d):
        pa[i] =  [i for i in range(0, d )]

    pa[N-1] = [i for i in range(d,2*d)]

    X_all = []

    int_set = np.ones(2**(d+1),dtype=int)


    #save regret
    regret = np.zeros(T)
    actions = np.zeros(T)

    s=1
    tag=0
    for t in range(T):
        At = None
        if sum(int_set)==1:
            At = np.where(int_set==1)[0][0]
        if tag==0 and At == None:
            # s=1
            # store UCB for all arms
            UCB, width = calculate_UCB_GA(hat_V, hat_b, hat_VN, hat_bN,int_set, pa, m_vec,m_pa_vec)
            if max(width[int_set==1])<m_vec[-1]*np.sqrt(1/T):
                location = np.where(int_set==1)[0]
                At  = location[np.argmax(UCB[location])]
                tag = 1
            while max(width[int_set==1]) < m_vec[-1] * 2**(-s):
                maxUCB = np.max(UCB)
                condition =  (UCB < maxUCB - m_vec[-1] * 2**(1-s))
                # Get the indices where the condition is True
                indices = np.where(condition)[0]
                # Perform the assignment
                int_set[indices] = 0
                print(t,int_set)
                s += 1
            if not At:
                under = np.where(width >= m_vec[-1] * 2**(-s))[0]
                At = np.random.choice(under)
        elif tag==1:
            while max(width[int_set==1])<m_vec[-1]* 2**(-s):
                maxUCB = np.max(UCB)
                condition =  (UCB < maxUCB - m_vec[-1] * 2**(1-s))

                # Get the indices where the condition is True
                indices = np.where(condition)[0]

                # Perform the assignment
                int_set[indices] = 0
                print(t, int_set)

                s += 1
            UCB, width = calculate_UCB_GA(hat_V, hat_b, hat_VN, hat_bN, int_set, pa, m_vec,m_pa_vec)
            location = np.where(int_set==1)[0]
            At  = location[np.argmax(UCB[location])]

        At_binary = [int(s) for s in "{0:b}".format(At)]
        At_binary[:0]=[0]*(d+1-len(At_binary))


        bAt=[]
        for i in range(d):
            bAt.append(b_true[i][At_binary[i]].copy())
        bNAt = bN_true[At_binary[-1]].copy()

        NAt = [Ni[At_binary[s]][s] for s in range(d+1)] 
        
        #calculate regret
        choose = 0
        for i in range(d):
            choose += bNAt[i]* (sum(bAt[i])+0.5)
        regret[t] = (best-choose)
        actions[t] = At


        X=generate_X(N,bAt,bNAt)
        
        if sum(int_set)>1:
            for i in range(d):
                weights = 1
                hat_V[i][At_binary[i]] += weights *  np.outer(np.array(X[:d]),np.array(X[:d]))
                g[i][At_binary[i]] += weights * np.array([X[:d]]).transpose() * (X[i+d]-0.5)

                
            weights = 1
            hat_VN[At_binary[-1]]  += weights *  np.outer(np.array(X[d:2*d]),np.array(X[d:2*d]))
            gN[At_binary[-1]]  += weights * np.array([X[d:2*d]]).transpose() * (X[-1]-0.5)
            

            for i in range(d):
                hat_b[i][At_binary[i]] =  np.matmul(np.linalg.inv(hat_V[i][At_binary[i]]), g[i][At_binary[i]])
            hat_bN[At_binary[-1]]  = np.matmul(np.linalg.inv(hat_VN[At_binary[-1]]), gN[At_binary[-1]])
            
            for i in range(d+1):
                Ni[At_binary[i]][i]+=1
            
    # return regret, hat_b21, hat_b22, hat_b23, hat_bN, actions, UCB
    return regret, hat_b, hat_bN, actions, UCB

def GCB(N,T,m1,m2):
    """
    N: number of nodes
    T: time horizon
    m: bound for the norm of X
    C: corruption level
    mc: contamination power
    """
    delta=0.1
    
    
    #initial the estimators
    # parameters for other variables
    
    hat_V=[]
    for i in range(d):
        hat_V.append(np.array((np.zeros((d,d)),np.zeros((d,d)))))
    hat_VN = np.array((np.zeros((d,d)),np.zeros((d,d))))
    hat_b=[]
    g = []
    for i in range(d):
        hat_b.append(np.zeros((2,d,1)))
        g.append(np.zeros((2,d,1)))
    hat_bN = np.zeros((2,d,1))
    gN = np.zeros((2,d,1))
    Ni = np.zeros((2,d+1))

    #save regret
    regret = np.zeros(T)
    actions = np.zeros(T)
    

    for t in range(T):
        # print(t)
        # store UCB for all arms
        beta_T_1 =  8 * np.log( 2*(1+m1*t)**d /delta) /4  + 2 * (8*m1 + np.sqrt(8*np.log(4*t**2/delta)/4))
        beta_T_2 =  8 * np.log( 2*(1+m2*t)**d /delta) /4  + 2 * (8*m2 + np.sqrt(8*np.log(4*t**2/delta)/4))
        try:
            UCB = calculate_UCB_GCB(hat_V, hat_b, hat_VN, hat_bN, np.sqrt(beta_T_1), np.sqrt(beta_T_2))
         
            # choose action that maxmize the UCB
            At = np.argmax(UCB)
        except:
            # print("except1:",t)
            At = np.random.randint(0,2**(d+1))
        At_binary = [int(s) for s in "{0:b}".format(At)]
        At_binary[:0]=[0]*(d+1-len(At_binary))
        
        # the B matrix of action At B_At
        # print(At_binary)
        # print(b_true)
        bAt=[]
        for i in range(d):
            bAt.append(b_true[i][At_binary[i]].copy())
        bNAt = bN_true[At_binary[-1]].copy()

        NAt = [Ni[At_binary[s]][s] for s in range(d+1)] 
        
        #calculate regret
        choose = 0
        for i in range(d):
            choose += bNAt[i]* (sum(bAt[i])+0.5)
        regret[t] = (best-choose)
        actions[t] = At


        X=generate_X(N,bAt,bNAt)
        # X=generate_corrupted_X_time(N,bAt,bNAt,C,NAt,mc)
        
        for i in range(d):
            hat_V[i][At_binary[i]] +=   np.outer(np.array(X[:d]),np.array(X[:d]))
            g[i][At_binary[i]] +=  np.array([X[:d]]).transpose() * (X[i+d]-0.5)

            
        hat_VN[At_binary[-1]]  +=  np.outer(np.array(X[d:2*d]),np.array(X[d:2*d]))
        gN[At_binary[-1]]  +=  np.array([X[d:2*d]]).transpose() * (X[-1]-0.5)
        

        
        for i in range(d):
            try:
                hat_b[i][At_binary[i]] =  np.matmul(np.linalg.inv(hat_V[i][At_binary[i]]), g[i][At_binary[i]])
            except:
                # print("except2:",t)
                hat_b[i][At_binary[i]] =  np.matmul(np.linalg.pinv(hat_V[i][At_binary[i]]), g[i][At_binary[i]])
        
        try:
            hat_bN[At_binary[-1]]  = np.matmul(np.linalg.inv(hat_VN[At_binary[-1]]), gN[At_binary[-1]])
        except:
            # print("except3:",t)
            hat_bN[At_binary[-1]]  = np.matmul(np.linalg.pinv(hat_VN[At_binary[-1]]), gN[At_binary[-1]])

        for i in range(d+1):
            Ni[At_binary[i]][i]+=1

    return regret, hat_b, hat_bN, actions, UCB

  
def LinSEMUCB(N,T,m):
    """
    N: number of nodes
    T: time horizon
    m: bound for the norm of X
    C: corruption level
    mc: contamination power
    """
    delta=1/(2*N*T)
    
    
    #initial the estimators
    # parameters for other variables
    
    hat_V=[]
    for i in range(d):
        hat_V.append(np.array((np.eye(d),np.eye(d))))
    hat_VN = np.array((np.eye(d),np.eye(d)))
    hat_b=[]
    g = []
    for i in range(d):
        hat_b.append(np.zeros((2,d,1)))
        g.append(np.zeros((2,d,1)))
    hat_bN = np.zeros((2,d,1))
    gN = np.zeros((2,d,1))
    Ni = np.zeros((2,d+1))

    #save regret
    regret = np.zeros(T)
    actions = np.zeros(T)
    

    for t in range(T):
        # print(t)
        # store UCB for all arms
        beta_T =  d+ np.sqrt(2*np.log(1/delta)+d*np.log((1+m**2*T/d)))
        UCB = calculate_UCB(hat_V, hat_b, hat_VN, hat_bN, beta_T)
     
        # choose action that maxmize the UCB
        At = np.argmax(UCB)
        At_binary = [int(s) for s in "{0:b}".format(At)]
        At_binary[:0]=[0]*(d+1-len(At_binary))
        
        # the B matrix of action At B_At
        # print(At_binary)
        # print(b_true)
        bAt=[]
        for i in range(d):
            bAt.append(b_true[i][At_binary[i]].copy())
        bNAt = bN_true[At_binary[-1]].copy()

        NAt = [Ni[At_binary[s]][s] for s in range(d+1)] 
        
        #calculate regret
        choose = 0
        for i in range(d):
            choose += bNAt[i]* (sum(bAt[i])+0.5)
        regret[t] = (best-choose)
        actions[t] = At


        X=generate_X(N,bAt,bNAt)
        # X=generate_corrupted_X_time(N,bAt,bNAt,C,NAt,mc)
        
        for i in range(d):
            weights = 1
            hat_V[i][At_binary[i]] += weights *  np.outer(np.array(X[:d]),np.array(X[:d]))
            g[i][At_binary[i]] += weights * np.array([X[:d]]).transpose() * (X[i+d]-0.5)

            
        weights = 1
        hat_VN[At_binary[-1]]  += weights *  np.outer(np.array(X[d:2*d]),np.array(X[d:2*d]))
        gN[At_binary[-1]]  += weights * np.array([X[d:2*d]]).transpose() * (X[-1]-0.5)
        


        for i in range(d):
            hat_b[i][At_binary[i]] =  np.matmul(np.linalg.inv(hat_V[i][At_binary[i]]), g[i][At_binary[i]])
        hat_bN[At_binary[-1]]  = np.matmul(np.linalg.inv(hat_VN[At_binary[-1]]), gN[At_binary[-1]])
        
        for i in range(d+1):
            Ni[At_binary[i]][i]+=1
            
    # return regret, hat_b21, hat_b22, hat_b23, hat_bN, actions, UCB
    return regret, hat_b, hat_bN, actions, UCB


np.random.seed(2024)
random.seed(2024)
T = 40000
print("d",d)

m= calculate_m(N)
print(m)
m1,m2 = calculate_m_GCB(N)
print(m1,m2)

m_vec,m_pa_vec = calculate_m_GALCB(N)

avgRegret_GA = np.zeros(T)
avgRegret_GA_known = np.zeros(T)
avgRegret_GCB = np.zeros(T)
avgRegret_Lin = np.zeros(T)


avgtime_GA = 0
avgtime_GA_known = 0
avgtime_GCB = 0
avgtime_Lin = 0


T_1 = 500
T_2 = 500

iteration=20

b_true = {}
for x in range(0, d):
        b_true[x] = {1:[0.5]*(d),0:[1]*(d)}

bN_true = {0: [0.5]*(d),1:[1]*(d)}

best = 0
for i in range(d):
    best += max(bN_true[0][i], bN_true[1][i]) * (max(sum(b_true[i][0]),sum(b_true[i][1]))+0.5)

for i in range(iteration):
    print(i)
    t1=time.time()
    regret_GA, hat_b_GA, hat_bN_GA, actions_GA, UCB_GA = GALCB_unknown(N, T, m_vec,m_pa_vec,T_1,T_2)
    t1_1=time.time()
    regret_GA_known, hat_b_GA_known, hat_bN_GA_known, actions_GA_known, UCB_GA_known = GALCB_known(N, T, m_vec,m_pa_vec,T_1,T_2)
    t1_2=time.time()
    regret_GCB, hat_b_GCB, hat_bN_GCB, actions_GCB, UCB_GCB = GCB(N, T, m1,m2)
    t1_3=time.time()
    regret_Lin, hat_b_Lin, hat_bN_Lin, actions_Lin, UCB_Lin = LinSEMUCB(N, T, m)
    t1_4=time.time()

    avgRegret_GA += regret_GA / iteration
    avgRegret_GA_known += regret_GA_known / iteration
    avgRegret_GCB += regret_GCB / iteration
    avgRegret_Lin += regret_Lin / iteration

    avgtime_GA += (t1_1-t1)/ iteration
    avgtime_GA_known += (t1_2-t1_1)/ iteration
    avgtime_GCB += (t1_3-t1_2)/ iteration
    avgtime_Lin += (t1_4-t1_3)/ iteration

    print(
    avgtime_GA,
    avgtime_GA_known,
    avgtime_GCB,
    avgtime_Lin
    )

    plt.figure()
    plt.plot(np.cumsum(avgRegret_GA)*iteration/(i+1),linewidth=2)
    plt.plot(np.cumsum(avgRegret_GA_known)*iteration/(i+1),linewidth=2)
    plt.plot(np.cumsum(avgRegret_GCB)*iteration/(i+1),linewidth=2)
    plt.plot(np.cumsum(avgRegret_Lin)*iteration/(i+1),linewidth=2)
    plt.grid()
    plt.ylabel('Cumulated Regret')
    plt.xlabel('Number of Iterations')
    plt.legend(["GA-LCB","GA-LCB(known)","GCB-UCB","LinSEM-UCB",'UCB'])
    plt.savefig('./result/plot/regret_he_%d.pdf'%d,bbox_inches='tight',dpi=200)
    plt.close()
    np.save('./result/data/regret_he_avgRegret_GA_%d.npy'%d, avgRegret_GA)
    np.save('./result/data/1_regret_he_avgRegret_GA_known_%d.npy'%d, avgRegret_GA_known)
    np.save('./result/data/regret_he_avgRegret_GCB_%d.npy'%d, avgRegret_GCB)
    np.save('./result/data/regret_he_avgRegret_Lin_%d.npy'%d, avgRegret_Lin)

plt.rc('font',size=17)
plt.rc('axes',titlesize=17)
plt.rc('axes',labelsize=17)
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('legend',fontsize=17)
plt.rc('figure',titlesize=17)

plt.figure()
plt.plot(np.cumsum(avgRegret_GA),linewidth=2)
plt.plot(np.cumsum(avgRegret_GA_known)*iteration/(i+1),linewidth=2)
plt.plot(np.cumsum(avgRegret_GCB),linewidth=2)
plt.plot(np.cumsum(avgRegret_Lin),linewidth=2)
plt.grid()
plt.ylabel('Cumulated Regret')
plt.xlabel('Number of Iterations')
plt.legend(["GA-LCB","GA-LCB(known)", "GCB-UCB","LinSEM-UCB",'UCB'])
plt.savefig('./result/plot/regret_he_%d.pdf'%d,bbox_inches='tight',dpi=200)
plt.close()
        
np.save('./result/data/regret_he_avgRegret_GA_%d.npy'%d, avgRegret_GA)
np.save('./result/data/1_regret_he_avgRegret_GA_known_%d.npy'%d, avgRegret_GA_known)
np.save('./result/data/regret_he_avgRegret_GCB_%d.npy'%d, avgRegret_GCB)
np.save('./result/data/regret_he_avgRegret_Lin_%d.npy'%d, avgRegret_Lin)

print(
    avgtime_GA,
    avgtime_GA_known,
    avgtime_GCB,
    avgtime_Lin
)

