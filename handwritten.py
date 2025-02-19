

# X = [step, (i1,i2), r, X_now]
    # i1 from 0 to N1
    # i2 from 0 to N2, then from N2+1 to N2+N0

def K_D(x, y):
    # x and y are both 2d locations
    pass

def w(x):
    # compute initial vorticity using initial velocity field
    pass

def A(i1, i2):
    # gives areas based on indexes i1 and i2
    pass

def u(x, k, X):
    # compute velocity field from given locations at this time step t_k
    pass
    # fcn u is also used for boats simulation

def X_update(u, X):
    # update new state in X tensor using current location X step t_k
    # will call fcn u
    pass
