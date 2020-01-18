from autograd import elementwise_grad as grad

import geomstats.backend as gs

def metric(q):

    return gs.array([
        [1., 0.],
        [0., q ]])

def exp(vector, point, n_steps):
    p = gs.einsum('ij,j', metric, vector)
    return symp_flow(kinetic_energy)(x)

def kinetic_energy(x): 
    q, p = x
    return 1/2 * gs.einsum('i,ij,j', p, GL.inv(metric), p)

def symp_flow(H): 
    return iterate(symp_euler(H))

def symp_grad(H):
    H_q, H_p = grad(H)
    return gs.array([H_p, - H_q])


def iterate(func, n_steps): 

    def flow(x):
        xs = [x] * n_steps
        for i in range(n_steps):
            xs.append(f(xs[i]))
        return gs.array(xs)

    return flow
        
def symp_euler(H):

    def step(x):
        dq, _ = symp_grad(H)(x)
        y = x + gs.array([dq, 0.])
        _, dp = symp_grad(H)(y)
        return y + gs.array([0., dp])

    return step

