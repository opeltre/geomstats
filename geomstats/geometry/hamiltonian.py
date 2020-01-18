from autograd import elementwise_grad as grad
import geomstats.backend as gs
import matplotlib.pyplot as plt

def main():
    x = exp(0.05 * gs.array([1.,1.]), gs.array([0.,0.]), 20)
    plt.plot(x[:,0,0], x[:,0,1])
    plt.show()
    return x
def metric(q):
    return gs.array([
        [1/(1 + q[1]**4) , q[0]**2],
        [q[0]**2, 1. + 2*q[1]**3]])

def exp(vector, point, n_steps=100):
    momentum = gs.einsum('ij,j', metric(point), vector)
    x = gs.array([point, momentum])
    return symp_flow(kinetic_energy)(x)

def kinetic_energy(x): 
    q, p = x
    cometric = gs.linalg.inv(metric(q))
    return 1/2 * gs.einsum('i,ij,j', p, cometric, p)

def symp_flow(H, n_steps=20): 
    return iterate(symp_euler(H), n_steps)

def symp_grad(H):
    def vector(x): 
        H_q, H_p = grad(H)(x)
        return gs.array([H_p, - H_q])
    return vector


def symp_euler(H):

    def step(x):
        q, p = x
        dq, _ = symp_grad(H)(x)
        y = gs.array([q + dq, p])
        _, dp = symp_grad(H)(y)
        return gs.array([q + dq, p + dp])

    return step


def iterate(func, n_steps): 

    def flow(x):
        xs = [x] 
        for i in range(n_steps):
            xs.append(func(xs[i]))
        return gs.array(xs)

    return flow
