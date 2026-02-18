import matplotlib.pyplot as plt
import numpy as np
import pylops 
import scipy.sparse.linalg

def load_image_option_I(file_name = "dog_rgb.npy"):
    sampling = 5
    im = np.load(file_name)[::sampling, ::sampling, 2]
    Nz, Nx = im.shape

    # Blurring Gaussian operator
    nh = [15, 25]
    hz = np.exp(-0.1 * np.linspace(-(nh[0] // 2), nh[0] // 2, nh[0]) ** 2)
    hx = np.exp(-0.3 * np.linspace(-(nh[1] // 2), nh[1] // 2, nh[1]) ** 2)
    hz /= np.trapz(hz)  # normalize the integral to 1
    hx /= np.trapz(hx)  # normalize the integral to 1
    h = hz[:, np.newaxis] * hx[np.newaxis, :]

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    him = ax.imshow(h)
    ax.set_title("Blurring operator")
    fig.colorbar(him, ax=ax)
    ax.axis("tight")
    plt.show()
    Cop = pylops.signalprocessing.Convolve2D(
        (Nz, Nx), h=h, offset=(nh[0] // 2, nh[1] // 2), dtype="float32"
    )

    imblur = Cop * im
    plt.imshow(im, cmap="viridis", vmin=0, vmax=255)
    plt.show()
    plt.imshow(imblur, cmap="viridis", vmin=0, vmax=255)
    plt.show()

    Wop = pylops.signalprocessing.DWT2D((Nz, Nx), wavelet="haar", level=3)

    # This is your A and b for your f1 cost!
    A = Cop * Wop.H
    b = imblur.ravel()

    return Wop, A, b, im, imblur

def load_image_option_II(file_name = "chateau.npy"):
    sampling = 2
    im = np.load(file_name)[::sampling, ::sampling, 1]
    Nz, Nx = im.shape

    # Blurring Gaussian operator
    nh = [15, 25]
    hz = np.exp(0.1 * np.linspace(-(nh[0] // 2), nh[0] // 2, nh[0]) ** 2)
    hx = np.exp(0.3 * np.linspace(-(nh[1] // 2), nh[1] // 2, nh[1]) ** 2)
    hz /= np.trapz(hz)  # normalize the integral to 1
    hx /= np.trapz(hx)  # normalize the integral to 1
    h = hz[:, np.newaxis] * hx[np.newaxis, :]

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    him = ax.imshow(h)
    ax.set_title("Blurring operator")
    fig.colorbar(him, ax=ax)
    ax.axis("tight")
    plt.show()
    Cop = pylops.signalprocessing.Convolve2D(
        (Nz, Nx), h=h, offset=(nh[0] // 2, nh[1] // 2), dtype="float32"
    )

    imblur = Cop * im
    plt.imshow(im, cmap="gray", vmin=0, vmax=255)
    plt.show()
    plt.imshow(imblur, cmap="gray", vmin=0, vmax=255)
    plt.show()

    Wop = pylops.signalprocessing.DWT2D((Nz, Nx), wavelet="haar", level=3)

    # This is your A and b for your f1 cost!
    A = Cop * Wop.H
    b = imblur.ravel()

    return Wop, A, b, im, imblur


def my_fista(A, b, opt_cost, eps=10**(-1), niter=10000, tol=1e-6, acceleration=False):
    """ Here you can code your ISTA and FISTA algorithm
        Return: optimal x, and opt_gap_cost (history of cost-optcost)
    """

    # alpha = 1/((A.T@A).eigs(neigs = 1, symmetric = True)[0])

    # if acceleration:
    #     print("Running FISTA...")
    #     l = 0
    #     gamma = 2*(1-l)/(1+np.sqrt(1+4*l**2))
    #     l = (1+np.sqrt(1+4*l**2))/2
    #     x = np.zeros(A.shape[1]) 
    #     grad_f = (A.T)@(A@x-b)
    #     v = x - alpha*grad_f
    #     k=0
    #     y = np.zeros(A.shape[1])
    #     cost= np.zeros(niter + 1)
    #     cost[k] = 1/2*(np.linalg.norm(A@x-b)**2) + eps*np.linalg.norm(x,1) 
    #     while(np.linalg.norm(grad_f) > tol and k<niter):
    #         x=gamma*y
    #         for i in range(A.shape[1]):    
    #             if v[i]>eps :
    #                 y[i] = v[i] - alpha*eps
    #             if v[i]<-eps :
    #                 y[i] = v[i] + alpha*eps
    #             if v[i] >= -eps and v[i] <= eps:
    #                 y[i] = 0
    #         x+=(1-gamma)*y 
    #         gamma = 2*(1-l)/(1+np.sqrt(1+4*l**2))
    #         l = (1+np.sqrt(1+4*l**2))/2 
    #         grad_f = A.T@(A@x-b)
    #         v = x - alpha*grad_f
    #         k+=1
    #         cost[k] = 1/2*(np.linalg.norm(A@x-b)**2) + eps*np.linalg.norm(x,1) 
    #     imdeblurfista0, n_eff_iter, cost_history = pylops.optimization.sparsity.fista(A,b,eps = eps,niter = niter)    
    #     opt_cost = cost_history[-1]
    #     opt_gap_cost = cost - opt_cost


    # else :
    #     print("Running ISTA...")

    #     x = np.zeros(A.shape[1]) 
    #     grad_f = (A.T)@(A@x-b)
    #     v = x - alpha*grad_f
    #     k=0
    #     cost= np.zeros(niter + 1)
    #     cost[k] = 1/2*(np.linalg.norm(A@x-b)**2) + eps*np.linalg.norm(x,1) 

    #     while(np.linalg.norm(grad_f) > tol and k<niter):
    #         for i in range(A.shape[1]):
    #             if v[i] > alpha*eps :
    #                 x[i] = v[i] - alpha*eps
    #             if v[i] < -alpha*eps :
    #                 x[i] = v[i] + alpha*eps
    #             if v[i] >= -alpha*eps and v[i] <= alpha*eps:
    #                 x[i] = 0
    #         grad_f = A.T@(A@x-b)
    #         v = x - alpha*grad_f
    #         k+=1
    #         cost[k] = 1/2*(np.linalg.norm(A@x-b)**2) + eps*np.linalg.norm(x,1)  
    #     imdeblurfista0, n_eff_iter, cost_history = pylops.optimization.sparsity.fista(A,b,eps = eps,niter = niter)    
    #     opt_cost = cost_history[-1]
    #     opt_gap_cost = cost - opt_cost

    alpha = 1.0 / np.abs((A.T@A).eigs(neigs = 1, symmetric = True)[0])

    print(f"Algorithm: {'FISTA' if acceleration else 'ISTA'}, Alpha: {alpha}, Eps: {eps}")

    x = np.zeros(A.shape[1])
    
    # Pour FISTA
    y = np.copy(x) # y_k (point d'inertie)
    l = 0.0        # lambda_k (paramètre de Nesterov)

    cost = np.zeros(niter + 1)    
    cost[0] = 0.5 * (np.linalg.norm(A @ x - b)**2) + eps * np.linalg.norm(x, 1)

    for k in range(niter):
        
        if acceleration:
            if k > 0:
                gamma = 2*(1-l)/(1+np.sqrt(1+4*l**2))
            else:
                gamma = 0
            pass
        
        grad_f = A.T @ (A @ x - b)
        v = x - alpha * grad_f
        
        threshold = alpha * eps
        
        # cette ligne remplace la boucle 'for'
        # on calcule le prox du L1 de manière vectorisée avec le max qui correspond à la partie positive
        y_next = np.sign(v) * np.maximum(np.abs(v) - threshold, 0)
        
        if acceleration:
            l_next = (1 + np.sqrt(1 + 4 * l**2)) / 2
            gamma = (1 - l) / l_next
            
            x_next = gamma * y + (1 - gamma) * y_next
            
            x = x_next
            y = y_next
            l = l_next
        else:
            x = y_next
            
        cost[k+1] = 0.5 * (np.linalg.norm(A @ x - b)**2) + eps * np.linalg.norm(x, 1)
        
        if np.linalg.norm(grad_f) < tol:
            print(f"Converged at iteration {k}")
            break

    cost = cost[:k+2] # on coupe les zéros en trop
    
    # baseline comparaison
    imdeblurfista0, n_eff_iter, cost_history = pylops.optimization.sparsity.fista(A,b,eps=eps,niter=niter)    
    opt_cost = cost_history[-1]
    opt_gap_cost = np.maximum(cost - opt_cost, 1e-16) # évite les valeurs négatives ou nulles pour le loglog

    return x, opt_gap_cost

def douglas_rashford_alg(A, b, opt_cost, eps=10**(-1), niter=10000, tol=1e-6, maxiter = 1000):
    # print("Running Douglas-Rachford...")
    # z = np.zeros(A.shape[1])
    # x=np.zeros(A.shape[1])
    # k = 0
    # cost= np.zeros(niter + 1)
    # cost[k] = 1/2*(np.linalg.norm(A@x-b)**2) + eps*np.linalg.norm(x,1) 
    # grad_f = (A.T)@(A@x-b)
    # while(k<niter and np.linalg.norm(grad_f)>eps):
    #     for i in range(A.shape[1]):
    #         if z[i]>eps :
    #             x[i] = z[i] - eps
    #         if z[i]<-eps :
    #             x[i] = z[i] + eps
    #         if (z[i]<=eps and z[i]>=-eps):
    #             x[i] = 0
    #     first_member = (A.T)@A + pylops.Identity(A.shape[1])
    #     second_member = (A.T)@b + 2*x-z
    #     y,conv = scipy.sparse.linalg.cg(first_member, second_member,rtol = tol,maxiter = maxiter)
    #     z = z + y -x
    #     k+=1
    #     cost[k] = 1/2*(np.linalg.norm(A@x-b)**2) + eps*np.linalg.norm(x,1) 

    #     # on avait oublié de mettre à jour grad_f avec le nouveau x, sans ça le test du while n'a pas de sens car on teste grad_f mais il n'est jamais mis à jour dans la boucle
    #     grad_f = (A.T) @ (A @ x - b)

    # imdeblurfista0, n_eff_iter, cost_history = pylops.optimization.sparsity.fista(A,b,eps = eps,niter = niter)    
    # opt_cost = cost_history[-1]
    # opt_gap_cost = cost - opt_cost

    print("Running Douglas-Rachford...")

    z = np.zeros(A.shape[1])
    x = np.zeros(A.shape[1])

    Op = (A.T @ A) + pylops.Identity(A.shape[1])

    cost = np.zeros(niter + 1)
    cost[0] = 0.5 * (np.linalg.norm(A @ x - b)**2) + eps * np.linalg.norm(x, 1)

    for k in range(niter):
        x = np.sign(z) * np.maximum(np.abs(z) - eps, 0)
        # threshold est 'eps' car pas de step size alpha

        rhs = A.T @ b + (2 * x - z)        
        y, info = scipy.sparse.linalg.cg(Op, rhs, tol=tol, maxiter=maxiter)
        z = z + y - x
        cost[k+1] = 0.5 * (np.linalg.norm(A @ x - b)**2) + eps * np.linalg.norm(x, 1)

    # baseline comparaison
    imdeblurfista0, n_eff_iter, cost_history = pylops.optimization.sparsity.fista(A,b,eps=eps,niter=niter)    
    opt_cost = cost_history[-1]
    opt_gap_cost = np.maximum(cost - opt_cost, 1e-16)

    return x, opt_gap_cost

def run_program(A, b, Wop, eps_value=10**(-1), baseline_iter=10000, my_iter=10000):
    
    # Baseline from pylops
    imdeblurfista0, n_eff_iter, cost_history = pylops.optimization.sparsity.fista(
        A, b, eps=eps_value, niter=baseline_iter
    )

    opt_cost = cost_history[-1]

    # ISTA
    my_imdeblurfista, opt_gap_cost = my_fista(
        A, b, opt_cost, eps=eps_value, niter=my_iter, acceleration=False)

    # FISTA
    my_imdeblurfista1, opt_gap_cost1 = my_fista(
        A, b, opt_cost, eps=eps_value, niter=my_iter, acceleration=True)

    #my_imdeblurfista1, opt_gap_cost1 = douglas_rashford_alg(A, b, opt_cost, eps=eps_value, niter=my_iter, tol=1e-5, maxiter = 1000)

    plt.loglog(opt_gap_cost, 'C0', label='ISTA')
    plt.loglog(opt_gap_cost1, 'C1', label='FISTA')
    #plt.loglog(opt_gap_cost1, 'C1', label='douglas rashford')
    plt.grid()
    plt.loglog([3, 30], [1e6, 1e5], 'C0--', label='1/k')
    plt.loglog([3, 30], [.5e5, .5e3], 'C1--', label='1/k2')

    plt.legend()
    plt.show()

    imdeblurfista = my_imdeblurfista1.reshape(A.dims)
    imdeblurfista = Wop.H * imdeblurfista

    return imdeblurfista

def visualise_results(im, imblur, imdeblurfista):
    #Change viridis into gray for castle image.

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Deblurring", fontsize=14, fontweight="bold", y=0.95)
    ax1 = plt.subplot2grid((2, 5), (0, 0))
    ax2 = plt.subplot2grid((2, 5), (0, 1))
    ax3 = plt.subplot2grid((2, 5), (0, 2))

    ax1.imshow(im, cmap="viridis", vmin=0, vmax=250)
    ax1.axis("tight")
    ax1.set_title("Original")
    ax2.imshow(imblur, cmap="viridis", vmin=0, vmax=250)
    ax2.axis("tight")
    ax2.set_title("Blurred")

    ax3.imshow(imdeblurfista, cmap="viridis", vmin=0, vmax=250)
    ax3.axis("tight")
    ax3.set_title("FISTA deblurred")

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)

    plt.show()

## Load the image according to your option
Wop, A, b, im, imblur = load_image_option_I()

## Run program you have coded:
imdeblurfista = run_program(A,b, Wop)

## Visualise your image results
visualise_results(im, imblur, imdeblurfista)









                         