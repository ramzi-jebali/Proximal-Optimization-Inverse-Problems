import matplotlib.pyplot as plt
import numpy as np
import pylops 
import scipy.sparse.linalg
import time
import os

def load_image_option_I(bz=0.1, bx=0.3):
    sampling = 5
    im = np.load("dog_rgb.npy")[::sampling, ::sampling, 2]
    Nz, Nx = im.shape

    # Blurring Gaussian operator
    nh = [15, 25]
    hz = np.exp(-bz * np.linspace(-(nh[0] // 2), nh[0] // 2, nh[0]) ** 2)
    hx = np.exp(-bx * np.linspace(-(nh[1] // 2), nh[1] // 2, nh[1]) ** 2)
    hz /= np.trapezoid(hz)  # normalize the integral to 1
    hx /= np.trapezoid(hx)  # normalize the integral to 1
    h = hz[:, np.newaxis] * hx[np.newaxis, :]

    # Commented out to prevent blocking execution
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    # him = ax.imshow(h)
    # ax.set_title("Blurring operator")
    # fig.colorbar(him, ax=ax)
    # ax.axis("tight")
    # plt.show()

    Cop = pylops.signalprocessing.Convolve2D(
        (Nz, Nx), h=h, offset=(nh[0] // 2, nh[1] // 2), dtype="float32")

    imblur = Cop * im
    
    # Commented out to prevent blocking execution
    # plt.imshow(im, cmap="viridis", vmin=0, vmax=255)
    # plt.show()
    # plt.imshow(imblur, cmap="viridis", vmin=0, vmax=255)
    # plt.show()

    Wop = pylops.signalprocessing.DWT2D((Nz, Nx), wavelet="haar", level=3)

    # This is your A and b for your f1 cost!
    A = Cop * Wop.H
    b = imblur.ravel()

    return Wop, A, b, im, imblur

def load_image_option_II(bz=0.1, bx=0.3):
    sampling = 2
    im = np.load("chateau.npy")[::sampling, ::sampling, 1]
    Nz, Nx = im.shape

    # Blurring Gaussian operator
    nh = [15, 25]
    hz = np.exp(bz * np.linspace(-(nh[0] // 2), nh[0] // 2, nh[0]) ** 2)
    hx = np.exp(bx * np.linspace(-(nh[1] // 2), nh[1] // 2, nh[1]) ** 2)
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


def my_fista(A, b, opt_cost, eps=10**(-1), niter=100, tol=1e-10, acceleration=False, p=1):
    """ Here you can code your ISTA and FISTA algorithm
        Return: optimal x, opt_gap_cost, execution_time
    """
    start_time = time.time() # Start Timer

    L = np.abs((A.T@A).eigs(neigs = 1, symmetric = True)[0])
    alpha = 1.0 / (p*L)

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
        
        grad_f = A.T @ (A @ x - b)
        v = x - alpha * grad_f
        
        threshold = alpha*eps
        
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
            break

    cost = cost[:k+2] # on coupe les zéros en trop
    
    # baseline comparaison
    opt_gap_cost = cost - opt_cost

    exec_time = time.time() - start_time # End Timer
    return x, opt_gap_cost, exec_time

def douglas_rachford_alg(A, b, opt_cost, eps=10**(-1), niter=100, tol=1e-10, maxiter = 100):
    start_time = time.time() # Start Timer

    z = np.zeros(A.shape[1])
    x = np.zeros(A.shape[1])

    Op = (A.T @ A) + pylops.Identity(A.shape[1])

    cost = np.zeros(niter + 1)
    cost[0] = 0.5 * (np.linalg.norm(A @ x - b)**2) + eps * np.linalg.norm(x, 1)

    # on garde en mémoire le 'y' précédent pour aider le solveur cg (warm start)
    y_guess = np.zeros(A.shape[1])
    for k in range(niter):
        x = np.sign(z) * np.maximum(np.abs(z) - eps, 0)

        rhs = A.T @ b + (2 * x - z)        
        y, info = scipy.sparse.linalg.cg(Op, rhs, x0=y_guess, rtol=tol, maxiter=maxiter)
        y_guess = y # on met à jour le guess pour le prochain solveur cg, ça peut aider à accélérer la convergence du solveur linéaire
        z = z + y - x
        cost[k+1] = 0.5 * (np.linalg.norm(A @ x - b)**2) + eps * np.linalg.norm(x, 1)

    # baseline comparaison
    opt_gap_cost = cost - opt_cost

    exec_time = time.time() - start_time # End Timer
    return x, opt_gap_cost, exec_time

def run_program(A, b, Wop, eps_value, baseline_iter, tol, my_iter, maxiter_DR, p, bz, bx):
    
    print(f"  -> Params: eps={eps_value}, p={p}, blur=({bz}, {bx})")

    filename = f"baseline_opt_costs/opt_cost_baseline_eps{eps_value:.0e}_iter{baseline_iter}_bz{bz}_bx{bx}.npy"
    
    if os.path.exists(filename):
        opt_cost = np.load(filename)
    else:
        print("     Calculating baseline...")
        # Baseline from pylops
        imdeblurfista0, n_eff_iter, cost_history = pylops.optimization.sparsity.fista(
            A, b, eps=eps_value, niter=baseline_iter)
        opt_cost = cost_history[-1]
        
        # save the baseline cost
        np.save(filename, opt_cost)

    # ISTA
    my_imdeblurfista, opt_gap_cost, t_ista = my_fista(
        A, b, opt_cost, eps=eps_value, niter=my_iter, tol=tol, acceleration=False, p=p)

    # FISTA
    my_imdeblurfista1, opt_gap_cost1, t_fista = my_fista(
        A, b, opt_cost, eps=eps_value, niter=my_iter, tol=tol, acceleration=True, p=p)  

    # Douglas-Rachford
    my_imdeblurfista2, opt_gap_cost2, t_dr = douglas_rachford_alg(
        A, b, opt_cost, eps=eps_value, niter=my_iter, tol=tol, maxiter=maxiter_DR)

    # Log times
    log_execution_time(bz, bx, eps_value, p, t_ista, t_fista, t_dr)

    fig_name = f"{OUTPUT_DIR}/convergence_eps{eps_value:.0e}_p{p}_bz{bz}_bx{bx}.png"

    plt.figure(figsize=(10, 6))
    plt.loglog(opt_gap_cost, 'C0', label='ISTA')
    plt.loglog(opt_gap_cost1, 'C1', label='FISTA')
    plt.loglog(opt_gap_cost2, 'C2', label='Douglas Rachford')

    plt.grid(True, which="both", ls="-")
    plt.loglog([3, 30], [1e6, 1e5], 'C0--', label='1/k')
    plt.loglog([3, 30], [.5e5, .5e3], 'C1--', label='1/k2')
    
    plt.xlabel("Number of iterations")
    plt.ylabel("Optimality gap: F - F*")
    plt.title(f"Convergence: eps={eps_value}, alpha=1/({p}*L), blur=({bz},{bx})")
    plt.legend()
    plt.savefig(fig_name)
    plt.close()

    imdeblurfista = my_imdeblurfista1.reshape(A.dims)
    imdeblurfista = Wop.H * imdeblurfista

    imdeblurDR = my_imdeblurfista2.reshape(A.dims)
    imdeblurDR = Wop.H * imdeblurDR

    return imdeblurfista, imdeblurDR

def visualise_and_save(im, imblur, imdeblurfista, imdeblurDR, eps, p, bz, bx):
    fig_name = f"{OUTPUT_DIR}/visual_eps{eps:.0e}_p{p}_bz{bz}_bx{bx}.png"

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f"Deblurring (eps={eps:.0e}, alpha=1/({p}*L), blur={bz}|{bx})", fontsize=14, fontweight="bold", y=0.95)
    ax1 = plt.subplot2grid((2, 5), (0, 0))
    ax2 = plt.subplot2grid((2, 5), (0, 1))
    ax3 = plt.subplot2grid((2, 5), (0, 2))
    ax4 = plt.subplot2grid((2, 5), (0, 3))

    ax1.imshow(im, cmap="viridis", vmin=0, vmax=250)
    ax1.axis("tight"); ax1.set_title("Original")
    
    ax2.imshow(imblur, cmap="viridis", vmin=0, vmax=250)
    ax2.axis("tight"); ax2.set_title("Blurred")

    ax3.imshow(imdeblurfista, cmap="viridis", vmin=0, vmax=250)
    ax3.axis("tight"); ax3.set_title("FISTA")

    ax4.imshow(imdeblurDR, cmap="viridis", vmin=0, vmax=250)
    ax4.axis("tight"); ax4.set_title("Douglas-Rachford")

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.savefig(fig_name)
    plt.close()


# create output directory if it doesn't exist
OUTPUT_DIR = "results_project"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# log file to keep track of execution times for each configuration
LOG_FILE = os.path.join(OUTPUT_DIR, "execution_log.txt")

# initialize log file with headers
with open(LOG_FILE, "w") as f:
    f.write("blur_z,blur_x,epsilon,p,time_ista,time_fista,time_dr\n")


def log_execution_time(bz, bx, eps, p, t_ista, t_fista, t_dr):
    """Save execution times for each configuration to a log file."""
    with open(LOG_FILE, "a") as f:
        f.write(f"{bz},{bx},{eps},{p},{t_ista:.5f},{t_fista:.5f},{t_dr:.5f}\n")

baseline_iter = 5000
tol = 1e-10
my_iter = 5000 
maxiter_DR = 1000

# variable parameters 
blur_values = [(0.1, 0.3), (1.0, 3.0)]
eps_values = [1e-4, 1e-8, 1e-12]
p_values = [1, 2, 4, 8, 12]

total_steps = len(blur_values) * len(eps_values) * len(p_values)
current_step = 0

print(f"\n\nStarting Batch Processing... Total configurations: {total_steps}")
start_global = time.time()

for (bz, bx) in blur_values:
    print(f"\n--- CHANGING BLUR to ({bz}, {bx}) ---")
    Wop, A, b, im, imblur = load_image_option_I(bz, bx)
    
    for eps in eps_values:        
        for p in p_values:

            current_step += 1
            print(f"\n[{current_step}/{total_steps}] Running: eps={eps}, p={p}...")
            
            res_fista, res_dr = run_program(A, b, Wop, eps, baseline_iter, tol, my_iter, maxiter_DR, p, bz, bx)
            
            visualise_and_save(im, imblur, res_fista, res_dr, eps, p, bz, bx)

end_global = time.time()
print(f"\nAll done in {end_global - start_global:.2f} seconds.")
print(f"Results are available in the folder: {OUTPUT_DIR}")