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

def run_program(A, b, Wop, eps_value, baseline_iter, tol, my_iter, maxiter_DR, bz, bx):
    
    print(f"  -> Params: eps={eps_value}, blur=({bz}, {bx})")

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

    # Douglas-Rachford
    my_imdeblurfista2, opt_gap_cost2, t_dr = douglas_rachford_alg(
        A, b, opt_cost, eps=eps_value, niter=my_iter, tol=tol, maxiter=maxiter_DR)

    # Log times
    log_execution_time(bz, bx, eps_value, t_dr)

    fig_name = f"{OUTPUT_DIR}/convergence_eps{eps_value:.0e}_bz{bz}_bx{bx}.png"

    plt.figure(figsize=(10, 6))
    plt.loglog(opt_gap_cost2, 'C2', label='Douglas Rachford')

    plt.grid(True, which="both", ls="-")
    plt.loglog([3, 30], [1e6, 1e5], 'C0--', label='1/k')
    plt.loglog([3, 30], [.5e5, .5e3], 'C1--', label='1/k2')
    
    plt.xlabel("Number of iterations")
    plt.ylabel("Optimality gap: F - F*")
    plt.title(f"Convergence: eps={eps_value}, blur=({bz},{bx})")
    plt.legend()
    plt.savefig(fig_name)
    plt.close()

    imdeblurDR = my_imdeblurfista2.reshape(A.dims)
    imdeblurDR = Wop.H * imdeblurDR

    return imdeblurDR

def visualise_and_save(im, imdeblurDR, eps, bz, bx):
    fig_name = f"{OUTPUT_DIR}/visual_eps{eps:.0e}_bz{bz}_bx{bx}.png"

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f"Deblurring (eps={eps:.0e}, blur={bz}|{bx})", fontsize=14, fontweight="bold", y=0.95)
    ax1 = plt.subplot2grid((2, 5), (0, 0))
    ax2 = plt.subplot2grid((2, 5), (0, 1))
    ax3 = plt.subplot2grid((2, 5), (0, 2))

    ax1.imshow(im, cmap="viridis", vmin=0, vmax=250)
    ax1.axis("tight"); ax1.set_title("Original")
    
    ax2.imshow(imblur, cmap="viridis", vmin=0, vmax=250)
    ax2.axis("tight"); ax2.set_title("Blurred")

    ax3.imshow(imdeblurDR, cmap="viridis", vmin=0, vmax=250)
    ax3.axis("tight"); ax3.set_title("Douglas-Rachford")

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.savefig(fig_name)
    plt.close()


# create output directory if it doesn't exist
OUTPUT_DIR = "results_DR"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# log file to keep track of execution times for each configuration
LOG_FILE = os.path.join(OUTPUT_DIR, "execution_log.txt")

# initialize log file with headers
with open(LOG_FILE, "w") as f:
    f.write("blur_z,blur_x,epsilon,time_dr\n")


def log_execution_time(bz, bx, eps, t_dr):
    """Save execution times for each configuration to a log file."""
    with open(LOG_FILE, "a") as f:
        f.write(f"{bz},{bx},{eps},{t_dr:.5f}\n")

baseline_iter = 5000
tol = 1e-10
my_iter = 5000 
maxiter_DR = 1000

# variable parameters 
blur_values = [(0.01, 0.03), (0.1, 0.3), (1.0, 3.0)]
eps_values = [1e-4, 1e-8, 1e-12, 1e-16]

total_steps = len(blur_values) * len(eps_values)
current_step = 0

print(f"\n\nStarting Batch Processing for DR... Total configurations: {total_steps}")
start_global = time.time()

for (bz, bx) in blur_values:
    print(f"\n--- CHANGING BLUR to ({bz}, {bx}) ---")
    Wop, A, b, im, imblur = load_image_option_I(bz, bx)
    
    for eps in eps_values:        
        current_step += 1
        print(f"\n[{current_step}/{total_steps}] Running: eps={eps}...")
            
        res_dr = run_program(A, b, Wop, eps, baseline_iter, tol, my_iter, maxiter_DR, bz, bx)
            
        visualise_and_save(im, res_dr, eps, bz, bx)

end_global = time.time()
print(f"\nAll done in {end_global - start_global:.2f} seconds.")
print(f"Results are available in the folder: {OUTPUT_DIR}")