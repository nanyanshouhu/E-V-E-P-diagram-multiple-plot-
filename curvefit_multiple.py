import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
N=1
# Define a function to calculate discrete point derivatives
def cal_deriv(x, y):                  # x and y are both lists
    diff_x = []                       # Used to store the difference between two numbers in the x list
    for i, j in zip(x[0::], x[1::]):
        diff_x.append(j - i)

    diff_y = []                       # Used to store the difference between two numbers in the y list
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)

    slopes = []                       # Used to store the slopes
    for i in range(len(diff_y)):
        slopes.append(diff_y[i] / diff_x[i])

    deriv = []                        # Used to store the first derivative
    for i, j in zip(slopes[0::], slopes[1::]):
        deriv.append((0.5 * (i + j)))  # Calculate and store the result according to the definition of the discrete point derivative
    deriv.insert(0, slopes[0])        # The derivative at the (left) end is the slope with its nearest point
    deriv.append(slopes[-1])          # The derivative at the (right) end is the slope with its nearest point

    for i in deriv:                   # Print the result for easy checking (can be commented out when calling)
        print(i)

    return deriv                      # Return the list storing the first derivative results
def read_data(filename):
    """ read the energy and Volume, the format shows as follow: 
    the first line are structure factors, the second line are volumes, and the third lines are energy:
        3.74 136.2000 -105.833996
        3.75 136.9300 -105.865334
        3.76 137.6600 -105.892136
        3.78 139.1300 -105.928546
        3.79 139.8600 -105.944722
        3.80 140.6000 -105.955402
        3.81 141.3400 -105.960574
        3.82 142.0900 -105.960563
        3.83 142.8300 -105.954437
        3.84 143.5800 -105.949877
    """
    # data = np.loadtxt(filename)
    data = np.loadtxt(filename)
    return data[:,1], data[:,2] 
def eos_murnaghan(vol, E0, B0, BP, V0):
    # First term in the equation
    term1 = (4 * B0 * V0) / (BP - 1)**2
    
    # Second term in the equation
    term2 = 1 - (3 / 2) * (BP - 1) * (1 - (vol / V0)**(1 / 3))
    
    # Exponential term
    exp_term = np.exp((3 / 2) * (BP - 1) * (1 - (vol / V0)**(1 / 3)))
    
    # Energy calculation
    E = E0 + term1 - term1 * term2 * exp_term
    
    return E
def fit_murnaghan(volume, energy):
    """ fittint Murnaghan equation，and return the optimized parameters 
    """
    # fitting with Quadratic first and then get the guess parameters.
    p_coefs = np.polyfit(volume, energy, 2)
    # the lowest point of parabola dE/dV = 0 ( p_coefs = [c,b,a] ) V(min) = -b/2a
    p_min = - p_coefs[1]/(2.*p_coefs[0])
    # warn if min volume not in result range 
    if (p_min < volume.min() or p_min > volume.max()):
        print ("Warning: minimum volume not in range of results")
    # estimate the energy based the the lowest point of parabola   
    E0 = np.polyval(p_coefs, p_min)
    # estimate the bulk modules
    B0 = 2.*p_coefs[2]*p_min
    # guess the parameter (set BP as 4)
    init_par = [E0, B0, 4, p_min]
    print ("guess parameters:")
    print (" V0     =  {:1.4f} A^3 ".format(init_par[3]))
    print (" E0     =  {:1.4f} eV  ".format(init_par[0]))
    print (" B(V0)  =  {:1.4f} eV/A^3".format(init_par[1]))
    print (" B'(VO) =  {:1.4f} ".format(init_par[2]))
    best_par, cov_matrix = curve_fit(eos_murnaghan, volume, energy, p0 = init_par)
    residuals = energy - eos_murnaghan(volume, *best_par)
    ssr = np.sum(residuals**2)
    s_sq = ssr / (len(volume) - len(best_par))
    cov_diag = np.diag(cov_matrix)
    std_errors = np.sqrt(cov_diag)
    return best_par
    
def fit_and_plot(filenames):
    colors = ['k', 'b', 'r', 'c', 'm', 'y', 'g']  # Additional colors can be added if needed
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    label_names = [
    '(Hf$_{0.5}$,Zr$_{0.5}$)TiO$_{4}$',
    '(Hf$_{0.25}$,Zr$_{0.25}$,Ti$_{0.5}$)$_{2}$O$_{4}$',
    'HfTiO$_{4}$',
    'ZrTiO$_{4}$',
    ]  # Add custom label names here
    volumes = []  # List to store volumes
    pressures = []  # List to store pressures
    for i, filename in enumerate(filenames):
        volume, energy = read_data(filename)
        pfit, perr = fit_curvefit(volume, energy)
        best_par = fit_murnaghan(volume, energy)
        m_volume = np.linspace(volume.min(), volume.max(), 550)
        m_energy = eos_murnaghan(m_volume, *best_par)
        P=np.array(cal_deriv(m_volume,m_energy)*(-m_volume))*160.218
        label_data = label_names[i] + ' (data)'  # Generate label for data plot
        label_fit = label_names[i] + ' (fit)'  # Generate label for fit plot
        A_cubed = best_par[3] / N
        E_eq=best_par[0]/N
        label_text = f"V$_{{eq}}$: {A_cubed:.3f} A$^{{3}}$ (E$_{{eq}}$: {E_eq:.3f} eV/atom)"
        volumes.append(m_volume/N)  # Store volume data
        pressures.append(P/N)  # Store pressure data
        axs[0].plot(volume/N, energy/N, 'o', fillstyle='none', color=colors[i % len(colors)], markersize=10, label=label_data)  # Assign label to scatter plot
        axs[0].plot(m_volume/N, m_energy/N, '-', color=colors[i % len(colors)], linewidth=1.5, label=label_fit)  # Assign label to curve plot
        axs[0].plot(best_par[3]/N, best_par[0]/N, '*', fillstyle='none', color=colors[i % len(colors)], markersize=10, label=label_text)
        axs[1].plot(P/N, m_energy/N, '-', color=colors[i % len(colors)], linewidth=1.5, label=label_fit)

        print("Fit parameters for", filename + ":")
        print(" V0     =  {:1.4f} A^3 ".format(best_par[3]))
        print(" E0     =  {:1.4f} eV  ".format(best_par[0]))
        print(" B(V0)  =  {:1.4f} eV/A^3".format(best_par[1]))
        print(" B'(VO) =  {:1.4f} ".format(best_par[2]))

        print("\n# Fit parameters and parameter errors from curve_fit method :")
        print("pfit = ", pfit)
        print("perr = ", perr)
        pd.DataFrame(np.concatenate((np.array(volumes).reshape(-1,1), np.array(pressures).reshape(-1,1)),axis=1)).to_csv('V-P_fit.csv')
    axs[0].set_xlabel(r"Volume [$\rm{A}^3$]", fontsize=10)
    axs[0].set_ylabel(r"Energy [$\rm{eV/atom}$]", fontsize=10)
    axs[0].tick_params(axis='both', which='major', labelsize=10)
    axs[1].set_xlabel(r"Pressure [$\rm{GPa}$]", fontsize=10)
    axs[1].set_ylabel(r"Energy [$\rm{eV/atom}$]", fontsize=10)
    axs[1].tick_params(axis='both', which='major', labelsize=10)
    axs[0].legend(fontsize=8, frameon=False)
    axs[1].legend(fontsize=8, frameon=False)

    return best_par, pfit, perr
def fit_curvefit(volume, energy):
    """
    Note: As per the current documentation (Scipy V1.1.0), sigma (yerr) must be:
        None or M-length sequence or MxM array, optional
    Therefore, replace:
        err_stdev = 0.2
    With:
        err_stdev = [0.2 for item in xdata]
    Or similar, to create an M-length sequence for this example.
    """
    # Remove the unnecessary line that reassigns volume and energy
    # volume, energy = read_data(filenames)

    # fitting with Quadratic first and then get the guess parameters.
    p_coefs = np.polyfit(volume, energy, 2)
    # the lowest point of parabola dE/dV = 0 ( p_coefs = [c,b,a] ) V(min) = -b/2a
    p_min = - p_coefs[1] / (2. * p_coefs[0])
    # warn if min volume not in result range
    if (p_min < volume.min() or p_min > volume.max()):
        print("Warning: minimum volume not in range of results")
    # estimate the energy based on the lowest point of parabola
    E0 = np.polyval(p_coefs, p_min)
    # estimate the bulk modulus
    B0 = 2. * p_coefs[2] * p_min
    # guess the parameter (set BP as 4)
    init_par = [E0, B0, 4, p_min]
    pfit, pcov = curve_fit(eos_murnaghan, volume, energy, p0=init_par)
    error = []
    for i in range(len(pfit)):
        try:
            error.append(np.absolute(pcov[i][i]) ** 0.5)
        except:
            error.append(0.00)

    pfit_curvefit = pfit
    perr_curvefit = np.array(error)
    return pfit_curvefit, perr_curvefit 

num_files=4
# filenames = [f"curvefit{i}.txt" for i in range(1, num_files+1)]
filenames = [f"curvefit{i}.txt" for i in range(1, num_files+1)]
# print(filenames)
# exit()

# volume, energy = read_data(filenames)
best_par, pfit, perr = fit_and_plot(filenames)
plt.draw()
plt.tight_layout()
plt.savefig('E-V_curve.png', format='png', dpi=330) 

