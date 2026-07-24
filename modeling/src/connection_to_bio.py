import numpy as np
import scipy.stats as stats

def simulation_sensitivity_organoids(steps,dt,Volume_L,params):
    """
    Simulates the biological dynamics of organoids over time with sensitivity analysis in mind.
    Tracks ATP, glucose (M), oxygen (OXY), and waste (W) levels based on provided parameters.

    Parameters:
    - steps (int): Number of time steps for the simulation.
    - dt (float): Time step size (in minutes).
    - Volume_L (float): Volume of the experimental setup (in liters).
    - params (list): List of parameters for the simulation. The parameters are:
        - M_0 (float): Initial glucose concentration (mM).
        - OXY_0 (float): Initial oxygen concentration (mM).
        - ORGANOID_VOLUME (float): Volume of the organoid (m³).
        - CELLDENSITY (float): Cell density (cells/m³).
        - myu_glucose (float): Glucose consumption rate (mol/cell*s).
        - myu_oxygen (float): Oxygen consumption rate (mol/cell*s).
        - myu_waste_yield (float): Waste production yield.
        - k_m_m (float): Michaelis-Menten constant for glucose.
        - k_m_o (float): Michaelis-Menten constant for oxygen.
        - k_m_i (float): Inhibition constant for waste.
        - k_m_a (float): Inhibition constant for ATP.
        - yield_aerobic (float, optional): Aerobic yield coefficient. If not provided, defaults to 0.6.
        - myu_fixed_costs (float): Fixed costs for ATP production.

    Returns:
    - tuple: (ATP_p, M, OXY, W, params)
      - ATP_p (numpy.ndarray): ATP levels over time.
      - M (numpy.ndarray): Glucose levels over time.
      - OXY (numpy.ndarray): Oxygen levels over time.
      - W (numpy.ndarray): Waste levels over time.
      - params (list): Input parameters used for the simulation.

    Notes:
    - Uses Michaelis-Menten kinetics for glucose and oxygen consumption.
    - Includes inhibition terms for waste and ATP.
    - All values are clamped to non-negative values.
    - Handles two analysis cases:
        1. `yield_aerobic` and `myu_fixed_costs` are free/sampled parameters (13 parameters).
        2. `yield_aerobic` is fixed to the ABC posterior mean (0.6), and `myu_fixed_costs` is a free parameter (12 parameters).
    """

    M_0             = params[0]
    OXY_0           = params[1]
    ORGANOID_VOLUME = params[2]
    CELLDENSITY     = params[3]
    myu_glucose     = params[4]
    myu_oxygen      = params[5]
    myu_waste_yield = params[6]
    k_m_m           = params[7]
    k_m_o           = params[8]
    k_m_i           = params[9]
    k_m_a           = params[10]

    if len(params) == 13:
        # Analysis 1: yield_aerobic is a free/sampled parameter
        yield_aerobic   = params[11]
        myu_fixed_costs = params[12]
    else:
        # Analysis 2: yield_aerobic fixed to ABC posterior mean
        yield_aerobic   = 0.6          # <- posterior mean, update if it changes
        myu_fixed_costs = params[11]

    # --- 2. Initial States ---
    # We start with 100% health, 2.5mL medium, and 0 waste.
    M = np.zeros(steps);  M[0] = M_0 # mM;
    W = np.zeros(steps);  W[0] = 0.0
    OXY = np.zeros(steps);  OXY[0] = OXY_0 # mM
    ATP_p = np.zeros(steps); ATP_p[0] = 0.9

    # volume_organoid =  4/3 * np.pi* (3/2)**3 *1e-9 # mm³ → m³
    # number_of_organoids = 18

    X = np.zeros(steps);   X[0] = ORGANOID_VOLUME # Biomass (Cell Viability)

    Volume_L =Volume_L# Volume of the Experimental Setup

    # we multiply the baserates by 60 to convert into minutes
    # we scale them by the amount of volume since in the simulation the should represent the effect on the concentration 
    # this effect will be smaller if volume is big
    #myu_m = base_myu_m*60 / Volume_ml *1e-3
    myu_m = myu_glucose *CELLDENSITY* (X[0] / Volume_L) * 60
    myu_o = myu_oxygen *CELLDENSITY* (X[0] / Volume_L) * 60
    myu_w = myu_waste_yield*myu_m #1-15, 6 would allow max aerobic metabolism, upper numbers would reflect lactic acid

    #K konstants
    Km_m_norm = k_m_m
    Km_o_norm = k_m_o

    Km_m_bio = Km_m_norm * M_0 #mM
    Km_o_bio = Km_o_norm * OXY_0 #mM

    #random_constants
    yield_aerobic = yield_aerobic
    myu_fixed_cost = myu_fixed_costs


    def inhibition(w, Ki=k_m_i):
        return (w) / (Ki + w)

    for t in range(1, steps):
        # STEP A: Normalization (0.0 to 1.0 scale)

        M_ratio = M[t-1] / (Km_m_bio+ M[t-1])       #Michealis- Menten Curve
        OXY_ratio = OXY[t-1] / (Km_o_bio+OXY[t-1])

        aerobic = M_ratio * OXY_ratio * (1-inhibition(W[t-1])) * (1-inhibition(ATP_p[t-1],k_m_a))

        prod = (aerobic * yield_aerobic) 
        cost = (myu_fixed_cost)

        dATP_p =  prod-cost
        
        dW =  myu_w   *aerobic
        dM =  -myu_m  *aerobic
        dOxy = -myu_o *aerobic

        ATP_p[t] = max(0,ATP_p[t-1] + dATP_p * dt)
        W[t] = max(0,W[t-1]+ dW*dt)
        M[t] = max(0,M[t-1]+ dM*dt)
        OXY[t] = max(0,OXY[t-1]+ dOxy*dt)

    return ATP_p,M,OXY,W,params



def bayes_prior():
    """
    Defines Bayesian prior distributions for the parameters used in the organoid simulation.
    These priors represent the initial beliefs about the parameter values before observing any data.

    Returns:
    - dict: A dictionary where keys are parameter names and values are `scipy.stats` distribution objects.

    Notes:
    - Each parameter is assigned a prior distribution based on literature values or reasonable assumptions.
    - The distributions include:
        - Normal distributions for parameters like glucose concentration, oxygen concentration, and cell density.
        - Uniform distributions for parameters like waste yield.
        - Truncated normal distributions for inhibition constants to ensure non-negative values.
    """

    # Mediumconcentraion
    medium_d = stats.norm(loc= 12.5,scale = 3)

    #Oxygenconcentration of 0.20-0.22mM
    oxygen_d = stats.norm(loc = 0.21,scale = 0.01)
    
    #orgnaoid_volume
    volume_organoid =  4/3 * np.pi* (3/2)**3 *1e-9 # mm³ → m³
    number_of_organoids = 18
    total_volume = volume_organoid*number_of_organoids
    total_organoid_volume = stats.norm(loc= total_volume,scale = total_volume*0.2)

    #cell densitiy
    cell_density = 4.82e11  # cells/m^3 (example density)
    cell_density_d = stats.norm(loc = cell_density,scale = cell_density*0.15)

    #myu medium
    glucose_consumption = 2.0e-16 # mol/cell*s 
    myu_glucose_consumption_d = stats.norm(loc= glucose_consumption,scale = glucose_consumption*0.2)

    #myu oxygen
    oxygen_consumption = 7.7e-16  # mol/cell*s
    myu_oxygen_consumption_d = stats.norm(loc= oxygen_consumption,scale= oxygen_consumption*0.2)

    #myu_waste 
    myu_waste_d = stats.uniform(loc=0, scale=60)  # samples 0 to 60
    #km_m
    km_m_d = stats.norm(loc= 0.2,scale= 0.02)
    #km_o
    km_o_d = stats.norm(loc= 0.2,scale = 0.02)
    #km_inhibition
    km_w_d = stats.truncnorm(0,1,loc= 0.2,scale = 0.2)
    #atp selfregulation
    km_a_d = stats.truncnorm(0,1,loc= 0.2,scale = 0.2)

    #yield
    aerobic_yield = stats.norm(loc = 0.6,scale = 0.3)

    #myu_fixed_cost
    fixed_cost = 0.07
    myu_fixed_costs_d= stats.norm(loc =fixed_cost,scale = fixed_cost*0.5)

    return {"M_0":medium_d,
            "OXY_0":oxygen_d,
            "ORGANOID_VOLUME":total_organoid_volume,
            "CELLDENSITY":cell_density_d,
            "myu_glucose":myu_glucose_consumption_d,
            "myu_oxygen":myu_oxygen_consumption_d,
            "myu_waste_yield":myu_waste_d,
            "k_m_m":km_m_d,
            "k_m_o":km_o_d,
            "k_m_i":km_w_d,
            "k_m_a":km_a_d,
            "yield_aerobic":aerobic_yield,
            "myu_fixed_costs":myu_fixed_costs_d}

