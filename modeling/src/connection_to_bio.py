import numpy as np
import scipy.stats as stats

def simulation_sensitivity_organoids(steps,dt,Volume_L,params):#

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
    #medium
    #Paper says medium has concentrations from 5-20mM
    medium_d = stats.norm(loc= 12.5,scale = 3)

    #oxygenconcentration of 0.20-0.22mM
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
    #myu_waste_d = stats.norm(loc= 26,scale = 10)
    myu_waste_d = stats.uniform(loc=0, scale=60)  # samples 0 to 60
    
    #km_m
    km_m_d = stats.norm(loc= 0.2,scale= 0.02)
    #km_o
    km_o_d = stats.norm(loc= 0.2,scale = 0.02)
    #km_inhibition
    km_w_d = stats.truncnorm(0,1,loc= 0.2,scale = 0.2)

    km_a_d = stats.truncnorm(0,1,loc= 0.2,scale = 0.2)


    #yield
    aerobic_yield = stats.norm(loc = 0.6,scale = 0.3)
    #aerobic_yield = stats.beta(a=3, b=2)
    #myu_fixed_cost
    fixed_cost = 0.07
    myu_fixed_costs_d= stats.norm(loc =fixed_cost,scale = fixed_cost*0.5)


    myu_activity_costs = stats.norm(0,0)
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

