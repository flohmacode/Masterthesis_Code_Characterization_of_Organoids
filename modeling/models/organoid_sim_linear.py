import numpy as np
import matplotlib.pyplot as plt
import src.connection_to_bio as ctb
import src.helper as helper

def organoid_sim_spec_params_linear(steps,dt,Volume_L,params):
    M_0 = params['M_0']
    OXY_0 = params['OXY_0']
    ORGANOID_VOLUME= params['ORGANOID_VOLUME']
    CELLDENSITY = params['CELLDENSITY']
    myu_glucose = params['myu_glucose']
    myu_oxygen = params['myu_oxygen']
    myu_waste_yield = params['myu_waste_yield']
    k_m_m = params['k_m_m']
    k_m_o = params['k_m_o']
    k_m_i = params['k_m_i']
    k_m_a = params['k_m_a']
    yield_aerobic = params['yield_aerobic']
    myu_fixed_costs = params['myu_fixed_costs']


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
    myu_w = myu_waste_yield*myu_m 

    #K konstants
    Km_m_norm = k_m_m
    Km_o_norm = k_m_o

    Km_m_bio = Km_m_norm * M_0 #mM
    Km_o_bio = Km_o_norm * OXY_0 #mM

    #random_constants
    yield_aerobic = yield_aerobic

    myu_fixed_cost = myu_fixed_costs
    #myu_activity_cost = myu_activity_costs

    def inhibition(w, Ki=k_m_i):
        return (w) / (Ki + w)

    for t in range(1, steps):
        # STEP A: Normalization (0.0 to 1.0 scale)

        M_ratio = M[t-1] / (Km_m_bio+ M[t-1])       #Michealis- Menten Curve
        OXY_ratio = OXY[t-1] / (Km_o_bio+OXY[t-1])

        aerobic = M_ratio * OXY_ratio * (1-inhibition(W[t-1],k_m_i)) * (1-inhibition(ATP_p[t-1],k_m_a))

        prod = (aerobic * yield_aerobic) 
        cost = (myu_fixed_cost) #+ myu_activity_cost * W[t-1])

        dATP_p =  prod-cost
        
        dW =  myu_w   *aerobic
        dM =  -myu_m  *aerobic
        dOxy = -myu_o *aerobic

        ATP_p[t] = max(0,ATP_p[t-1] + dATP_p * dt)
        W[t] = max(0,W[t-1]+ dW*dt)
        M[t] = max(0,M[t-1]+ dM*dt)
        OXY[t] = max(0,OXY[t-1]+ dOxy*dt)

    return ATP_p,M,OXY,W

def run_bayes_linear(steps,dt,Volume_L):

    p = ctb.bayes_prior()

    M_0 = 20                                            #fixed bc measureable
    OXY_0 = 0.21                                        #fixed bc measureable
    ORGANOID_VOLUME = 4/3 * np.pi* (3/2)**3 *1e-9 *22   #fixed bc measureable
    CELLDENSITY = 4.82e11                               #fixed bc measureable
    myu_glucose = 2.0e-16 # mol/cell*s                  #fixed bc literature
    myu_oxygen = p["myu_oxygen"].rvs()                 #fixed bc literature, showed as example for uninformative prior
    myu_waste_yield = p["myu_waste_yield"].rvs()
    k_m_m = 0.2                                         #fixed bc im always saturated this shouldnt be important as i dont enter this regime
    k_m_o = 0.2                                         #fixed bc im always saturated this shouldnt be important as i dont enter this regime
    k_m_i = p["k_m_i"].rvs()
    k_m_a = p["k_m_a"].rvs()
    yield_aerobic = p["yield_aerobic"].rvs()
    myu_fixed_costs = p["myu_fixed_costs"].rvs()


    params = {'M_0':M_0,
              'OXY_0':OXY_0,
              'ORGANOID_VOLUME':ORGANOID_VOLUME,
              'CELLDENSITY':CELLDENSITY,
              'myu_glucose':myu_glucose,
              'myu_oxygen':myu_oxygen,
              'myu_waste_yield':myu_waste_yield,
              'k_m_m':k_m_m,
              'k_m_o':k_m_o,
              'k_m_i':k_m_i,
              'k_m_a':k_m_a,
              'yield_aerobic':yield_aerobic,
              'myu_fixed_costs':myu_fixed_costs}

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
    myu_m = myu_glucose *CELLDENSITY* (X[0] / Volume_L) * 60 #*60 to convert to minutes
    myu_o = myu_oxygen *CELLDENSITY* (X[0] / Volume_L) * 60
    myu_w = myu_waste_yield*myu_m

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

        aerobic = M_ratio * OXY_ratio * (1-inhibition(W[t-1],k_m_i)) * (1-inhibition(ATP_p[t-1],k_m_a))

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

def eval_accepted_parameters(steps,dt,Volume,accepted_parameters):
    trajectories = {'ATP_p':np.empty((len(accepted_parameters),steps)),
              'M':np.empty((len(accepted_parameters),steps)),
              'OXY':np.empty((len(accepted_parameters),steps)),
              'W':np.empty((len(accepted_parameters),steps))}
    
    for key,value in enumerate(accepted_parameters):
        ATP_p,M,OXY,W = organoid_sim_spec_params_linear(steps,dt,Volume,value)
        trajectories['ATP_p'][key,:] = ATP_p
        trajectories['M'][key,:] = M
        trajectories['OXY'][key,:] = OXY
        trajectories['W'][key,:] = W
    return trajectories

def run_and_plot_linear(name,steps,dt,total_time,Volume,best_parameter,accepted_parameters,rmse_best_param= [9999,9999],constraint = False):

    rmse_best_param0 = rmse_best_param[0]
    rmse_best_param1 = rmse_best_param[1]

    ATP_p,M,OXY,W = organoid_sim_spec_params_linear(steps,dt,Volume,best_parameter)

    trajectories = eval_accepted_parameters(steps,dt,Volume,accepted_parameters)

    ATP_p_mean = np.mean(trajectories['ATP_p'],axis=0)
    M_mean = np.mean(trajectories['M'],axis=0)
    OXY_mean = np.mean(trajectories['OXY'],axis=0)
    W_mean = np.mean(trajectories['W'],axis=0)



    #FOR HDI Bands
    ATP_p_low,ATP_p_high = np.percentile(trajectories['ATP_p'], [5, 95], axis=0)
    M_low,M_high = np.percentile(trajectories['M'], [5, 95], axis=0)
    OXY_low,OXY_high = np.percentile(trajectories['OXY'], [5, 95], axis=0)
    W_low,W_high = np.percentile(trajectories['W'], [5, 95], axis=0)
    print('trajectories',trajectories['M'])

    time_exp,normed_datapoints= helper.get_real_data(name)

    time_exp1,ip_data = helper.get_inorganic_phosphate(name)

    if name == 'leupold_feb_integral' or name == 'leupold_feb':
        normed_datapoints = normed_datapoints[0:39]

    print('sim_rmse',helper.sim_rmse(ATP_p,normed_datapoints,time_exp))


    my_colors = {
    'blueblue':    '#003366',  # Deep Navy (Primary, highly professional)
    'redred':      "#F8720C",  # Burnt Orange / Ochre (Excellent contrast against navy)
    'greengreen':  "#1B9E5A",  # Deep Teal (Distinct, colorblind-friendly accent)
    'orange':      '#4B0082'   # Deep Indigo / Violet (Dark, distinct from navy and black)
    }

    time_axis = np.linspace(0, total_time, steps)
    # Create a figure with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Organoid Biological Simulation", fontsize=16)

    # 1. ATP Plot (The MRS Peak Scale)
    axs[0, 0].plot(time_axis, ATP_p, color=my_colors["greengreen"], label=f"ATP (Best fit-RMSE: {rmse_best_param0:.2f})")
    #axs[0, 0].plot(time_axis, ATP_p_mean, color=my_colors["greengreen"], label="ATP mean")
    axs[0, 0].fill_between(time_axis, ATP_p_low,ATP_p_high,alpha = 0.4,label = 'ATP HDI Band',color = my_colors["greengreen"])
    axs[0, 0].scatter(time_exp,normed_datapoints,label = 'Recorded ATP Data',color = my_colors['redred'])
    # If you want to overlay the real data:
    #axs[0, 0].scatter(timeaxis_exp, datapoints, color="darkgreen", s=10, label="Real MRS Data")
    axs[0, 0].set_ylabel("Peak Height (Unitless)")
    axs[0, 0].set_title("ATP Levels")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Oxygen Plot (0-1 or Concentration Scale)
    axs[0, 1].plot(time_axis, OXY, color=my_colors["blueblue"], linestyle='--', label="Oxygen (Best fit)")
    if constraint == True:
        axs[0, 1].scatter(300,0.1)

    axs[0, 1].fill_between(time_axis, OXY_low, OXY_high, alpha = 0.4,label = "Oxygen HDI Band")
    axs[0, 1].set_ylabel("Relative Level")
    axs[0, 1].set_title("Oxygen Saturation (mM)")
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    axs[0, 1].set_ylim(0,0.25)


    # 3. Medium/Nutrients Plot
    axs[1, 0].plot(time_axis, M, color=my_colors["orange"], linestyle='--', linewidth=2, label="Medium (Best fit)")
    axs[1, 0].fill_between(time_axis, M_low, M_high, alpha = 0.4,label = "Medium mean")
    axs[1, 0].set_ylabel("Relative Level (mM)")
    axs[1, 0].set_xlabel("Time (min)")
    axs[1, 0].set_title("Nutrient Availability")
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    axs[1, 0].set_ylim(0,25)

    # 4. Waste Plot
    axs[1, 1].plot(time_axis, W, color=my_colors["redred"], linestyle='--', label=f"Waste (Best fit-RMSE:{rmse_best_param1:.2f})")
    axs[1, 1].scatter(time_exp,ip_data,label = 'Recorded Pi Data')
    axs[1, 1].fill_between(time_axis,W_low, W_high, alpha = 0.4,label = "Waste HDI Band")
    axs[1, 1].set_ylabel("Accumulation")
    axs[1, 1].set_xlabel("Time (min)")
    axs[1, 1].set_title("Metabolic Waste")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for title
    plt.show()
