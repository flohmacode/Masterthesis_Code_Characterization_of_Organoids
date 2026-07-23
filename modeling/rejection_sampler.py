
import src.helper as helper
import numpy as np
import models.organoid_sim_linear as mosl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import src.connection_to_bio as ctb

def run_abc(name,steps,dt,Volume_L,model,target_samples = 10):

    time_exp,normed_datapoints = helper.get_real_data(name)
    time_sim = np.arange(steps)
    atp_real_interp = np.interp(time_sim,time_exp,normed_datapoints)
    print(np.var(atp_real_interp))


    time_exp,datapoints_ip = helper.get_inorganic_phosphate(name)
    ip_real_interp = np.interp(time_sim,time_exp,datapoints_ip)


    startpoint = 60
    endpoint = 320

    cnt= 0
    old_distance = 999
    best_parameter = {}

    num_target_samples = target_samples
    
    accepted_samples = []
    if model =='waste':
        print('wastemodel')
    if model =='linear':
        print('linear')
    if model == 'atp_cost':
        print('atp_cost')

    ATP_p = np.empty(steps)
    M = np.empty(steps)
    OXY = np.empty(steps)
    W = np.empty(steps)

    modelmap = {'linear':(mosl.run_bayes_linear,mosl.organoid_sim_spec_params_linear)}

    bayes_func,spec_func = modelmap[model]

    best_parameter = {}
    rmse_best_param = 999

    var_atp = np.var(atp_real_interp[startpoint:endpoint])
    var_ip = np.var(ip_real_interp[startpoint:endpoint])

    while len(accepted_samples)<num_target_samples:
        cnt += 1
        ATP_p,M,OXY,W,parameter = bayes_func(steps,dt,Volume_L)
        distance_atp = helper.mse(ATP_p[startpoint:endpoint],atp_real_interp[startpoint:endpoint])
        distance_ip = helper.mse(W[startpoint:endpoint],ip_real_interp[startpoint:endpoint])

        com_distance = distance_atp/var_atp + distance_ip/var_ip

        if distance_atp<0.15 and distance_ip <0.15:
            print('here')

            accepted_samples.append(parameter)
            if com_distance<old_distance:
                old_distance = com_distance
                best_parameter = parameter
            
        else:

            continue

        #cnt+=1
        atp,m,oxy,w = spec_func(steps,dt,Volume_L,best_parameter)
        rmse_atp = helper.sim_rmse(atp,normed_datapoints,time_exp)
        rmse_ip = helper.sim_rmse(w,datapoints_ip,time_exp)
        rmse_best_param = [rmse_atp,rmse_ip]
        #print('rmse',rmse_best_param)
        #np.save(f'./parameters/{name}_{model}',best_parameter)

    return accepted_samples,best_parameter,rmse_best_param,cnt

dt = 1 #min
total_time =700 #min
steps = int(total_time/dt)
Volume = 7.0  * 1e-6

name  = 'jojo_april'
model = 'linear'

accepted_parameters,best_parameter,rmse_best_param,count = run_abc(name,steps,dt,Volume,model,target_samples=1000)
mosl.run_and_plot_linear(name,steps,dt,total_time,Volume,best_parameter,accepted_parameters,rmse_best_param)

p = ctb.bayes_prior()

# Convert list of dicts to a DataFrame
df = pd.DataFrame(accepted_parameters)

np.save(f'./parameters/{name}_{model}_best_parameter_numbers',best_parameter)
np.save(f'./parameters/{name}_{model}_accepted_parameter_numbers',accepted_parameters)

inter = ['myu_oxygen', 'yield_aerobic', 'k_m_i','k_m_a','myu_fixed_costs','myu_waste_yield']
for key in inter:
    # 1. Academic Configuration
    # 'paper' context sets font sizes appropriate for documents
    # 'whitegrid' makes the chart easier to read
    #sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

    # sns.pairplot(df[['myu_glucose', 'myu_oxygen', 'yield_aerobic', 'k_m_i']])
    # plt.show()

    # 2. Setup Figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # 1. Plot the Prior (Re-sample from your biocon object to get the 'before' picture)
    priors = ctb.bayes_prior()
    prior_samples = priors[key].rvs(size=1000)
    sns.kdeplot(prior_samples, label="Prior (Before Data)",fill=True,alpha= 0.3, color="gray")

    # 2. Plot the Posterior (From your accepted_parameters dataframe)
    sns.kdeplot(df[key], label="Posterior (After Data)", fill=True,alpha = 0.3, color="C0", linewidth=2)

    ax.set_title(f"Bayesian Update: {key}", fontsize=16, pad=15)
    ax.set_xlabel("Parameter Value", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.legend(frameon=True, fontsize=12)
    plt.savefig(f'./figs/abc/{name}/{key}_{model[1]}_numbers')
    plt.show()
