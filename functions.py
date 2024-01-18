# # # # # # #
# Functions #
# # # # # # #

# # #
# Import
import numpy as np
import pandas as pd
#

# # #
# Density of an aqueous potassium hydroxide solution
# Haug et al. (DOI: 10.1016/j.ijhydene.2017.05.031)
def dens_KOH(m,T,w_KOH):
    if(m==0):
        m = np
    return((1001.53053-0.08343*(T-273.15)-0.00401*(T-273.15)**2+5.51232*1e-6*(T-273.15)**3-8.20994*1e-10*(T-273.15)**4) * m.exp(0.86*w_KOH)) # kg/m^3
#

# # #
# Density of liquid water
# Kell (DOI: 10.1021/je60064a005)
def dens_H2O(T):
    return( ((999.83952 + 16.945176*(T-273.15) - 7.9870401*(1e-3)*(T-273.15)**2 - 46.170461*(1e-6)*(T-273.15)**3 + 105.56302*(1e-9)*(T-273.15)**4 - 280.54253*(1e-12)*(T-273.15)**5)/(1+16.87985*(1e-3)*(T-273.15)))) # kg/m^3
#

# # #
# Molality
# Haug et al. (DOI: 10.1016/j.ijhydene.2017.05.031)
def molal_KOH(w_KOH):
    M_KOH = 0.0561056 # kg/mol
    return (w_KOH / (M_KOH * (1 - w_KOH)))
#

# # #
# Molarity
# Gilliam et al. (DOI: 10.1016/j.ijhydene.2006.10.062)
def molar_KOH(m,T,w_KOH):
    M_KOH = 0.0561056 # kg/mol
    return (( (w_KOH * 100) * dens_KOH(m=m,T=T,w_KOH=w_KOH) ) / ( 100 * (1000 * M_KOH) ))
#

# # #
# Water vapor pressure above an aqueous potassium hydroxide solution
# Haug et al. (DOI: 10.1016/j.ijhydene.2017.05.031)
def p_sat(m,T,w_KOH):
    if(m==0):
        m = np
    return ((10**(-0.01508*molal_KOH(w_KOH=w_KOH) - 0.0016788*molal_KOH(w_KOH=w_KOH)**2 +2.25887*1e-5* molal_KOH(w_KOH=w_KOH)**3 + ( 1 - 0.0012062*molal_KOH(w_KOH=w_KOH) +5.6024*1e-4*molal_KOH(w_KOH=w_KOH)**2 - 7.8228*1e-6*molal_KOH(w_KOH=w_KOH)**3 )*(35.4462 - 3343.93/T -10.9*m.log10(T) +0.0041645*T) ))*1e5)
#

# # #
# Ionic conductivity of an aqueous potassium hydroxide solution
# Gilliam et al. (DOI: 10.1016/j.ijhydene.2006.10.062)
def cond_KOH(m,T,w_KOH):
    if(m==0):
        m = np
    return ((-2.041 * molar_KOH(m=m,T=T,w_KOH=w_KOH) - 0.0028 * molar_KOH(m=m,T=T,w_KOH=w_KOH)**2 + 0.005332 * molar_KOH(m=m,T=T,w_KOH=w_KOH) * T + 207.2 * molar_KOH(m=m,T=T,w_KOH=w_KOH) / T + 0.001043 * molar_KOH(m=m,T=T,w_KOH=w_KOH)**3 - 0.0000003 * molar_KOH(m=m,T=T,w_KOH=w_KOH)**2 * T**2) * 100) # S/m
#

# # #
# Dynamic viscosity of an aqueous potassium hydroxide solution
# Self-developed correlation with data from https://koh.olinchloralkali.com/TechnicalInformation/KOH%20Viscosity.pdf (accessed 2017-02-09), inspired by the Equation from Laliberte (DOI: 10.1021/je0604075 and 10.1021/je700232s)
def eta_KOH(m,T,w_KOH):
    if(m==0):
        m = np
    return ((m.exp( (9.206736611 * w_KOH**(2.64439258) + 3.515143754) / ( 9.956060084e-3 * (T-273.15) + 1 ) ) / ( 4.543513881 * w_KOH**(-5.552437316e-1) + 1 )) * 1e-3)
#

# # #
# Surface tension of an aqueous ptoassium hydroxide solution
# Feldkamp (DOI: 10.1002/cite.330412107)
def gamma_KOH(T, w_KOH):
    _alpha_ik = [[75.4787, -0.138489, -0.336392e-3, 0.475362e-6, -0.264479e-9],
                 [-32.8890, 1.34382, -0.910138e-2, 0.396124e-4, -0.573565e-7],
                 [614.527, -12.8736, 0.104855, -0.449076e-3, 0.651193e-6],
                 [-1455.06, 39.8511, -0.344234, 0.144383e-2, -0.207599e-5],
                 [1333.62, -38.3316, 0.335129, -0.137313e-2, 0.194911e-5]]
    return((sum((_alpha_ik[i][k]*(T-273.15)**k)*w_KOH**i for i in range(5) for k in range(5) ))*1e-3)
#

# # #
# Increase of the gas bubble pressure (Young-Laplace Equation)
def Delta_p_b(gamma_fl, d_b):
    return((4 * gamma_fl) / (d_b))
#

# # # 
# Binary diffusion coefficients of the product gases in an aqueous potassium hydroxide solution
# Self-developed correlation with data from Tham et al. (DOI: 10.1021/j100703a015)
def D(m, i, T, w_KOH):
    if(m==0):
        m = np
    if(i=="H2"):
        _k = [2.23248e5, 6.45173e3, 7.4774e-1, 2.11675, 7.21703e-1, 3.67543e1, 9.45732e-1]
    if(i=="O2"):
        _k = [4.19821e10, 1.08583e3, 2.92527e-1, 3.50104, 0.806288, 0.0134882, 0]
    return(( _k[0] * m.exp( - (_k[1])/(8.314 * T**_k[2])  - _k[3] * w_KOH**_k[4] + (w_KOH**_k[5])/(T**_k[6]) ) )*1e-9)
#

# # #
# Heat capacity of steel
# Mills et al. (DOI: 10.2355/isijinternational.44.1661)
def cp_steel(m, T):
    if(m==0):
        m = np
    return (472+13.6e-2 * T - 2.82e6 / T**2)
#

# # #
# Heat capacity of an aqueous potassium hydroxide solution
# LeBideau et al. (DOI: 10.1016/j.ijhydene.2018.12.222 and 10.1016/j.ijhydene.2021.04.137)
def cp_El(m, T, w_KOH):
    if(m==0):
        m = np
    _k = [4.236e3, 1.075, -4.831e3, 1.576e3, 8]
    return ( _k[0] + _k[1] * m.log((T-273.15)/100) + (_k[2] + _k[3] * w_KOH + _k[4] * (T-273.15)) * w_KOH )
#

# # #
# Density of an ideal gas
def dens_gas(m, M, p, T):
    if(m==0):
        m = np
    R = 8.314 # J/(mol K)
    return ( ( p * M )/( R * T ) )
#

# # #
# Heat capacity of the gaseous components
# VDI-Waermeatlas (DOI: 10.1007/978-3-662-52991-1)
def cp_gas(i, T):
    if(i=="H2"):
        _k = [392.8422,2.4906,-3.6262,-1.9624,35.6197,-81.3691,62.6668]
        _R = 8.314 / 0.00201588
    if(i=="O2"):
        _k = [2122.2098,3.5302,-7.1076,-1.4542,30.6057,-83.6696,79.4375]
        _R = 8.314 / 0.0319988
    if(i=="H2O"):
        _k = [706.3032,5.1703,-6.0865,-6.6011,36.2723,-63.0965,46.2085]
        _R = 8.314 / 0.01801528
    return ( _R * ( _k[1] + (_k[2]-_k[1]) * (T/(_k[0]+T))**2 * ( 1 - (_k[0]/(_k[0]+T)) * ( _k[3] + _k[4] * (T/(_k[0]+T)) + _k[5] * (T/(_k[0]+T))**2 + _k[6] * (T/(_k[0]+T))**3 ) ) ) ) # J / (kg * K)
#

# # #
# Heat capacity of liquid water
# VDI-Waermeatlas (DOI: 10.1007/978-3-662-52991-1)
def cp_H2O_liq(T):
    _k = [0.2399,12.8647,-33.6392,104.7686,-155.4709,92.3726]
    _Tc = 647.10 # K
    _R = 8.314 / 0.01801528
    return ( _R * ( (_k[0]/(1-(T/_Tc))) + _k[1] + _k[2] * (1-(T/_Tc)) + _k[3] * (1-(T/_Tc))**2 + _k[4] * (1-(T/_Tc))**3 + _k[5] * (1-(T/_Tc))**4 ) ) # J / (kg * K)
#

# # #
# Vaporization enthalpy of water
# VDI-Waermeatlas (DOI: 10.1007/978-3-662-52991-1)
def Delta_h_v_H2O(m, T):
    if(m==0):
        m = np
    _k = [6.85307,7.43804,-2.937595,-3.282093,8.397378]
    _Tc = 647.10 # K
    _R = 8.314 / 0.01801528
    return ( _R * _Tc * ( _k[0] * (1-(T/_Tc))**(1/3) + _k[1] * (1-(T/_Tc))**(2/3) + _k[2] * (1-(T/_Tc)) + _k[3] * (1-(T/_Tc))**(2) + _k[4] * (1-(T/_Tc))**(6) ) ) # J / kg
#

# # #
# Henry coefficient
# Self-developed correlation with data from Himmelblau et al. (DOI: 10.1021/je60005a003)
def He_chb(i, T):
    if(i=="H2"):
        _k = [-2.22162e2, 2.11949, -7.04452e-3, 9.93116e-06, -5.01539e-09]
    if(i=="O2"):
        _k = [3.64396e2, -4.64227, 2.14978e-2, -4.26680e-05, 3.08677e-08]
    return(_k[0]+_k[1]*T+_k[2]*T**2+_k[3]*T**3+_k[4]*T**4)
#

# # #
# Setchenov constant
# Self-developed correlation with data from Shoor et al. (DOI: 10.1021/j100722a006)
def C_Sech(i, T):
    if(i=="H2"):
        _k = [1.29e-4, 0, 0, 0, 0]
    if(i=="O2"):
        _k = [1.80e-4, 9.86994e-06, -7.99744e-08, 2.13832e-10, -1.89936e-13]
    return(_k[0] + _k[1] * (T) + _k[2] * T**2 + _k[3]* T**3 + _k[4] * T**4)
#

# # #
# Gas bubble diameter
# Self-developed correlation, inspired by the modified Fritz Equation from Vogt et al. (DOI: 10.1016/j.electacta.2004.09.025)
def d_b(m, i, j, gamma_KOH, dens_KOH, dens_gas, p_abs):
    if(m==0):
        m = np
    if(i=="H2"):
        _k = [3.4,0.2,-0.25,0,0]
    if(i=="O2"):
        # _k = [1.14,0.2,-0.25,0,0]
        _k = [1.2,0.2,-0.25,0,0]
    if(type(i)==list):
        _k = i
    _C_Pi = 3.14159265359
    _C_g = 9.81
    _C_p_atm = 1.01325e5
    return (1.2 * (_k[0]*_C_Pi/180) * m.sqrt((gamma_KOH)/(_C_g * (dens_KOH - dens_gas))) * (1 + _k[1] * (j))**(_k[2]) * (1 + _k[3] * (p_abs/_C_p_atm))**(_k[4]))
#

# # #
# Supersaturation factor
# Self-developed correlation
def fS(i, j, p_abs):
    if(i=="H2"):
        _k = [0.21, -0.54]
    if(i=="O2"):
        _k = [0.11, -0.40]
    if(type(i)==list):
        _k = i
    _C_p_atm = 1.01325e5
    return (1 + j**(_k[0]) * (1 + (p_abs/_C_p_atm))**(_k[1]))
#

# # #
# Exchange current density
# Self-developed correlation, inspired by the equation from Adibi et al. (DOI: 10.1016/j.ijft.2021.100126)
def j_0(m, i, T):
    if(i=="H2"):
        _k = [43.746, 29.516]
    if(i=="O2"):
        _k = [69.766, 5.356e-05]
    if(type(i)==list):
        _k = i
    _C_R = 8.314
    return (m.exp(-(_k[0]*1e3/_C_R) * ((1/T) - (1/(273.15+25))) ) * _k[1])
#

# # #
# Symmetry factor
# Self-developed correlation
def alpha(i, T):
    if(i=="H2"):
        _k = [0.2769, 0, 0]
    if(i=="O2"):
        _k = [0.3959, 0, 0]
    if(type(i)==list):
        _k = i
    return (_k[0] + _k[1] * T + _k[2] * T**2)
#

# # #
# Heat transfer coefficient (natural convection) of cylinders
# Holman (ISBN: 978-0-07-352936-3)
def h(i, Delta_T, d, insulation_factor):
    if(i=="h"): # horizontal
        _k = 1.32
    if(i=="v"): # vertical
        _k = 1.42
    if(i=="hv_mean"): # mean of horizontal and vertical
        _k = (1.32+1.42)/2
    if(type(i)==list):
        _k = i
    return (_k * (Delta_T/d)**(0.25) * (1-insulation_factor))
#

# # #
# Diffusion coefficient of the ions in an aqueous potassium hydroxide solution
# LeBideau et al. (DOI: 10.1016/j.ijhydene.2018.12.222)
def D_KOH(w_KOH, T):
    _k = [-1.05e-1, 2.45, 9.20e-2, 1.148e-2]
    theta = T-273.15
    return ((_k[0] + _k[1] * w_KOH + _k[2] * theta + _k[3] * theta * w_KOH)*1e-9)
#

# # #
# GC data (H2 in O2) interface
def gc_data(name, m_time, start_time_hour=0, time_steps_per_hour=1, time_offset=0, gc_calib_factor=0.0037):
    if(name != ""):
        gc_data = pd.read_parquet('data/gc/' + name + '.parquet', engine='fastparquet')
        gc_data["time"] = start_time_hour * 3600 + time_offset + (gc_data.index-gc_data.index[0])/pd.Timedelta('1 second')
        gc_data["H2_in_O2"] = gc_data["Area"]*gc_calib_factor
        gc_data.set_index('time', inplace=True)
        gc_data = gc_data.drop(columns=['Area'])
        m_process_time = pd.DataFrame({'time':m_time})
        m_process_time.set_index('time', inplace=True)
        est_data = pd.merge_asof(m_process_time, gc_data, on="time", direction="forward", tolerance=3600/time_steps_per_hour)
        est_data["H2_in_O2_zero"] = est_data['H2_in_O2'].apply(lambda x: x if pd.notnull(x) else 0)
        est_data["valid"] = est_data['H2_in_O2'].apply(lambda x: 1 if pd.notnull(x) else 0)
    else:
        est_data = pd.DataFrame(index=m_time, columns=["H2_in_O2", "H2_in_O2_zero", "valid"])
        est_data["valid"] = 0
    return est_data
#

# # #
# Datalog interface
def datalog_data(name, m_time, start_time_hour=0, time_steps_per_hour=1, time_offset=0):
    if(name != ""):
        datalog_data = pd.read_parquet('data/ael_system/' + name + '.parquet', engine='fastparquet')
        datalog_data = datalog_data.resample(str(int(3600/time_steps_per_hour)) + 's').mean()
        datalog_data["time"] = datalog_data["time"] + start_time_hour * 3600 + time_offset
        datalog_data.set_index('time', inplace=True)
        datalog_data.index = datalog_data.index.astype(float)
        m_process_time = pd.DataFrame({'time':m_time})
        m_process_time.set_index('time', inplace=True)
        return_data = pd.merge_asof(m_process_time, datalog_data, on="time", direction="forward", tolerance=3600/time_steps_per_hour)
    else:
        return_data = 0
    return return_data
#

# # #
# Electrolyte concentration estimation data interface
def w_KOH_data(m_time, start_time_hour=0, time_steps_per_hour=1, w_KOH_time=0, w_KOH_A=0.32, w_KOH_C=0.32):
    w_KOH_data = pd.DataFrame([{"time":float(w_KOH_time), "w_KOH_A":w_KOH_A, "w_KOH_C":w_KOH_C}])
    if(len(w_KOH_data) > 0):
        w_KOH_data["time"] = w_KOH_data["time"] + start_time_hour * 3600
        w_KOH_data.set_index('time', inplace=True)
        m_process_time = pd.DataFrame({'time':m_time})
        m_process_time.set_index('time', inplace=True)
        est_data = pd.merge_asof(m_process_time, w_KOH_data, on="time", direction="forward", tolerance=3600/time_steps_per_hour)
        est_data["w_KOH_A_zero"] = est_data['w_KOH_A'].apply(lambda x: x if pd.notnull(x) else 0)
        est_data["w_KOH_C_zero"] = est_data['w_KOH_C'].apply(lambda x: x if pd.notnull(x) else 0)
        est_data["valid"] = est_data['w_KOH_A'].apply(lambda x: 1 if pd.notnull(x) else 0)
    else:
        est_data = pd.DataFrame(index=m_time, columns=["w_KOH_A", "w_KOH_C", "w_KOH_C_zero", "w_KOH_A_zero", "valid"])
        est_data["valid"] = 0
    return est_data
#

# # #
# Temperature estimation data interface
def T_sys_data(name, m_time, start_time_hour=0, time_steps_per_hour=1, time_offset=0):
    if(name != ""):
        est_data = pd.read_parquet('data/ael_system/' + name + '.parquet', engine='fastparquet')[['time', 'temp_anode_electrolyte_inlet', 'temp_cathode_electrolyte_inlet', 'temp_anode_electrolyte_outlet', 'temp_cathode_electrolyte_outlet', 'temp_anode_endplate']]
        est_data = est_data.resample(str(int(3600/time_steps_per_hour)) + 's').mean()
        est_data["time"] = est_data["time"] + start_time_hour * 3600 + time_offset
        est_data.set_index('time', inplace=True)
        est_data.index = est_data.index.astype(float)
        est_data["T_sys"] = est_data[['temp_anode_electrolyte_inlet', 'temp_cathode_electrolyte_inlet']].mean(axis=1)
        est_data["T_amb"] = est_data['temp_anode_endplate']
        m_process_time = pd.DataFrame({'time':m_time})
        m_process_time.set_index('time', inplace=True)
        return_data = pd.merge_asof(m_process_time, est_data, on="time", direction="forward", tolerance=3600/time_steps_per_hour)
        return_data["T_sys_zero"] = return_data['T_sys'].apply(lambda x: x if pd.notnull(x) else 0)
        return_data["valid"] = return_data['T_sys'].apply(lambda x: 1 if pd.notnull(x) else 0)
    else:
        return_data = 0
    return return_data
#

# # #
# Datalog gas impurity (O2 in H2) interface
def O2_in_H2_data(name, m_time, start_time_hour=0, time_steps_per_hour=1, time_offset=0, o2_in_h2_calib_factor=0.5):
    if(name != ""):
        est_data = pd.read_parquet('data/ael_system/' + name + '.parquet', engine='fastparquet')[['time', 'signal_in_o2_in_h2_sensor']]
        est_data = est_data.resample(str(int(3600/time_steps_per_hour)) + 's').mean()
        est_data["time"] = est_data["time"] + start_time_hour * 3600 + time_offset
        est_data.set_index('time', inplace=True)
        est_data.index = est_data.index.astype(float)
        est_data["O2_in_H2"] = est_data['signal_in_o2_in_h2_sensor']*o2_in_h2_calib_factor
        m_process_time = pd.DataFrame({'time':m_time})
        m_process_time.set_index('time', inplace=True)
        return_data = pd.merge_asof(m_process_time, est_data, on="time", direction="forward", tolerance=3600/time_steps_per_hour)
        return_data["O2_in_H2_zero"] = return_data['O2_in_H2'].apply(lambda x: x if pd.notnull(x) else 0)
        return_data["valid"] = return_data['O2_in_H2'].apply(lambda x: 1 if pd.notnull(x) else 0)
    else:
        return_data = 0
    return return_data
#

# # #
# Initial value selector
def initial_value(value):
    try:
        initial = value[0]
    except:
        initial = value
    return initial
