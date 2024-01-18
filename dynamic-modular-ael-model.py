# # # # # # # # # # # # # # #
# Dynamic Modular AEL Model #
# Author: Joern Brauns      #
# # # # # # # # # # # # # # #
#
# # #
# Import
#
import numpy as np
import matplotlib.pyplot as plt
import toml
import includes.functions as Functions
import includes.constants as Constants
import pandas as pd
import concurrent.futures
import time
import logging
import sys
#
# # #
# Global
#
class EmptyClass:
    pass
#        
global_config = toml.load(r'options/config.toml')
#
if (global_config["options"]["model_export"]):
    import os
    from zipfile import ZipFile
#
if(global_config["options"]["gekko_patched"]):
    from gekko_patched import GEKKO
else:
    from gekko import GEKKO
#
# # #
# Log
#
run_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
#
logger = logging.getLogger('logger')
if (logger.hasHandlers()): logger.handlers.clear()
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
#
logger_ch = logging.StreamHandler(sys.stdout)
logger_ch.setLevel(logging.INFO)
logger_ch.setFormatter(logger_formatter)
logger.addHandler(logger_ch)
#
if(global_config["options"]["log_export"]):
    logger_fh_filename = "log/" + str(run_time) + "_log.txt"
    logger_fh = logging.FileHandler(logger_fh_filename, mode='w')
    logger_fh.setLevel(logging.INFO)
    logger_fh.setFormatter(logger_formatter)
    logger.addHandler(logger_fh)
    logger.info("Logging to " + str(logger_fh_filename) + " started")
#
# # #
# Export archive of current state to log directory
#
if (global_config["options"]["model_export"]):
    zip_model = ZipFile('log/' + str(run_time) + '_model.zip', 'w')
    zip_model.write('dynamic-modular-ael-model.py')
    zip_model.write('includes/functions.py')
    zip_model.write('includes/constants.py')
    zip_model.write('options/config.toml')
#
# # #
# Config Parse
#
num_process = len(global_config["process"])
num_system = [None for pp in range(num_process)]
#
for pp in global_config["process"]:
    num_system[int(pp)] = len(global_config["process"][pp]["system"])
#
Process = [None for pp in range(num_process)]
process_config = [None for pp in range(num_process)]
#
for pp in range(num_process):
    process_config[pp] = EmptyClass()
    process_config[pp].system = global_config["process"][str(pp)]["system"]
    for index, name in enumerate(global_config["default"]["process"]):
        if(name not in global_config["options"]["no_eval_process_parameters"]):
            setattr(process_config[pp], name, eval(global_config["process"][str(pp)].get(str(name), global_config["default"]["process"][str(name)])))
        else:
            setattr(process_config[pp], name, global_config["process"][str(pp)].get(str(name), global_config["default"]["process"][str(name)]))
    process_config[pp].m_time_pre = np.linspace(0, (process_config[pp].time_pre_end_hour * 3600), ((process_config[pp].time_pre_end_hour * process_config[pp].time_pre_steps_per_hour) + 1))
    process_config[pp].m_time_post =  np.linspace((process_config[pp].time_pre_end_hour * 3600 + 3600/process_config[pp].time_steps_per_hour), ((process_config[pp].time_pre_end_hour+process_config[pp].time_end_hour) * 3600), ((process_config[pp].time_end_hour * process_config[pp].time_steps_per_hour))) # s
    process_config[pp].m_time = np.concatenate((process_config[pp].m_time_pre,process_config[pp].m_time_post)) # s
    #
    process_config[pp].num_system = len(global_config["process"][str(pp)]["system"])
    #
    try:
        process_config[pp].systems_selectors = [selector for selector in global_config["process"][str(pp)]["systems"]]
        process_config[pp].systems_selectors_range = []
        for ss in process_config[pp].systems_selectors:
            process_config[pp].systems_selectors_range.append(list(range(int(ss.split("-")[0]),int(ss.split("-")[1])+1)))
    except:
        process_config[pp].systems_selectors = []
        process_config[pp].systems_selectors_range = []
    #
    process_config[pp].system_config = [None for ss in range(process_config[pp].num_system)]
    for ss in range(process_config[pp].num_system):
        process_config[pp].system_config[ss] = EmptyClass()
        process_config[pp].system_config[ss].solve_mode = process_config[pp].solve_mode
        #
        process_config[pp].system_config[ss].systems_selector = ""
        for index, value in enumerate(process_config[pp].systems_selectors_range):
            if (ss in value):
                process_config[pp].system_config[ss].systems_selector = process_config[pp].systems_selectors[index]
        #
        for index, name in enumerate(global_config["default"]["system"]):
            if(process_config[pp].system_config[ss].systems_selector != "" and name in global_config["process"][str(pp)]["systems"][process_config[pp].system_config[ss].systems_selector]):
                logger.info("Process." + str(pp) + ".System." + str(ss) + "." + str(name) + ": Using value from summarized systems definition: " + str(process_config[pp].system_config[ss].systems_selector))
                if(name not in global_config["options"]["no_eval_system_parameters"]):
                    setattr(process_config[pp].system_config[ss], name, eval(global_config["process"][str(pp)]["systems"][process_config[pp].system_config[ss].systems_selector].get(str(name), global_config["default"]["system"][str(name)])))
                else:
                    setattr(process_config[pp].system_config[ss], name, global_config["process"][str(pp)]["systems"][process_config[pp].system_config[ss].systems_selector].get(str(name), global_config["default"]["system"][str(name)]))
            else:
                if(name not in global_config["options"]["no_eval_system_parameters"]):
                    setattr(process_config[pp].system_config[ss], name, eval(global_config["process"][str(pp)]["system"][str(ss)].get(str(name), global_config["default"]["system"][str(name)])))
                else:
                    setattr(process_config[pp].system_config[ss], name, global_config["process"][str(pp)]["system"][str(ss)].get(str(name), global_config["default"]["system"][str(name)]))
        if(process_config[pp].solve_mode == 1 and (process_config[pp].pe_H2_in_O2 == 1 or process_config[pp].plot_vd == 1)): 
            process_config[pp].system_config[ss].est_data_H2_in_O2 = Functions.gc_data(name=process_config[pp].system_config[ss].gc_validation_name, m_time=process_config[pp].m_time, start_time_hour=process_config[pp].time_pre_end_hour, time_steps_per_hour=process_config[pp].time_steps_per_hour, time_offset=process_config[pp].system_config[ss].gc_validation_time_offset, gc_calib_factor=process_config[pp].system_config[ss].gc_calib_factor)
        #
        if(process_config[pp].solve_mode == 1 and (process_config[pp].plot_datalog == 1)): 
            process_config[pp].system_config[ss].datalog_data = Functions.datalog_data(name=process_config[pp].system_config[ss].datalog_validation_name, m_time=process_config[pp].m_time, start_time_hour=process_config[pp].time_pre_end_hour, time_steps_per_hour=process_config[pp].time_steps_per_hour, time_offset=process_config[pp].system_config[ss].datalog_validation_time_offset)
        #
        if(process_config[pp].solve_mode == 1 and (process_config[pp].pe_w_KOH == 1)): 
            process_config[pp].system_config[ss].est_data_w_KOH = Functions.w_KOH_data(m_time=process_config[pp].m_time, start_time_hour=process_config[pp].time_pre_end_hour, time_steps_per_hour=process_config[pp].time_steps_per_hour, w_KOH_time=process_config[pp].system_config[ss].exp_w_KOH_time, w_KOH_A=process_config[pp].system_config[ss].exp_w_KOH_A, w_KOH_C=process_config[pp].system_config[ss].exp_w_KOH_C)
        #
        if(process_config[pp].solve_mode == 1 and (process_config[pp].pe_energy == 1 or process_config[pp].dl_T_amb == 1)): 
            process_config[pp].system_config[ss].est_data_T_sys = Functions.T_sys_data(name=process_config[pp].system_config[ss].datalog_validation_name, m_time=process_config[pp].m_time, start_time_hour=process_config[pp].time_pre_end_hour, time_steps_per_hour=process_config[pp].time_steps_per_hour, time_offset=process_config[pp].system_config[ss].datalog_energy_validation_time_offset)
            process_config[pp].system_config[ss].T_ini = process_config[pp].system_config[ss].est_data_T_sys['T_sys_zero'][0]+273.15
            logger.info("Process." + str(pp) + ".System." + str(ss) + ": " + "The initial system temperature was automatically adjusted according the experimental validation data.")
        #
        if(process_config[pp].solve_mode == 1 and (process_config[pp].pe_O2_in_H2 == 1)): 
            process_config[pp].system_config[ss].est_data_O2_in_H2 = Functions.O2_in_H2_data(name=process_config[pp].system_config[ss].datalog_validation_name, m_time=process_config[pp].m_time, start_time_hour=process_config[pp].time_pre_end_hour, time_steps_per_hour=process_config[pp].time_steps_per_hour, time_offset=process_config[pp].system_config[ss].datalog_o2_in_h2_validation_time_offset, o2_in_h2_calib_factor=process_config[pp].system_config[ss].datalog_o2_in_h2_calib_factor)
#
# # #
# Class Definitions
#
components = ["O2", "H2"]      


class HalfcellClass:
    def __init__(self, name, system_config, m):
        self.name = name
        self.name_pre = self.name + "_"
        self.system_config = system_config
        self.Initialize(m=m)
        
    def Initialize(self, m):
        # # #
        # Parameters
        #
        self.A_elec = None # m^2
        self.A_sep = None # m^2
        self.d_sep = None # m
        self.eps_sep = None # -
        self.F = Constants.F # C/mol
        self.g = Constants.g # m/s^2
        self.j = None # A/m^2
        self.M_H2 = Constants.M_H2 # kg/mol
        self.M_H2O = Constants.M_H2O # kg/mol
        self.M_KOH = Constants.M_KOH # kg/mol
        self.M_O2 =  Constants.M_O2 # kg/mol
        self.nu = {ii: None for ii in components} # -
        self.p = None # Pa
        self.p_atm = Constants.p_atm # Pa
        self.p_in = {ii: None for ii in components} # Pa
        self.Pi = Constants.Pi # -
        self.R = Constants.R # J/(mol K)
        self.tau_sep = None # -
        self.VR = None # m^3
        self.VS_G_in = None # m^3/s
        self.z = None # -
        #
        self.nu["O2"] = None # -
        self.nu["H2"] = None # -
        #
        # # #
        # Intermediates
        #
        self.A_gl = None # m^2
        self.A_sb = None # m^2
        self.c_eq = {ii: None for ii in components} # mol/m^3
        self.c_eq0 = {ii: None for ii in components} # mol/m^3
        self.c_in = {ii: None for ii in components} # mol/m^3
        self.c_KOH_25C = None # mol/m^3
        self.D = {ii: None for ii in components} # m^2/s
        self.D_eff = {ii: None for ii in components} # m^2/s
        self.Delta_p = None # Pa
        self.dens_gas = None # kg/m^3
        self.dens_H2O = None # kg/m^3
        self.dens_KOH = None # kg/m^3
        self.dens_KOH_25C = None # kg/m^3
        self.eps = None # -
        self.eta_KOH = None # Pa s
        self.fG = {ii: None for ii in components} # -
        self.fS = {ii: None for ii in components} # -
        self.gamma_fl = None # N/m
        self.He_chb = {ii: None for ii in components} # (-)
        self.He_hb = {ii: None for ii in components} # (atm)
        self.C_sech = {ii: None for ii in components} # -
        self.kL = {ii: None for ii in components}
        self.M_gas = None # kg/mol
        self.M_gas_wet = None # kg/mol
        self.n_cross = {ii: None for ii in components} # mol/s
        self.n_phys = {ii: None for ii in components} # mol/s
        self.n_R = {ii: None for ii in components} # mol/s
        self.p_b = None # Pa
        self.p_sat = None # Pa 
        self.Re = None # -
        self.Sc = {ii: None for ii in components} # -
        self.Sh = {ii: None for ii in components} # -
        self.T_chb = None #
        self.u_sb = None # m/s
        self.u_sw = None # m/s
        self.V_gas = None # m^3
        self.V_L = None
        self.V_sb = None # m^3
        self.VS_L_in = None # m^3/s
        self.x = {ii: None for ii in components} # -
        self.x_percent = {ii: None for ii in components} # %
        self.x_wet = {ii: None for ii in components} # -
        # 
        # # #
        # Variables
        #
        self.c_out = {ii: m.Var(value=0, lb=0, name=self.name_pre+"c_out" + "_" + str(ii)) for ii in components} # mol/m^3
        self.d_b = m.Var(lb=1e-7, ub=1000e-6, name=self.name_pre+"d_b") # m
        self.ns_ve_sep = {ii:m.Var(value=0, name=self.name_pre+"ns_ve_sep" + "_" + str(ii)) for ii in components} # mol/s
        self.ns_drag = {ii:m.Var(value=0, name=self.name_pre+"ns_drag" + "_" + str(ii)) for ii in components} # mol/s
        self.p_abs = m.Var(lb=Constants.p_atm, name=self.name_pre+"p_abs") # Pa
        self.p_b = m.Var(lb=Constants.p_atm, name=self.name_pre+"p_b") # Pa
        self.p_out = {ii: m.Var(value=0, lb=0, name=self.name_pre+"p_out" + "_" + str(ii)) for ii in components} # Pa
        self.T = m.Var(value=self.system_config.T_ini, lb=273.15, ub=373.15, name=self.name_pre+"T") # K
        self.VS_G_out = m.Var(lb=0, name=self.name_pre+"VS_G_out") # m^3/s
        self.VS_L_out = m.Var(lb=0, name=self.name_pre+"VS_L_out") # m^3/s
        self.w_KOH = m.Var(lb=0, ub=0.55, name=self.name_pre+"w_KOH") # -

    # # #
    # Intermediate Equations
    #
    def Intermediates(self, m):
        #
        self.T_chb = m.Intermediate(self.T * 1e-3, name=self.name_pre+"T_chb")
        self.C_sech["O2"] = m.Intermediate( Functions.C_Sech(i="O2", T=self.T) , name=self.name_pre+"C_sech_O2") 
        self.C_sech["H2"] = m.Intermediate( Functions.C_Sech(i="H2", T=self.T) , name=self.name_pre+"C_sech_H2")
        self.He_chb["O2"] = m.Intermediate( Functions.He_chb(i="O2", T=self.T) , name=self.name_pre+"He_chb_O2")
        self.He_chb["H2"] = m.Intermediate( Functions.He_chb(i="H2", T=self.T) , name=self.name_pre+"He_chb_H2")
        self.p_sat = m.Intermediate( Functions.p_sat(m=m, T=self.T, w_KOH=self.w_KOH) , name=self.name_pre+"p_sat")
        for ii in components:
            self.n_R[ii] = m.Intermediate(self.nu[ii] * (self.j * self.A_elec)/(self.z*self.F), name=self.name_pre+"n_R_" + str(ii))
            self.x_wet[ii] = m.Intermediate(self.p_out[ii] / (self.p_b), name=self.name_pre+"x_wet_" + str(ii))
        self.eps = m.Intermediate((self.VS_G_out)/(self.VS_G_out + self.VS_L_out), name=self.name_pre+"eps")
        self.dens_H2O = m.Intermediate( Functions.dens_H2O(T=self.T) , name=self.name_pre+"dens_H2O")
        self.dens_KOH = m.Intermediate( Functions.dens_KOH(m=m, T=self.T, w_KOH=self.w_KOH) , name=self.name_pre+"dens_KOH")
        self.eta_KOH = m.Intermediate( Functions.eta_KOH(m=m, T=self.T, w_KOH=self.w_KOH), name=self.name_pre+"eta_KOH")
        self.D["O2"] = m.Intermediate( Functions.D(m=m, i="O2", T=self.T, w_KOH=self.w_KOH), name=self.name_pre+"D_O2")
        self.D["H2"] = m.Intermediate( Functions.D(m=m, i="H2", T=self.T, w_KOH=self.w_KOH), name=self.name_pre+"D_H2")
        self.dens_KOH_25C = m.Intermediate( Functions.dens_KOH(m=m, T=273.15+25, w_KOH=self.w_KOH) , name=self.name_pre+"dens_KOH_25C")
        self.c_KOH_25C = m.Intermediate(self.w_KOH * self.dens_KOH_25C / (self.M_KOH), name=self.name_pre+"c_KOH_25C")
        self.A_sb = m.Intermediate(self.Pi * self.d_b**2, name=self.name_pre+"A_sb")
        self.V_sb = m.Intermediate((self.Pi/6) * self.d_b**3, name=self.name_pre+"V_sb")
        self.gamma_fl = m.Intermediate( Functions.gamma_KOH(T=self.T, w_KOH=self.w_KOH), name=self.name_pre+"gamma_fl")
        self.Delta_p_b = m.Intermediate( Functions.Delta_p_b(gamma_fl=self.gamma_fl, d_b=self.d_b) , name=self.name_pre+"Delta_p_b")
        self.V_gas = m.Intermediate(self.eps * self.VR, name=self.name_pre+"V_gas")
        self.V_L = m.Intermediate((1-self.eps) * self.VR, name=self.name_pre+"V_L")
        self.A_gl = m.Intermediate((self.V_gas * self.A_sb) / (self.V_sb), name=self.name_pre+"A_gl")
        self.M_gas = m.Intermediate(self.x_wet["H2"] * self.M_H2 + self.x_wet["O2"] * self.M_O2, name=self.name_pre+"M_gas")
        self.M_gas_wet = m.Intermediate(self.x_wet["H2"] * self.M_H2 + self.x_wet["O2"] * self.M_O2 + (1-self.x_wet["H2"]-self.x_wet["O2"]) * self.M_H2O, name=self.name_pre+"M_gas_wet")
        self.dens_gas = m.Intermediate(Functions.dens_gas(m=0, M=self.M_gas_wet, p=self.p_b, T=self.T), name=self.name_pre+"dens_gas")
        self.u_sb = m.Intermediate((2 * (self.d_b/2)**2 * (self.dens_KOH - self.dens_gas) * self.g) / (9 * self.eta_KOH), name=self.name_pre+"u_sb")
        self.u_sw = m.Intermediate(self.u_sb * 1/(1+self.eps/(1-self.eps)**2) * (1-self.eps)/(1+1.05/((1+0.0685/(self.eps**2))**0.5 - 0.5)), name=self.name_pre+"u_sw")
        self.Re = m.Intermediate(self.dens_KOH * self.d_b * (self.u_sw) / (self.eta_KOH), name=self.name_pre+"Re")
        #
        for ii in components:
            self.D_eff[ii] = m.Intermediate(self.D[ii] * (self.eps_sep/self.tau_sep), name=self.name_pre+"D_eff_" + str(ii))
            self.He_hb[ii] = m.Intermediate(self.He_chb[ii] / 1e-4, name=self.name_pre+"He_hb_" + str(ii))
            self.c_eq0[ii] = m.Intermediate(self.dens_H2O * self.p_out[ii] / (self.M_H2O * 101325 * self.He_hb[ii]), name=self.name_pre+"c_eq0_" + str(ii))
            self.c_eq[ii] = m.Intermediate(self.c_eq0[ii] / (10**(self.C_sech[ii]*(self.c_KOH_25C))), name=self.name_pre+"c_eq_" + str(ii))
            self.Sc[ii] = m.Intermediate(self.eta_KOH / (self.dens_KOH * self.D[ii]), name=self.name_pre+"Sc_" + str(ii))
            self.Sh[ii] = m.Intermediate(2 + (0.651 * (self.Re * self.Sc[ii])**1.72) / (1 + (self.Re * self.Sc[ii])**1.22), name=self.name_pre+"Sh_" + str(ii))
            self.kL[ii] = m.Intermediate(self.Sh[ii] * self.D[ii] / self.d_b, name=self.name_pre+"kL_" + str(ii))
            self.n_phys[ii] = m.Intermediate(self.A_gl * self.kL[ii] * (self.c_eq[ii] - self.c_out[ii]), name=self.name_pre+"n_phys_" + str(ii))
            self.x[ii] = m.Intermediate(self.p_out[ii] / (self.p_b - self.p_sat), name=self.name_pre+"x_" + str(ii))
            self.x_percent[ii] = m.Intermediate(self.x[ii] * 100, name=self.name_pre+"x_percent_" + str(ii))

    # # #
    # Equations
    #
    def Equations(self, m):
        m.Equation(self.p_abs == self.p)
        m.Equation(self.p_b == self.p_abs + self.Delta_p_b)
        m.Equation(self.p_b == self.p_out["O2"] + self.p_out["H2"] + self.p_sat) 
        for ii in components:
            m.Equation(self.VR * (1-self.eps) * self.c_out[ii].dt() == self.VS_L_in * self.c_in[ii] - self.VS_L_out * self.c_out[ii] + self.n_cross[ii] + self.n_phys[ii] + (1 - self.fG[ii]) * self.n_R[ii] + self.ns_ve_sep[ii] + self.ns_drag[ii])
            #
            m.Equation((self.VR * self.eps)/(self.R * self.T) * self.p_out[ii].dt() == (self.VS_G_in)/(self.R * self.T) * self.p_in[ii] - (self.VS_G_out)/(self.R * self.T) * self.p_out[ii] - self.n_phys[ii] + self.fG[ii] * self.n_R[ii])
            
        
class ElectrolysisCellClass:
    def __init__(self, name, system_config, m):
        self.name = name
        self.name_pre = self.name + "_"
        self.system_config = system_config
        self.Initialize(m=m)
        
    def Initialize(self, m):
        self.Anode = HalfcellClass(name=self.name_pre+"Anode", system_config=self.system_config, m=m)
        self.Cathode = HalfcellClass(name=self.name_pre+"Cathode", system_config=self.system_config, m=m)
        # # #
        # Parameters
        #
        # System Config Parameter Overwrites
        for vv in ["A_elec", "A_sep", "p", "d_sep", "eps_sep", "tau_sep"]:
            setattr(self.Anode, str(vv), getattr(self.system_config, vv))
            setattr(self.Cathode, str(vv), getattr(self.system_config, vv))
        self.Anode.VR = self.system_config.HC_VR
        self.Cathode.VR = self.system_config.HC_VR
        #
        self.Anode.nu["O2"] = 1
        self.Anode.nu["H2"] = 0
        self.Anode.z = Constants.z_A
        self.Cathode.nu["O2"] = 0
        self.Cathode.nu["H2"] = 1
        self.Cathode.z = Constants.z_C
        #
        # Variables
        self.T = m.Var(value=self.system_config.T_ini, lb=273.15, ub=373.15, name=self.name_pre+"T")
        
    def Initials(self, m):
        # Initials
        self.Anode.p_abs.value = self.Anode.p
        self.Cathode.p_abs.value = self.Cathode.p
        self.Anode.d_b.value = Functions.d_b(m=0, i="O2", j=Functions.initial_value(self.system_config.j), gamma_KOH=Functions.gamma_KOH(self.Anode.T.value, self.Anode.w_KOH.value), dens_KOH=Functions.dens_KOH(m=0,T=self.Anode.T.value,w_KOH=self.Anode.w_KOH.value), dens_gas=Functions.dens_gas(m=0, M=self.Anode.M_O2, p=self.Anode.p_abs.value, T=self.Anode.T.value), p_abs=self.Anode.p_abs.value)
        self.Cathode.d_b.value = Functions.d_b(m=0, i="H2", j=Functions.initial_value(self.system_config.j), gamma_KOH=Functions.gamma_KOH(self.Cathode.T.value, self.Cathode.w_KOH.value), dens_KOH=Functions.dens_KOH(m=0,T=self.Cathode.T.value,w_KOH=self.Cathode.w_KOH.value), dens_gas=Functions.dens_gas(m=0, M=self.Cathode.M_H2, p=self.Cathode.p_abs.value, T=self.Cathode.T.value), p_abs=self.Cathode.p_abs.value)
        self.Anode.p_b.value = self.Anode.p + Functions.Delta_p_b(gamma_fl=Functions.gamma_KOH(T=self.Anode.T.value, w_KOH=self.Anode.w_KOH.value), d_b=self.Anode.d_b.value)
        self.Cathode.p_b.value = self.Cathode.p + Functions.Delta_p_b(gamma_fl=Functions.gamma_KOH(T=self.Cathode.T.value, w_KOH=self.Cathode.w_KOH.value), d_b=self.Cathode.d_b.value)
        self.Anode.p_out["O2"].value = self.Anode.p - Functions.p_sat(m=0, T=self.Anode.T.value, w_KOH=self.Anode.w_KOH.value)
        self.Anode.p_out["H2"].value = 0
        self.Cathode.p_out["O2"].value = 0
        self.Cathode.p_out["H2"].value = self.Cathode.p - Functions.p_sat(m=0, T=self.Cathode.T.value, w_KOH=self.Cathode.w_KOH.value)
        #
        if(self.system_config.solve_mode == 0):
            self.Anode.VS_G_out.value = 0
            self.Cathode.VS_G_out.value = 0
        else:
            self.Anode.VS_G_out.value = ( ((Functions.initial_value(self.system_config.j)*self.Anode.A_elec)/(self.Anode.z*self.Anode.F))/(self.Anode.p_b.value/(self.Anode.R*self.Anode.T.value)) )
            self.Cathode.VS_G_out.value = ( ((Functions.initial_value(self.system_config.j)*self.Cathode.A_elec)/(self.Cathode.z*self.Cathode.F))/(self.Cathode.p_b.value/(self.Cathode.R*self.Cathode.T.value)) )
        #
        # Boundary Conditions
        self.Anode.VS_G_in = 0
        self.Cathode.VS_G_in = 0
        for ii in components:
            self.Anode.p_in[ii] = 0
            self.Cathode.p_in[ii] = 0
        
    
    def Intermediates(self, m):
        # Pre Intermediates
        self.Anode.fG["O2"] = m.Intermediate(0, name=self.name_pre+"Anode_fG_O2")
        self.Anode.fG["H2"] = m.Intermediate(0, name=self.name_pre+"Anode_fG_H2")
        self.Cathode.fG["O2"] = m.Intermediate(0, name=self.name_pre+"Cathode_fG_O2")
        self.Cathode.fG["H2"] = m.Intermediate(0, name=self.name_pre+"Cathode_fG_H2")
        #
        self.Anode.Intermediates(m=m)
        self.Cathode.Intermediates(m=m)
        #
        # Post Intermediates
        self.Anode.n_cross["O2"] = m.Intermediate(self.Anode.A_sep * (self.Anode.D_eff["O2"]/self.Anode.d_sep) * (self.Cathode.c_out["O2"] - self.Anode.fS["O2"] * self.Anode.c_out["O2"]), name=self.name_pre+"Anode_n_cross_O2")
        self.Cathode.n_cross["H2"] = m.Intermediate(self.Cathode.A_sep * (self.Cathode.D_eff["H2"]/self.Cathode.d_sep) * (self.Anode.c_out["H2"] - self.Cathode.fS["H2"] * self.Cathode.c_out["H2"]), name=self.name_pre+"Cathode_n_cross_H2")
        self.Anode.n_cross["H2"] = m.Intermediate(-self.Cathode.n_cross["H2"], name=self.name_pre+"Anode_n_cross_H2")
        self.Cathode.n_cross["O2"] = m.Intermediate(-self.Anode.n_cross["O2"], name=self.name_pre+"Cathode_n_cross_O2")
        
    def Equations(self, m):
        m.Equation(self.Anode.T == self.T)
        m.Equation(self.Cathode.T == self.T)
        self.Anode.Equations(m=m)
        self.Cathode.Equations(m=m)
        

class StackCompartmentClass:
    def __init__(self, name, system_config, m):
        self.m = m
        self.name = name
        self.name_pre = name + "_"
        self.system_config = system_config
        self.Initialize(m=m)
        
    def Initialize(self, m):
        self.c_in = {ii: None for ii in components}
        self.c_out = {ii: None for ii in components}
        self.p_out = {ii: None for ii in components}
        self.VS_G_out = None # m^3/s
        self.VS_L_in = None # m^3/s
        
    def Intermediates(self, m):
        pass
    
    def Equations(self, m):
        pass


class StackClass:
    def __init__(self, name, system_config, m):
        self.m = m
        self.name = name
        self.name_pre = name + "_"
        self.system_config = system_config
        self.Initialize(m=m)
        
    def Initialize(self, m):
        self.ElectrolysisCell = ElectrolysisCellClass(name=self.name_pre+"ElectrolysisCell", system_config=self.system_config, m=m)
        self.Anode = StackCompartmentClass(name=self.name_pre+"Anode", system_config=self.system_config, m=m)
        self.Cathode = StackCompartmentClass(name=self.name_pre+"Cathode", system_config=self.system_config, m=m)
        self.num_cells = self.system_config.num_cells # -
    
    def Initials(self, m):
        self.ElectrolysisCell.Initials(m=m)
    
    def Intermediates(self, m):
        # Inlets
        self.ElectrolysisCell.Anode.VS_L_in = m.Intermediate(self.Anode.VS_L_in/self.num_cells, name=self.name_pre+"ElectrolysisCell_Anode_VS_L_in")
        self.ElectrolysisCell.Cathode.VS_L_in = m.Intermediate(self.Cathode.VS_L_in/self.num_cells, name=self.name_pre+"ElectrolysisCell_Cathode_VS_L_in")
        #
        self.ElectrolysisCell.Intermediates(m=m)
        self.Anode.Intermediates(m=m)
        self.Cathode.Intermediates(m=m)
        #
        # Outlets
        self.Anode.VS_G_out = m.Intermediate(self.ElectrolysisCell.Anode.VS_G_out*self.num_cells, name=self.name_pre+"Anode_VS_G_out")
        self.Cathode.VS_G_out = m.Intermediate(self.ElectrolysisCell.Cathode.VS_G_out*self.num_cells, name=self.name_pre+"Cathode_VS_G_out")
        for ii in components:
            self.Anode.c_out[ii] = m.Intermediate(self.ElectrolysisCell.Anode.c_out[ii], name=self.name_pre+"Anode_c_out_" + str(ii))
            self.Cathode.c_out[ii] = m.Intermediate(self.ElectrolysisCell.Cathode.c_out[ii], name=self.name_pre+"Cathode_c_out_" + str(ii))
            self.Anode.p_out[ii] = m.Intermediate(self.ElectrolysisCell.Anode.p_out[ii], name=self.name_pre+"Anode_p_out_" + str(ii))
            self.Cathode.p_out[ii] = m.Intermediate(self.ElectrolysisCell.Cathode.p_out[ii], name=self.name_pre+"Cathode_p_out_" + str(ii))
            
    def Equations(self, m):
        self.ElectrolysisCell.Equations(m=m)
        self.Anode.Equations(m=m)
        self.Cathode.Equations(m=m)

class GasSeparatorVesselClass:
    def __init__(self, name, system_config, m):
        self.m = m
        self.name = name
        self.name_pre = name + "_"
        self.system_config = system_config
        self.Initialize(m=m)
        
    def Initialize(self, m):
        # Parameters
        self.c_in = {ii: None for ii in components} # mol/m^3
        self.M_H2 = Constants.M_H2 # kg/mol
        self.M_H2O = Constants.M_H2O # kg/mol
        self.M_O2 =  Constants.M_O2 # kg/mol
        self.p = None # Pa
        self.p_in = {ii: None for ii in components} # Pa
        self.R = Constants.R # J/(mol K)
        self.VR = None # m^3
        self.VS_L_out = None # m^3/s 
        #
        # Intermediates
        self.dens_KOH = None # kg/m^3
        self.M_gas = None # kg/mol
        self.M_gas_wet = None # kg/mol
        self.p_sat = None # Pa
        self.V_G = None # m^3
        self.V_L = None # m^3
        self.x = {ii: None for ii in components} # -
        self.x_percent = {ii: None for ii in components} # %
        self.x_wet = {ii: None for ii in components} # -
        #
        # Variables
        self.c_out = {ii: m.Var(value=0, lb=0, name=self.name_pre+"c_out" + "_" + str(ii)) for ii in components} # mol/m^3
        self.eps = m.Var(lb=0, ub=1, name=self.name_pre+"eps") # -
        self.ns_ve_mix = {ii: m.Var(value=0, name=self.name_pre+"ns_ve_mix" + "_" + str(ii)) for ii in components} # mol/s
        self.p_abs = m.Var(value=0, lb=Constants.p_atm, name=self.name_pre+"p_abs") # Pa
        self.p_out = {ii: m.Var(value=0, lb=0, name=self.name_pre+"p_out" + "_" + str(ii)) for ii in components} # Pa
        self.T = m.Var(value=self.system_config.T_ini, lb=273.15, ub=373.15, name=self.name_pre+"T") # K
        self.VS_G_in = m.Var(value=0, lb=0, name=self.name_pre+"VS_G_in") # m^3/s
        self.VS_G_out = m.Var(value=0, lb=0, name=self.name_pre+"VS_G_out") # m^3/s
        self.VS_L_in = m.Var(value=0, lb=0, name=self.name_pre+"VS_L_in") # m^3/s
        self.w_KOH = m.Var(lb=0, ub=0.55, name=self.name_pre+"w_KOH") # -
        
    def Intermediates(self, m):
        self.p_sat = m.Intermediate( Functions.p_sat(m=m, T=self.T, w_KOH=self.w_KOH) , name=self.name_pre+"p_sat")
        for ii in components:
            self.x[ii] = m.Intermediate(self.p_out[ii] / (self.p_out["O2"] + self.p_out["H2"]), name=self.name_pre+"x_" + str(ii))
            self.x_wet[ii] = m.Intermediate(self.p_out[ii] / (self.p_out["O2"] + self.p_out["H2"] + self.p_sat), name=self.name_pre+"x_wet_" + str(ii))
            self.x_percent[ii] = m.Intermediate(self.x[ii] * 100, name=self.name_pre+"x_percent_" + str(ii))
        self.M_gas = m.Intermediate(self.x["H2"] * self.M_H2 + self.x["O2"] * self.M_O2, name=self.name_pre+"M_gas")
        self.M_gas_wet = m.Intermediate(self.x_wet["H2"] * self.M_H2 + self.x_wet["O2"] * self.M_O2 + (1-self.x_wet["H2"]-self.x_wet["O2"]) * self.M_H2O, name=self.name_pre+"M_gas_wet")
        self.dens_KOH = m.Intermediate(Functions.dens_KOH(m=m,T=self.T,w_KOH=self.w_KOH) , name=self.name_pre+"dens_KOH")
        self.V_G = m.Intermediate(self.eps * self.VR , name=self.name_pre+"V_G")
        self.V_L = m.Intermediate((1-self.eps) * self.VR, name=self.name_pre+"V_L")
        
    def Equations(self, m):
        m.Equation(self.p_abs == self.p_out["O2"] + self.p_out["H2"] + self.p_sat)
        m.Equation(self.p_abs == self.p)
        for ii in components:
            m.Equation(self.VR * (1-self.eps) * self.c_out[ii].dt() == self.VS_L_in * self.c_in[ii] - self.VS_L_out * self.c_out[ii] + self.ns_ve_mix[ii])
            m.Equation((self.VR * self.eps)/(self.R * self.T) * self.p_out[ii].dt() == ((self.VS_G_in)/(self.R * self.T)) * self.p_in[ii] - ((self.VS_G_out)/(self.R * self.T)) * self.p_out[ii])
            

class GasSeparatorClass:
    def __init__(self, name, system_config, m):
        self.name = name
        self.name_pre = self.name + "_"
        self.system_config = system_config
        self.Initialize(m=m)
        
    def Initialize(self, m):
        self.Anode = GasSeparatorVesselClass(name=self.name_pre+"Anode", system_config=self.system_config, m=m)
        self.Cathode = GasSeparatorVesselClass(name=self.name_pre+"Cathode", system_config=self.system_config, m=m)
        #
        self.Anode.VR = self.system_config.GS_VR
        self.Cathode.VR = self.system_config.GS_VR
        self.Anode.p = self.system_config.p
        self.Cathode.p = self.system_config.p
        self.Anode.L_level = self.system_config.GS_L_ini
        self.Cathode.L_level = self.system_config.GS_L_ini
        #
        # Variables
        self.T = m.Var(value=self.system_config.T_ini, lb=273.15, ub=373.15, name=self.name_pre+"T") # K
        #
        
    def Initials(self, m):
        self.Anode.p_abs.value = self.Anode.p
        self.Cathode.p_abs.value = self.Cathode.p
        self.Anode.p_out["O2"].value = self.Anode.p - Functions.p_sat(m=0, T=self.Anode.T.value, w_KOH=self.Anode.w_KOH.value)
        self.Anode.p_out["H2"].value = 0
        self.Cathode.p_out["O2"].value = 0
        self.Cathode.p_out["H2"].value = self.Cathode.p - Functions.p_sat(m=0, T=self.Cathode.T.value, w_KOH=self.Cathode.w_KOH.value)
        
    def Intermediates(self, m):
        self.Anode.Intermediates(m=m)
        self.Cathode.Intermediates(m=m)
    
    def Equations(self, m):
        m.Equation(self.Anode.T == self.T)
        m.Equation(self.Cathode.T == self.T)
        self.Anode.Equations(m=m)
        self.Cathode.Equations(m=m)
        
        
class MixerClass:
    def __init__(self, name, system_config, m):
        self.m = m
        self.name = name
        self.name_pre = name + "_"
        self.Initialize(m=m)
        
    def Initialize(self, m):
        self.c_in_A = {ii: None for ii in components} # mol/m^3
        self.c_in_C = {ii: None for ii in components} # mol/m^3
        self.c_out_mix = {ii:  m.Var(value=0, lb=0, name=self.name_pre+"c_out_mix" + "_" + str(ii)) for ii in components} # mol/m^3
        self.VS_L_A = None # m^3/s
        self.VS_L_C = None # m^3/s
        self.VS_L_mix = None # m^3/s
        
    def Intermediates(self, m):
        self.VS_L_mix = m.Intermediate(self.VS_L_A + self.VS_L_C, name=self.name_pre+"VS_L_mix")         

    def Equations(self, m):
        for ii in components:
            m.Equation(self.c_out_mix[ii] * self.VS_L_mix == (self.c_in_A[ii] * self.VS_L_A + self.c_in_C[ii] * self.VS_L_C))   

class PotentialClass:
    def __init__(self, name, system_config, m):
        self.name = name
        self.name_pre = self.name + "_"
        self.system_config = system_config
        self.Initialize(m=m)
    
    def Initialize(self, m):
        # Parameters
        self.sides = ['Anode', 'Cathode'] # -
        self.A_elec = self.system_config.A_elec # m^2
        self.A_elec_eff = self.system_config.A_elec_eff # m^2
        self.d_eg = None # m
        self.d_sep = self.system_config.d_sep # m
        self.eps_sep = self.system_config.eps_sep # -
        self.F = Constants.F # C/mol
        self.j = None # A/m^2
        self.M_KOH = Constants.M_KOH # kg/mol
        self.num_cells = self.system_config.num_cells # -
        self.p = None # Pa
        self.p_0 = Constants.p_0 # Pa
        self.R = Constants.R # J/(mol K)
        self.R_add = None # Ohm
        self.tau_sep = self.system_config.tau_sep # -
        self.z = {jj: None for jj in self.sides} # -
        self.z["Anode"] = Constants.z_A
        self.z["Cathode"] = Constants.z_C
        #
        # Intermediates
        self.alpha = {jj: None for jj in self.sides} # -
        self.cond_KOH = {jj: None for jj in self.sides} # S/m
        self.cond_KOH_eg = {jj: None for jj in self.sides} # S/m
        self.cond_KOH_mean = None # S/m
        self.dens_KOH = None # kg/m^3
        self.eff_cell = None # -
        self.eff_cell_percent = None # -
        self.eta_act = {jj: None for jj in self.sides} # V
        self.eps = {jj: None for jj in self.sides} # -
        self.fS = {jj: None for jj in self.sides} # -
        self.I = None # A
        self.j_0 = {jj: None for jj in self.sides} # A/m^2
        self.j_eff = None # A/m^2
        self.p_H2O = None # Pa
        self.p_hc_b = {jj: None for jj in self.sides} # Pa
        self.p_hc_p = {jj: None for jj in self.sides} # Pa
        self.p_sat = {jj: None for jj in self.sides} # Pa
        self.phi = None # -
        self.R_eg = {jj: None for jj in self.sides} # Ohm
        self.R_ohm = None # Ohm
        self.R_sep = None # Ohm
        self.U_cell = None # V
        self.U_eta_act = None # V
        self.U_hhv_0 = None # V
        self.U_nernst = None # V
        self.U_nernst_c = None # V
        self.U_nernst_p = None # V
        self.U_rev = None # V
        self.U_rev_0 = None # V
        self.U_stack = None # V
        self.U_tn = None # V
        self.w_KOH = {jj: None for jj in self.sides} # -
        self.w_KOH_mean = None # -
        self.y = None # J/mol
        #
        # Variables
        self.T = m.Var(value=self.system_config.T_ini, lb=273.15, ub=373.15, name=self.name_pre+"T") # K

    
    def Intermediates(self, m):
        self.w_KOH_mean = m.Intermediate( (self.w_KOH['Anode'] + self.w_KOH['Cathode'])/2 )
        self.U_rev_0 = m.Intermediate(1.50342 - 9.956E-4 * self.T + 2.5E-7 * self.T**2, name=self.name_pre+"U_rev_0") # V
        self.p_H2O = m.Intermediate( Functions.p_sat(m=m, T=self.T, w_KOH=0) , name=self.name_pre+"p_H2O")
        self.U_rev = m.Intermediate( self.U_rev_0, name=self.name_pre+"U_rev")
        self.U_nernst_p = m.Intermediate( ((self.R * self.T)/(self.z["Cathode"] * self.F)) * m.log( ( (self.p_hc_p["Cathode"]/self.p_0) * (self.p_hc_p["Anode"]/self.p_0)**(1/2) * (self.p_sat["Anode"]/self.p_H2O) ) / ( (self.p_sat["Cathode"]/self.p_H2O)**2 ) ) , name=self.name_pre+"U_nernst_p")
        self.U_nernst_c = m.Intermediate( ((self.R * self. T)/(self.F)) * ( (1/self.z["Cathode"]) * m.log(self.fS["Cathode"]) + (1/self.z["Anode"]) * m.log(self.fS["Anode"]) )  , name=self.name_pre+"U_nernst_c")
        self.U_nernst = m.Intermediate(self.U_nernst_p + self.U_nernst_c, name=self.name_pre+"U_nernst")
        self.I = m.Intermediate(self.j * self.A_elec, name=self.name_pre+"I")      
        self.j_eff = m.Intermediate(self.I/self.A_elec_eff, name=self.name_pre+"j_eff")
        self.cond_KOH_mean = m.Intermediate( Functions.cond_KOH(m=m, T=self.T, w_KOH=self.w_KOH_mean), name=self.name_pre+"cond_KOH_mean")
        self.R_sep = m.Intermediate((1/(self.cond_KOH_mean * self.eps_sep/self.tau_sep)) * (self.d_sep/self.A_elec_eff), name=self.name_pre+"R_sep")
        #
        for jj in self.sides:
            self.cond_KOH[jj] = m.Intermediate( Functions.cond_KOH(m=m, T=self.T, w_KOH=self.w_KOH[jj]), name=self.name_pre+"cond_KOH_"+ str(jj))
            self.cond_KOH_eg[jj] = m.Intermediate( self.cond_KOH[jj] * (1 - self.eps[jj])**(3/2), name=self.name_pre+"cond_KOH_eg_"+ str(jj))
            self.eta_act[jj] = m.Intermediate(((2.303 * self.R * self.T)/(self.z[jj] * self.F * self.alpha[jj])) * m.log10(self.j_eff/self.j_0[jj]), name=self.name_pre+"eta_act_" + str(jj)) # INFO: Overpotentials only valid for j_eff >= j_0
            self.R_eg[jj] = m.Intermediate((1/self.cond_KOH_eg[jj]) * (self.d_eg/self.A_elec_eff), name=self.name_pre+"R_eg_" + str(jj))
        #
        self.R_ohm = m.Intermediate(self.R_sep + self.R_eg["Anode"] + self.R_eg["Cathode"] + self.R_add, name=self.name_pre+"R_ohm")
        self.U_eta_act = m.Intermediate(self.eta_act["Anode"] + self.eta_act["Cathode"], name=self.name_pre+"U_eta_act")
        #
        self.U_hhv_0 = m.Intermediate(1.4756 + 2.252E-4 * (self.T-273.15) + 1.52E-8 * (self.T-273.15)**2, name=self.name_pre+"U_hhv_0")
        self.phi = m.Intermediate(1 * (self.p_sat["Cathode"]/(self.p_hc_b["Cathode"] - self.p_sat["Cathode"])) + 0.5 * (self.p_sat["Anode"]/(self.p_hc_b["Anode"] - self.p_sat["Anode"])), name=self.name_pre+"phi")
        self.y = m.Intermediate(42960 + 40.762 * (self.T-273.15) - 0.06682 * (self.T-273.15)**2, name=self.name_pre+"y")
        self.U_tn = m.Intermediate(self.U_hhv_0 + (self.phi/(self.z["Cathode"] * self.F)) * self.y, name=self.name_pre+"U_tn")
        self.U_cell = m.Intermediate(self.U_rev + self.U_nernst + self.U_eta_act + (self.R_ohm * self.I), name=self.name_pre+"U_cell")
        self.U_stack = m.Intermediate(self.num_cells * self.U_cell, name=self.name_pre+"U_stack")
        #
        self.eff_cell = m.Intermediate(self.U_tn / self.U_cell, name=self.name_pre+"eff_cell")
        self.eff_cell_percent = m.Intermediate(self.eff_cell * 100, name=self.name_pre+"eff_cell_percent")
    
    def Equations(self, m):
        pass


class ElectrolyteConcentrationHalfCellClass:
    def __init__(self, name, system_config, m):
        self.name = name
        self.name_pre = self.name + "_"
        self.system_config = system_config
        self.Initialize(m=m)

    def Initialize(self, m):
        # Parameters
        self.A_elec = self.system_config.A_elec # m^2
        self.F = Constants.F # C/mol
        self.M_H2O = Constants.M_H2O # kg/mol
        self.M_KOH = Constants.M_KOH # kg/mol
        self.nu_H2O = None # -
        self.VR = self.system_config.HC_VR # m^2
        self.VS_L_in = None # m^3/s
        self.VS_L_out = None # m^3/s
        self.z = None # -
        #
        # Intermediates
        self.dens_H2O = None # kg/m^3
        self.dens_KOH = None # kg/m^3
        self.dens_KOH_in = None # kg/m^3
        self.dens_KOH_ve = None # kg/m^3
        self.eps = None # -
        self.j = None # A/m^2
        self.m_El = None # kg
        self.m_H2O = None # kg
        self.m_KOH = None # kg
        self.m_R_H2O = None # kg/s
        self.m_D_H2O = None # kg/s
        self.mf_H2O = None # kg/s
        self.mf_in = None # kg/s
        self.mf_KOH = None # kg/s
        self.mf_out = None # kg/s
        self.T = None # K
        #
        # Variables
        self.mf_KOH_cross = m.Var(value=0, name=self.name_pre+"mf_KOH_cross") # kg/s
        self.mf_ve = m.Var(value=0, name=self.name_pre+"mf_ve") # kg/s
        self.Vf_ve = m.Var(value=0, name=self.name_pre+"Vf_ve") # m^3/s
        self.Vf_drag = m.Var(value=0, name=self.name_pre+"Vf_drag") # m^3/s
        self.w_KOH = m.Var(value=self.system_config.w_KOH_ini, lb=0, ub=0.55, name=self.name_pre+"w_KOH") # -
        self.w_KOH_in = m.Var(value=self.system_config.w_KOH_ini, lb=0, ub=0.55, name=self.name_pre+"w_KOH_in") # -
        self.w_KOH_ve = m.Var(value=self.system_config.w_KOH_ini, lb=0, ub=0.55, name=self.name_pre+"w_KOH_ve") # -

    def Intermediates(self, m):
        self.dens_H2O = m.Intermediate( Functions.dens_H2O(T=self.T) , name=self.name_pre+"dens_H2O")
        self.dens_KOH = m.Intermediate(Functions.dens_KOH(m=m, T=self.T, w_KOH=self.w_KOH), name=self.name_pre+"dens_KOH")
        self.dens_KOH_in = m.Intermediate(Functions.dens_KOH(m=m, T=self.T, w_KOH=self.w_KOH_in), name=self.name_pre+"dens_KOH_in")
        self.dens_KOH_ve = m.Intermediate(Functions.dens_KOH(m=m, T=self.T, w_KOH=self.w_KOH_ve), name=self.name_pre+"dens_KOH_ve")
        self.m_R_H2O = m.Intermediate((self.nu_H2O * self.j * self.A_elec) / (self.z * self.F) * self.M_H2O, name=self.name_pre+"m_R_H2O")
        self.mf_in = m.Intermediate(self.VS_L_in * self.dens_KOH_in, name=self.name_pre+"mf_in")
        self.mf_KOH = m.Intermediate(self.mf_in * self.w_KOH_in + self.mf_ve * self.w_KOH_ve + self.mf_KOH_cross, name=self.name_pre+"mf_KOH")
        self.mf_H2O = m.Intermediate(self.mf_in * (1-self.w_KOH_in) + self.mf_ve * (1-self.w_KOH_ve) + self.m_R_H2O + self.m_D_H2O, name=self.name_pre+"mf_H2O")
        self.mf_out = m.Intermediate(self.mf_KOH + self.mf_H2O, name=self.name_pre+"mf_out")
        self.VS_L_out = m.Intermediate(self.mf_out / self.dens_KOH, name=self.name_pre+"VS_L_out")
        self.m_El = m.Intermediate((1-self.eps) * self.VR * self.dens_KOH, name=self.name_pre+"m_El")
        self.m_H2O = m.Intermediate((1-self.w_KOH) * self.m_El, name=self.name_pre+"m_H2O")
        self.m_KOH = m.Intermediate(self.w_KOH * self.m_El, name=self.name_pre+"m_KOH")

    def Equations(self, m):
        m.Equation(self.w_KOH == self.mf_KOH / (self.mf_KOH + self.mf_H2O))
        

class ElectrolyteConcentrationGasSeparatorClass:
    def __init__(self, name, system_config, m):
        self.name = name
        self.name_pre = self.name + "_"
        self.system_config = system_config
        self.Initialize(m=m)

    def Initialize(self, m):
        # Constants
        self.L_ini = self.system_config.GS_L_ini # -
        self.M_H2O = Constants.M_H2O # kg/mol
        self.VR = self.system_config.GS_VR # m^3
        self.w_KOH_ini = self.system_config.w_KOH_ini # -
        #
        # Parameters
        self.dens_KOH_ini = Functions.dens_KOH(m=0, T=self.system_config.T_ini, w_KOH=self.w_KOH_ini) # kg/m^3
        self.m_El_ini = self.VR * self.L_ini * self.dens_KOH_ini # kg
        self.m_H2O_ini = (1-self.w_KOH_ini) * self.m_El_ini # kg
        self.m_KOH_ini = self.w_KOH_ini * self.m_El_ini # kg
        self.VS_L_in = None # m^3/s
        self.VS_L_out = None # m^3/s
        #
        # Intermediates
        self.dens_KOH = None # kg/m^3
        self.dens_KOH_in = None # kg/m^3
        self.eps = None # -
        self.m_El = None # kg
        self.m_El_in = None # kg/s
        self.m_El_out = None # kg/s
        self.mf_G_H2O = None # kg/s
        self.mf_G_H2O_cond = None # kg/s
        self.mf_H2O_WD = None # kg/s
        self.p_out = {ii: None for ii in components} # Pa
        self.p_sat = None # Pa
        self.p_sat_cond = None # Pa
        self.T = None # K
        self.T_gas = None # K
        self.V_L = None # m^3
        self.VS_G_H2O = None # m^3/s
        self.VS_G_H2O_cond = None # m^3/s
        self.VS_G_out = None # m^3/s
        self.x_wet = {ii: None for ii in components} # -
        self.x_wet_cond = {ii: None for ii in components} # -
        #
        # Variables
        self.m_H2O = m.Var(value=self.m_H2O_ini, lb=0, name=self.name_pre+"m_H2O") # kg
        self.m_KOH = m.Var(value=self.m_KOH_ini, lb=0, name=self.name_pre+"m_KOH") # kg
        self.mf_ve = m.Var(value=0, name=self.name_pre+"mf_ve") # kg/s
        self.Vf_ve = m.Var(value=0, name=self.name_pre+"Vf_ve") # m^3/s
        self.w_KOH = m.Var(value=self.system_config.w_KOH_ini, lb=0, ub=0.55, name=self.name_pre+"w_KOH") # -
        self.w_KOH_in = m.Var(value=self.system_config.w_KOH_ini, lb=0, ub=0.55, name=self.name_pre+"w_KOH_in") # -
        self.w_KOH_ve = m.Var(value=self.system_config.w_KOH_ini, lb=0, ub=0.55, name=self.name_pre+"w_KOH_ve") # -

    def Intermediates(self, m):
        self.m_El = m.Intermediate(self.m_KOH + self.m_H2O, name=self.name_pre+"m_El")
        self.dens_KOH = m.Intermediate(Functions.dens_KOH(m=m, T=self.T, w_KOH=self.w_KOH), name=self.name_pre+"dens_KOH")
        self.dens_KOH_in = m.Intermediate(Functions.dens_KOH(m=m, T=self.T, w_KOH=self.w_KOH_in), name=self.name_pre+"dens_KOH_in")
        self.V_L = m.Intermediate(self.m_El / self.dens_KOH, name=self.name_pre+"V_L")
        self.m_El_in = m.Intermediate(self.VS_L_in * self.dens_KOH_in, name=self.name_pre+"m_El_in")
        self.m_El_out = m.Intermediate(self.VS_L_out * self.dens_KOH, name=self.name_pre+"m_El_out")
        self.eps = m.Intermediate(1 - self.V_L / self.VR, name=self.name_pre+"eps")
        #
        self.p_sat_cond = m.Intermediate( Functions.p_sat(m=m, T=self.T_gas, w_KOH=self.w_KOH) , name=self.name_pre+"p_sat_cond") 
        self.p_abs = m.Intermediate( self.p_out["H2"] + self.p_out["O2"] + self.p_sat , name=self.name_pre+"p_abs")
        self.p_abs_cond = m.Intermediate( self.p_out["H2"] + self.p_out["O2"] + self.p_sat_cond , name=self.name_pre+"p_abs_cond")
        self.dens_H2O_vapor = m.Intermediate(Functions.dens_gas(m=0, M=self.M_H2O, p=self.p_abs, T=self.T), name=self.name_pre+"dens_H2O_vapor")
        self.dens_H2O_vapor_cond = m.Intermediate(Functions.dens_gas(m=0, M=self.M_H2O, p=self.p_abs_cond, T=self.T_gas), name=self.name_pre+"dens_H2O_vapor_cond")
        for ii in components:
            self.x_wet[ii] = m.Intermediate(self.p_out[ii] / (self.p_out["O2"] + self.p_out["H2"] + self.p_sat), name=self.name_pre+"x_wet_" + str(ii))
            self.x_wet_cond[ii] = m.Intermediate(self.p_out[ii] / (self.p_out["O2"] + self.p_out["H2"] + self.p_sat_cond), name=self.name_pre+"x_wet_cond" + str(ii))
        self.VS_G_H2O = m.Intermediate( self.VS_G_out * ((1/(self.x_wet["H2"]+self.x_wet["O2"])) - 1), name=self.name_pre+"VS_G_H2O")
        self.VS_G_H2O_cond = m.Intermediate( self.VS_G_out * ((1/(self.x_wet_cond["H2"]+self.x_wet_cond["O2"])) - 1), name=self.name_pre+"VS_G_H2O_cond")
        self.mf_G_H2O = m.Intermediate( self.VS_G_H2O * self.dens_H2O_vapor , name=self.name_pre+"mf_G_H2O")
        self.mf_G_H2O_cond = m.Intermediate( self.VS_G_H2O_cond * self.dens_H2O_vapor_cond, name=self.name_pre+"mf_G_H2O_cond")
        
    def Equations(self, m):
        m.Equation(self.m_KOH.dt() == self.m_El_in * self.w_KOH_in - self.m_El_out * self.w_KOH + self.mf_ve * self.w_KOH_ve)
        m.Equation(self.m_H2O.dt() == self.m_El_in * (1-self.w_KOH_in) - self.m_El_out * (1-self.w_KOH) + self.mf_ve * (1-self.w_KOH_ve) + self.mf_H2O_WD - 0 * self.mf_G_H2O_cond) # INFO: Assumption of complete condensation of H2O
        m.Equation(self.w_KOH == self.m_KOH / (self.m_KOH + self.m_H2O))


class ElectrolyteConcentrationMixerClass:
    def __init__(self, name, system_config, m):
        self.name = name
        self.name_pre = self.name + "_"
        self.system_config = system_config
        self.Initialize(m=m)

    def Initialize(self, m):
        # Intermediates
        self.mf_A = None # kg/s
        self.mf_C = None # kg/s
        self.mf_mix = None # kg/s
        #
        # Variables
        self.w_KOH_A = m.Var(value=self.system_config.w_KOH_ini, lb=0, ub=0.55, name=self.name_pre+"w_KOH_A") # -
        self.w_KOH_C = m.Var(value=self.system_config.w_KOH_ini, lb=0, ub=0.55, name=self.name_pre+"w_KOH_C") # -
        self.w_KOH_mix = m.Var(value=self.system_config.w_KOH_ini, lb=0, ub=0.55, name=self.name_pre+"w_KOH_mix") # -

    def Intermediates(self, m):
        self.mf_mix = m.Intermediate(self.mf_A + self.mf_C)

    def Equations(self, m):
        m.Equation(self.mf_mix * self.w_KOH_mix == self.mf_A * self.w_KOH_A + self.mf_C * self.w_KOH_C)


class ElectrolyteConcentrationClass:
    def __init__(self, name, system_config, m):
        self.name = name
        self.name_pre = self.name + "_"
        self.system_config = system_config
        self.Initialize(m=m)

    def Initialize(self, m):
        #
        self.Anode = ElectrolyteConcentrationHalfCellClass(name=self.name_pre+"Anode", system_config=self.system_config, m=m)
        self.GasSeparatorAnode = ElectrolyteConcentrationGasSeparatorClass(name=self.name_pre+"GasSeparatorAnode", system_config=self.system_config, m=m)
        #
        self.Cathode = ElectrolyteConcentrationHalfCellClass(name=self.name_pre+"Cathode", system_config=self.system_config, m=m)
        self.GasSeparatorCathode = ElectrolyteConcentrationGasSeparatorClass(name=self.name_pre+"GasSeparatorCathode", system_config=self.system_config, m=m)
        #
        self.Mixer = ElectrolyteConcentrationMixerClass(name=self.name_pre+"Mixer", system_config=self.system_config, m=m)
        #
        self.T = None # K
        self.w_KOH_ini = self.system_config.w_KOH_ini
        self.dens_KOH_ini = Functions.dens_KOH(m=0, T=self.system_config.T_ini, w_KOH=self.w_KOH_ini)
        self.num_cells = self.system_config.num_cells
        self.A_sep = self.system_config.A_sep
        self.d_sep = self.system_config.d_sep
        self.eps_sep = self.system_config.eps_sep
        self.tau_sep = self.system_config.tau_sep
        self.tau_ve_s = self.system_config.tau_ve_s
        self.tau_ve_m = self.system_config.tau_ve_m
        #
        # Parameters
        self.Anode.z = Constants.z_A
        self.Anode.nu_H2O = 2
        #
        self.Cathode.z = Constants.z_C
        self.Cathode.nu_H2O = -2
        #
        # Intermediates
        self.D_KOH = None # m^2/s
        self.Delta_V_L_GS = None # m^3
        self.f_W_drag = None # -
        self.f_D_KOH = None # -
        self.j = None # A/m^2
        self.mf_G_H2O_cond_sum = None # kg/s
        self.S_mixing = None # -
        self.S_VE = None # -
        self.S_WD = None # -
        self.T = None # K
        #
        # Overall system
        self.m_El = None # kg
        self.m_H2O = None # kg
        self.m_KOH = None # kg
        #
        # Variables
        self.w_KOH_sys = m.Var(value=self.system_config.w_KOH_ini, lb=0, ub=0.55, name=self.name_pre+"w_KOH_sys") # -

    def Intermediates(self, m):
        self.Anode.j = m.Intermediate(self.j, name=self.name_pre+"Anode_j")
        self.Cathode.j = m.Intermediate(self.j, name=self.name_pre+"Cathode_j")
        #
        self.Anode.T = m.Intermediate(self.T, name=self.name_pre+"Anode_T")
        self.GasSeparatorAnode.T = m.Intermediate(self.T, name=self.name_pre+"GasSeparatorAnode_T")
        self.Cathode.T = m.Intermediate(self.T, name=self.name_pre+"Cathode_T")
        self.GasSeparatorCathode.T = m.Intermediate(self.T, name=self.name_pre+"GasSeparatorCathode_T")
        #
        self.Cathode.m_D_H2O = m.Intermediate(self.f_W_drag * (self.Cathode.nu_H2O * self.Cathode.j * self.Cathode.A_elec) / (self.Cathode.z * self.Cathode.F) * self.Cathode.M_H2O, name=self.name_pre+"Cathode_m_D_H2O")
        self.Anode.m_D_H2O = m.Intermediate(-self.Cathode.m_D_H2O, name=self.name_pre+"Anode_m_D_H2O")
        #
        self.Anode.Intermediates(m=m)
        self.Cathode.Intermediates(m=m)
        #
        self.GasSeparatorAnode.VS_L_in = m.Intermediate(self.Anode.VS_L_out * self.num_cells, name=self.name_pre+"GasSeparatorAnode_VS_L_in")
        self.GasSeparatorAnode.Intermediates(m=m)
        self.GasSeparatorAnode.mf_H2O_WD = m.Intermediate(0, name=self.name_pre+"GasSeparatorAnode_mf_H2O_WD")
        #
        self.GasSeparatorCathode.VS_L_in = m.Intermediate(self.Cathode.VS_L_out  * self.num_cells, name=self.name_pre+"GasSeparatorCathode_VS_L_in")
        self.GasSeparatorCathode.Intermediates(m=m)
        #
        self.mf_G_H2O_cond_sum = m.Intermediate(self.GasSeparatorAnode.mf_G_H2O_cond + self.GasSeparatorCathode.mf_G_H2O_cond, name=self.name_pre+"mf_G_H2O_cond_sum")
        #
        self.GasSeparatorCathode.mf_H2O_WD = m.Intermediate(self.S_WD * (self.num_cells * (-0.5 * self.Cathode.m_R_H2O) + 0 * self.mf_G_H2O_cond_sum), name=self.name_pre+"GasSeparatorCathode_mf_H2O_WD") # INFO: Assumption of complete condensation
        #
        self.Delta_V_L_GS = m.Intermediate(self.GasSeparatorCathode.V_L - self.GasSeparatorAnode.V_L, name=self.name_pre+"Delta_V_L_GS")
        #
        self.Mixer.mf_A = m.Intermediate(self.GasSeparatorAnode.m_El_out, name=self.name_pre+"Mixer_mf_A")
        self.Mixer.mf_C = m.Intermediate(self.GasSeparatorCathode.m_El_out, name=self.name_pre+"Mixer_mf_C")
        self.Mixer.Intermediates(m=m)
        #
        self.m_El = m.Intermediate(self.num_cells * (self.Anode.m_El + self.Cathode.m_El) + self.GasSeparatorAnode.m_El +  self.GasSeparatorCathode.m_El, name=self.name_pre+"m_El")
        self.m_KOH = m.Intermediate(self.GasSeparatorAnode.m_KOH +  self.GasSeparatorCathode.m_KOH, name=self.name_pre+"m_KOH")
        self.m_H2O = m.Intermediate(self.GasSeparatorAnode.m_H2O +  self.GasSeparatorCathode.m_H2O, name=self.name_pre+"m_H2O")
        #
        self.D_KOH = m.Intermediate(Functions.D_KOH(w_KOH=self.w_KOH_sys, T=self.T) , name=self.name_pre+"D_KOH")

    def Equations(self, m):
        m.Equation(self.Anode.w_KOH_in == self.S_mixing * self.Mixer.w_KOH_mix + (1-self.S_mixing) * self.GasSeparatorAnode.w_KOH)
        m.Equation(self.Anode.w_KOH_ve == self.Anode.w_KOH)
        m.Equation(self.Cathode.w_KOH_in == self.S_mixing * self.Mixer.w_KOH_mix + (1-self.S_mixing) * self.GasSeparatorCathode.w_KOH)
        m.Equation(self.Cathode.w_KOH_ve == self.Anode.w_KOH)
        #
        m.Equation(self.GasSeparatorAnode.w_KOH_in == self.Anode.w_KOH)
        m.Equation(self.GasSeparatorCathode.w_KOH_in == self.Cathode.w_KOH)
        #
        m.Equation(self.GasSeparatorAnode.w_KOH_ve == self.GasSeparatorAnode.w_KOH)
        m.Equation(self.GasSeparatorCathode.w_KOH_ve == self.GasSeparatorAnode.w_KOH)
        #
        m.Equation(self.GasSeparatorAnode.mf_ve == (self.S_mixing) * self.S_VE * (self.Delta_V_L_GS * self.GasSeparatorAnode.dens_KOH)/(self.tau_ve_m))
        m.Equation(self.GasSeparatorCathode.mf_ve == -self.GasSeparatorAnode.mf_ve)
        #
        m.Equation(self.GasSeparatorAnode.Vf_ve == self.GasSeparatorAnode.mf_ve / self.GasSeparatorAnode.dens_KOH)
        m.Equation(self.GasSeparatorCathode.Vf_ve == -self.GasSeparatorAnode.Vf_ve)
        #
        m.Equation(self.Anode.mf_ve == (1-self.S_mixing) * self.S_VE * (self.Delta_V_L_GS * self.Anode.dens_KOH_ve)/(self.num_cells * self.tau_ve_s))
        m.Equation(self.Cathode.mf_ve == -self.Anode.mf_ve)
        #
        m.Equation(self.Anode.Vf_ve == self.Anode.mf_ve / self.Anode.dens_KOH)
        m.Equation(self.Cathode.Vf_ve == -self.Anode.Vf_ve)
        #
        m.Equation(self.Cathode.Vf_drag == self.Cathode.m_D_H2O / self.Cathode.dens_H2O)
        m.Equation(self.Anode.Vf_drag == -self.Cathode.Vf_drag)
        #
        m.Equation(self.Anode.mf_KOH_cross ==  self.A_sep * ((self.D_KOH*self.eps_sep/self.tau_sep)/self.d_sep) * self.f_D_KOH * (self.Cathode.w_KOH * self.Cathode.dens_KOH - self.Anode.w_KOH*self.Anode.dens_KOH) ) 
        m.Equation(self.Cathode.mf_KOH_cross ==  -self.Anode.mf_KOH_cross )
        #
        m.Equation(self.Mixer.w_KOH_A == self.GasSeparatorAnode.w_KOH)
        m.Equation(self.Mixer.w_KOH_C == self.GasSeparatorCathode.w_KOH)
        #
        self.Anode.Equations(m=m)
        self.GasSeparatorAnode.Equations(m=m)
        self.Cathode.Equations(m=m)
        self.GasSeparatorCathode.Equations(m=m)
        self.Mixer.Equations(m=m)
        #
        m.Equation(self.w_KOH_sys == ( self.num_cells * (self.Anode.m_KOH + self.Cathode.m_KOH) + self.GasSeparatorAnode.m_KOH + self.GasSeparatorCathode.m_KOH)/( self.num_cells * (self.Anode.m_KOH + self.Cathode.m_KOH + self.Anode.m_H2O + self.Cathode.m_H2O) + self.GasSeparatorAnode.m_KOH + self.GasSeparatorCathode.m_KOH + self.GasSeparatorAnode.m_H2O + self.GasSeparatorCathode.m_H2O) )

class EnergyBalanceClass:
    def __init__(self, name, system_config, m):
        self.name = name
        self.name_pre = self.name + "_"
        self.system_config = system_config
        self.Initialize(m=m)
    
    def Initialize(self, m):
        # Constants
        self.sides = ['Anode', 'Cathode']
        self.num_cells = self.system_config.num_cells # -
        self.M_H2 = Constants.M_H2 # kg/mol
        self.M_H2O = Constants.M_H2O # kg/mol
        self.M_O2 =  Constants.M_O2 # kg/mol
        self.d_cell_stack = self.system_config.EB_d_cell_stack # m
        self.d_gs = self.system_config.EB_d_gs # m
        self.d_tube = self.system_config.EB_d_tube # m
        self.EB_man_hc_power = self.system_config.EB_man_hc_power # W
        self.A_cell_stack = self.system_config.EB_A_cell_stack # m^2
        self.A_gs = self.system_config.EB_A_gs # m^2
        self.A_tube = self.system_config.EB_A_tube # m^2/m
        self.insulation_factor_cell_stack = self.system_config.EB_insulation_factor_cell_stack # -
        self.insulation_factor_gs = self.system_config.EB_insulation_factor_gs # -
        self.insulation_factor_tube = self.system_config.EB_insulation_factor_tube # -
        self.m_cell_stack = self.system_config.EB_m_cell_stack # kg
        self.m_gs = self.system_config.EB_m_gs # kg
        self.m_tube = self.system_config.EB_m_tube # kg/m
        #
        # Parameters
        self.S_hc = m.Param(value=self.system_config.EB_hc, name=self.name_pre+"S_hc") # -
        #
        # Intermediates
        self.C_t = None # J/K
        self.cp_El = None # J/(kg K)
        self.cp_gas = {jj: None for jj in self.sides} # J/(kg K)
        self.cp_H2 = None # J/(kg K)
        self.cp_O2 = None # J/(kg K)
        self.cp_steel = None # J/(kg K)
        self.Delta_h_v_H2O = None # J/kg
        self.dens_gas = {jj: None for jj in self.sides} # kg/m^3
        self.dens_H2O_vapor = {jj: None for jj in self.sides} # kg/m^3
        self.dens_KOH = None # kg/m^3
        self.h_cell_stack = None # W/(m^2 K)
        self.h_gs = None # W/(m^2 K)
        self.h_tube = None # W/(m^2 K)
        self.I = None # A
        self.m_El = None # kg
        self.M_gas = {jj: None for jj in self.sides} # kg/mol
        self.m_steel_sys = None # kg
        self.mf_gas = {jj: None for jj in self.sides} # kg/s
        self.mf_gas_H2O_vapor = {jj: None for jj in self.sides} # kg/s
        self.mf_H2O_vapor_sum = None # kg/s
        self.mf_WD_H2O = None # kg/s
        self.L_tube = None # m
        self.p_abs = {jj: None for jj in self.sides} # Pa
        self.Qf_out_gas_latent = None # W
        self.Qf_out_gas_sens = None # W
        self.Qf_WD_warm_sens = None # W
        self.R_t_cell_stack = None # K/W
        self.R_t_gs_A = None # K/W
        self.R_t_gs_C = None # K/W
        self.R_t_tube = None # K/W
        self.R_t_sys = None # K/W
        self.T_gas = None # K
        self.U_cell = None # V
        self.U_tn = None # V
        self.Vf_gas = {jj: None for jj in self.sides} # m^3/s
        self.Vf_gas_H2O_cond = {jj: None for jj in self.sides} # m^3/s
        self.w_KOH = None # -
        self.x_Anode = {ii: None for ii in components} # -
        self.x_Cathode = {ii: None for ii in components} # -
        self.x_wet_Anode = {ii: None for ii in components} # -
        self.x_wet_Cathode = {ii: None for ii in components} # -
        #
        # Variables
        self.Q_hc = m.Var(value=0, name=self.name_pre+"Q_hc") # W
        self.Q_loss = m.Var(value=0, name=self.name_pre+"Q_loss") # W
        self.Q_R = m.Var(value=0, lb=0, name=self.name_pre+"Q_R") # W
        self.T = m.Var(value=self.system_config.T_ini, lb=273.15, ub=373.15, name=self.name_pre+"T") # K

    def Intermediates(self, m):
        self.cp_El = m.Intermediate(Functions.cp_El(m=m, T=self.T, w_KOH=self.w_KOH), name=self.name_pre+"cp_KOH")
        self.cp_steel = m.Intermediate(Functions.cp_steel(m=0, T=self.T), name=self.name_pre+"cp_steel")
        self.m_steel_sys = m.Intermediate(self.m_cell_stack + 2 * self.m_gs + self.L_tube * self.m_tube, name=self.name_pre+"m_steel_sys")
        self.C_t = m.Intermediate(self.m_El * self.cp_El + self.m_steel_sys * self.cp_steel, name=self.name_pre+"C_t")
        self.dens_KOH = m.Intermediate( Functions.dens_KOH(m=m, T=self.T, w_KOH=self.w_KOH) , name=self.name_pre+"dens_KOH")
        self.dens_gas["Anode"] = m.Intermediate(Functions.dens_gas(m=0, M=self.M_gas["Anode"], p=self.p_abs["Anode"], T=self.T), name=self.name_pre+"dens_gas_Anode")
        self.dens_gas["Cathode"] = m.Intermediate(Functions.dens_gas(m=0, M=self.M_gas["Cathode"], p=self.p_abs["Cathode"], T=self.T), name=self.name_pre+"dens_gas_Cathode")
        self.cp_H2 = m.Intermediate(Functions.cp_gas(i="H2", T=self.T), name=self.name_pre+"cp_H2")
        self.cp_O2 = m.Intermediate(Functions.cp_gas(i="O2", T=self.T), name=self.name_pre+"cp_O2")
        self.cp_gas["Anode"] = m.Intermediate((self.x_Anode["H2"] * (self.cp_H2*self.M_H2) + self.x_Anode["O2"] * (self.cp_O2*self.M_O2) )/self.M_gas["Anode"], name=self.name_pre+"cp_gas_Anode")
        self.cp_gas["Cathode"] = m.Intermediate((self.x_Cathode["H2"] * (self.cp_H2*self.M_H2) + self.x_Cathode["O2"] * (self.cp_O2*self.M_O2) )/self.M_gas["Cathode"], name=self.name_pre+"cp_gas_Cathode")
        self.cp_H2O_liq = m.Intermediate(Functions.cp_H2O_liq(T=self.T), name=self.name_pre+"cp_H2O_liq")
        for jj in self.sides:
            self.mf_gas[jj] = m.Intermediate(self.Vf_gas[jj] * self.dens_gas[jj] ,name=self.name_pre+"mf_gas_"+ str(jj))
        self.mf_WD_H2O = m.Intermediate(self.mf_gas["Cathode"] + self.mf_gas["Anode"], name=self.name_pre+"mf_WD_H2O")
        self.mf_H2O_vapor_sum = m.Intermediate(self.mf_gas_H2O_vapor["Cathode"] + self.mf_gas_H2O_vapor["Anode"], name=self.name_pre+"mf_H2O_vapor_sum")
        self.Qf_out_gas_sens = m.Intermediate((self.mf_gas["Cathode"] * self.cp_gas["Cathode"] + self.mf_gas["Anode"] * self.cp_gas["Anode"]) * (self.T - self.T_gas), name=self.name_pre+"Qf_out_sens")
        self.Qf_WD_warm_sens = m.Intermediate(self.mf_WD_H2O * self.cp_H2O_liq * (self.T - self.T_amb), name=self.name_pre+"Qf_WD_warm_sens")
        self.Delta_h_v_H2O = m.Intermediate( Functions.Delta_h_v_H2O(m=0, T=self.T) , name=self.name_pre+"Delta_h_v_H2O")
        self.Qf_out_gas_latent = m.Intermediate(self.mf_H2O_vapor_sum  * self.Delta_h_v_H2O, name=self.name_pre+"Qf_out_gas_latent")
        #
        self.h_cell_stack = m.Intermediate( Functions.h(i="h", Delta_T=self.T-self.T_amb, d=self.system_config.EB_d_cell_stack, insulation_factor=self.system_config.EB_insulation_factor_cell_stack), name=self.name_pre+"h_cell_stack")
        self.h_gs = m.Intermediate( Functions.h(i="h", Delta_T=self.T-self.T_amb, d=self.system_config.EB_d_gs, insulation_factor=self.system_config.EB_insulation_factor_gs), name=self.name_pre+"h_gs")
        self.h_tube = m.Intermediate( Functions.h(i="h", Delta_T=self.T-self.T_amb, d=self.system_config.EB_d_tube, insulation_factor=self.system_config.EB_insulation_factor_tube), name=self.name_pre+"h_tube")
        #
        self.R_t_cell_stack = m.Intermediate( 1/(self.h_cell_stack * self.A_cell_stack), name=self.name_pre+"R_t_cell_stack")
        self.R_t_gs_A = m.Intermediate(1 / (self.h_gs * (self.A_gs)) , name=self.name_pre+"R_t_gs_A")
        self.R_t_gs_C = m.Intermediate(1 / (self.h_gs * (self.A_gs)) , name=self.name_pre+"R_t_gs_C")
        self.R_t_tube = m.Intermediate( 1 / (self.h_tube * (self.L_tube * self.A_tube)), name=self.name_pre+"R_t_tube")
        self.R_t_sys = m.Intermediate( 1 / ( (1/self.R_t_cell_stack) + (1/self.R_t_gs_A) + (1/self.R_t_gs_C) + (1/self.R_t_tube) ), name=self.name_pre+"R_t_sys")

    def Equations(self, m):
        m.Equation(self.C_t * self.T.dt() == self.Q_R - self.Q_loss + self.Q_hc - self.Qf_out_gas_sens - self.Qf_WD_warm_sens - self.Qf_out_gas_latent)
        m.Equation(self.Q_hc == self.S_hc * (-(self.Q_R - self.Q_loss - self.Qf_out_gas_sens - self.Qf_WD_warm_sens - self.Qf_out_gas_latent)) + self.EB_man_hc_power)
        m.Equation(self.Q_R == self.num_cells * (self.U_cell - self.U_tn) * self.I)
        m.Equation(self.Q_loss == (1/self.R_t_sys) * (self.T - self.T_amb))


class SystemClass:
    def __init__(self, name, system_config, m):
        self.name = name
        self.name_pre = self.name + "_"
        self.system_config = system_config
        self.Initialize(m=m)
    
    def Initialize(self, m):
        self.Stack = StackClass(name=self.name_pre+"Stack", system_config=self.system_config, m=m)
        self.GasSeparator = GasSeparatorClass(name=self.name_pre+"GasSeparator", system_config=self.system_config, m=m)
        self.Mixer = MixerClass(name=self.name_pre+"Mixer", system_config=self.system_config, m=m)
        #
        # System Variables
        self.S_mixing = m.Param(value=self.system_config.mixing, name=self.name_pre+"S_mixing") # -
        self.S_WD = m.Param(value=self.system_config.WD, name=self.name_pre+"S_WD") # -
        self.S_VE = m.Param(value=self.system_config.VE, name=self.name_pre+"S_VE") # -
        self.S_VE_dg = m.Param(value=self.system_config.VE, name=self.name_pre+"S_VE_dg") # -
        self.S_potential = self.system_config.potential # -
        self.S_electrolyte_concentration = self.system_config.electrolyte_concentration # -
        self.S_energy_balance = self.system_config.energy_balance # -
        if(not self.S_potential and self.S_energy_balance): self.S_energy_balance = 0 # -
        #
        self.GS_L_ini = self.system_config.GS_L_ini # -
        self.j = m.Param(value=self.system_config.j, name=self.name_pre+"j") # A/m^2
        self.num_cells = self.system_config.num_cells # -
        self.p = self.system_config.p # Pa
        self.T = m.Var(value=self.system_config.T_ini, lb=273.15, ub=373.15, name=self.name_pre+"T") # K
        self.VS_L_A = m.Param(value=self.system_config.VS_L_A, name=self.name_pre+"VS_L_A") # m^3/s
        self.VS_L_C = m.Param(value=self.system_config.VS_L_C, name=self.name_pre+"VS_L_C") # m^3/s
        self.w_KOH_ini = self.system_config.w_KOH_ini # -
        #
        self.eff_gas = None # -
        self.eff_gas_percent = None # %
        self.gas_purity_h2 = None # -
        self.gas_purity_h2_percent = None # %
        self.gas_purity_o2 = None # -
        self.gas_purity_o2_percent = None # %
        self.eff_w_KOH_C = None # -
        self.eff_w_KOH_C_percent = None # %
        #
        # System Variable Connection
        #
        # Intermediates
        # Current Density
        self.Stack.ElectrolysisCell.Anode.j = m.Intermediate(self.j, name=self.name_pre+"Stack_ElectrolysisCell_Anode_j")
        self.Stack.ElectrolysisCell.Cathode.j = m.Intermediate(self.j, name=self.name_pre+"Stack_ElectrolysisCell_Cathode_j")
        #
        # Electrolyte Volume Flow Rate
        self.Stack.Anode.VS_L_in = m.Intermediate(self.VS_L_A, name=self.name_pre+"Stack_Anode_VS_L_in")
        self.Stack.Cathode.VS_L_in = m.Intermediate(self.VS_L_C, name=self.name_pre+"Stack_Cathode_VS_L_in")
        self.GasSeparator.Anode.VS_L_out = m.Intermediate(self.VS_L_A, name=self.name_pre+"Stack_GasSeparator_Anode_VS_L_out")
        self.GasSeparator.Cathode.VS_L_out = m.Intermediate(self.VS_L_C, name=self.name_pre+"Stack_GasSeparator_Cathode_VS_L_out")
        #
        # Equations
        # Temperaure
        m.Equation(self.Stack.ElectrolysisCell.T == self.T)
        m.Equation(self.GasSeparator.T == self.T)
        #
        # Potential
        if(self.S_potential):
            self.Potential = PotentialClass(name=self.name_pre+"Potential", system_config=self.system_config, m=m)
            m.Equation(self.Potential.T == self.T)
            self.Potential.p = m.Intermediate(self.p, name=self.name_pre+"Potential_p")
            self.Potential.j = m.Intermediate(self.j, name=self.name_pre+"Potential_j")
            self.Potential.p_hc_p["Anode"] = m.Intermediate(self.Stack.ElectrolysisCell.Anode.p_out["O2"], name=self.name_pre+"Potential_p_hc_p_Anode")
            self.Potential.p_hc_p["Cathode"] = m.Intermediate(self.Stack.ElectrolysisCell.Cathode.p_out["H2"], name=self.name_pre+"Potential_p_hc_p_Cathode")
            self.Potential.p_hc_b["Anode"] = m.Intermediate(self.Stack.ElectrolysisCell.Anode.p_b, name=self.name_pre+"Potential_p_hc_b_Anode")
            self.Potential.p_hc_b["Cathode"] = m.Intermediate(self.Stack.ElectrolysisCell.Cathode.p_b, name=self.name_pre+"Potential_p_hc_b_Cathode")
            #
            self.eff_system = None # -
            self.eff_system_percent = None # -
            self.eff_objective = None # -
            self.eff_objective_percent = None # -
        
        # Electrolyte Concentration
        self.Stack.ElectrolysisCell.Anode.w_KOH.value = self.w_KOH_ini
        self.Stack.ElectrolysisCell.Cathode.w_KOH.value = self.w_KOH_ini
        self.GasSeparator.Anode.w_KOH.value = self.w_KOH_ini
        self.GasSeparator.Cathode.w_KOH.value = self.w_KOH_ini
        self.Stack.ElectrolysisCell.Anode.VS_L_out.value = Functions.initial_value(self.system_config.VS_L_A) / self.Stack.num_cells
        self.Stack.ElectrolysisCell.Cathode.VS_L_out.value = Functions.initial_value(self.system_config.VS_L_C) / self.Stack.num_cells
        self.GasSeparator.Anode.VS_L_in.value = Functions.initial_value(self.system_config.VS_L_A)
        self.GasSeparator.Cathode.VS_L_in.value = Functions.initial_value(self.system_config.VS_L_C)
        self.GasSeparator.Anode.eps.value = 1 - self.GS_L_ini
        self.GasSeparator.Cathode.eps.value = 1 - self.GS_L_ini
        #
        if(self.S_electrolyte_concentration):
            self.ElectrolyteConcentration = ElectrolyteConcentrationClass(name=self.name_pre+"ElectrolyteConcentration", system_config=self.system_config, m=m)
        #
        if(self.S_energy_balance):
            self.EnergyBalance = EnergyBalanceClass(name=self.name_pre+"EnergyBalance", system_config=self.system_config, m=m)
            m.Equation(self.T == self.EnergyBalance.T)
        else:
            m.Equation(self.T == self.system_config.T_ini)      
        
    def Initials(self, m):
        self.Stack.Initials(m=m)
        self.GasSeparator.Initials(m=m)
    
    def Intermediates(self, m):
        self.Mixer.VS_L_A = m.Intermediate(self.GasSeparator.Anode.VS_L_out, name=self.name_pre+"Mixer_VS_L_A")
        self.Mixer.VS_L_C = m.Intermediate(self.GasSeparator.Cathode.VS_L_out, name=self.name_pre+"Mixer_VS_L_C")
        self.Stack.Intermediates(m=m)
        #
        for ii in components:
            self.GasSeparator.Anode.c_in[ii] = m.Intermediate(self.Stack.Anode.c_out[ii], name=self.name_pre+"GasSeparator_Anode_c_in_" + str(ii))
            self.GasSeparator.Cathode.c_in[ii] = m.Intermediate(self.Stack.Cathode.c_out[ii], name=self.name_pre+"GasSeparator_Cathode_c_in_" + str(ii))
            self.GasSeparator.Anode.p_in[ii] = m.Intermediate(self.Stack.Anode.p_out[ii], name=self.name_pre+"GasSeparator_Anode_p_in_" + str(ii))
            self.GasSeparator.Cathode.p_in[ii] = m.Intermediate(self.Stack.Cathode.p_out[ii], name=self.name_pre+"GasSeparator_Cathode_p_in_" + str(ii))
            #
            self.Mixer.c_in_A[ii] = m.Intermediate(self.GasSeparator.Anode.c_out[ii], name=self.name_pre+"Mixer_c_in_A_" + str(ii)) 
            self.Mixer.c_in_C[ii] = m.Intermediate(self.GasSeparator.Cathode.c_out[ii], name=self.name_pre+"Mixer_c_in_C_" + str(ii)) 
            #
            self.Stack.Anode.c_in[ii] = m.Intermediate( (self.S_mixing * self.Mixer.c_out_mix[ii]) + ((1-self.S_mixing) * self.GasSeparator.Anode.c_out[ii]), name=self.name_pre+"Stack_Anode_c_in_" + str(ii))
            self.Stack.Cathode.c_in[ii] = m.Intermediate( (self.S_mixing * self.Mixer.c_out_mix[ii]) + ((1-self.S_mixing) * self.GasSeparator.Cathode.c_out[ii]), name=self.name_pre+"Stack_Cathode_c_in_" + str(ii))
            self.Stack.ElectrolysisCell.Anode.c_in[ii] = m.Intermediate(self.Stack.Anode.c_in[ii], name=self.name_pre+"Stack_ElectrolysisCell_Anode_c_in_" + str(ii))
            self.Stack.ElectrolysisCell.Cathode.c_in[ii] = m.Intermediate(self.Stack.Cathode.c_in[ii], name=self.name_pre+"Stack_ElectrolysisCell_Cathode_c_in_" + str(ii))
        #
        self.GasSeparator.Intermediates(m=m)
        self.Mixer.Intermediates(m=m)
        #
        self.eff_gas = m.Intermediate((1 - self.GasSeparator.Anode.x["H2"]) * (1 - self.GasSeparator.Cathode.x["O2"]), name=self.name_pre+"eff_gas")
        self.eff_gas_percent = m.Intermediate(self.eff_gas * 100, name=self.name_pre+"eff_gas_percent")
        self.gas_purity_h2 = m.Intermediate(self.GasSeparator.Cathode.x["H2"], name=self.name_pre+"gas_purity_h2")
        self.gas_purity_o2 = m.Intermediate(self.GasSeparator.Anode.x["O2"], name=self.name_pre+"gas_purity_o2")
        self.gas_purity_h2_percent = m.Intermediate(self.gas_purity_h2 * 100, name=self.name_pre+"gas_purity_h2_percent")
        self.gas_purity_o2_percent = m.Intermediate(self.gas_purity_o2 * 100, name=self.name_pre+"gas_purity_o2_percent")
        self.eff_w_KOH_C = m.Intermediate(1 - ((self.Stack.ElectrolysisCell.Cathode.w_KOH - self.w_KOH_ini)/(self.w_KOH_ini)), name=self.name_pre+"eff_w_KOH_C")
        self.eff_w_KOH_C_percent = m.Intermediate(self.eff_w_KOH_C * 100, name=self.name_pre+"eff_w_KOH_C_percent")
        #
        # Potential
        if(self.S_potential):
            if(self.S_electrolyte_concentration):
                self.Potential.w_KOH['Anode'] = m.Intermediate(self.ElectrolyteConcentration.Anode.w_KOH, name=self.name_pre+"Potential_w_KOH_Anode")
                self.Potential.w_KOH['Cathode'] = m.Intermediate(self.ElectrolyteConcentration.Cathode.w_KOH, name=self.name_pre+"Potential_w_KOH_Cathode")
            else:
                self.Potential.w_KOH['Anode'] = m.Intermediate(self.w_KOH_ini, name=self.name_pre+"Potential_w_KOH_Anode")
                self.Potential.w_KOH['Cathode'] = m.Intermediate(self.w_KOH_ini, name=self.name_pre+"Potential_w_KOH_Cathode")
            self.Potential.p_sat["Anode"] = m.Intermediate(self.Stack.ElectrolysisCell.Anode.p_sat, name=self.name_pre+"Potential_p_sat_Anode")
            self.Potential.p_sat["Cathode"] = m.Intermediate(self.Stack.ElectrolysisCell.Cathode.p_sat, name=self.name_pre+"Potential_p_sat_Cathode")
            #
            self.Potential.eps["Anode"] = m.Intermediate(self.Stack.ElectrolysisCell.Anode.eps, name=self.name_pre+"Potential_eps_Anode")
            self.Potential.eps["Cathode"] = m.Intermediate(self.Stack.ElectrolysisCell.Cathode.eps, name=self.name_pre+"Potential_eps_Cathode")
            self.Potential.Intermediates(m=m)
            #
            self.eff_system = m.Intermediate(self.Potential.eff_cell * self.eff_gas, name=self.name_pre+"eff_system")
            self.eff_system_percent = m.Intermediate(self.eff_system * 100, name=self.name_pre+"eff_system_percent")
            self.eff_objective = m.Intermediate((1 - self.GasSeparator.Anode.x["H2"]) * self.eff_w_KOH_C**(1/6), name=self.name_pre+"eff_objective")
            self.eff_objective_percent = m.Intermediate(self.eff_objective * 100, name=self.name_pre+"eff_objective_percent")
        #    
        # Electrolyte Concentration
        if(self.S_electrolyte_concentration):
            #
            self.ElectrolyteConcentration.j = m.Intermediate(self.j, name=self.name_pre+"ElectrolyteConcentration_j")
            self.ElectrolyteConcentration.T = m.Intermediate(self.T, name=self.name_pre+"ElectrolyteConcentration_T")
            self.ElectrolyteConcentration.S_WD = m.Intermediate(self.S_WD, name=self.name_pre+"ElectrolyteConcentration_S_WD")
            self.ElectrolyteConcentration.S_VE = m.Intermediate(self.S_VE, name=self.name_pre+"ElectrolyteConcentration_S_VE")
            self.ElectrolyteConcentration.S_VE_dg = m.Intermediate(self.S_VE_dg, name=self.name_pre+"ElectrolyteConcentration_S_VE_dg")
            self.ElectrolyteConcentration.S_mixing = m.Intermediate(self.S_mixing, name=self.name_pre+"ElectrolyteConcentration_S_mixing")
            #
            self.ElectrolyteConcentration.Anode.VS_L_in = m.Intermediate(self.VS_L_A / self.num_cells, name=self.name_pre+"ElectrolyteConcentration_Anode_VS_L_in")
            self.ElectrolyteConcentration.Cathode.VS_L_in = m.Intermediate(self.VS_L_C / self.num_cells, name=self.name_pre+"ElectrolyteConcentration_Cathode_VS_L_in")
            self.ElectrolyteConcentration.GasSeparatorAnode.VS_L_out = m.Intermediate(self.VS_L_A, name=self.name_pre+"ElectrolyteConcentration_GasSeparatorAnode_VS_L_out")
            self.ElectrolyteConcentration.GasSeparatorCathode.VS_L_out = m.Intermediate(self.VS_L_C, name=self.name_pre+"ElectrolyteConcentration_GasSeparatorCathode_VS_L_out")
            #
            self.ElectrolyteConcentration.Anode.eps = m.Intermediate(self.Stack.ElectrolysisCell.Anode.eps, name=self.name_pre+"ElectrolyteConcentration_Anode_eps")
            self.ElectrolyteConcentration.Cathode.eps = m.Intermediate(self.Stack.ElectrolysisCell.Cathode.eps, name=self.name_pre+"ElectrolyteConcentration_Cathode_eps")
            #
            for ii in components:
                self.ElectrolyteConcentration.GasSeparatorAnode.p_out[ii] = m.Intermediate(self.GasSeparator.Anode.p_out[ii], name=self.name_pre+"ElectrolyteConcentration_GasSeparatorAnode_p_out_" + str(ii))
                self.ElectrolyteConcentration.GasSeparatorCathode.p_out[ii] = m.Intermediate(self.GasSeparator.Cathode.p_out[ii], name=self.name_pre+"ElectrolyteConcentration_GasSeparatorCathode_p_out_" + str(ii))
            self.ElectrolyteConcentration.GasSeparatorAnode.p_sat = m.Intermediate(self.GasSeparator.Anode.p_sat, name=self.name_pre+"ElectrolyteConcentration_GasSeparatorAnode_p_sat")
            self.ElectrolyteConcentration.GasSeparatorCathode.p_sat = m.Intermediate(self.GasSeparator.Cathode.p_sat, name=self.name_pre+"ElectrolyteConcentration_GasSeparatorCathode_p_sat")
            self.ElectrolyteConcentration.GasSeparatorAnode.VS_G_out = m.Intermediate(self.GasSeparator.Anode.VS_G_out, name=self.name_pre+"ElectrolyteConcentration_GasSeparatorAnode_VS_G_out")
            self.ElectrolyteConcentration.GasSeparatorCathode.VS_G_out = m.Intermediate(self.GasSeparator.Cathode.VS_G_out, name=self.name_pre+"ElectrolyteConcentration_GasSeparatorCathode_VS_G_out")
            #
            if(self.S_energy_balance):
                self.ElectrolyteConcentration.GasSeparatorAnode.T_gas = m.Intermediate(self.EnergyBalance.T_gas, name=self.name_pre+"ElectrolyteConcentration_GasSeparatorAnode_T_gas")
                self.ElectrolyteConcentration.GasSeparatorCathode.T_gas = m.Intermediate(self.EnergyBalance.T_gas, name=self.name_pre+"ElectrolyteConcentration_GasSeparatorCathode_T_gas")
            else:
                self.ElectrolyteConcentration.GasSeparatorAnode.T_gas = m.Intermediate(self.system_config.EB_T_amb, name=self.name_pre+"ElectrolyteConcentration_GasSeparatorAnode_T_gas")
                self.ElectrolyteConcentration.GasSeparatorCathode.T_gas = m.Intermediate(self.system_config.EB_T_amb, name=self.name_pre+"ElectrolyteConcentration_GasSeparatorCathode_T_gas")
            self.ElectrolyteConcentration.Intermediates(m=m)
        #
        # EnergyBalance
        if(self.S_energy_balance):
            self.EnergyBalance.U_cell = m.Intermediate(self.Potential.U_cell, name=self.name_pre+"EnergyBalance_U_cell")
            self.EnergyBalance.U_tn = m.Intermediate(self.Potential.U_tn, name=self.name_pre+"EnergyBalance_U_tn")
            self.EnergyBalance.I = m.Intermediate(self.Potential.I, name=self.name_pre+"EnergyBalance_I")
            if(self.S_electrolyte_concentration):
                self.EnergyBalance.w_KOH = m.Intermediate(self.ElectrolyteConcentration.w_KOH_sys, name=self.name_pre+"EnergyBalance_w_KOH")
                #
                self.EnergyBalance.mf_gas_H2O_vapor["Anode"] = m.Intermediate(self.ElectrolyteConcentration.GasSeparatorAnode.mf_G_H2O, name=self.name_pre+"EnergyBalance_mf_gas_H2O_vapor_Anode") # INFO: Assumption of complete H2O condensation
                self.EnergyBalance.mf_gas_H2O_vapor["Cathode"] = m.Intermediate(self.ElectrolyteConcentration.GasSeparatorCathode.mf_G_H2O, name=self.name_pre+"EnergyBalance_mf_gas_H2O_vapor_Cathode") # INFO: Assumption of complete H2O condensation
            else:
                self.EnergyBalance.w_KOH = m.Intermediate(self.w_KOH_ini, name=self.name_pre+"EnergyBalance_w_KOH_ini")
                #
                self.EnergyBalance.mf_gas_H2O_vapor["Anode"] = m.Intermediate(0, name=self.name_pre+"EnergyBalance_mf_gas_H2O_vapor_Anode")
                self.EnergyBalance.mf_gas_H2O_vapor["Cathode"] = m.Intermediate(0, name=self.name_pre+"EnergyBalance_mf_gas_H2O_vapor_Cathode")
            #
            self.EnergyBalance.Vf_gas["Anode"] = m.Intermediate(self.GasSeparator.Anode.VS_G_out, name=self.name_pre+"EnergyBalance_Vf_gas_Anode")
            self.EnergyBalance.Vf_gas["Cathode"] = m.Intermediate(self.GasSeparator.Cathode.VS_G_out, name=self.name_pre+"EnergyBalance_Vf_gas_Cathode")
            self.EnergyBalance.p_abs["Anode"] = m.Intermediate(self.GasSeparator.Anode.p_abs, name=self.name_pre+"EnergyBalance_p_abs_Anode")
            self.EnergyBalance.p_abs["Cathode"] = m.Intermediate(self.GasSeparator.Cathode.p_abs, name=self.name_pre+"EnergyBalance_p_abs_Cathode")
            self.EnergyBalance.M_gas["Anode"] = m.Intermediate(self.GasSeparator.Anode.M_gas, name=self.name_pre+"EnergyBalance_M_gas_Anode")
            self.EnergyBalance.M_gas["Cathode"] = m.Intermediate(self.GasSeparator.Cathode.M_gas, name=self.name_pre+"EnergyBalance_M_gas_Cathode")
            for ii in components:
                self.EnergyBalance.x_Anode[ii] = m.Intermediate(self.GasSeparator.Anode.x[ii], name=self.name_pre+"EnergyBalance_x_Anode_"+str(ii))
                self.EnergyBalance.x_Cathode[ii]= m.Intermediate(self.GasSeparator.Cathode.x[ii], name=self.name_pre+"EnergyBalance_x_Cathode_"+str(ii))
                self.EnergyBalance.x_wet_Anode[ii] = m.Intermediate(self.GasSeparator.Anode.x_wet[ii], name=self.name_pre+"EnergyBalance_x_wet_Anode_"+str(ii))
                self.EnergyBalance.x_wet_Cathode[ii]= m.Intermediate(self.GasSeparator.Cathode.x_wet[ii], name=self.name_pre+"EnergyBalance_x_wet_Cathode_"+str(ii))
            #
            self.EnergyBalance.m_El = m.Intermediate(self.EnergyBalance.num_cells * (self.Stack.ElectrolysisCell.Anode.V_L * self.Stack.ElectrolysisCell.Anode.dens_KOH + self.Stack.ElectrolysisCell.Cathode.V_L * self.Stack.ElectrolysisCell.Cathode.dens_KOH) + self.GasSeparator.Anode.V_L * self.GasSeparator.Anode.dens_KOH + self.GasSeparator.Cathode.V_L * self.GasSeparator.Cathode.dens_KOH, name=self.name_pre+"EnergyBalance_m_El")
            #
            self.EnergyBalance.Intermediates(m=m)
               
        
    def Equations(self, m):
        self.Mixer.Equations(m=m)
        self.Stack.Equations(m=m)
        self.GasSeparator.Equations(m=m)
        #
        m.Equation(self.GasSeparator.Anode.VS_G_in == self.Stack.Anode.VS_G_out)
        m.Equation(self.GasSeparator.Cathode.VS_G_in == self.Stack.Cathode.VS_G_out)
        #
        # Potential
        if(self.S_potential):
            self.Potential.Equations(m=m)
        #
        # Electrolyte Concentration
        if(self.S_electrolyte_concentration):
            self.ElectrolyteConcentration.Equations(m=m)
            #
            m.Equation(self.Stack.ElectrolysisCell.Anode.w_KOH == self.ElectrolyteConcentration.Anode.w_KOH)
            m.Equation(self.Stack.ElectrolysisCell.Cathode.w_KOH == self.ElectrolyteConcentration.Cathode.w_KOH)
            #
            m.Equation(self.GasSeparator.Anode.w_KOH == self.ElectrolyteConcentration.GasSeparatorAnode.w_KOH)
            m.Equation(self.GasSeparator.Cathode.w_KOH == self.ElectrolyteConcentration.GasSeparatorCathode.w_KOH)
            #
            m.Equation(self.Stack.ElectrolysisCell.Anode.VS_L_out == self.ElectrolyteConcentration.Anode.VS_L_out)
            m.Equation(self.Stack.ElectrolysisCell.Cathode.VS_L_out == self.ElectrolyteConcentration.Cathode.VS_L_out)
            #
            m.Equation(self.GasSeparator.Anode.VS_L_in == self.ElectrolyteConcentration.GasSeparatorAnode.VS_L_in)
            m.Equation(self.GasSeparator.Cathode.VS_L_in == self.ElectrolyteConcentration.GasSeparatorCathode.VS_L_in)
            m.Equation(self.GasSeparator.Anode.eps == self.ElectrolyteConcentration.GasSeparatorAnode.eps)
            m.Equation(self.GasSeparator.Cathode.eps == self.ElectrolyteConcentration.GasSeparatorCathode.eps)
            #
            for ii in components:
                m.Equation(self.Stack.ElectrolysisCell.Anode.ns_ve_sep[ii] == self.ElectrolyteConcentration.S_VE_dg * self.ElectrolyteConcentration.Anode.Vf_ve * self.Stack.ElectrolysisCell.Anode.fS[ii] * self.Stack.ElectrolysisCell.Anode.c_out[ii])
                m.Equation(self.Stack.ElectrolysisCell.Cathode.ns_ve_sep[ii] == -self.Stack.ElectrolysisCell.Anode.ns_ve_sep[ii])
                m.Equation(self.GasSeparator.Anode.ns_ve_mix[ii] == self.ElectrolyteConcentration.S_VE_dg * self.ElectrolyteConcentration.GasSeparatorAnode.Vf_ve * self.GasSeparator.Anode.c_out[ii])
                m.Equation(self.GasSeparator.Cathode.ns_ve_mix[ii] == -self.GasSeparator.Anode.ns_ve_mix[ii])
                #
                m.Equation(self.Stack.ElectrolysisCell.Cathode.ns_drag[ii] == self.ElectrolyteConcentration.Cathode.Vf_drag * self.Stack.ElectrolysisCell.Cathode.fS[ii] * self.Stack.ElectrolysisCell.Cathode.c_out[ii])
                m.Equation(self.Stack.ElectrolysisCell.Anode.ns_drag[ii] == -self.Stack.ElectrolysisCell.Cathode.ns_drag[ii])        
        else:
            m.Equation(self.Stack.ElectrolysisCell.Anode.w_KOH == self.w_KOH_ini)
            m.Equation(self.Stack.ElectrolysisCell.Cathode.w_KOH == self.w_KOH_ini)
            m.Equation(self.GasSeparator.Anode.w_KOH == self.w_KOH_ini)
            m.Equation(self.GasSeparator.Cathode.w_KOH == self.w_KOH_ini)
            m.Equation(self.Stack.ElectrolysisCell.Anode.VS_L_out == self.VS_L_A / self.Stack.num_cells)
            m.Equation(self.Stack.ElectrolysisCell.Cathode.VS_L_out == self.VS_L_C / self.Stack.num_cells)
            m.Equation(self.GasSeparator.Anode.VS_L_in == self.VS_L_A)
            m.Equation(self.GasSeparator.Cathode.VS_L_in == self.VS_L_C)
            m.Equation(self.GasSeparator.Anode.eps == 1 - self.GS_L_ini)
            m.Equation(self.GasSeparator.Cathode.eps == 1 - self.GS_L_ini)
            for ii in components:
                m.Equation(self.Stack.ElectrolysisCell.Anode.ns_ve_sep[ii] == (1-self.S_mixing) * (-((1 * 2 * self.Stack.ElectrolysisCell.Cathode.n_R["H2"] / self.Stack.ElectrolysisCell.Cathode.dens_H2O) + (self.Stack.ElectrolysisCell.Anode.n_R["O2"]*2 / (2 * self.Stack.ElectrolysisCell.Anode.dens_H2O))) * self.Stack.ElectrolysisCell.Anode.M_H2O * self.Stack.ElectrolysisCell.Anode.fS[ii] * self.Stack.ElectrolysisCell.Anode.c_out[ii]))
                m.Equation(self.Stack.ElectrolysisCell.Cathode.ns_ve_sep[ii] == -self.Stack.ElectrolysisCell.Anode.ns_ve_sep[ii])
                m.Equation(self.GasSeparator.Anode.ns_ve_mix[ii] == self.S_mixing * (-self.Stack.num_cells*((1 * 2 * self.Stack.ElectrolysisCell.Cathode.n_R["H2"] / self.Stack.ElectrolysisCell.Cathode.dens_H2O) + (self.Stack.ElectrolysisCell.Anode.n_R["O2"]*2 / (2 * self.Stack.ElectrolysisCell.Anode.dens_H2O))) * self.Stack.ElectrolysisCell.Anode.M_H2O * self.GasSeparator.Anode.c_out[ii]))
                m.Equation(self.GasSeparator.Cathode.ns_ve_mix[ii] == -self.GasSeparator.Anode.ns_ve_mix[ii])
                #
                m.Equation(self.Stack.ElectrolysisCell.Cathode.ns_drag[ii] == -(1 * 2 * self.Stack.ElectrolysisCell.Cathode.n_R["H2"] * self.Stack.ElectrolysisCell.Cathode.M_H2O / self.Stack.ElectrolysisCell.Cathode.dens_H2O) * self.Stack.ElectrolysisCell.Cathode.fS[ii] * self.Stack.ElectrolysisCell.Cathode.c_out[ii])
                m.Equation(self.Stack.ElectrolysisCell.Anode.ns_drag[ii] == -self.Stack.ElectrolysisCell.Cathode.ns_drag[ii])
        #
        # EnergyBalance
        if(self.S_energy_balance):
            self.EnergyBalance.Equations(m=m)
        

class ProcessClass:
    def __init__(self, name, process_config):
        self.name = name
        self.name_pre = self.name + "_"
        self.process_config = process_config
        if(self.process_config.remote == 1):
            self.m = GEKKO(remote = True, server=self.process_config.remote_server, name=self.process_config.remote_name+"_"+self.name)
        else:
            self.m = GEKKO(remote = False)
        m = self.m
        if (self.process_config.solve_mode == 1):
            self.m.time = self.process_config.m_time # s
        #
        # Parameter Estimation
        if (self.process_config.pe_H2_in_O2):
            # d_b
            self.C_d_b_H2_p1 = m.FV(value=3.37, lb=3, ub=4, name=self.name_pre + "C_db_H2_p1")
            self.C_d_b_H2_p21 = m.FV(value=0.2, lb=0.1, ub=0.2, name=self.name_pre + "C_db_H2_p21")
            self.C_d_b_H2_p22 = m.FV(value=-0.25, lb=-0.45, ub=-0.15, name=self.name_pre + "C_db_H2_p22")
            self.C_d_b_H2_p31 = m.FV(value=0, lb=0, ub=1, name=self.name_pre + "C_db_H2_p31")
            self.C_d_b_H2_p32 = m.FV(value=0, lb=0, ub=1, name=self.name_pre + "C_db_H2_p32")
            self.C_d_b_H2_p1.STATUS = 1
            self.C_d_b_H2_p21.STATUS = 0
            self.C_d_b_H2_p22.STATUS = 0
            self.C_d_b_H2_p31.STATUS = 0 
            self.C_d_b_H2_p32.STATUS = 0
            #
            # fS
            self.C_fS_H2_in_O2_p1 = m.FV(value=1, lb=1, ub=1, name=self.name_pre + "C_fS_H2_in_O2_p1")
            self.C_fS_H2_in_O2_p21 = m.FV(value=1, lb=0, ub=1, name=self.name_pre + "C_fS_H2_in_O2_p21")
            self.C_fS_H2_in_O2_p22 = m.FV(value=0.21, lb=0, ub=0.35, name=self.name_pre + "C_fS_H2_in_O2_p22")
            self.C_fS_H2_in_O2_p31 = m.FV(value=1, lb=0, ub=1, name=self.name_pre + "C_fS_H2_in_O2_p31")
            self.C_fS_H2_in_O2_p32 = m.FV(value=-0.54, lb=-0.8, ub=0, name=self.name_pre + "C_fS_H2_in_O2_p32")
            self.C_fS_H2_in_O2_p1.STATUS = 0
            self.C_fS_H2_in_O2_p21.STATUS = 0
            self.C_fS_H2_in_O2_p22.STATUS = 1
            self.C_fS_H2_in_O2_p31.STATUS = 0
            self.C_fS_H2_in_O2_p32.STATUS = 1
        #
        if(self.process_config.solve_mode == 1 and (process_config.pe_H2_in_O2 == 1 or process_config.plot_vd == 1)):
            self.est_data_H2_in_O2 = [None for ss in range(self.process_config.num_system)]
            self.est_data_H2_in_O2_valid = [None for ss in range(self.process_config.num_system)]
            for ss in range(self.process_config.num_system):
                self.est_data_H2_in_O2[ss] = m.Param(value=self.process_config.system_config[ss].est_data_H2_in_O2["H2_in_O2_zero"].values, name=self.name_pre + "est_data_H2_in_O2_"+str(ss))
                self.est_data_H2_in_O2_valid[ss] = m.Param(value=self.process_config.system_config[ss].est_data_H2_in_O2["valid"].values, name=self.name_pre + "est_data_H2_in_O2_valid_"+str(ss))
        #
        if(self.process_config.solve_mode == 1 and (process_config.pe_w_KOH == 1)):
            self.est_data_w_KOH_A = [None for ss in range(self.process_config.num_system)]
            self.est_data_w_KOH_C = [None for ss in range(self.process_config.num_system)]
            self.est_data_w_KOH_valid = [None for ss in range(self.process_config.num_system)]
            for ss in range(self.process_config.num_system):
                self.est_data_w_KOH_A[ss] = m.Param(value=self.process_config.system_config[ss].est_data_w_KOH["w_KOH_A_zero"].values, name=self.name_pre + "est_data_w_KOH_A_"+str(ss))
                self.est_data_w_KOH_C[ss] = m.Param(value=self.process_config.system_config[ss].est_data_w_KOH["w_KOH_C_zero"].values, name=self.name_pre + "est_data_w_KOH_C_"+str(ss))
                self.est_data_w_KOH_valid[ss] = m.Param(value=self.process_config.system_config[ss].est_data_w_KOH["valid"].values, name=self.name_pre + "est_data_w_KOH_valid_"+str(ss))
        #
        if(self.process_config.solve_mode == 1 and (process_config.pe_energy == 1 or process_config.dl_T_amb == 1)):
            self.est_data_T_sys = [None for ss in range(self.process_config.num_system)]
            self.est_data_T_sys_valid = [None for ss in range(self.process_config.num_system)]
            for ss in range(self.process_config.num_system):
                self.est_data_T_sys[ss] = m.Param(value=self.process_config.system_config[ss].est_data_T_sys["T_sys_zero"].add(273.15).values.tolist(), name=self.name_pre + "est_data_T_sys_"+str(ss)) 
                self.est_data_T_sys_valid[ss] = m.Param(value=self.process_config.system_config[ss].est_data_T_sys["valid"].values.tolist(), name=self.name_pre + "est_data_T_sys_valid_"+str(ss))
        #
        if(self.process_config.solve_mode == 1 and (process_config.pe_O2_in_H2 == 1)):
            self.est_data_O2_in_H2 = [None for ss in range(self.process_config.num_system)]
            self.est_data_O2_in_H2_valid = [None for ss in range(self.process_config.num_system)]
            for ss in range(self.process_config.num_system):
                self.est_data_O2_in_H2[ss] = m.Param(value=self.process_config.system_config[ss].est_data_O2_in_H2["O2_in_H2_zero"].values.tolist(), name=self.name_pre + "est_data_O2_in_H2"+str(ss)) 
                self.est_data_O2_in_H2_valid[ss] = m.Param(value=self.process_config.system_config[ss].est_data_O2_in_H2["valid"].values.tolist(), name=self.name_pre + "est_data_O2_in_H2_valid"+str(ss))
        #
        if (self.process_config.pe_potential):
            self.C_j_0_A_p1 = m.FV(value=1, lb=1, ub=1.5, name=self.name_pre + "C_j_0_A_p1")
            self.C_j_0_A_p1.STATUS = 0
            self.C_j_0_A_p2 = m.FV(value=100, lb=60, ub=130, name=self.name_pre + "C_j_0_A_p2")
            self.C_j_0_A_p2.STATUS = 1
            self.C_j_0_A_p3 = m.FV(value=6e-5, lb=1e-8, ub=1e-4, name=self.name_pre + "C_j_0_A_p3")
            self.C_j_0_A_p3.STATUS = 1
            #
            self.C_j_0_C_p1 = m.FV(value=1, lb=1, ub=1.5, name=self.name_pre + "C_j_0_C_p1")
            self.C_j_0_C_p1.STATUS = 0
            self.C_j_0_C_p2 = m.FV(value=70, lb=40, ub=90, name=self.name_pre + "C_j_0_C_p2")
            self.C_j_0_C_p2.STATUS = 1
            self.C_j_0_C_p3 = m.FV(value=1, lb=1e-1, ub=50, name=self.name_pre + "C_j_0_C_p3")
            self.C_j_0_C_p3.STATUS = 1
            #
            self.C_alpha_A_p1 = m.FV(value=0.35, lb=0.25, ub=0.5, name=self.name_pre + "C_alpha_A_p1")
            self.C_alpha_A_p1.STATUS = 1
            self.C_alpha_A_p2 = m.FV(value=0, lb=0, ub=5e-4, name=self.name_pre + "C_alpha_A_p2")
            self.C_alpha_A_p2.STATUS = 0
            self.C_alpha_A_p3 = m.FV(value=0, lb=0, ub=1e-4, name=self.name_pre + "C_alpha_A_p3")
            self.C_alpha_A_p3.STATUS = 0
            #
            self.C_alpha_C_p1 = m.FV(value=0.4, lb=0.25, ub=0.5, name=self.name_pre + "C_alpha_C_p1")
            self.C_alpha_C_p1.STATUS = 1
            self.C_alpha_C_p2 = m.FV(value=0, lb=0, ub=1e-2, name=self.name_pre + "C_alpha_C_p2")
            self.C_alpha_C_p2.STATUS = 0
            self.C_alpha_C_p3 = m.FV(value=0, lb=0, ub=1e-4, name=self.name_pre + "C_alpha_C_p3")
            self.C_alpha_C_p3.STATUS = 0
            #
            self.C_d_eg = m.FV(value=250-6, lb=250e-6, ub=1000e-6, name=self.name_pre + "C_d_eg")
            self.C_d_eg.STATUS = 1
            self.C_R_add = m.FV(value=4.18e-3, lb=1e-3, ub=10e-3, name=self.name_pre + "C_R_add")
            self.C_R_add.STATUS = 1
        #
        if (self.process_config.pe_w_KOH):
            self.C_EC_f_W_p1 = m.FV(value=1, lb=0, ub=4, name=self.name_pre + "C_EC_f_W_p1")
            self.C_EC_f_W_p1.STATUS = 1
            self.C_EC_f_D_p1 = m.FV(value=0.3538, lb=0.1, ub=1, name=self.name_pre + "C_EC_f_D_p1")
            self.C_EC_f_D_p1.STATUS = 1
        #
        if (self.process_config.pe_energy):
            self.C_EB_L_tube_p1 = m.FV(value=5, lb=0, name=self.name_pre + "C_EB_L_tube_p1")
            self.C_EB_L_tube_p1.STATUS = 1
        #
        if (self.process_config.pe_O2_in_H2 == 1):
            # d_b
            self.C_d_b_O2_p1 = m.FV(value=1.2, lb=0.4, ub=4, name=self.name_pre + "C_db_O2_p1")
            self.C_d_b_O2_p21 = m.FV(value=0.2, lb=0.1, ub=0.2, name=self.name_pre + "C_db_O2_p21")
            self.C_d_b_O2_p22 = m.FV(value=-0.25, lb=-0.45, ub=-0.15, name=self.name_pre + "C_db_O2_p22")
            self.C_d_b_O2_p31 = m.FV(value=1, lb=0, ub=1, name=self.name_pre + "C_db_O2_p31")
            self.C_d_b_O2_p32 = m.FV(value=0, lb=-1, ub=0, name=self.name_pre + "C_db_O2_p32")
            self.C_d_b_O2_p1.STATUS = 1
            self.C_d_b_O2_p21.STATUS = 0
            self.C_d_b_O2_p22.STATUS = 0
            self.C_d_b_O2_p31.STATUS = 0
            self.C_d_b_O2_p32.STATUS = 0
            #
            # fS
            self.C_fS_O2_in_H2_p1 = m.FV(value=1, lb=1, ub=1, name=self.name_pre + "C_fS_O2_in_H2_p1") # 1
            self.C_fS_O2_in_H2_p21 = m.FV(value=1, lb=0, ub=1, name=self.name_pre + "C_fS_O2_in_H2_p21") # 1
            self.C_fS_O2_in_H2_p22 = m.FV(value=0.11, lb=0.05, ub=0.2, name=self.name_pre + "C_fS_O2_in_H2_p22")
            self.C_fS_O2_in_H2_p31 = m.FV(value=1, lb=0, ub=1, name=self.name_pre + "C_fS_O2_in_H2_p31") # 1
            self.C_fS_O2_in_H2_p32 = m.FV(value=-0.40, lb=-0.6, ub=0, name=self.name_pre + "C_fS_O2_in_H2_p32")
            self.C_fS_O2_in_H2_p1.STATUS = 0
            self.C_fS_O2_in_H2_p21.STATUS = 0
            self.C_fS_O2_in_H2_p22.STATUS = 1
            self.C_fS_O2_in_H2_p31.STATUS = 0
            self.C_fS_O2_in_H2_p32.STATUS = 1
        #
        # System Initialization
        self.System = [None for ss in range(self.process_config.num_system)]
        for ss in range(self.process_config.num_system):
            self.System[ss] = SystemClass(name=self.name_pre+"System_" + str(ss), system_config=self.process_config.system_config[ss], m=self.m)
            self.System[ss].Initials(m=self.m)
            #
            # Parameter Estimation
            # fS
            self.System[ss].Stack.ElectrolysisCell.Anode.fS["H2"] = m.Intermediate(1, name=self.name_pre+"System_"+str(ss)+"_ElectrolysisCell_Anode_fS_H2")
            self.System[ss].Stack.ElectrolysisCell.Cathode.fS["O2"] = m.Intermediate(1, name=self.name_pre+"System_"+str(ss)+"_ElectrolysisCell_Cathode_fS_O2")
            if(self.process_config.pe_H2_in_O2 == 1):
                self.System[ss].Stack.ElectrolysisCell.Cathode.fS["H2"] = m.Intermediate(Functions.fS(i=[self.C_fS_H2_in_O2_p22, self.C_fS_H2_in_O2_p32], j=self.System[ss].Stack.ElectrolysisCell.Cathode.j, p_abs=self.System[ss].Stack.ElectrolysisCell.Cathode.p_abs), name=self.name_pre+"System_"+str(ss)+"_ElectrolysisCell_Cathode_fS_H2")
            else:
                self.System[ss].Stack.ElectrolysisCell.Cathode.fS["H2"] = m.Intermediate(Functions.fS(i="H2", j=self.System[ss].Stack.ElectrolysisCell.Cathode.j, p_abs=self.System[ss].Stack.ElectrolysisCell.Cathode.p_abs), name=self.name_pre+"System_"+str(ss)+"_ElectrolysisCell_Cathode_fS_H2")
            #
            if(self.process_config.pe_O2_in_H2 == 1):
                self.System[ss].Stack.ElectrolysisCell.Anode.fS["O2"] = m.Intermediate(Functions.fS(i=[self.C_fS_O2_in_H2_p22, self.C_fS_O2_in_H2_p32], j=self.System[ss].Stack.ElectrolysisCell.Anode.j, p_abs=self.System[ss].Stack.ElectrolysisCell.Anode.p_abs), name=self.name_pre+"System_"+str(ss)+"_ElectrolysisCell_Anode_fS_O2")
            else:
                self.System[ss].Stack.ElectrolysisCell.Anode.fS["O2"] = m.Intermediate(Functions.fS(i="O2", j=self.System[ss].Stack.ElectrolysisCell.Anode.j, p_abs=self.System[ss].Stack.ElectrolysisCell.Anode.p_abs), name=self.name_pre+"System_"+str(ss)+"_ElectrolysisCell_Anode_fS_O2")
            #
            if(self.System[ss].S_potential):
                self.System[ss].Potential.fS['Anode'] = m.Intermediate(self.System[ss].Stack.ElectrolysisCell.Anode.fS["O2"], name=self.name_pre+"System_"+str(ss)+"_Potential_Anode_fS")
                self.System[ss].Potential.fS['Cathode'] = m.Intermediate(self.System[ss].Stack.ElectrolysisCell.Cathode.fS["H2"], name=self.name_pre+"System_"+str(ss)+"_Potential_Cathode_fS")
            #
            # Potential
            if(self.System[ss].system_config.potential == 1):
                if(self.process_config.pe_potential== 1):
                    self.System[ss].Potential.j_0["Anode"] = m.Intermediate( Functions.j_0(m=m, i=[self.C_j_0_A_p2, self.C_j_0_A_p3], T=self.System[ss].Potential.T) , name=self.name_pre+"System_"+str(ss)+"_Potential_j_0_Anode")
                    self.System[ss].Potential.j_0["Cathode"] = m.Intermediate( Functions.j_0(m=m, i=[self.C_j_0_C_p2, self.C_j_0_C_p3], T=self.System[ss].Potential.T), name=self.name_pre+"System_"+str(ss)+"_Potential_j_0_Cathode")
                    self.System[ss].Potential.alpha["Anode"] = m.Intermediate( Functions.alpha(i=[self.C_alpha_A_p1, self.C_alpha_A_p2, self.C_alpha_A_p3], T=self.System[ss].Potential.T) , name=self.name_pre+"System_"+str(ss)+"_Potential_alpha_Anode")
                    self.System[ss].Potential.alpha["Cathode"] = m.Intermediate(Functions.alpha(i=[self.C_alpha_C_p1, self.C_alpha_C_p2, self.C_alpha_C_p3], T=self.System[ss].Potential.T) , name=self.name_pre+"System_"+str(ss)+"_Potential_alpha_Cathode")
                    self.System[ss].Potential.d_eg = m.Intermediate(self.C_d_eg, name=self.name_pre+"System_"+str(ss)+"_Potential_d_eg")
                    self.System[ss].Potential.R_add = m.Intermediate(self.C_R_add, name=self.name_pre+"System_"+str(ss)+"_Potential_Radd")
                else:
                    self.System[ss].Potential.j_0["Anode"] = m.Intermediate( Functions.j_0(m=m, i="O2", T=self.System[ss].Potential.T), name=self.name_pre+"System_"+str(ss)+"_Potential_j_0_Anode")
                    self.System[ss].Potential.j_0["Cathode"] = m.Intermediate( Functions.j_0(m=m, i="H2", T=self.System[ss].Potential.T), name=self.name_pre+"System_"+str(ss)+"_Potential_j_0_Cathode")
                    self.System[ss].Potential.alpha["Anode"] = m.Intermediate( Functions.alpha(i="O2", T=self.System[ss].Potential.T), name=self.name_pre+"System_"+str(ss)+"_Potential_alpha_Anode")
                    self.System[ss].Potential.alpha["Cathode"] = m.Intermediate( Functions.alpha(i="H2", T=self.System[ss].Potential.T), name=self.name_pre+"System_"+str(ss)+"_Potential_alpha_Cathode")
                    self.System[ss].Potential.d_eg = m.Intermediate(self.System[ss].system_config.d_eg, name=self.name_pre+"System_"+str(ss)+"_Potential_d_eg")
                    self.System[ss].Potential.R_add = m.Intermediate(self.System[ss].system_config.R_add, name=self.name_pre+"System_"+str(ss)+"_Potential_Radd")
            #
            # w_KOH Estimation
            if(self.System[ss].system_config.electrolyte_concentration == 1):
                if(self.process_config.pe_w_KOH == 1):
                    self.System[ss].ElectrolyteConcentration.f_D_KOH = m.Intermediate(self.C_EC_f_D_p1, name=self.name_pre+"System_"+str(ss)+"_ElectrolyteConcentration_f_D_KOH")
                    self.System[ss].ElectrolyteConcentration.f_W_drag = m.Intermediate(self.C_EC_f_W_p1, name=self.name_pre+"System_"+str(ss)+"_ElectrolyteConcentration_f_W_drag")
                else:
                    self.System[ss].ElectrolyteConcentration.f_D_KOH = m.Intermediate(0.3538, name=self.name_pre+"System_"+str(ss)+"_ElectrolyteConcentration_f_D_KOH")
                    self.System[ss].ElectrolyteConcentration.f_W_drag = m.Intermediate(1, name=self.name_pre+"System_"+str(ss)+"_ElectrolyteConcentration_f_W_drag")
            #
            # Energy Balance Estimation
            if(self.System[ss].system_config.energy_balance == 1 and self.System[ss].system_config.potential == 1):
                if(self.process_config.pe_energy == 1):
                    self.System[ss].EnergyBalance.T_amb = m.Param(value=self.process_config.system_config[ss].est_data_T_sys["T_amb"].add(273.15).values.tolist(), name=self.name_pre+"System_"+str(ss)+"_T_amb")
                    self.System[ss].EnergyBalance.L_tube = m.Intermediate(self.C_EB_L_tube_p1, name=self.name_pre+"System_"+str(ss)+"_EnergyBalance_L_tube")
                else:
                    if(process_config.dl_T_amb == 1):
                         self.System[ss].EnergyBalance.T_amb = m.Param(value=self.process_config.system_config[ss].est_data_T_sys["T_amb"].add(273.15).values.tolist(), name=self.name_pre+"System_"+str(ss)+"_T_amb")  
                    else:
                        self.System[ss].EnergyBalance.T_amb = m.Param(value=self.process_config.system_config[ss].EB_T_amb, name=self.name_pre+"System_"+str(ss)+"_T_amb")
                    self.System[ss].EnergyBalance.L_tube  = m.Intermediate(self.process_config.system_config[ss].EB_L_tube, name=self.name_pre+"System_"+str(ss)+"_EnergyBalance_L_tube")
                #
                self.System[ss].EnergyBalance.T_gas = m.Intermediate(self.System[ss].EnergyBalance.T_amb, name=self.name_pre+"T_gas")
            #
            self.System[ss].Intermediates(m=self.m)
            #
            # Parameter Estimation
            if(self.process_config.pe_H2_in_O2 == 1):
                # d_b
                m.Equation(self.System[ss].Stack.ElectrolysisCell.Cathode.d_b == (Functions.d_b(m=m, i=[self.C_d_b_H2_p1, self.C_d_b_H2_p21, self.C_d_b_H2_p22, self.C_d_b_H2_p31, self.C_d_b_H2_p32], j=self.System[ss].Stack.ElectrolysisCell.Cathode.j, gamma_KOH=self.System[ss].Stack.ElectrolysisCell.Cathode.gamma_fl, dens_KOH=self.System[ss].Stack.ElectrolysisCell.Cathode.dens_KOH, dens_gas=self.System[ss].Stack.ElectrolysisCell.Cathode.dens_gas, p_abs=self.System[ss].Stack.ElectrolysisCell.Cathode.p_abs)))
            else:
                m.Equation(self.System[ss].Stack.ElectrolysisCell.Cathode.d_b == (Functions.d_b(m=m, i="H2", j=self.System[ss].Stack.ElectrolysisCell.Cathode.j, gamma_KOH=self.System[ss].Stack.ElectrolysisCell.Cathode.gamma_fl, dens_KOH=self.System[ss].Stack.ElectrolysisCell.Cathode.dens_KOH, dens_gas=self.System[ss].Stack.ElectrolysisCell.Cathode.dens_gas, p_abs=self.System[ss].Stack.ElectrolysisCell.Cathode.p_abs)))
            #
            if(self.process_config.pe_O2_in_H2 == 1):
                # d_b
                m.Equation(self.System[ss].Stack.ElectrolysisCell.Anode.d_b == (Functions.d_b(m=m, i=[self.C_d_b_O2_p1, self.C_d_b_O2_p21, self.C_d_b_O2_p22, self.C_d_b_O2_p31, self.C_d_b_O2_p32], j=self.System[ss].Stack.ElectrolysisCell.Anode.j, gamma_KOH=self.System[ss].Stack.ElectrolysisCell.Anode.gamma_fl, dens_KOH=self.System[ss].Stack.ElectrolysisCell.Anode.dens_KOH, dens_gas=self.System[ss].Stack.ElectrolysisCell.Anode.dens_gas, p_abs=self.System[ss].Stack.ElectrolysisCell.Anode.p_abs)))
            else:
                m.Equation(self.System[ss].Stack.ElectrolysisCell.Anode.d_b == (Functions.d_b(m=m, i="O2", j=self.System[ss].Stack.ElectrolysisCell.Anode.j, gamma_KOH=self.System[ss].Stack.ElectrolysisCell.Anode.gamma_fl, dens_KOH=self.System[ss].Stack.ElectrolysisCell.Anode.dens_KOH, dens_gas=self.System[ss].Stack.ElectrolysisCell.Anode.dens_gas, p_abs=self.System[ss].Stack.ElectrolysisCell.Anode.p_abs)) )
            #
            # Objectives
            # Gas impurity
            if(self.process_config.pe_H2_in_O2):   
                if(self.process_config.solve_mode == 0):
                    m.Obj((100*self.System[ss].system_config.exp_H2_in_O2_valid/(self.process_config.num_system)) * m.abs((self.System[ss].system_config.exp_H2_in_O2 - self.System[ss].GasSeparator.Anode.x_percent["H2"])/(self.System[ss].system_config.exp_H2_in_O2)))
                else:
                    m.Obj((self.est_data_H2_in_O2_valid[ss]/(np.sum(self.est_data_H2_in_O2_valid[ss]))) * m.abs((self.est_data_H2_in_O2[ss] - self.System[ss].GasSeparator.Anode.x_percent["H2"])/(self.est_data_H2_in_O2_valid[ss] * self.est_data_H2_in_O2[ss] + (1-self.est_data_H2_in_O2_valid[ss]) * 1e-20)))
            #
            if(self.process_config.pe_O2_in_H2):   
                    m.Obj((self.est_data_O2_in_H2_valid[ss]/(np.sum(self.est_data_O2_in_H2_valid[ss]))) * m.abs((self.est_data_O2_in_H2[ss] - self.System[ss].GasSeparator.Cathode.x_percent["O2"])/(self.est_data_O2_in_H2_valid[ss] * self.est_data_O2_in_H2[ss] + (1-self.est_data_O2_in_H2_valid[ss]) * 1e-20)))
            #
            # w_KOH
            if(self.process_config.pe_w_KOH):
                m.Obj((self.est_data_w_KOH_valid[ss]/(np.sum(self.est_data_w_KOH_valid[ss]))) * m.abs((self.est_data_w_KOH_A[ss] - self.System[ss].ElectrolyteConcentration.Anode.w_KOH)/(self.est_data_w_KOH_valid[ss] * self.est_data_w_KOH_A[ss] + (1-self.est_data_w_KOH_valid[ss]) * 1e-20)))
                m.Obj((self.est_data_w_KOH_valid[ss]/(np.sum(self.est_data_w_KOH_valid[ss]))) * m.abs((self.est_data_w_KOH_C[ss] - self.System[ss].ElectrolyteConcentration.Cathode.w_KOH)/(self.est_data_w_KOH_valid[ss] * self.est_data_w_KOH_C[ss] + (1-self.est_data_w_KOH_valid[ss]) * 1e-20)))
            #
            # Temperature
            if(self.process_config.pe_energy):
                m.Obj((self.est_data_T_sys_valid[ss]/(np.sum(self.est_data_T_sys_valid[ss]))) * m.abs((self.est_data_T_sys[ss] - self.System[ss].EnergyBalance.T)/(self.est_data_T_sys_valid[ss] * self.est_data_T_sys[ss] + (1-self.est_data_T_sys_valid[ss]) * 1e-20)))
            #
            # Potential
            if(self.process_config.pe_potential):
                m.Obj(self.System[ss].system_config.exp_potential_valid * m.abs((self.System[ss].system_config.exp_potential - self.System[ss].Potential.U_cell)/(self.System[ss].system_config.exp_potential))**2) 
            self.System[ss].Equations(m=self.m)
            
    def apm_export(self):
        m = self.m
        if(global_config["options"]["apm_export"] and global_config["options"]["model_export"]):
            apm_filename = "log/" + str(run_time) + "_apm_" + str(self.name.lower())
            for folderName, subfolders, filenames in os.walk(m._path):
                for filename in filenames:
                    filePath = os.path.join(folderName, filename)
                    zip_model.write(filePath, 'apm_output/' + str(self.name.lower()) + '/' + os.path.basename(filePath))
            logger.info("APMonitor output files of " + str(self.name) + " were added to the model archive")

    def dyn_solve(self):
        m = self.m
        m.options.NODES = 7
        m.options.REDUCE = 3
        m.options.DIAGLEVEL = 0
        m.options.WEB = 0
        m.options.MAX_TIME = 600
        if(np.sum([ss.system_config.electrolyte_concentration for ss in self.System]) >= 1):
            m.options.RTOL = 1e-4
        else:
            m.options.RTOL = 1e-6
        m.options.OTOL = 1e-6
        m.options.IMODE = 7 #
        m.options.SOLVER = 1
        # m.open_folder()
        m.solve(disp=True)
        self.apm_export()
        m.cleanup()
        
    def stat_solve(self):
        m = self.m
        m.options.REDUCE = 0
        m.options.DIAGLEVEL = 0
        m.options.WEB = 0
        # m.options.MAX_TIME = 60
        # m.options.MAX_ITER = 1000
        m.options.RTOL = 1e-6
        m.options.OTOL = 1e-6
        m.options.IMODE = 1
        m.options.SOLVER = 2
        # m.open_folder()
        m.solve(disp=True)
        self.apm_export()
        m.cleanup()
    
    def stat_est(self):
        m = self.m
        m.options.REDUCE = 0
        m.options.DIAGLEVEL = 0
        m.options.WEB = 0
        if(self.process_config.pe_H2_in_O2):
            m.options.MAX_TIME = 60
            m.options.MAX_ITER = 1000
            m.options.RTOL = 1e-6
            m.options.OTOL = 1e-1
            m.options.SOLVER = 1
        if(self.process_config.pe_potential):
            m.options.MAX_TIME = 1800
            m.options.MAX_ITER = 10000
            m.options.RTOL = 1e-6
            m.options.OTOL = 1e-6
            m.options.SOLVER = 2
        m.options.IMODE = 2
        # m.open_folder()
        m.solve(disp=True)
        self.apm_export()
        m.cleanup()
    
    def dyn_est(self):
        m = self.m
        m.options.REDUCE = 0
        m.options.DIAGLEVEL = 0
        m.options.WEB = 0
        # m.options.MAX_TIME = 600
        # m.options.MAX_ITER = 1000
        if(self.process_config.pe_H2_in_O2):
            m.options.RTOL = 1e-6 
            m.options.OTOL = 1e-6
        if(self.process_config.pe_O2_in_H2):
            m.options.RTOL = 1e-3
            m.options.OTOL = 1e-3
        if(self.process_config.pe_w_KOH):
            m.options.RTOL = 1e-3
            m.options.OTOL = 1e-3
        if(self.process_config.pe_energy):
            m.options.RTOL = 1e-6
            m.options.OTOL = 1e-6
        m.options.IMODE = 5
        m.options.SOLVER = 1
        # m.open_folder()
        #
        m.options.COLDSTART = 2
        m.solve(disp=True, GUI=False)
        m.options.TIME_SHIFT = 0
        m.solve(disp=True, GUI=False)
        #
        m.solve(disp=True, GUI=False)
        #
        self.apm_export()
        m.cleanup()

# # #
# Solve Process
#
#
# Process object initialization
for pp in range(num_process):
    Process[pp] = ProcessClass(name="Process" + "_" + str(pp), process_config=process_config[pp])
#
def data_export_writer(Process, pp):
    #
    if (Process[pp].process_config.solve_mode == 0):
        df_export = pd.DataFrame()
        if(Process[pp].process_config.process_export_name != ""):
                process_export_name = str(Process[pp].process_config.process_export_name)
        else:
            process_export_name = 'Stat_Results_Process_' + str(pp)
        for ss in range(Process[pp].process_config.num_system):
            export_dict = {"j": Process[pp].System[ss].j.value[0]}
            export_dict.update({"p": Process[pp].System[ss].p})
            export_dict.update({"T": Process[pp].System[ss].T.value[0]})
            export_dict.update({"w_KOH": Process[pp].System[ss].w_KOH_ini})
            export_dict.update({"VS_L_A": Process[pp].System[ss].VS_L_A.value[0]})
            export_dict.update({"VS_L_C": Process[pp].System[ss].VS_L_C.value[0]})
            export_dict.update({"H2inO2": Process[pp].System[ss].GasSeparator.Anode.x_percent["H2"].value[0]})
            export_dict.update({"exp_H2inO2": Process[pp].System[ss].system_config.exp_H2_in_O2})
            export_dict.update({"O2inH2": Process[pp].System[ss].GasSeparator.Cathode.x_percent["O2"].value[0]})
            export_dict.update({"flow_mode": Process[pp].System[ss].S_mixing.value[0]})
            if(Process[pp].System[ss].system_config.potential == 1):
                export_dict.update({"U_cell": Process[pp].System[ss].Potential.U_cell.value[0]})
            df_export = df_export.append(export_dict, ignore_index=True)
        df_export.to_csv('results/' + process_export_name + '.txt', sep='\t', index=False)
    #
    if (Process[pp].process_config.solve_mode == 1):
        df_export = [None for ss in range(Process[pp].process_config.num_system)]
        for ss in range(Process[pp].process_config.num_system):
            if(Process[pp].process_config.system_config[ss].system_export_name != ""):
                system_export_name = str(Process[pp].process_config.system_config[ss].system_export_name)
            else:
                system_export_name = 'Dyn_Results_Process_' + str(pp) + '_System_' + str(ss)
            export_dict = {'t': Process[pp].process_config.m_time}
            export_dict.update({"j": Process[pp].System[ss].j.value})
            export_dict.update({"p": Process[pp].System[ss].p})
            export_dict.update({"T": Process[pp].System[ss].T.value})
            export_dict.update({"w_KOH": Process[pp].System[ss].w_KOH_ini})
            export_dict.update({"H2inO2": Process[pp].System[ss].GasSeparator.Anode.x_percent["H2"].value})
            export_dict.update({"O2inH2": Process[pp].System[ss].GasSeparator.Cathode.x_percent["O2"].value})
            export_dict.update({"mixing": Process[pp].System[ss].S_mixing.value})
            export_dict.update({"opt_j_switching_value": Process[pp].System[ss].system_config.opt_j_switching_value})
            export_dict.update({"eff_gas_percent": Process[pp].System[ss].eff_gas_percent.value})
            export_dict.update({"gas_purity_h2_percent": Process[pp].System[ss].gas_purity_h2_percent.value})
            export_dict.update({"gas_purity_o2_percent": Process[pp].System[ss].gas_purity_o2_percent.value})
            export_dict.update({"eff_w_KOH_C_percent": Process[pp].System[ss].eff_w_KOH_C_percent.value})
            if(Process[pp].System[ss].system_config.potential == 1):
                export_dict.update({"U_cell": Process[pp].System[ss].Potential.U_cell.value})
                export_dict.update({"eff_cell_percent": Process[pp].System[ss].Potential.eff_cell_percent.value})
                export_dict.update({"eff_system_percent": Process[pp].System[ss].eff_system_percent.value})
                export_dict.update({"eff_objective_percent": Process[pp].System[ss].eff_objective_percent.value})
            if(Process[pp].System[ss].system_config.electrolyte_concentration == 1):
                export_dict.update({"w_KOH_A": Process[pp].System[ss].ElectrolyteConcentration.Anode.w_KOH.value})
                export_dict.update({"w_KOH_C": Process[pp].System[ss].ElectrolyteConcentration.Cathode.w_KOH.value})
                export_dict.update({"w_KOH_GA": Process[pp].System[ss].ElectrolyteConcentration.GasSeparatorAnode.w_KOH.value})
                export_dict.update({"w_KOH_GC": Process[pp].System[ss].ElectrolyteConcentration.GasSeparatorCathode.w_KOH.value})
                export_dict.update({"V_L_GA": Process[pp].System[ss].ElectrolyteConcentration.GasSeparatorAnode.V_L.value})
                export_dict.update({"V_L_GC": Process[pp].System[ss].ElectrolyteConcentration.GasSeparatorCathode.V_L.value})
                export_dict.update({"w_KOH_mix": Process[pp].System[ss].ElectrolyteConcentration.Mixer.w_KOH_mix.value})
                export_dict.update({"m_El_sys": Process[pp].System[ss].ElectrolyteConcentration.m_El.value})
                export_dict.update({"m_KOH_sys": Process[pp].System[ss].ElectrolyteConcentration.m_KOH.value})
                export_dict.update({"m_H2O_sys": Process[pp].System[ss].ElectrolyteConcentration.m_H2O.value})
            if(Process[pp].System[ss].system_config.energy_balance == 1):
                export_dict.update({"T_amb": Process[pp].System[ss].EnergyBalance.T_amb.value})
            df_export[ss] = pd.DataFrame(export_dict)
            df_export[ss].to_csv('results/' + system_export_name + '.txt', sep='\t', index=False)
#    
def parasolve(Process, pp):
    Process[pp].solve_time_start = time.time()
    logger.info("Process." + str(pp) + ": Start Solving")
    if (Process[pp].process_config.solve_mode == 0):
        if(Process[pp].process_config.pe_H2_in_O2 == 1 or Process[pp].process_config.pe_potential):
            Process[pp].stat_est()
        else:
            Process[pp].stat_solve()
        if(global_config["options"]["data_export"]==1):
            data_export_writer(Process=Process, pp=pp)
    if (Process[pp].process_config.solve_mode == 1):
        if(Process[pp].process_config.pe_H2_in_O2 == 1 or Process[pp].process_config.pe_w_KOH == 1 or Process[pp].process_config.pe_energy == 1 or Process[pp].process_config.pe_O2_in_H2 == 1):
            Process[pp].dyn_est()
        else:
            Process[pp].dyn_solve()
            if(global_config["options"]["data_export"]==1):
                data_export_writer(Process=Process, pp=pp)
    Process[pp].solve_time_end = time.time()
    Process[pp].solve_time = np.round(Process[pp].solve_time_end - Process[pp].solve_time_start,2)
    logger.info("Process." + str(pp) + ": End Solving")
#
executor = concurrent.futures.ThreadPoolExecutor()
futures = {executor.submit(parasolve, Process, count): count for count in range(num_process)}
#
concurrent.futures.wait(futures)
logger.info("----------------------------------------")
logger.info("All simulations completed")
logger.info("----------------------------------------")
logger.info("Solve times:")
for pp in range(num_process):
    logger.info("process." + str(pp) + ": " + str(Process[pp].solve_time) + " s")
logger.info("----------------------------------------")
if(global_config["options"]["data_plot"] == 1):
    for pp in range(num_process):
        if(Process[pp].process_config.solve_mode == 0):
            for ss in range(num_system[pp]):
                if(Process[pp].process_config.pe_H2_in_O2 == 1 or Process[pp].process_config.pe_potential == 1 ):
                    if(Process[pp].process_config.pe_H2_in_O2 == 1):
                        logger.info("p." + str(pp) + ".s." + str(ss) + ": p=" + str(Process[pp].System[ss].Stack.ElectrolysisCell.Anode.p_abs.value[0]/1e5) + ", j=" + str(Process[pp].System[ss].Stack.ElectrolysisCell.Anode.j.value[0]/1e3) + ": " + str(np.round(Process[pp].System[ss].GasSeparator.Anode.x_percent["H2"].value[0], 4)) + " / " + str(Process[pp].System[ss].system_config.exp_H2_in_O2) + ", Diff: " + str(np.round((100*(Process[pp].System[ss].GasSeparator.Anode.x_percent["H2"].value[0] - Process[pp].System[ss].system_config.exp_H2_in_O2)/Process[pp].System[ss].system_config.exp_H2_in_O2),0)) + "%" + " (Valid: " + str(Process[pp].System[ss].system_config.exp_H2_in_O2_valid) + ")" )
                    if(Process[pp].process_config.pe_potential == 1):
                        logger.info("p." + str(pp) + ".s." + str(ss) + ": p=" + str(Process[pp].System[ss].Stack.ElectrolysisCell.Anode.p_abs.value[0]/1e5) + ", j=" + str(Process[pp].System[ss].Stack.ElectrolysisCell.Anode.j.value[0]/1e3) + ": " +  str(np.round(Process[pp].System[ss].GasSeparator.Anode.x_percent["H2"].value[0], 4)) + ", " + str(np.round(Process[pp].System[ss].Potential.U_cell.value[0], 4)) + " / " + str(Process[pp].System[ss].system_config.exp_potential) + ", Diff: " + str(np.round((100*(Process[pp].System[ss].Potential.U_cell.value[0] - Process[pp].System[ss].system_config.exp_potential)/Process[pp].System[ss].system_config.exp_potential),0)) + "%" + " (Valid: " + str(Process[pp].System[ss].system_config.exp_potential_valid) + ")")
                else:
                    logger.info("p." + str(pp) + ".s." + str(ss) + ": p=" + str(Process[pp].System[ss].Stack.ElectrolysisCell.Anode.p_abs.value[0]/1e5) + ", j=" + str(Process[pp].System[ss].Stack.ElectrolysisCell.Anode.j.value[0]/1e3) + ": " + str(Process[pp].System[ss].GasSeparator.Anode.x_percent["H2"].value[0]) + " / " + str(Process[pp].System[ss].GasSeparator.Cathode.x_percent["O2"].value[0]))
            #
            if(Process[pp].process_config.pe_H2_in_O2 == 1):
                logger.info("----------------------------------------")
                logger.info("Parameter Estimation Results:")
                logger.info("C_d_b_H2_p1: " + str(Process[pp].C_d_b_H2_p1.value[0]))
                logger.info("C_d_b_H2_p21: " + str(Process[pp].C_d_b_H2_p21.value[0]))
                logger.info("C_d_b_H2_p22: " + str(Process[pp].C_d_b_H2_p22.value[0]))
                logger.info("C_d_b_H2_p31: " + str(Process[pp].C_d_b_H2_p31.value[0]))
                logger.info("C_d_b_H2_p32: " + str(Process[pp].C_d_b_H2_p32.value[0]))
                logger.info("C_fS_H2_in_O2_p1: " + str(Process[pp].C_fS_H2_in_O2_p1.value[0]))
                logger.info("C_fS_H2_in_O2_p21: " + str(Process[pp].C_fS_H2_in_O2_p21.value[0]))
                logger.info("C_fS_H2_in_O2_p22: " + str(Process[pp].C_fS_H2_in_O2_p22.value[0]))
                logger.info("C_fS_H2_in_O2_p31: " + str(Process[pp].C_fS_H2_in_O2_p31.value[0]))
                logger.info("C_fS_H2_in_O2_p32: " + str(Process[pp].C_fS_H2_in_O2_p32.value[0]))
                logger.info("----------------------------------------")
            #
            if(Process[pp].process_config.pe_potential == 1):
                logger.info("----------------------------------------")
                logger.info("Parameter Estimation Results:")
                logger.info("C_j_0_A_p1: " + str(Process[pp].C_j_0_A_p1.value[0]))
                logger.info("C_j_0_A_p2: " + str(Process[pp].C_j_0_A_p2.value[0]))
                logger.info("C_j_0_A_p3: " + str(Process[pp].C_j_0_A_p3.value[0]))
                logger.info("----------------------------------------")
                logger.info("C_j_0_C_p1: " + str(Process[pp].C_j_0_C_p1.value[0]))
                logger.info("C_j_0_C_p2: " + str(Process[pp].C_j_0_C_p2.value[0]))
                logger.info("C_j_0_C_p3: " + str(Process[pp].C_j_0_C_p3.value[0]))
                logger.info("----------------------------------------")
                logger.info("C_alpha_A_p1: " + str(Process[pp].C_alpha_A_p1.value[0]))
                logger.info("C_alpha_A_p2: " + str(Process[pp].C_alpha_A_p2.value[0]))
                logger.info("C_alpha_A_p3: " + str(Process[pp].C_alpha_A_p3.value[0]))
                logger.info("----------------------------------------")
                logger.info("C_alpha_C_p1: " + str(Process[pp].C_alpha_C_p1.value[0]))
                logger.info("C_alpha_C_p2: " + str(Process[pp].C_alpha_C_p2.value[0]))
                logger.info("C_alpha_C_p3: " + str(Process[pp].C_alpha_C_p3.value[0]))
                logger.info("----------------------------------------")
                logger.info("C_d_eg: " + str(Process[pp].C_d_eg.value[0]))
                logger.info("C_R_add: " + str(Process[pp].C_R_add.value[0]))
                logger.info("----------------------------------------")
        #
        if(Process[pp].process_config.solve_mode == 1):
            #
            # H2 in O2 Plot
            for ss in range(num_system[pp]):
                logger.info("process." + str(pp) + ".system." + str(ss) + ": p=" + str(Process[pp].System[ss].Stack.ElectrolysisCell.Anode.p_abs.value[0]) + ", j=" + str(Process[pp].System[ss].Stack.ElectrolysisCell.Anode.j.value[0]) + ": " + str(Process[pp].System[ss].GasSeparator.Anode.x_percent["H2"].value[-1]))
                plt.plot(Process[pp].m.time/3600 - Process[pp].process_config.time_pre_end_hour, Process[pp].System[ss].GasSeparator.Anode.x_percent["H2"], label="Process: " + str(pp) + ", " "System: " + str(ss))
                if(process_config[pp].pe_H2_in_O2 == 1 or process_config[pp].plot_vd == 1):
                    plt.plot(Process[pp].m.time/3600 - Process[pp].process_config.time_pre_end_hour, Process[pp].System[ss].system_config.est_data_H2_in_O2["H2_in_O2"], 'ro', markersize=1, label="Process: " + str(pp) + ", " "System: " + str(ss))
                if(process_config[pp].plot_datalog == 1 and type(Process[pp].System[ss].system_config.datalog_data) != int):
                    plt.plot(Process[pp].m.time/3600 - Process[pp].process_config.time_pre_end_hour, Process[pp].System[ss].system_config.datalog_data["signal_in_h2_in_o2_sensor"], 'bo', markersize=1, label="Process: " + str(pp) + ", " "System: " + str(ss))
            plt.xlim(-2, Process[pp].process_config.time_end_hour)
            plt.xlabel("time / h")
            plt.ylabel("H$_2$ in O$_2$ / vol.%")
            #
            if(global_config["options"]["fig_export"]):
                plt.savefig("log/" + str(run_time) + "_fig_process_" + str(pp) + ".png", dpi=300)
                if (global_config["options"]["model_export"]): zip_model.write("log/" + str(run_time) + "_fig_process_" + str(pp) + ".png")
            plt.clf()
            #
            # Electrolyte Concentration Plot
            if(Process[pp].process_config.pe_w_KOH == 1):
                for ss in range(num_system[pp]):
                    plt.plot(Process[pp].m.time/3600 - Process[pp].process_config.time_pre_end_hour, Process[pp].System[ss].ElectrolyteConcentration.Anode.w_KOH, label="Process: " + str(pp) + ", " "System: " + str(ss))
                    plt.plot(Process[pp].m.time/3600 - Process[pp].process_config.time_pre_end_hour, Process[pp].System[ss].ElectrolyteConcentration.Cathode.w_KOH, label="Process: " + str(pp) + ", " "System: " + str(ss))
                    plt.plot(Process[pp].m.time/3600 - Process[pp].process_config.time_pre_end_hour, Process[pp].System[ss].system_config.est_data_w_KOH["w_KOH_A"], 'ro', markersize=1, label="Process: " + str(pp) + ", " "System: " + str(ss))
                    plt.plot(Process[pp].m.time/3600 - Process[pp].process_config.time_pre_end_hour, Process[pp].System[ss].system_config.est_data_w_KOH["w_KOH_C"], 'ro', markersize=1, label="Process: " + str(pp) + ", " "System: " + str(ss))
                plt.xlabel("time / h")
                plt.ylabel("electrolyte conc. / wt.%")
                if(global_config["options"]["fig_export"]):
                    plt.savefig("log/" + str(run_time) + "_fig_w_KOH_process_" + str(pp) + ".png", dpi=300)
                    if (global_config["options"]["model_export"]): zip_model.write("log/" + str(run_time) + "_fig_w_KOH_process_" + str(pp) + ".png")
                plt.clf()
            #
            # Temperature Plot
            if(Process[pp].process_config.pe_energy == 1):
                for ss in range(num_system[pp]):
                    plt.plot(Process[pp].m.time/3600 - Process[pp].process_config.time_pre_end_hour, np.add(Process[pp].System[ss].EnergyBalance.T.value,-273.15), label="Process: " + str(pp) + ", " "System: " + str(ss))
                    plt.plot(Process[pp].m.time/3600 - Process[pp].process_config.time_pre_end_hour, Process[pp].System[ss].system_config.est_data_T_sys["T_sys"], 'ro', markersize=1, label="Process: " + str(pp) + ", " "System: " + str(ss))
                    plt.plot(Process[pp].m.time/3600 - Process[pp].process_config.time_pre_end_hour, np.add(Process[pp].System[ss].EnergyBalance.T_amb.value,-273.15), label="Process: " + str(pp) + ", " "System: " + str(ss))
                plt.xlabel("time / h")
                plt.ylabel("temperature / °C")
                if(global_config["options"]["fig_export"]):
                    plt.savefig("log/" + str(run_time) + "_fig_energy_process_" + str(pp) + ".png", dpi=300)
                    if (global_config["options"]["model_export"]): zip_model.write("log/" + str(run_time) + "_fig_energy_process_" + str(pp) + ".png")
                plt.clf()
            #
            # O2 in H2 Plot
            if(Process[pp].process_config.pe_O2_in_H2 == 1):
                for ss in range(num_system[pp]):
                    plt.plot(Process[pp].m.time/3600 - Process[pp].process_config.time_pre_end_hour, Process[pp].System[ss].GasSeparator.Cathode.x_percent["O2"], label="Process: " + str(pp) + ", " "System: " + str(ss))
                    plt.plot(Process[pp].m.time/3600 - Process[pp].process_config.time_pre_end_hour, Process[pp].System[ss].system_config.est_data_O2_in_H2["O2_in_H2"], 'ro', markersize=1, label="Process: " + str(pp) + ", " "System: " + str(ss))
                plt.xlim(-2, Process[pp].process_config.time_end_hour)
                plt.xlabel("time / h")
                plt.ylabel("O$_2$ in H$_2$ / vol.%")
                if(global_config["options"]["fig_export"]):
                    plt.savefig("log/" + str(run_time) + "_fig_o2_in_h2_process_" + str(pp) + ".png", dpi=300)
                    if (global_config["options"]["model_export"]): zip_model.write("log/" + str(run_time) + "_fig_o2_in_h2_process_" + str(pp) + ".png")
                plt.clf()
            #
            if(Process[pp].process_config.pe_H2_in_O2 == 1):
                logger.info("----------------------------------------")
                logger.info("Parameter Estimation Results:")
                logger.info("C_d_b_H2_p1: " + str(Process[pp].C_d_b_H2_p1.value[0]))
                logger.info("C_d_b_H2_p21: " + str(Process[pp].C_d_b_H2_p21.value[0]))
                logger.info("C_d_b_H2_p22: " + str(Process[pp].C_d_b_H2_p22.value[0]))
                logger.info("C_d_b_H2_p31: " + str(Process[pp].C_d_b_H2_p31.value[0]))
                logger.info("C_d_b_H2_p32: " + str(Process[pp].C_d_b_H2_p32.value[0]))
                logger.info("C_fS_H2_in_O2_p1: " + str(Process[pp].C_fS_H2_in_O2_p1.value[0]))
                logger.info("C_fS_H2_in_O2_p21: " + str(Process[pp].C_fS_H2_in_O2_p21.value[0]))
                logger.info("C_fS_H2_in_O2_p22: " + str(Process[pp].C_fS_H2_in_O2_p22.value[0]))
                logger.info("C_fS_H2_in_O2_p31: " + str(Process[pp].C_fS_H2_in_O2_p31.value[0]))
                logger.info("C_fS_H2_in_O2_p32: " + str(Process[pp].C_fS_H2_in_O2_p32.value[0]))
                logger.info("----------------------------------------")
            #
            if(Process[pp].process_config.pe_O2_in_H2 == 1):
                logger.info("----------------------------------------")
                logger.info("Parameter Estimation Results:")
                logger.info("C_d_b_O2_p1: " + str(Process[pp].C_d_b_O2_p1.value[0]))
                logger.info("C_d_b_O2_p21: " + str(Process[pp].C_d_b_O2_p21.value[0]))
                logger.info("C_d_b_O2_p22: " + str(Process[pp].C_d_b_O2_p22.value[0]))
                logger.info("C_d_b_O2_p31: " + str(Process[pp].C_d_b_O2_p31.value[0]))
                logger.info("C_d_b_O2_p32: " + str(Process[pp].C_d_b_O2_p32.value[0]))
                logger.info("C_fS_O2_in_H2_p1: " + str(Process[pp].C_fS_O2_in_H2_p1.value[0]))
                logger.info("C_fS_O2_in_H2_p21: " + str(Process[pp].C_fS_O2_in_H2_p21.value[0]))
                logger.info("C_fS_O2_in_H2_p22: " + str(Process[pp].C_fS_O2_in_H2_p22.value[0]))
                logger.info("C_fS_O2_in_H2_p31: " + str(Process[pp].C_fS_O2_in_H2_p31.value[0]))
                logger.info("C_fS_O2_in_H2_p32: " + str(Process[pp].C_fS_O2_in_H2_p32.value[0]))
                logger.info("----------------------------------------")
            #
            if(Process[pp].process_config.pe_w_KOH == 1):
                logger.info("----------------------------------------")
                logger.info("Parameter Estimation Results:")
                logger.info("C_EC_f_W_p1: " + str(Process[pp].C_EC_f_W_p1.value[0]))
                logger.info("C_EC_f_D_p1: " + str(Process[pp].C_EC_f_D_p1.value[0]))
                logger.info("----------------------------------------")
            #
            if(Process[pp].process_config.pe_energy == 1):
                logger.info("----------------------------------------")
                logger.info("Parameter Estimation Results:")
                logger.info("C_EB_L_tube_p1: " + str(Process[pp].C_EB_L_tube_p1.value[0]))
                logger.info("----------------------------------------")
#
logging.shutdown()
if (global_config["options"]["model_export"]):
    if(global_config["options"]["log_export"]):
        zip_model.write(logger_fh_filename)
    zip_model.close()
