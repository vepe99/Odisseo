from fireworks import *

import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

np.random.seed(9725)
path = '.'

def evolve_3body():
    ## INITIAL CONDITIONS
    position = np.array([[1,3,0],
                    [-2,-1,0],
                    [1,-1,0]])
    vel = np.array([[0,0,0],
                [0,0,0],
                [0,0,0]])
    mass = np.array([3,4,5])


    # CREATE INSTANCES OF THE PARTICLES
    part = Particles(position, vel, mass)
    partcom = part.com_pos()
    Etot_0, _, _ = part.Etot()
    print(Etot_0)


    ## INTEGRATION
    Tperiod = 2*np.pi
    N_end = 50 # -> N_end*Tperiod

    #define number of time steps per time increment
    time_increments = np.array([0.0001])
    # n_ts = np.floor(N_end*Tperiod/time_increments)

    # config file
    # ic_param = np.array([mass1, mass2, rp, e, a, Etot_0, Tperiod, N_end])
    # np.savetxt(path + '/data/ass_3/ic_param_all'+'_e_'+str(e)+'_rp_'+str(rp)+'.txt', ic_param)

    for dt in time_increments:
        N_ts = int(np.floor(N_end*Tperiod/dt))
        file_name = path + '/data1/3body_dt_'+str(dt)
    
        
        tot_time = 0 # init flags count to 0
        N_ts_cum = 0

        
        array = np.zeros(shape=(N_ts, 10))
        # part = ic_two_body(mass1=mass1, mass2=mass2, rp=rp, e=e)
        part.pos = part.pos - partcom
        dt_copy = dt.copy()
        # for t_i in range(N_ts):
        for t_i in tqdm(range(N_ts), desc=str(dt_copy)):
            part, dt_copy, acc = integrator_rk4(part,
                                            tstep=dt_copy,
                                            acceleration_estimator=acceleration_direct, 
                                            softening=10e-04)

            Etot_i, _, _ = part.Etot()
            
            array[t_i, :3] = part.pos[0, :3]
            array[t_i, 3:6]= part.pos[1, :3]
            array[t_i, 6:9]= part.pos[2, :3]
            array[t_i, 9]  = Etot_i
            # array[t_i, 10]  = dt_copy

            dt_copy = adaptive_timestep_vel(particles=part, acc=acc, eta=0.02, 
                                            tmax=None, tmin=None)

            tot_time += dt_copy
            N_ts_cum += 1

            if tot_time >= N_end*Tperiod:
                print('Exceeded time limit')
                break
            elif N_ts_cum >= 10*N_ts:
                print('Exceeded number of time steps')
                break
            
        
            
        np.savez(file_name, array=array)
        

def evolve_2body():
    ## INITIAL CONDITIONS
    mass1 = 8
    mass2 = 2
    rp = 0.1
    e = 0.0 # 0.0 for circular orbit
    part = ic_two_body(mass1=mass1, mass2=mass2, rp=rp, e=e) # particles initialization
    part.pos = part.pos - part.com_pos() # correcting iniziatial position by C.O.M
    # print(part.pos, part.vel, part.mass)
    Etot_0, _, _ = part.Etot() # total energy of the system

    # Calculate the binary period Tperiod
    a = rp / (1 - e)  # Semi-major axis
    Tperiod = 2 * np.pi * np.sqrt(a**3 / (mass1 + mass2))


## INTEGRATION 
    N_end = 50 # -> N_end*Tperiod

    #define number of time steps per time increment
    time_increments = np.array([0.00001])
    # n_ts = np.floor(N_end*Tperiod/time_increments)

    # config file
    ic_param = np.array([mass1, mass2, rp, e, a, Etot_0, Tperiod, N_end])
    np.savetxt(path + '/data1/ic_param_all'+'_e_'+str(e)+'_rp_'+str(rp)+'.txt', ic_param)

    for dt in time_increments:
        N_ts = int(np.floor(N_end*Tperiod/dt))
        file_name = path + '/data1/dt_'+str(dt)+'_e_'+str(e)+'_rp_'+str(rp)
        
        tot_time = 0 # init flags count to 0
        N_ts_cum = 0

        
        array = np.zeros(shape=(N_ts, 6))
        part = ic_two_body(mass1=mass1, mass2=mass2, rp=rp, e=e)
        part.pos = part.pos - part.com_pos()
        dt_copy = dt.copy()
        # for t_i in range(N_ts):
        for t_i in tqdm(range(N_ts), desc=str(dt_copy)):
            part, dt_copy, acc = integrator_rk4(part,
                                            tstep=dt_copy,
                                            acceleration_estimator=acceleration_direct, softening=0.0001)

            Etot_i, _, _ = part.Etot()
            
            array[t_i, :2] = part.pos[0, :2]
            array[t_i, 2:4]= part.pos[1, :2]
            array[t_i, 4]  = Etot_i
            array[t_i, 5]  = dt_copy

            tot_time += dt_copy
            N_ts_cum += 1

            if tot_time >= N_end*Tperiod:
                print('Exceeded time limit')
                break
            elif N_ts_cum >= 10*N_ts:
                print('Exceeded number of time steps')
                break

        np.savez(file_name, array=array)


if __name__ == '__main__':
    # evolve_2body()
    evolve_3body()