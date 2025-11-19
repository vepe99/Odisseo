# from autocvd import autocvd
# autocvd(num_gpus = 1)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import matplotlib.pyplot as plt

import jax.random as jr

from unxt import Quantity
import galax.coordinates as gc
import galax.potential as gp
import galax.dynamics as gd

import agama

import time


if __name__ == "__main__":

    pot = gp.BovyMWPotential2014()

    w = gc.PhaseSpacePosition(q=Quantity([40, 0, 0], "kpc"),
                              p=Quantity([0, 100, 0], "km/s"),
                            )

    t_array = Quantity(-np.arange(500+1)*5, "Myr")
    prog_mass = Quantity(1e4, "Msun")
    print("The first run is slow. This is a feature of JIT.")

    # Chen+25 (no prog.)
    df = gd.ChenStreamDF()
    gen = gd.MockStreamGenerator(df, pot)
    start = time.time()
    gen.run(jr.key(0), t_array, w, prog_mass)
    end = time.time()
    print(f"Elapsed time: {end - start} seconds")

    print("The second run should be much faster.")

    # Chen+25 (no prog.)
    df = gd.ChenStreamDF()
    gen = gd.MockStreamGenerator(df, pot)
    start = time.time()
    gen.run(jr.key(0), t_array, w, prog_mass)
    end = time.time()
    print(f"Elapsed time: {end - start} seconds")

    stream_c25, prog_c25 = gen.run(jr.key(0), t_array, w, prog_mass)

    plt.scatter(stream_c25.q.x.value, stream_c25.q.y.value, s=1, label="Chen+25", alpha=0.5)
    plt.xlabel(r'$x\ ({\rm kpc})$')
    plt.ylabel(r'$y\ ({\rm kpc})$')
    plt.xlim(25, 50)
    plt.ylim(-15, 15)
    plt.legend(loc='lower left')
    plt.gca().set_aspect(1)
    plt.savefig('./plot/particlespray_chen25/stream_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    import agama # to calculate action
    agama.setUnits(length=1, velocity=1, mass=1) # working units: 1 Msun, 1 kpc, 1 km/s

    actFinder = agama.ActionFinder(agama.Potential('./MWPotential2024.ini'))


    def get_action(stream, prog, actFinder):
        pos_prog = np.array([prog.q.x.to('kpc').value, prog.q.y.to('kpc').value, prog.q.z.to('kpc').value])
        vel_prog = np.array([prog.p.x.to('km/s').value, prog.p.y.to('km/s').value, prog.p.z.to('km/s').value])
        posvel_prog = np.r_[pos_prog.squeeze(),vel_prog.squeeze()]
        action_prog = actFinder(posvel_prog)
        Jphi_prog = action_prog[2]
        Jr_prog = action_prog[0]

        pos = np.array([stream.q.x.to('kpc').value, stream.q.y.to('kpc').value, stream.q.z.to('kpc').value])
        vel = np.array([stream.p.x.to('km/s').value, stream.p.y.to('km/s').value, stream.p.z.to('km/s').value])
        posvel = np.column_stack((pos.T,vel.T))
        actions = actFinder(posvel)
        Jphi = actions[:,2]
        Jr = actions[:,0]
        
        # DLtot = Ltot - Ltot_prog
        DJphi = Jphi - Jphi_prog
        DJr = Jr - Jr_prog
        return DJphi, DJr
        
    # DJphi, DJr = get_action(stream_f15, prog_f15, actFinder)
    # plt.scatter(DJphi, DJr, s=1, alpha=0.5, label='Fardal+15')

    DJphi, DJr = get_action(stream_c25, prog_c25, actFinder)
    plt.figure()
    plt.scatter(DJphi, DJr, s=1, alpha=0.5, label='Chen+25 no prog.')
    plt.xlabel(r'$\Delta J_\phi\ ({\rm kpc\,km/s})$')
    plt.ylabel(r'$\Delta J_r\ ({\rm kpc\,km/s})$')
    plt.xlim(-120, 120)
    plt.ylim(-100, 100)
    plt.legend(loc='lower right')
    plt.gca().set_aspect(1)
    plt.savefig('./plot/particlespray_chen25/stream_action_plot.png', dpi=300, bbox_inches='tight')
    plt.show()




