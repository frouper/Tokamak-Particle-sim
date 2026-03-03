#H1 Tokamak Particle Tracker 

A 3D simulation of charged particles magnetically confined within a spherical tokamak. 

The simulation loads a .npz file containing a 2D MHD equilbrium solved with [FreeGS](https://github.com/freegs-plasma/freegs) using MAST-U like dimensions. The equilibrium is interpolated onto a 3d grid where up to thousands of particles are integrated with a boris pusher. 

[Taichi Lang](https://www.taichi-lang.org/) is used to render the wall geometry, coils and particles. 


