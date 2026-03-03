import taichi as ti 
ti.init(arch=ti.cpu) 

# ----------------------------
# Imports
# ----------------------------
import equilibrium_taichi as eq
import geometry
import particles
import numpy as np
import matplotlib as plt 

# ----------------------------
# Load equilibrium from freegs solution
# ----------------------------
eq.load_equilibrium("eq_fields.npz")


#geometry.build_coils()
geometry.build_wall_wireframe()


coil_verts = geometry.coils_verts
coil_inds  = geometry.coils_inds
wall_verts = geometry.wall_verts
wall_inds  = geometry.wall_inds

particles.init_particles(speed=7e5)

# ----------------------------
# testing
# ----------------------------
for step in range(1):
    particles.step_particles()

    if step % 1000 == 0:
        particles.diagnostic_particle(0)
        particles.diagnostic_guiding_center(0)

        print(
            "step", step,
            "phi", particles.diag_phi[None],
            "v_parallel", particles.diag_vpar[None],
            "|B|", particles.diag_Bmag[None],
            "phi_gc", particles.diag_phi_gc[None]
        )




# ----------------------------
# window setup
# ----------------------------
window = ti.ui.Window("Tokamak particles", (1980, 1280))
scene = window.get_scene()
canvas = window.get_canvas()
camera = ti.ui.Camera()

camera.position(3.0, 3.0, 2.0)
camera.lookat(0.0, 0.0, 0.0)
camera.up(0.0, 0.0, 1.0)




# ----------------------------
# runtime
# ----------------------------
while window.running:
    # advance + record
    for _ in range(5):
        particles.step_particles()
       

    
    # camera
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0.4, 0.4, 0.4))
    scene.point_light((5, 5, 5), (1, 1, 1))


    # draw machine geometry
    scene.lines(vertices=geometry.wall_verts, indices=geometry.wall_inds, width=1.2, color=(0.8, 0.8, 0.8))
    scene.lines(vertices=geometry.coils_verts, indices=geometry.coils_inds, width=2.0, color=(0.2, 0.8, 1.0))

    # draw particle
    scene.particles(particles.pos, radius=0.01, color=(0.2, 0.8, 1.0))
    particles.build_tail_lines()
    scene.lines(vertices=particles.tail_verts, indices=particles.tail_inds, width=1.0, color=(1.0, 0.4, 0.1))

    canvas.set_background_color((0.05, 0.05, 0.05))
    canvas.scene(scene)
    window.show()

