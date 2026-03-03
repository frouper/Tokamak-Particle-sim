import taichi as ti
import math



# ----------------------------
# Geometry from MASTU
# ----------------------------
pf_coils = [
    (0.49,  1.76), (0.49, -1.76),   # P2
    (1.10,  1.10), (1.10, -1.10),   # P3
    (1.51,  1.095),(1.51, -1.095),  # P4
    (1.66,  0.52), (1.66, -0.52),   # P5
    (1.50,  0.90), (1.50, -0.90),   # P6
]

rings = pf_coils

# D shape 
N_COILS = len(rings)
R0 = 0.9
a  = 0.7
d  = 0.5
k  = 1.9


WALL_RES = 1000 #points per ring

N_POLODIAL = 12 # number of polodial rings (WALL)
N_TORODIAL = 12 # number of torodial rings (WALL)



N_COILS_RES = 256
N_VERTS = N_COILS * N_COILS_RES

# ----------------------------
# construction 
# ----------------------------

# COILS 
coils_verts = ti.Vector.field(3, dtype=ti.f32, shape=N_VERTS)
coils_inds  = ti.field(dtype=ti.i32, shape=(N_VERTS, 2))

ring_centers = ti.Vector.field(2, dtype=ti.f32, shape=N_COILS)
for i, (R, Z) in enumerate(rings):
    ring_centers[i] = ti.Vector([R, Z])


# WALLS 
N_WALL_VERTS = WALL_RES * (N_POLODIAL + N_TORODIAL)

wall_verts = ti.Vector.field(3, dtype=ti.f32, shape=N_WALL_VERTS)
wall_inds  = ti.field(dtype=ti.i32, shape=(N_WALL_VERTS , 2))

# ----------------------------
# coils (MASTU geo)
# ----------------------------

@ti.kernel
def build_coils():
    for k in range(N_COILS):
        R = ring_centers[k][0]
        Z = ring_centers[k][1]
        base = k * N_COILS_RES

        for i in range(N_COILS_RES):
            theta = 2.0 * math.pi * i / N_COILS_RES
            idx = base + i
            next_idx = base + ((i + 1) % N_COILS_RES)

            x = R * ti.cos(theta)
            y = R * ti.sin(theta)
            z = Z

            coils_verts[idx] = ti.Vector([x, y, z])
            coils_inds[idx, 0] = idx
            coils_inds[idx, 1] = next_idx


@ti.kernel
def build_wall_wireframe():
    # --------------------------------
    # Poloidal rings 
    # --------------------------------
    for p in range(N_POLODIAL):
        phi = 2.0 * math.pi * p / N_POLODIAL
        cphi = ti.cos(phi)
        sphi = ti.sin(phi)

        base = p * WALL_RES

        for i in range(WALL_RES):
            t = 2.0 * math.pi * i / WALL_RES

            R = R0 + a * ti.cos(t + d * ti.sin(t))
            Z = k * a * ti.sin(t)

            x = R * cphi
            y = R * sphi
            z = Z

            idx = base + i
            nxt = base + (i + 1) % WALL_RES

            wall_verts[idx] = ti.Vector([x, y, z])
            wall_inds[idx, 0] = idx
            wall_inds[idx, 1] = nxt

    # --------------------------------
    # Toroidal rings 
    # --------------------------------
    offset = N_POLODIAL * WALL_RES

    for q in range(N_TORODIAL):
        t = 2.0 * math.pi * q / N_TORODIAL

        R = R0 + a * ti.cos(t + d * ti.sin(t))
        Z = k * a * ti.sin(t)

        base = offset + q * WALL_RES

        for i in range(WALL_RES):
            phi = 2.0 * math.pi * i / WALL_RES

            x = R * ti.cos(phi)
            y = R * ti.sin(phi)
            z = Z

            idx = base + i
            nxt = base + (i + 1) % WALL_RES

            wall_verts[idx] = ti.Vector([x, y, z])
            wall_inds[idx, 0] = idx
            wall_inds[idx, 1] = nxt











