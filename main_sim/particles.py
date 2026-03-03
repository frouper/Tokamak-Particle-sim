import taichi as ti
import math 
import numpy as np 
import equilibrium_taichi as eq 




# ----------------------------
# particle params
# ----------------------------

N_PART = 1

dt = 1e-8
qm = 4.79e7
qm_dt = dt * qm 

pos   = ti.Vector.field(3, ti.f32, shape=N_PART)
vel   = ti.Vector.field(3, ti.f32, shape=N_PART)
alive = ti.field(ti.i32, shape=N_PART)

TAIL_LEN = 5000

tail_pos = ti.Vector.field(3, ti.f32, shape=(N_PART, TAIL_LEN))
tail_head = ti.field(ti.i32, shape=N_PART)  


diag_phi  = ti.field(ti.f32, shape=())
diag_vpar = ti.field(ti.f32, shape=())
diag_Bmag = ti.field(ti.f32, shape=())
diag_phi_gc = ti.field(ti.f32, shape=())

# ----------------------------
# tokamak shape
# ----------------------------
R0 = 0.9
a  = 0.7
d  = 0.5
k  = 1.9

POLY_N = 256
polyR = ti.field(ti.f32, shape=POLY_N)
polyZ = ti.field(ti.f32, shape=POLY_N)

import numpy as np

Rcpu = np.zeros(POLY_N, np.float32)
Zcpu = np.zeros(POLY_N, np.float32)

for i in range(POLY_N):
    t = 2.0 * math.pi * i / POLY_N
    Rcpu[i] = R0 + a * math.cos(t + d * math.sin(t))
    Zcpu[i] = k * a * math.sin(t)

polyR.from_numpy(Rcpu)
polyZ.from_numpy(Zcpu)


@ti.func
def inside_poloidal(Rq: ti.f32, Zq: ti.f32) -> ti.i32:
    inside = 0
    j = POLY_N - 1
    for i in range(POLY_N):
        Ri, Zi = polyR[i], polyZ[i]
        Rj, Zj = polyR[j], polyZ[j]
        cond = ((Zi > Zq) != (Zj > Zq))
        Rint = (Rj - Ri) * (Zq - Zi) / (Zj - Zi + 1e-12) + Ri
        if cond and (Rq < Rint):
            inside = 1 - inside
        j = i
    return inside

@ti.func
def B_equilibrium(pos: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    return eq.B_cartesian(pos.x, pos.y, pos.z)

# ----------------------------
# particle integrator
# ----------------------------
@ti.func
def boris_push(v, B):
    t = 0.5 * qm_dt * B
    t2 = t.dot(t)
    s = (2.0 * t) / (1.0 + t2)

    v_prime = v + v.cross(t)
    v_plus  = v + v_prime.cross(s)

    return v_plus

@ti.kernel
def init_particles(speed: ti.f32):
    for p in range(N_PART):
        alive[p] = 1

        # random near magnetic axis
        x = R0 + (ti.random() -0.5 ) * a
        y = (ti.random() -0.5) * a
        z = (ti.random() -0.5) * k * a

        Rq = ti.sqrt(x*x + y*y)
        if inside_poloidal(Rq, z) == 0:
            alive[p] = 0

        pos[p] = ti.Vector([x, y, z])

        for i in range(TAIL_LEN):
            tail_pos[p, i] = ti.Vector([x, y, z])
            tail_head[p] = 0

        # random velocity direction
        vx = ti.random() - 0.5
        vy = ti.random() - 0.5
        vz = ti.random() - 0.5
        v = ti.Vector([vx, vy, vz])
        v = v / (ti.sqrt(v.dot(v)) + 1e-12)

        vel[p] = speed * v



@ti.kernel
def step_particles():
    for p in range(N_PART):
        if alive[p] == 0:
            continue

        x = pos[p]
        v = vel[p]

        B = B_equilibrium(x)
        v = boris_push(v, B)
        x = x + dt * v

        # wall loss
        Rq = ti.sqrt(x.x*x.x + x.y*x.y)
        if inside_poloidal(Rq, x.z) == 0:
            alive[p] = 0
            x = ti.Vector([999.0, 999.0, 999.0])

        pos[p] = x
        vel[p] = v
        h = tail_head[p]
        tail_pos[p, h] = x
        tail_head[p] = (h + 1) % TAIL_LEN

tail_verts = ti.Vector.field(3, ti.f32, shape=(N_PART * TAIL_LEN))
tail_inds  = ti.field(ti.i32, shape=(N_PART * TAIL_LEN, 2))

@ti.kernel
def build_tail_lines():
    for p in range(N_PART):
        for i in range(TAIL_LEN - 1):
            h  = (tail_head[p] + i    ) % TAIL_LEN
            hn = (tail_head[p] + i + 1) % TAIL_LEN
            idx = p * TAIL_LEN + i

            tail_verts[idx] = tail_pos[p, h]

            if alive[p] == 1:
                tail_inds[idx, 0] = idx
                tail_inds[idx, 1] = idx + 1  
            else:
                tail_inds[idx, 0] = idx  
                tail_inds[idx, 1] = idx

        
        last_idx = p * TAIL_LEN + TAIL_LEN - 1
        h = (tail_head[p] + TAIL_LEN - 1) % TAIL_LEN
        tail_verts[last_idx] = tail_pos[p, h]
        tail_inds[last_idx, 0] = last_idx
        tail_inds[last_idx, 1] = last_idx  



@ti.kernel
def diagnostic_particle(p: ti.i32):
    x = pos[p]
    v = vel[p]
    B = eq.B_cartesian(x.x, x.y, x.z)
    Bmag = ti.sqrt(B.dot(B)) + 1e-30
    b = B / Bmag

    diag_Bmag[None] = Bmag
    diag_vpar[None] = v.dot(b)
    diag_phi[None]  = ti.atan2(x.y, x.x)
    
@ti.kernel
def diagnostic_guiding_center(p: ti.i32):
    x = pos[p]
    v = vel[p]
    B = eq.B_cartesian(x.x, x.y, x.z)
    Bmag = ti.sqrt(B.dot(B)) + 1e-30
    b = B / Bmag

    vpar = v.dot(b)
    Xgc = x - (v - vpar * b).cross(b) / (qm * Bmag)

    diag_phi_gc[None] = ti.atan2(Xgc.y, Xgc.x)
    




















































