import numpy as np
import taichi as ti

# declares
NR = 0
NZ = 0
ti_BR = None
ti_BZ = None
ti_Bphi = None
ti_Rmin = None
ti_Zmin = None
ti_inv_dR = None
ti_inv_dZ = None

#load eq from freegs solve 
def load_equilibrium(npz_path="eq_fields.npz"):
   
    global NR, NZ, ti_BR, ti_BZ, ti_Bphi, ti_Rmin, ti_Zmin, ti_inv_dR, ti_inv_dZ

    data = np.load(npz_path)
    R = data["R"].astype(np.float32)          # (NR,)
    Z = data["Z"].astype(np.float32)          # (NZ,)
    BR = data["BR"].astype(np.float32)        # (NR, NZ)
    BZ = data["BZ"].astype(np.float32)        # (NR, NZ)
    Bphi = data["Bphi"].astype(np.float32)    # (NR, NZ)

    NR = R.shape[0]
    NZ = Z.shape[0]

    
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])
    Rmin = float(R[0])
    Zmin = float(Z[0])

    #creating taichi fields
    ti_BR = ti.field(dtype=ti.f32, shape=(NR, NZ))
    ti_BZ = ti.field(dtype=ti.f32, shape=(NR, NZ))
    ti_Bphi = ti.field(dtype=ti.f32, shape=(NR, NZ))

    ti_BR.from_numpy(BR)
    ti_BZ.from_numpy(BZ)
    ti_Bphi.from_numpy(Bphi)

    
    ti_Rmin = ti.field(dtype=ti.f32, shape=())
    ti_Zmin = ti.field(dtype=ti.f32, shape=())
    ti_inv_dR = ti.field(dtype=ti.f32, shape=())
    ti_inv_dZ = ti.field(dtype=ti.f32, shape=())

    ti_Rmin[None] = Rmin
    ti_Zmin[None] = Zmin
    ti_inv_dR[None] = 1.0 / dR
    ti_inv_dZ[None] = 1.0 / dZ


#bilinear for B grid
@ti.func
def interp2d_uniform(Rq: ti.f32, Zq: ti.f32, F) -> ti.f32:
    # Convert to grid coordinates
    gx = (Rq - ti_Rmin[None]) * ti_inv_dR[None]
    gz = (Zq - ti_Zmin[None]) * ti_inv_dZ[None]

    i0 = ti.cast(ti.floor(gx), ti.i32)
    j0 = ti.cast(ti.floor(gz), ti.i32)

    # Clamp to valid cell range
    i0 = ti.max(0, ti.min(i0, ti.static(NR - 2)))
    j0 = ti.max(0, ti.min(j0, ti.static(NZ - 2)))

    i1 = i0 + 1
    j1 = j0 + 1

    fx = gx - ti.cast(i0, ti.f32)
    fz = gz - ti.cast(j0, ti.f32)

    c00 = F[i0, j0]
    c10 = F[i1, j0]
    c01 = F[i0, j1]
    c11 = F[i1, j1]

    c0 = c00 * (1.0 - fx) + c10 * fx
    c1 = c01 * (1.0 - fx) + c11 * fx
    return c0 * (1.0 - fz) + c1 * fz


@ti.func
def B_cartesian(x: ti.f32, y: ti.f32, z: ti.f32) -> ti.types.vector(3, ti.f32):
    Rq = ti.sqrt(x * x + y * y)

    B = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
    if Rq >= 1e-6:
        phi = ti.atan2(y, x)

        BRq = interp2d_uniform(Rq, z, ti_BR)
        BZq = interp2d_uniform(Rq, z, ti_BZ)
        Bphiq = interp2d_uniform(Rq, z, ti_Bphi)

        Bx = BRq * ti.cos(phi) - Bphiq * ti.sin(phi)
        By = BRq * ti.sin(phi) + Bphiq * ti.cos(phi)
        B = ti.Vector([Bx, By, BZq], dt=ti.f32)

    return B
    