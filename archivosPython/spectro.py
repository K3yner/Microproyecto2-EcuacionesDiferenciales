#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autor: Christian Ramírez
Departamento de Matemática
Universidad del Valle de Guatemala
Curso: Ecuaciones diferenciales 1
Realizado para hacer un cálculo rápido del espectro de respuesta
"""

from obspy import read
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ==============================
# 1) Leer el archivo MiniSEED
#    (ajusta el nombre/ubicación a tu archivo)
st = read("ESCTL.HNE.2025.189.21.41.10")
tr = st[0]
data = tr.data.astype(np.float64)

# ⚠️ Datos ya están en aceleración
# Si tus cuentas están en nm/s² por count y la constante 2365.26382 convierte a nm/seg por count,
# mantén esta línea tal como la tenías: counts * (nm/seg² por count) * 1e-9 => m/seg²
accel = data * 1e-9 * 2365.26382   # m/s²

# Vector de tiempo
dt = tr.stats.delta
t = np.linspace(0, dt * (tr.stats.npts - 1), tr.stats.npts)

# ==============================
# 2) PGA
t_pga = t[np.argmax(np.abs(accel))]
PGA = np.max(np.abs(accel))              # m/s²
PGA_g = PGA / 9.80665                    # en g

# Interpolador de aceleración del suelo
ug = interp1d(t, accel, bounds_error=False, fill_value=(accel[0], accel[-1]))

# ==============================
# 3) Parámetros del SDOF
m = 1.0        # masa unitaria
xi = 0.05      # amortiguamiento 5% crítico

# Periodos a evaluar
periodos = np.linspace(0.04, 4.0, 120)   # s
Sa = np.zeros_like(periodos)

# ==============================
# 4) Resolver respuesta y obtener Sa (aceleración relativa máxima en g)
for i, T in enumerate(periodos):
    wn = 2 * np.pi / T
    k = m * wn**2
    c = 2 * xi * m * wn

    def edo(ti, y):
        x, v = y
        a = (-c*v - k*x - m*ug(ti)) / m
        return [v, a]

    y0 = [0.0, 0.0]
    sol = solve_ivp(edo, [t[0], t[-1]], y0, t_eval=t, method='RK45')
    x = sol.y[0]
    v = sol.y[1]
    a_rel = (-c*v - k*x) / m            # aceleración relativa (m/s²)
    Sa[i] = np.max(np.abs(a_rel)) / 9.80665   # en g

# ==============================
# 5) Graficar SOLO espectro de aceleración
plt.figure(figsize=(6,4))
plt.plot(periodos, Sa, marker='o', label='Sa (g)')
# Marcar el máximo
imax = np.argmax(Sa)
plt.plot(periodos[imax], Sa[imax], '*', markersize=12)
plt.text(periodos[imax], Sa[imax]*(1.05), f"Max: {Sa[imax]:.3f} g en {periodos[imax]:.2f}s",
         fontsize=8, ha='left', va='bottom')
# Marcar PGA en T=0
plt.plot(0, PGA_g, 's', markersize=8, label=f'PGA: {PGA_g*100:.2f}% g')
plt.xlabel('Periodo natural (s)')
plt.ylabel('Aceleración (g)')
plt.title('Espectro de Respuesta - Aceleración relativa (5% amort.)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
