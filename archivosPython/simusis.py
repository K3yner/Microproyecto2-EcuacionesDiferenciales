# Simulación de un sistema masa-resorte-amortiguador (SDOF)
# usando un registro sísmico real en MiniSEED (HNE) convertido a aceleración.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autor: Christian Ramírez
Departamento de Matemática
Universidad del Valle de Guatemala
Curso: Ecuaciones diferenciales 1
Realizado una simulacion del movimiento puntual de un edificio
"""
from obspy import read
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d

# ==============================
# Leer MiniSEED (HNE)
st = read("ESCTL.HNE.2025.189.21.41.10")
tr = st[0]
data = tr.data.astype(np.float64)

# Velocidad (nm/s) -> m/s con calibración; luego derivar a aceleración
dt = tr.stats.delta
vel_m_s = data * 1e-9 * 2365.26382
accel = np.gradient(vel_m_s, dt)  # m/s²
t = np.linspace(0, dt * (tr.stats.npts-1), tr.stats.npts)

# ==============================
# Parámetros del sistema (SDOF)
m = 2000.0   # kg
b = 500.0    # N·s/m
k = 80000.0  # N/m

ug_func = interp1d(t, accel, bounds_error=False, fill_value=(accel[0], accel[-1]))

def sistema(ti, y):
    x, v = y
    a = (-b*v - k*x - m*ug_func(ti)) / m
    return [v, a]

y0 = [0.0, 0.0]
sol = solve_ivp(sistema, [t[0], t[-1]], y0, t_eval=t, method='RK45')
x = sol.y[0]
v = sol.y[1]
a_rel = (-b*v - k*x)/m

# ==============================
# Gráfica rápida de aceleración
plt.figure()
plt.plot(t, a_rel)
plt.xlabel('Tiempo (s)')
plt.ylabel('Aceleración relativa (m/s²)')
plt.title('Aceleración relativa')
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================
# --- ANIMACIÓN MÁS RÁPIDA ---
DUR_ANIM_S = 12          # anima solo los primeros 12 s (ajusta)
STRIDE_FRAMES = 5        # salta 5 muestras por frame (ajusta)
FPS = 60                 # cuadros por segundo
INTERVAL_MS = 1000 // FPS

n_win = int(DUR_ANIM_S / dt)
n_win = min(n_win, len(t))
frames_idx = np.arange(0, n_win, STRIDE_FRAMES)

# Escala visual (para que el bloque se vea dentro del eje)
# Si el desplazamiento es muy pequeño, amplifícalo:
ESCALA = 10.0           # multiplica x visualmente (ajusta a gusto)
x_vis = ESCALA * x

# Animación horizontal
fig, ax = plt.subplots(figsize=(5, 4))
ax.set_xlim(-0.3, 0.3)
ax.set_ylim(-0.2, 0.2)
ax.hlines(0, -0.3, 0.3, colors='black')
ax.grid(True)
rect, = ax.plot([], [], 's', markersize=40)

ax.set_title('Movimiento en una dirección del Edificio')

def init():
    rect.set_data([], [])
    return rect,

def update(i):
    idx = frames_idx[i]
    # recorta/“clip” por si x_vis sale de los límites
    xv = np.clip(x_vis[idx], -0.28, 0.28)
    rect.set_data([xv], [0])
    return rect,

ani = animation.FuncAnimation(
    fig, update, frames=len(frames_idx),
    init_func=init, blit=True, interval=INTERVAL_MS
)

plt.tight_layout()
plt.show()
