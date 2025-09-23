import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Extensión del gráfico 2D
y_min, y_max = -1.5, 1.5
x_min, x_max = -1.5, 1.5
x = np.linspace(x_min, x_max, 100)
y = np.linspace(y_min, y_max, 100)

# Definición de las cargas y sus posiciones
cargas = [
    {"q": 2e-9, "xq": 1, "yq": 0},  # Carga positiva de 2 nC en (1, 0)
    {"q": 5e-9, "xq": -1, "yq": 0},  # Carga positiva de 5 nC en (-1, 0)
    {"q": -3e-9, "xq": 0, "yq": 0},  # Carga negativa de -3 nC en (0, 0)
]

# Constantes
epsilon_0 = 8.854e-12  # F/m
k = 1 / (4 * np.pi * epsilon_0)  # Constante de Coulomb


# Funciones auxiliares
def campo_electrico(q, xq, yq, x, y):
    r = np.sqrt((x - xq) ** 2 + (y - yq) ** 2)
    Ex = (k * q * (x - xq)) / (r**3)
    Ey = (k * q * (y - yq)) / (r**3)
    return Ex, Ey


def potencial_electrico(q, xq, yq, x, y):
    r = np.sqrt((x - xq) ** 2 + (y - yq) ** 2)
    return k * q / r


def campo_electrico_x(q, xq, yq, x):
    r = np.sqrt((x - xq) ** 2 + (0 - yq) ** 2)
    Ex = k * q * (x - xq) / (r**3)
    return Ex


def potencial_electrico_x(q, xq, yq, x):
    r = np.sqrt((x - xq) ** 2 + (0 - yq) ** 2)
    return k * q / r


# Gráficos individuales Ex(x) para cada carga
for i, carga in enumerate(cargas):
    # Evitar la singularidad en la posición exacta de la carga
    x_filtrado = x[np.abs(x - carga["xq"]) > 1e-3]

    Ex = campo_electrico_x(carga["q"], carga["xq"], carga["yq"], x_filtrado)

    plt.figure(figsize=(8, 5))
    plt.plot(
        x_filtrado,
        Ex,
        label=f"q{i} = {carga['q']:.1e} C en ({carga['xq']}, {carga['yq']})",
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.axvline(carga["xq"], color="red", linestyle=":", label="posición de la carga")
    plt.xlabel("x (m)")
    plt.ylabel("Ex (N/C)")
    plt.title(f"Campo eléctrico en el eje x debido a q{i}")
    plt.legend()
    plt.grid()
    plt.savefig(f"campo_electrico_x_q{i}.png", bbox_inches="tight")
    plt.show()


# Graficos individuales V(x) para cada carga
for i, carga in enumerate(cargas):
    # Evitar la singularidad en la posición exacta de la carga
    x_filtrado = x[np.abs(x - carga["xq"]) > 1e-3]

    V = potencial_electrico_x(carga["q"], carga["xq"], carga["yq"], x_filtrado)

    plt.figure(figsize=(8, 5))
    plt.plot(
        x_filtrado,
        V,
        label=f"q{i} = {carga['q']:.1e} C en ({carga['xq']}, {carga['yq']})",
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.axvline(carga["xq"], color="red", linestyle=":", label="posición de la carga")
    plt.xlabel("x (m)")
    plt.ylabel("V (V)")
    plt.title(f"Potencial eléctrico en el eje x debido a q{i}")
    plt.legend()
    plt.grid()
    plt.savefig(f"potencial_electrico_x_q{i}.png", bbox_inches="tight")
    plt.show()
    
    
# Campo y potencial total en 2D
X, Y = np.meshgrid(x, y)

Ex_total = np.zeros_like(X)
Ey_total = np.zeros_like(Y)
V_total = np.zeros_like(X)

for carga in cargas:
    Ex, Ey = campo_electrico(carga["q"], carga["xq"], carga["yq"], X, Y)
    Ex_total += Ex
    Ey_total += Ey
    V_total += potencial_electrico(carga["q"], carga["xq"], carga["yq"], X, Y)

fig1 = plt.figure(figsize=(10, 10))
ax = fig1.add_subplot()

# Líneas de campo
color = 2 * np.log(np.hypot(Ex_total, Ey_total))
ax.streamplot(
    x,
    y,
    Ex_total,
    Ey_total,
    color=color,
    linewidth=1,
    cmap=plt.cm.inferno,
    density=1.5,
    arrowstyle="->",
    arrowsize=1.5,
)

# Contornos de potencial
plt.contour(X, Y, V_total, 400, cmap="viridis", linewidths=0.7)

# Dibujar las cargas
for indice, carga in enumerate(cargas):
    pos = carga["xq"], carga["yq"]
    ref = f"q{indice} = {carga['q']} C en ({carga['xq']}, {carga['yq']})"
    ax.add_artist(Circle(pos, 0.05, label=ref))
plt.legend(loc="upper left", handlelength=0)

# Etiquetas y formato
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect("equal")
plt.grid()
plt.title("Campo y potencial eléctrico")
plt.colorbar(label="Potencial (V)")
plt.savefig("campo_y_potencial.png", bbox_inches="tight")
plt.show()
