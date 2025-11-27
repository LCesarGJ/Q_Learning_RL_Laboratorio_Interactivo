# %% [markdown]
# # Q-Learning con Gymnasium: Laboratorio Interactivo de Aprendizaje por Refuerzo
# 
# Este notebook tiene como objetivo explicar de forma **académica y práctica** el algoritmo de **Q-Learning**, utilizando entornos clásicos de la librería `gymnasium`.
# 
# La idea es que puedas:
# 
# - Entender los conceptos básicos de **aprendizaje por refuerzo (RL)**.
# - Ver cómo se implementa **Q-Learning** paso a paso.
# - Jugar con **hiperparámetros** mediante widgets interactivos.
# - Visualizar cómo evoluciona el aprendizaje del agente con diferentes configuraciones.

# %% [markdown]
# ## Q-Learning en pocas palabras
# 
# **Q-Learning** es un algoritmo de RL basado en **valores**. En lugar de aprender directamente una política, aprende una función:
# 
# \[
# Q(s, a) \approx \text{qué tan buena es la acción } a \text{ si estoy en el estado } s.
# \]
# 
# La actualización clásica de Q-Learning es:
# 
# \[
# Q(s,a) \leftarrow Q(s,a) + \alpha \big( r + \gamma \max_{a'}Q(s',a') - Q(s,a)\big)
# \]
# 
# Donde:
# 
# - `α` (**alpha**) es la **tasa de aprendizaje**.
# - `γ` (**gamma**) es el **factor de descuento** (cuánto valoramos recompensas futuras).
# - `r` es la recompensa inmediata.
# - `\max_{a'}Q(s',a')` es la mejor estimación de valor en el siguiente estado `s'`.
# 
# Además usaremos una **política ε-greedy**:
# 
# - Con probabilidad `ε` elegimos una acción **aleatoria** (exploración).
# - Con probabilidad `1 - ε` elegimos la acción con **mejor Q** (explotación).
# 
# Con el tiempo, reducimos `ε` para explorar menos y explotar más lo aprendido.

# %%
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium.envs.registration import registry

import ipywidgets as widgets
from ipywidgets import interactive

# Asegurar que las gráficas se muestren en el notebook
%matplotlib inline


# %% [markdown]
# ## Implementación de Q-Learning
# 
# A continuación definimos una función general `train_q_agent` que:
# 
# - Crea el entorno (`env_name`).
# - Inicializa una tabla `Q` con ceros.
# - Entrena por `num_episodes` episodios.
# - Aplica política ε-greedy.
# - Actualiza la Q-table con la ecuación de Q-Learning.
# - Guarda:
#   - Recompensa por episodio.
#   - Pasos por episodio.
#   - Éxito (1 si logró el objetivo, 0 si no).
#   - Historial de `ε`.

# %%
def train_q_agent(
    env_name: str,
    num_episodes: int = 2000,
    max_steps: int = 100,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_initial: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
):
    """Entrena un agente Q-Learning en un entorno discreto de Gymnasium.

    Parámetros
    ----------
    env_name : str
        Nombre del entorno (por ejemplo: 'FrozenLake-v1').
    num_episodes : int
        Número de episodios de entrenamiento.
    max_steps : int
        Máximo de pasos por episodio.
    alpha : float
        Tasa de aprendizaje.
    gamma : float
        Factor de descuento.
    epsilon_initial : float
        Valor inicial de epsilon para la política ε-greedy.
    epsilon_min : float
        Valor mínimo de epsilon.
    epsilon_decay : float
        Factor multiplicativo de decaimiento de epsilon por episodio.

    Retorna
    -------
    q_table : np.ndarray
        Tabla Q aprendida de forma tabular.
    rewards_per_episode : np.ndarray
        Recompensa obtenida en cada episodio.
    steps_per_episode : np.ndarray
        Número de pasos usados en cada episodio.
    success_per_episode : np.ndarray
        Indicador 1/0 de si el agente logró el objetivo en cada episodio.
    epsilon_history : np.ndarray
        Valor de epsilon utilizado en cada episodio.
    """

    env = gym.make(env_name)

    # Validamos que el entorno tenga espacios discretos
    assert hasattr(env.observation_space, 'n'), "El entorno debe tener estados discretos (observation_space.n)."
    assert hasattr(env.action_space, 'n'), "El entorno debe tener acciones discretas (action_space.n)."

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Inicializamos la tabla Q con ceros
    q_table = np.zeros((n_states, n_actions))

    rewards_per_episode = np.zeros(num_episodes)
    steps_per_episode = np.zeros(num_episodes)
    success_per_episode = np.zeros(num_episodes)
    epsilon_history = np.zeros(num_episodes)

    epsilon = epsilon_initial

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        epsilon_history[episode] = epsilon

        for step in range(max_steps):
            # Política ε-greedy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            next_state, reward, terminated, truncated, _ = env.step(action)

            # Actualización Q-Learning
            best_next_action = np.argmax(q_table[next_state, :])
            td_target = reward + gamma * q_table[next_state, best_next_action]
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error

            total_reward += reward
            state = next_state

            if terminated or truncated:
                # Para entornos tipo FrozenLake, reward suele ser 1 si llegó al objetivo
                success_per_episode[episode] = 1 if reward > 0 else 0
                steps_per_episode[episode] = step + 1
                break

            # Si no se terminó el episodio por done/trunc, y se acabaron los pasos:
            if step == max_steps - 1:
                steps_per_episode[episode] = max_steps

        rewards_per_episode[episode] = total_reward

        # Actualizamos epsilon (decaimiento multiplicativo)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    env.close()
    return (
        q_table,
        rewards_per_episode,
        steps_per_episode,
        success_per_episode,
        epsilon_history,
    )


# %% [markdown]
# ## Visualizaciones del aprendizaje
# 
# Usaremos varias gráficas para entender el comportamiento del agente:
# 
# - **Recompensa por episodio** (con media móvil).
# - **Número de pasos por episodio** (eficiencia).
# - **Tasa de éxito** (probabilidad de llegar al objetivo).
# - **Curva de epsilon** (cuánto explora vs explota).
# - **Mapa de valores Q** por estado-acción.
# - **Política aprendida** en forma de flechas (para entornos tipo tablero).

# %%
def moving_average(x, window=100):
    if len(x) < window:
        return x  # si hay pocos episodios, devolvemos tal cual
    weights = np.ones(window) / window
    return np.convolve(x, weights, mode='valid')

# %%
def plot_rewards(rewards, window=100):
    episodes = np.arange(len(rewards))
    ma = moving_average(rewards, window)

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, rewards, alpha=0.3, label='Recompensa por episodio')
    if len(rewards) >= window:
        plt.plot(episodes[window-1:], ma, label=f'Media móvil ({window})')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.title('Curva de Aprendizaje: Recompensa')
    plt.grid(True)
    plt.legend()
    plt.show()

# %%
def plot_steps(steps, window=100):
    episodes = np.arange(len(steps))
    ma = moving_average(steps, window)

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, steps, alpha=0.3, label='Pasos por episodio')
    if len(steps) >= window:
        plt.plot(episodes[window-1:], ma, label=f'Media móvil ({window})')
    plt.xlabel('Episodio')
    plt.ylabel('Pasos')
    plt.title('Eficiencia del Agente (menos pasos = mejor)')
    plt.grid(True)
    plt.legend()
    plt.show()

# %%
def plot_success(success, window=100):
    episodes = np.arange(len(success))
    ma = moving_average(success, window)

    plt.figure(figsize=(8, 4))
    if len(success) >= window:
        plt.plot(episodes[window-1:], ma, label=f'Tasa de éxito móvil ({window})')
    else:
        plt.plot(episodes, success, label='Éxito por episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Probabilidad de éxito')
    plt.ylim(0, 1.05)
    plt.title('Curva de Aprendizaje: Tasa de Éxito')
    plt.grid(True)
    plt.legend()
    plt.show()

# %%
def plot_epsilon(epsilon_history):
    episodes = np.arange(len(epsilon_history))
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, epsilon_history)
    plt.xlabel('Episodio')
    plt.ylabel('Epsilon')
    plt.title('Evolución de Epsilon (Exploración vs Explotación)')
    plt.grid(True)
    plt.show()

# %%
def plot_q_table(q_table, env_name):
    plt.figure(figsize=(8, 4))
    plt.imshow(q_table, aspect='auto')
    plt.colorbar(label='Valor Q')
    plt.xlabel('Acción')
    plt.ylabel('Estado')
    plt.title(f'Mapa de Valores Q - {env_name}')
    plt.show()

# %%
def plot_policy_grid(q_table, env_name, grid_size=None):
    """Muestra la política greedy como flechas en una grilla.

    Solo tiene sentido para entornos donde los estados son casillas de una grilla cuadrada,
    como FrozenLake 4x4 por defecto.

    Parámetros
    ----------
    q_table : np.ndarray
        Tabla Q aprendida.
    env_name : str
        Nombre del entorno (solo para el título).
    grid_size : int, opcional
        Tamaño de la grilla (por ejemplo, 4 para 4x4).
        Si es None, se intenta inferir a partir de la cantidad de estados.
    """
    n_states = q_table.shape[0]

    if grid_size is None:
        root = int(np.sqrt(n_states))
        if root * root == n_states:
            grid_size = root
        else:
            print("No se pudo inferir un tamaño de grilla cuadrada a partir del número de estados.")
            return

    policy = np.argmax(q_table, axis=1).reshape(grid_size, grid_size)
    arrow_map = {
        0: '←',
        1: '↓',
        2: '→',
        3: '↑'
    }

    print(f"Política greedy aprendida para {env_name}:\n")
    for i in range(grid_size):
        row = ""
        for j in range(grid_size):
            a = policy[i, j]
            row += arrow_map.get(a, '.') + " "
        print(row)


# %% [markdown]
# ## Evaluación de la política aprendida
# 
# Una vez entrenada la Q-table, podemos probar al agente usando una **política totalmente greedy** (siempre elige la acción con mejor valor Q) y medir su desempeño.

# %%
def run_greedy_episode(env_name, q_table, max_steps=100, render=False):
    """Ejecuta un episodio usando la política greedy derivada de la Q-table.

    Retorna la recompensa total y los pasos utilizados.
    """
    env = gym.make(env_name, render_mode='human' if render else None)
    state, _ = env.reset()
    total_reward = 0

    for t in range(max_steps):
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break

    env.close()
    return total_reward, t + 1

# %% [markdown]
# ## Laboratorio Interactivo de Q-Learning
# 
# En esta sección puedes jugar con:
# 
# - El entorno.
# - El número de episodios.
# - El máximo de pasos por episodio.
# - Los hiperparámetros `α` (alpha), `γ` (gamma), `ε` inicial, mínimo y su decaimiento.
# 
# Y observar cómo cambian las curvas de:
# 
# - Recompensa.
# - Pasos.
# - Tasa de éxito.
# - Epsilon.
# - Q-table.
# 
# Además, se prueba la política aprendida en varios episodios para ver su desempeño promedio.

# %%
def run_simulation_dashboard(
    env_name='FrozenLake-v1',
    num_episodes=2000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon_initial=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
):
    print(f"Entrenando agente Q-Learning en el entorno: {env_name}\n")
    print(f"Episodios: {num_episodes}, Max pasos: {max_steps}")
    print(f"alpha={alpha}, gamma={gamma}, epsilon_initial={epsilon_initial}, epsilon_min={epsilon_min}, epsilon_decay={epsilon_decay}\n")


    q_table, rewards, steps, success, epsilon_hist = train_q_agent(
        env_name=env_name,
        num_episodes=num_episodes,
        max_steps=max_steps,
        alpha=alpha,
        gamma=gamma,
        epsilon_initial=epsilon_initial,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    # Métricas generales
    overall_success_rate = success.mean()
    last_window = min(200, len(success))
    recent_success_rate = success[-last_window:].mean() if last_window > 0 else np.nan

    print(f"Tasa de éxito global: {overall_success_rate:.3f}")
    print(f"Tasa de éxito en los últimos {last_window} episodios: {recent_success_rate:.3f}\n")


    # Gráficas
    plot_rewards(rewards)
    plot_steps(steps)
    plot_success(success)
    plot_epsilon(epsilon_hist)
    plot_q_table(q_table, env_name)

    # Política en grilla (si aplica)
    print("\nPolítica (si el entorno es de tipo grilla cuadrada):\n")
    plot_policy_grid(q_table, env_name)

    # Evaluamos la política greedy unas cuantas veces
    n_test_episodes = 20
    test_rewards = []
    test_steps = []

    for _ in range(n_test_episodes):
        r, s = run_greedy_episode(env_name, q_table, max_steps=max_steps, render=False)
        test_rewards.append(r)
        test_steps.append(s)

    print("\nEvaluación de la política greedy (sin exploración):")
    print(f"Episodios de prueba: {n_test_episodes}")
    print(f"Recompensa promedio: {np.mean(test_rewards):.3f}")
    print(f"Pasos promedio: {np.mean(test_steps):.2f}\n")

# %%
# Definimos los widgets para el laboratorio interactivo

ENVIRONMENT = widgets.Dropdown(
    options=['FrozenLake-v1', 'Taxi-v3', 'CliffWalking-v1'],
    value='FrozenLake-v1',
    description='Entorno:',
)

NUM_EPISODES = widgets.IntSlider(
    value=2000,
    min=100,
    max=10000,
    step=100,
    description='Episodios:',
)

MAX_STEPS = widgets.IntSlider(
    value=100,
    min=10,
    max=500,
    step=10,
    description='Max pasos:',
)

ALPHA = widgets.FloatSlider(
    value=0.1,
    min=0.01,
    max=1.0,
    step=0.01,
    description='Alpha:',
)

GAMMA = widgets.FloatSlider(
    value=0.99,
    min=0.5,
    max=0.999,
    step=0.01,
    description='Gamma:',
)

EPSILON_INITIAL = widgets.FloatSlider(
    value=1.0,
    min=0.0,
    max=1.0,
    step=0.05,
    description='Epsilon0:',
)

EPSILON_MIN = widgets.FloatSlider(
    value=0.01,
    min=0.0,
    max=1.0,
    step=0.01,
    description='Eps. min:',
)

EPSILON_DECAY = widgets.FloatSlider(
    value=0.995,
    min=0.90,
    max=0.999,
    step=0.001,
    description='Eps. decay:',
)

# %% [markdown]
# ## Ejecute el entorno interactivo

# %%
interactive_plot = interactive(
    run_simulation_dashboard,
    env_name=ENVIRONMENT,
    num_episodes=NUM_EPISODES,
    max_steps=MAX_STEPS,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon_initial=EPSILON_INITIAL,
    epsilon_min=EPSILON_MIN,
    epsilon_decay=EPSILON_DECAY,
)

# %%
display(interactive_plot)

