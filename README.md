# Q_Learning_RL_Laboratorio_Interactivo
Este proyecto implementa un laboratorio interactivo de Q-Learning, diseñado para aprender, visualizar y experimentar con los fundamentos del aprendizaje por refuerzo (Reinforcement Learning – RL) usando entornos clásicos de la librería Gymnasium.

El objetivo es que puedas entender cómo aprende un agente, cómo se comportan los hiperparámetros, y cómo se traduce eso en desempeño real dentro de un entorno controlado.

## ¿Qué contiene este laboratorio?

Implementación completa de Q-Learning

  - Ecuación de actualización tabular.
  
  - Política ε-greedy configurable.
  
  - Decaimiento exponencial de epsilon.

Registro detallado: recompensas, pasos, tasa de éxito, exploración, Q-Table.

  Visualizaciones clave
  
  - Este proyecto incluye funciones para graficar:
  
  - Recompensa por episodio + media móvil
  
  - Pasos utilizados por episodio
  
  - Tasa de éxito del agente
  
  - Curva de epsilon (exploración vs explotación)
  
  - Mapa de calor de la Q-Table
  
  - Política aprendida en formato de flechas (para entornos tipo grilla)
  
** Estas visualizaciones permiten entender qué aprende el agente, cómo lo aprende y cuándo converge. **

Widgets interactivos

  El notebook incluye un laboratorio completo con ipywidgets, donde puedes manipular:
  
  alpha (tasa de aprendizaje)
  
  gamma (factor de descuento)
  
  epsilon_initial, epsilon_min, epsilon_decay
  
  Número de episodios
  
  Máximo de pasos
  
  Selección de entorno (FrozenLake-v1, Taxi-v3, CliffWalking-v0)
  
  Cada cambio actualiza automáticamente las curvas, métricas y Q-Table.

Evaluación de la política (modo greedy)

  Después del entrenamiento, puedes evaluar qué tan bien funciona el agente sin exploración, repitiendo episodios y midiendo:
  
  Recompensa promedio

## ¿Qué aprende el usuario con este proyecto?

Cómo funciona el ciclo agente → acción → entorno → recompensa.

Por qué la exploración es necesaria en RL.

Cómo afecta cada hiperparámetro al aprendizaje.

Cómo interpretar una Q-Table.

Cómo se ve una política óptima en entornos simples.

Cómo evaluar RL más allá del “reward final”.

Este proyecto sirve tanto para estudiantes como para analistas que quieran migrar hacia áreas de IA, optimización o automatización inteligente.
  
  Pasos promedio
  
  Estabilidad de la política

## Requisitos

Instalar dependencias:

pip install gymnasium
pip install gymnasium[classic-control]
pip install ipywidgets


En Jupyter, activar widgets:

jupyter nbextension enable --py widgetsnbextension


En JupyterLab:

pip install jupyterlab_widgets

## Experimentos recomendados

Para sacarle jugo al laboratorio:

1. Explorar el impacto de los hiperparámetros

Cambia alpha de 0.05 a 0.9:
¿es más estable? ¿más ruidoso?

Prueba gamma=0.5 vs gamma=0.99:
¿el agente planea a corto o largo plazo?

Juega con epsilon_decay:
¿explora suficiente o se “estanca”?

2. Comparar entornos

FrozenLake-v1 (estocástico, resbaloso)

Taxi-v3 (mucho más complejo)

CliffWalking-v0 (entorno castigador)

3. Analizar la política aprendida

¿La política greedy converge?

¿El mapa de valores Q tiene sentido?

¿Hay estados donde el agente no aprende?
