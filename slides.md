---
# You can also start simply with 'default'
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://images.unsplash.com/photo-1583074890416-7fd25dbafbdf?q=100&w=1920&auto=format&fit=crop
# some information about your slides (markdown enabled)
title: Deep Q-Network (DQN) y Double DQN
# apply unocss classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
---

# Deep Q-Network (DQN) y Double DQN

Playing Atari Game with DQN

<div class="abs-br m-6 text-xl">
  <a href="https://arxiv.org/abs/1312.5602" target="_blank" class="slidev-icon-btn">
    <academicons:arxiv-square/>
  </a>
  <a href="https://arxiv.org/abs/1509.06461" target="_blank" class="slidev-icon-btn">
    <academicons:arxiv-square/>
  </a>
</div>

---
layout: image-right
image: https://ale.farama.org/_images/breakout.gif
---
# Introducción

- Aprendizaje por refuerzo profundo aplicado a juegos Atari.
- Llegando a superar a humanos en varios juegos.
- Misma arquitectura y entrenamiento para diferentes juegos. 
- Input: imagen de 210x160x3 píxeles (frames).

---

# ¿Por qué DQN?

<v-clicks>

- Q-learning no escala en imágenes (entradas de alta dimensión).
- Problemas de estabilidad al entrenar redes profundas.
- Necesidad de generalización.

</v-clicks>

<v-click>
<div class="flex flex-col items-center">
  <img src="http://production-media.paperswithcode.com/methods/dqn.png" alt="DQN" class="rounded-lg shadow-lg w-1/2">
</div>
</v-click>

---

# De Q-Tabular a Q-Función Parametrizada

- **Q-Función Parametrizada**  
Sustituimos la tabla por una red con parámetros $\theta$:  
$$Q(s,a)\;\approx\;Q(s,a;\theta)$$

- **Actualización en Q-Learning Tabular**  
   $$Q(s,a)\;\leftarrow\;Q(s,a)\;+\;\alpha\;\bigl[r + \gamma\,\max_{a'}Q(s',a') - Q(s,a)\bigr]$$

- **Actualización en DQN**  
   $$\mathcal{L}(\theta)\;=\;\mathbb{E}_{s,a,r,s'}\Bigl[\bigl(r + \gamma\,\max_{a'}Q(s',a';\theta)\;-\;Q(s,a;\theta)\bigr)^2\Bigr]$$

---
layout: image
image: https://miro.medium.com/v2/resize:fit:4800/format:webp/1*hydIcU2SPTr5lYDtVeS_cw.png 
backgroundSize: 50em
---

---
clicks: 5 
---

# Preprocesamiento de Imagen - Paso a Paso

<v-switch>

  <template #1>
    <div class="text-center">
      <img src="/original-frame.png" class="rounded shadow-md mx-auto h-[420px]" />
      <p class="mt-4">1. Imagen Original (210x160 RGB)</p>
    </div>
  </template>

  <template #2>
    <div class="text-center">
      <img src="/grayscale-frame.png" class="rounded shadow-md mx-auto h-[420px]" />
      <p class="mt-4">2. Convertido a Escala de Grises</p>
    </div>
  </template>

  <template #3>
    <div class="text-center">
      <img src="/resized-frame.png" class="rounded shadow-md mx-auto h-[220px]" />
      <p class="mt-4">3. Redimensionado a 110x84</p>
    </div>
  </template>

  <template #4>
    <div class="text-center">
      <img src="/cropped-frame.png" class="rounded shadow-md mx-auto h-[168px]" />
      <p class="mt-4">4. Recortado Final (84x84)</p>
    </div>
  </template>

  <template #5>
    <div class="text-center">
      <img src="/stacked-frames.png" class="rounded shadow-md mx-auto h-[168px]" />
      <p class="mt-4">5. Stack Frames (4x84x84)</p>
    </div>
  </template>


</v-switch>

---
layout: two-cols 
---

# Replay Memory

<ul class="text-left mx-auto max-w-lg">
  <li v-click class="mb-2">
    <strong>¿Por qué?</strong><br>
    • Rompe la correlación temporal de las muestras<br>
    • Reutiliza experiencias para mejorar la eficiencia  
  </li>
  <li v-click class="mb-2">
    <strong>¿Qué guarda?</strong><br>
    Tuplas de transición:  
    <code class="block"> (state, action, reward, next_state, done) </code>
  </li>
  <li v-click class="mb-2">
    <strong>Buffer circular</strong><br>
    Cuando está lleno, sobrescribe las experiencias más antiguas
  </li>
  <li v-click class="mb-2">
    <strong>¿Cuándo se usa?</strong><br>
    Antes de cada update: muestreo aleatorio de un minibatch para entrenar la red
  </li>
</ul>

::right::

```python
# Insertar una experiencia en la memoria
memory.add(state, action, reward, done, next_state)

# Durante cada paso de entrenamiento:
if len(memory) > batch_size:
    # Muestreamos un minibatch aleatorio
    batch = memory.sample(batch_size)
    # Entrenamos la red con ese batch
    agent.update_weights(batch)
```

---

# Arquitectura de la CNN en DQN

- Entrada: tensor de $84 \times 84 \times 4$.
- Capas convolucionales para extraer características espaciales.
- Capas completamente conectadas para estimar valores Q.
- Salida: vector de tamaño $|A|$ (número de acciones posibles).

<br>

<v-click>
  <div class="flex flex-col items-center">
    <img src="https://miro.medium.com/v2/format:webp/1*yfrF2jnI3zspkZELq2rw9g.png" alt="DQN" class="rounded-lg shadow-lg w-200">
  </div>
</v-click>

---

# Algoritmo DQN

<div class="flex flex-col items-center">
  <img src="/dqn_algo.png" alt="DQN" class="rounded-lg shadow-lg w-200">
</div>

---

# Hiperpámetros  

<v-clicks>

- Entrenamiento de 50M de pasos
- $\epsilon$ con decrecimiento lineal: 1 -> 0.1 durante el primer millón de pasos
- $\gamma$ de 0.99
- Replay memory de 1M transiciones
- Batch size de 32

</v-clicks>
