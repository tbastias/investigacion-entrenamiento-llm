# 🧱 Capas del entrenamiento de un modelo

Entrenar un modelo suele involucrar varias capas:
- Cálculo numérico (tensores, operaciones)
- Uso de GPU (aceleración)
- Framework de deep learning
- Modelos preentrenados
- Fine-tuning optimizado

Cada librería vive en una de esas capas.

## 🔢 NumPy

**¿Qué es?**  
Librería base para cálculo numérico en Python.

**¿Para qué se usa?**  
- Operaciones matemáticas.
- Arreglos y matrices.

**Importante saber:**

❌ No se usa para entrenar modelos grandes  
✔️ Es la base conceptual de todo

## 🔥 PyTorch (torch)

**¿Qué es?**  
El framework de deep learning más usado hoy.

**¿Para qué sirve?**
- Definir redes neuronales
- Entrenar modelos
- Manejar tensores
- Backpropagation automático

Ejemplo mental: _“Con PyTorch escribo el modelo y el entrenamiento”_

✔️ Muy flexible  
✔️ Ideal para investigación y producción  
✔️ Base de casi todo lo moderno

## 🚀 CUDA

**¿Qué es?**  
Tecnología de NVIDIA para usar la GPU.

**¿Para qué sirve?**
- Acelerar cálculos
- Entrenar modelos mucho más rápido

**Importante:**

❌ No es Python  
❌ No la usás directamente  
✔️ PyTorch la usa por debajo  

Ejemplo mental: _“CUDA es el motor, PyTorch es el volante”_

## ⚙️ cuDNN

**¿Qué es?**  
Librería de NVIDIA optimizada para redes neuronales.

**¿Para qué sirve?**  
Operaciones rápidas para deep learning.

✔️ Totalmente invisible para el usuario  
✔️ Se activa sola si tenés GPU compatible

## 🤗 Transformers (Hugging Face)

**¿Qué es?**  
Librería con modelos preentrenados listos para usar.

**¿Para qué sirve?**
- Usar LLaMA, BERT, GPT-like, etc.
- Fine-tuning
- Inferencia

Ejemplo mental: _“No entreno desde cero, empiezo con un modelo ya inteligente”_

✔️ Ahorra meses de entrenamiento  
✔️ Ideal para principiantes

## 📦 Datasets (Hugging Face)

**¿Qué es?**  
Librería para manejar datasets grandes.

**¿Para qué sirve?**
- Descargar datasets
- Procesarlos
- Stream de datos

✔️ Muy usada junto con transformers

## ⚡ Accelerate

**¿Qué es?**  
Herramienta para escalar el entrenamiento.

**¿Para qué sirve?**
- Multi-GPU
- CPU + GPU
- Configuración simple

Ejemplo mental: _“Quiero entrenar sin preocuparme por la infraestructura”_

## 🧠 PEFT

**¿Qué es?**  
Parameter Efficient Fine-Tuning.

**¿Para qué sirve?**
- Fine-tuning sin entrenar todo el modelo
- Técnicas como LoRA

✔️ Consume menos GPU  
✔️ Ideal para LLMs grandes

## 🦥 Unsloth

**¿Qué es?**  
Framework optimizado para fine-tuning rápido de LLMs.

**¿Para qué sirve?**
- Fine-tuning de LLaMA, Mistral, etc.
- Mucho menos VRAM
- Más velocidad

Ejemplo mental: _“Fine-tuning rápido en una GPU chica”_

✔️ Usa PyTorch + Transformers por debajo  
✔️ Muy popular para setups locales

## 🧩 BitsAndBytes

**¿Qué es?**  
Librería para cuantización.

**¿Para qué sirve?**
- Modelos en 4-bit / 8-bit
- Menor uso de memoria

✔️ Clave para GPUs con poca VRAM

## 📊 Trainer (Transformers)

**¿Qué es?**  
Clase que simplifica el entrenamiento.

**¿Para qué sirve?**
- Entrenar sin escribir mucho código
- Manejar epochs, logs, evaluación

✔️ Ideal para empezar  
❌ Menos control fino

---

# 🧠 Conceptos base del entrenamiento

## Dataset

Conjunto de datos usados para entrenar el modelo.

En LLMs suele ser texto (contenido en archivos `.json` o `.csv`), que luego se transforma en tokens.

## Token

Unidad mínima de texto que entiende el modelo.

No son palabras exactas: pueden ser partes de palabras, símbolos o signos.

## Tokenizer

Herramienta que convierte texto → tokens (y viceversa).

_Ejemplo: SentencePiece, BPE, LLaMA tokenizer._

## Embedding

Representación numérica (vector) de cada token.

Permite que el modelo “entienda” relaciones semánticas.

_Ejemplo: Word2Vec, LLaMA / Transformer embeddings._

---

# 📉 Durante el entrenamiento

## Loss

Función que mide qué tan mal está prediciendo el modelo.

Cuanto más baja la loss → mejor aprende.

_Ejemplo simple: “¿Qué tan lejos está la respuesta del modelo de la respuesta correcta?”_

## Backward Pass (Backpropagation)

Proceso donde el modelo:
- Calcula el error (loss)
- Propaga ese error hacia atrás
- Ajusta los pesos

Todo esto lo maneja **PyTorch** (autograd).

## Optimizer

Algoritmo que usa los gradientes para actualizar los pesos.

_Ejemplos:_ Adam, AdamW, SGD

## Learning Rate

Tamaño de los pasos al ajustar los pesos.

- Muy alto → no converge
- Muy bajo → entrena lento

## Epoch / Batch / Step

- **Epoch:** una pasada completa al dataset  
- **Batch:** subconjunto de datos  
- **Step:** una actualización del modelo

---

# 🗣️ Inferencia y generación de texto (LLMs)
Estos conceptos **no afectan el entrenamiento**, sino **cómo responde el modelo** una vez entrenado.

## attention_mask

Indica qué tokens debe atender el modelo (`1`) y cuáles ignorar (`0`).

Se usa principalmente para ignorar tokens de padding.

## pad_token_id

ID del token usado como relleno (`<PAD>`).

Permite que todas las secuencias tengan la misma longitud.

Normalmente se ignora usando `attention_mask`.

## eos_token_id

Token que indica el final de una secuencia.

Cuando el modelo lo genera, **la respuesta se detiene**.

## apply_chat_template

Función que convierte mensajes tipo chat (`system`, `user`, `assistant`) al formato exacto que el modelo espera.

Cada modelo tiene su propio template.

### LLaMA 3

- Roles explícitos (`system`, `user`, `assistant`)
- Orden estricto
- Muy sensible al formato
- Usa tokens especiales propios de Meta

**Formato conceptual:**

```
<|begin_of_text|>
<|system|>
Eres un asistente útil
<|user|>
Hola
<|assistant|>
```

### Mistral 

- Usa bloques tipo instruct
- El mensaje del usuario va dentro de `[INST] ... [/INST]`
- System prompt opcional
- Más tolerante a errores

**Formato conceptual:**
```
<s>[INST] Hola [/INST]
```

Con System Prompt:

```text
<s>[INST] <<SYS>>
Eres un asistente útil
<</SYS>>
Hola
[/INST]
```

### Qwen / Qwen2 / Qwen2.5stilo ChatML

- Estilo ChatML
- Roles claramente delimitados
- Muy consistente para chat largo
- Menos tolerante que Mistral, más que LLaMA

```text
[BOS]
<role=system>
  instrucciones del sistema
</role>

<role=user>
  mensaje del usuario
</role>

<role=assistant>
  respuesta del asistente
</role>

<role=user>
  siguiente mensaje del usuario
</role>

<role=assistant>
  ...
</role>
```

### Gemma (Google)

- Estilo instruct simple
- No tan orientado a chat multi-turno
- Más cercano a “prompt → respuesta”
- Menos tokens de rol explícitos

**Formato conceptual:**

```text
[BOS]
INSTRUCCIÓN:Qué tiene que hacer el modelo

CONTEXTO (opcional):
  Información adicional

RESPUESTA:


❗``` U

sar el template incorrecto produce respuestas erráticas.

---

# 🎛️ Control de generación (sampling)

Estos parámetros definen **qué tan creativa o determinista** es la respuesta.

## do_sample

Indica si el modelo debe usar muestreo probabilístico.

- `False` → siempre elige el token más probable
- `True` → elige tokens según probabilidades

## Multinomial Sampling

Método de muestreo usado cuando `do_sample=True`.

El token se elige mediante un sorteo ponderado por probabilidad.

## temperature

Controla la aleatoriedad de las probabilidades.

- Baja (`0.2`) → respuestas más seguras
- Alta (`1.0+`) → más creatividad

## top_k

Limita la elección a los `k` tokens más probables.

Reduce ruido y tokens raros.

## top_p (nucleus sampling)

Elige tokens cuya probabilidad acumulada sea menor a `p`.

Más dinámico que `top_k`.

---

# ⚙️ Implementación de atención

## attn_implementation="eager"

Usa la implementación clásica de atención.

✔️ Más compatible  
✔️ Ideal para debugging o CPU  
❌ Más lenta que flash attention

Otras opciones:
- `sdpa`
- `flash_attention_2`

---

# 🧠 LoRA (Low-Rank Adaptation)

Método para adaptar modelos grandes congelando sus pesos y entrenando matrices pequeñas adicionales.

✔️ Menor consumo de memoria  
✔️ Ideal para LLMs grandes  
✔️ Base de PEFT
