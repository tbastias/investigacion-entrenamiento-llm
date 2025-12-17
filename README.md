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

Operaciones rápidas para deep learning

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

Ejemplo mental: _**“Quiero entrenar sin preocuparme por la infraestructura”**_

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

Ejemplo mental: _**“Fine-tuning rápido en una GPU chica”**_

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

En LLMs suele ser texto (contenido en archivos .json o .csv), que luego se transforma en tokens.

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

**Herramientas:** `torch.nn.CrossEntropyLoss`, `transformers.Trainer` (la maneja internamente)

## Loss Function

Fórmula matemática que calcula la loss.

En LLMs normalmente es Cross-Entropy Loss.

## Backward Pass (Backpropagation)

Proceso donde el modelo:
- Calcula el error (loss)
- Propaga ese error hacia atrás
- Ajusta los pesos

**Ejemplos en uso:**
- `loss.backward()`
- `optimizer.step()`
- `optimizer.zero_grad()`

Todo esto lo maneja **PyTorch** (autograd).

## Gradient

Indica en qué dirección y cuánto ajustar los pesos para reducir la loss.

## Optimizer

Algoritmo que usa los gradientes para actualizar los pesos.
_Ejemplos:_
- _Adam_
- _AdamW_
- _SGD_

## Learning Rate

Qué tan grandes son los pasos al ajustar los pesos.

- Muy alto → el modelo no converge
- Muy bajo → entrena lento

## Epoch

Una pasada completa del modelo sobre todo el dataset.

## Batch

Subconjunto del dataset usado en cada paso de entrenamiento.

## Step / Iteration

Una actualización del modelo usando un batch.

---

# LoRA (Low-Rank Adaptation)

## ¿Qué es? 

Un método para adaptar modelos grandes de IA (como LLMs o modelos de generación de imágenes) a tareas específicas de forma eficiente.
    
**¿Cómo funciona?** 

Congela los pesos originales del modelo y añade matrices pequeñas de bajo rango que se entrenan para la nueva tarea, reduciendo drásticamente los recursos (memoria, tiempo).
- **Aplicaciones:** Personalizar chatbots para dominios específicos, crear estilos artísticos únicos en IA generativa, o mejorar la detección de objetos en visión por computadora.
- **Beneficios:** Mantiene el rendimiento del modelo completo pero con una fracción del entrenamiento y almacenamiento. 
