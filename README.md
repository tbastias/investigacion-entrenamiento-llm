# 🧱 Capas del entrenamiento de un modelo (idea clave)

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
