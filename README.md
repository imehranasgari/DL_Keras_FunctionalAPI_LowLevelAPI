## 1) Project Title

Keras Functional API: CNNs, Multi-Input/Output Models, and a Toy ResNet (TFDS: MNIST & CIFAR-10)

## 2) Problem Statement and Goal of Project

Demonstrate practical use of TensorFlow/Keras **Functional API** for:

* A baseline **CNN classifier** on **MNIST**,
* A **multi-input, multi-output** model (ticket triage example with synthetic data),
* A small **ResNet-style (“skip connection”)** model trained briefly on **CIFAR-10**,

with **tf.data** pipelines via **TensorFlow Datasets (TFDS)**. The notebook focuses on correct API usage, modeling patterns, and explainability over squeezing out top metrics.

## 3) Solution Approach

The notebook implements three modeling patterns:

1. **Baseline CNN (MNIST)**

   * Input shape: **(28, 28, 1)**
   * Architecture (Functional API):
     `Conv2D(32, 3x3, ReLU) → MaxPool → Conv2D(64, 3x3, ReLU) → MaxPool → Flatten → Dropout(0.5) → Dense(num_classes, softmax)`
   * Loss set via `CategoricalCrossentropy(from_logits=True)` with `metrics=["accuracy"/"acc"]` in different cells.
   * Data via TFDS, normalized and one-hot encoded.

2. **Multi-Input, Multi-Output model (synthetic)**

   * Perso-language markdown describes a ticket-ranking use-case (title/body/tags → priority/department).
   * Code builds a Keras model with **multiple inputs** and **two outputs** (e.g., `"priority"`, `"department"`).
   * Trains on **synthetic NumPy arrays** (dummy data).
   * Uses **dictionary losses** and **`loss_weights`** to balance heads.
   * A diagram is generated with `keras.utils.plot_model("multi_input_and_output_model.png")`.

3. **Toy ResNet-style model (CIFAR-10)**

   * CIFAR-10 loaded from **TFDS** with splits: `train[:80%]` (train), `train[80%:]` (val), and `test`.
   * A small **skip-connection** (“Rednet/ResNet”) topology is defined using the Functional API (residual pattern).
   * Training is run for **a very small number of epochs** (e.g., `epochs=1`) to keep runtime short.
   * A diagram is generated with `keras.utils.plot_model("resnet_me.png", show_shapes=True)`.

Across all parts:

* **TFDS** utilities (e.g., `tfds.visualization.show_examples`) are used to inspect samples.
* **Preprocessing** uses `tf.divide` for **normalization** and `tf.one_hot(..., depth=10)` for labels.
* Batching and shuffling are handled via `tf.data` (e.g., `.map(...).shuffle(1000).batch(64)`).
* Optimizer examples include **`keras.optimizers.RMSprop(1e-3)`**.

> Note: Some code paths intentionally restrict sample counts or epochs (e.g., 1–2 epochs) to demonstrate API usage and keep execution time low.

## 4) Technologies & Libraries

* **Python** (conda kernel name indicates `conda-env-tf-py`)
* **TensorFlow / Keras**
* **TensorFlow Datasets (tfds)**
* **NumPy**

## 5) Description about Dataset

* **MNIST** (from TFDS)

  * Split in code: `train`, `test`
  * Input shape used: **(28, 28, 1)**
  * Labels one-hot encoded with depth **10** (matches MNIST classes).
* **CIFAR-10** (from TFDS)

  * Splits in code: `train[:80%]` (train), `train[80%:]` (validation), `test`
  * Model expects input shape **(32, 32, 3)**.
* Sample visualization via `tfds.visualization.show_examples(...)`.

> If additional dataset specifics (e.g., class names) are needed, they’re standard to MNIST/CIFAR-10 but **not explicitly enumerated** in the notebook.

## 6) Installation & Execution Guide

**Prerequisites**

* Python 3.x
* A working TensorFlow environment (GPU optional)

**Install (pip)**

```bash
pip install tensorflow tensorflow-datasets numpy pydot graphviz
```

> `graphviz` and `pydot` are required to save the model diagrams via `keras.utils.plot_model`.
> On some systems you must also install Graphviz binaries (e.g., `sudo apt-get install graphviz`).

**Run**

1. Open `functional_API_me.ipynb` in Jupyter/VS Code.
2. Run cells sequentially.
3. The notebook will:

   * Download MNIST and CIFAR-10 from TFDS on first run.
   * Train each demo briefly.
   * Save model diagrams (e.g., `my_first_model_with_shape_info.png`, `multi_input_and_output_model.png`, `resnet_me.png`) **when those cells are executed**.

## 7) Key Results / Performance

* **Not provided.**
  The notebook compiles and fits models (often for **1–2 epochs**). It prints evaluation for the MNIST CNN (`model.evaluate(...)`) and runs a brief fit on CIFAR-10, but **no final metrics are stored in the repository**.

## 8) Screenshots / Sample Output

* **Not provided in the repository.**
  The code includes calls to `keras.utils.plot_model(...)` which will create diagrams locally if executed.

## 9) Additional Learnings / Reflections

* **Why Functional API?** It enables **non-linear graphs**, **multiple inputs/outputs**, and **skip connections**—patterns that aren’t possible (or are awkward) in `Sequential`.
* **Loss dictionaries & loss weights:** The multi-head example shows how to balance objectives with `{"priority": ..., "department": ...}` and `loss_weights`.
* **Data pipelines with TFDS + `tf.data`:** The project demonstrates clean mapping, normalization, one-hot encoding, shuffling, and batching directly from TFDS.
* **Exploration over SOTA:** Several cells intentionally run **short trainings** or use **synthetic data** to **teach modeling mechanics** (e.g., model wiring, compilation options, plotting architectures) rather than to maximize accuracy.

---

## 10) 👤 Author

**Mehran Asgari**
**Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com)
**GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari)

---

## 📄 License

This project is licensed under the Apache 2.0 License – see the `LICENSE` file for details.

---

> 💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*
