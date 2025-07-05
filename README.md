# TinyNetLab

[中文版 README](docs/README.zh.md)

**A minimal neural network built from scratch in pure NumPy to intuitively understand how neural networks learn nonlinear decision boundaries. Visualize. Experiment. Build your intuition.**

## 🌟 What is TinyNetLab?

**TinyNetLab** is a hands-on educational project where you build and train a neural network from the ground up — no frameworks, no magic. Just pure NumPy code that shows you:

✅ How forward propagation works
✅ How gradients flow in backpropagation
✅ How gradient descent updates parameters
✅ How neural networks learn to draw complex decision boundaries

You’ll see it all in action by training on a simple **2D moons dataset** (a classic for testing nonlinear classifiers) and visualizing the decision boundary as it evolves.

## 🧠 Why use TinyNetLab?

* **Understand, don’t just use:** Go beyond black-box libraries — see the math at work.
* **Minimal but meaningful:** A small network (2-4-1) that’s powerful enough to learn nonlinear boundaries.
* **Visual + interactive:** Watch your neural net *draw* the boundary as it learns.
* **Easy to hack:** Modify the number of neurons, activations, layers, learning rates — and instantly see what changes.

## 🚀 Features

* One hidden layer, ReLU activations
* Output sigmoid + cross-entropy loss
* Manual forward and backward propagation
* Fully vectorized NumPy implementation
* Decision boundary visualization
* Clear, well-commented code

## 🎨 Example decision boundary

*(You can insert a generated plot here, e.g., after training 2000 epochs)*
![Decision boundary example](./boundary_example.png)

## 🔧 Requirements

```bash
pip install numpy matplotlib scikit-learn
```

## 📂 Project structure

```
TinyNetLab/
 ├── tinynet.py        # Main code: network definition, train loop, plotting
 ├── README.md         # This file
 └── boundary_example.png # Example plot (optional)
```

## 🌱 Get started

```bash
python tinynet.py
```

Or import parts of the code into your own experiments!

## 📌 License

MIT License.
