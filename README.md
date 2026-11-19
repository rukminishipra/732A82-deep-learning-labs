# Deep Learning labs (732A82, Linköping University)

Lab work for the course 732A82 Deep Learning in the Master's Programme in Statistics and Machine Learning at Linköping University, spring 2026. All five labs are graded Pass.

Author of the solutions: Rukmini Shipra Sundaraneedi.

The notebooks are the course's lab templates, written by the course staff. My part is the code in the marked places, the experiments and the written answers to the questions in each notebook.

## Labs

| Folder | Lab | What I implemented and studied |
|---|---|---|
| `1_Back_propagation/` | Back-propagation | Forward pass, gradients and training of single-layer and multi-layer networks in NumPy, tuned on four data sets |
| `2_DNN/` | Deep neural networks | Classification of benign and malicious network traffic (Mirai data set) with Keras: class weights, batch normalization, dropout, activation functions and optimizers, hyperparameter tuning with Ray Tune, uncertainty quantification, and a regression network |
| `3_CNN/` | Convolutional networks | Convolutions by hand, then a 2D CNN for CIFAR-10 image classification with batch normalization, dropout and other regularization, and a carbon-footprint measurement of the training |
| `4_RNN/` | Recurrent networks | Swedish to English machine translation with a GRU encoder-decoder, extended with an attention mechanism, evaluated with BLEU, with attention visualizations |
| `5_Transformers/` | Transformers | Scaled dot-product and multi-head attention, a Vision Transformer for CIFAR-10, attention visualization, carbon footprint compared with the CNN, and a text summarization example with a pretrained model |

Each notebook is saved with its outputs, so the results and figures can be read directly on GitHub.

## Data

The data sets supplied by the course (Mirai network traffic, the translation corpus and the back-propagation data) are not included. CIFAR-10 is downloaded by the notebooks.

## Technologies

Python, NumPy, TensorFlow and Keras, Ray Tune, scikit-learn, Hugging Face Transformers, PyTorch (for the pretrained models), sacreBLEU, CarbonTracker, Matplotlib, Jupyter.
