
```markdown
## Prerequisites

- Python >= 3.6 (recommended: >= 3.9)
- Miniconda or Anaconda is recommended for creating a virtual Python environment.

## Dependencies

The model is built on top of PyTorch. To install PyTorch, follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) suitable for your system. For example:

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

After ensuring that PyTorch is installed correctly, proceed to install the other required dependencies by running:

```bash
pip install -r requirements.txt
```

The `requirements.txt` should include the following dependencies:

```
easy-torch==1.3.2
easydict==1.10
pandas==1.3.5
packaging==23.1
setuptools==59.5.0
scipy==1.7.3
tables==3.7.0
sympy==1.10.1
setproctitle==1.3.2
scikit-learn==1.0.2
```

This setup will ensure that all necessary libraries are installed and compatible with your environment.

## Installation Steps

1. **Set up a virtual environment (recommended):**

   Using Anaconda or Miniconda:

   ```bash
   conda create -n your_env_name python=3.9
   conda activate your_env_name
   ```

   Or using `virtualenv`:

   ```bash
   python -m venv your_env_name
   source your_env_name/bin/activate  # On Windows, use: your_env_name\Scripts\activate
   ```

2. **Install PyTorch:**

   Follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) or use the command provided above.

3. **Install other dependencies:**

   Once inside your environment, run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify the installation:**

   You can check if everything is correctly installed by running:

   ```bash
   python -c "import torch; import pandas; print('Setup complete')"
   ```

This should print `Setup complete` without any errors if everything is installed properly.

Now, you're ready to proceed with your model development and experiments!
