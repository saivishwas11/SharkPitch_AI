# Athena_Assignment

## Setup Conda Environment

1. Create the conda environment from the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the conda environment:
   ```bash
   conda activate athena_assignment
   ```

3. To deactivate the environment when you're done:
   ```bash
   conda deactivate
   ```

4. To update the environment if you modify `environment.yml`:
   ```bash
   conda env update -f environment.yml --prune
   ```

5. To remove the environment:
   ```bash
   conda env remove -n athena_assignment
   ```