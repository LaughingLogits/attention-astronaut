This folder contains the code to run the clustering pipeline used in the paper (Section 5).

We strongly recommend running this code in the [cuml docker image](https://docs.rapids.ai/install#selector) provided by rapidsai.

Pip packages are provided in the requirements.txt file.

We provide a database that saves all attention patterns generated by each head in a separate h5py dataset.
You can query the database by changing the selectors (marked in the notebook) from "*" to a given set of keys you want to select.


run.ipynb contains a notebook that runs the entire pipeline from start to finish. It is important to note that running with the settings from the paper requires 12.5TB of storage, the bottleneck is also disk write speed, and GPU -> disk transfer rates. This is mainly due to saving all attention patterns, so we can evaluate the clusters. For future versions, we will provide an option to only save the encodings.

dimension_selection.ipynb contains a notebook that generates the trustworthiness scores for multiple target dimensions as mentioned in Section 5. This takes more than a week to run on an A100 GPU (80GB VRAM), we don't know how long it will take to run without the cuml acceleration so we strongly advise using the cuml docker image or library.
