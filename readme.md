# TicketSwap Interview Project

<div align="center">

![](images/ts_logo.png)

</div>

This project contains 4 problems. Your code doesn't need to run, but it should be clear how it would be run. Please use Python syntax. The questions are contained in the `questions.ipynb` file located in this directory. You'll need to load up the notebook to see the questions. It's assumed you have python installed on your system. Information about submission is at the end of this document and included in the email you received with the link to this repository.

> [!TIP]
> No prior experience in the domain is necessary. The purpose of the questions is to evaluate your problem-solving abilities and your capacity to research and adapt. You are welcome to use any resources at your disposal to assist in answering the questions. However, you should not seek help from others, submit code that is not your own creation, or directly copy and paste content from the internet or GPT. Should you have any inquiries about the project, please do not hesitate to contact us.

## Project Structure
```
│
├── src                       <- Source folder
│   ├── etl.py                <- ETL functions
│   └── gradientboost.py      <- Model object
│
├── cli.py                    <- The starting point for the fraud model.
│
├── questions.ipynb           <- Main notebook with questions. Place your answers here.
│
└── readme.md                 <- This file.
```
You can ignore the other files as they're not relevant to the project.

## Installing JupyterLab
JupyterLab can be installed using pip, Python’s package manager. To install JupyterLab, open your terminal and run:

```bash
pip install jupyterlab
```

or if you are using conda, you can install JupyterLab with:

```bash
conda install -c conda-forge jupyterlab
```
## Cloning the repository
To clone the repository, open your terminal, navigate to the directory where you want to store the project, and run:

```bash
git clone git@github.com:TicketSwap/MLE_Interview.git
```

or if you are using HTTPS:

```bash
git clone https://github.com/TicketSwap/MLE_Interview.git
```
Inside the directory a folder called `MLE_Interview` will be created. Navigate to this folder to find the `questions.ipynb`. Next you can start Jupyter Lab or run the notebook from VSCode.


## Running JupyterLab
First make sure you are in the directory where the `questions.ipynb` file is located. You can start JupyterLab by running the following command in your terminal:

```bash
jupyter lab
```

## OPTIONAL: Running Notebook from VSCode
If you are using VSCode, you can install the extension `Jupyter` by Microsoft. Once installed, you can open the `questions.ipynb` file and run the notebook from there.

## Submitting your work
Your work should be submitted by attaching the `questions.ipynb` file to the email linked with this repository.

> [!CAUTION]  
> Ensure the file is saved before sending it. Please do not include any additional files or compress the file using rar, zip, or tar formats.
