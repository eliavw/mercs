# mercs

Deployment information.

1 Development workflows
=======================

1.1 Start project
-----------------

Using the power of [cookiecutter](https://cookiecutter.readthedocs.io/en/latest/), this single command provides a pretty solid starting point for any new project.

```bash
cookiecutter gh:eliavw/cookiecutter-datascience
```

1.2 git
-------

Version control goes without saying. For the local repository, do;

```bash
git init
git add .
git commit -m "First commit"
```

For the remote repository, do;

```bash
git remote add origin git@github.com:eliavw/mercs.git
git remote -v
git push origin master
```

And that's it for git.

### One-liners

We can summarize the above procedure in two one-liners, should you really care about doing this fast.

```bash
git init; git add .; git commit -m "First commit";
```

and

```bash
git remote add origin git@github.com:eliavw/mercs.git; git remote -v; git push origin master
```

1.3 Conda Environments
----------------------

### Introduction

This cookiecutter is set up for optimal use with [conda](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf), for **local dependency managment**. The takeaway is this; _for local dependency managment, we rely on conda and nothing else._

Note that this has nothing to do with **remote dependency managment**. This is what you need to take care of when preparing a _release_ of your code which goes via [PyPi](https://pypi.org/) or alternatives. We treat that as an independent problem. Mixing remote and local dependency managment tends to add complexity instead of removing it.

### Workflow

To create our default environment, do:

```bash
conda env create -f dependencies-deploy.yaml -n mercs
```

To additionally add the packages which are relevant for the development phase, do:

```bash
conda activate mercs
conda env update -n mercs -f dependencies-develop.yaml
```

### Jupyterlab

To add your isolated python installation (i.e., the one in your new conda environment) to the list of "kernels" found by Jupyter, execute the following.

```bash
conda activate mercs
python -m ipykernel install --user --name mercs --display-name "mercs"
```

1.4 Local Installation
----------------------

One fundamental assumption is the following; 

> All code in this repository belongs to one of the two following categories: source code or standalone scripts.

- Code in [src](./src) is considered source code. It composes a Python package.
- Code in [scripts](./scripts) or [note](./note) acts as standalone scripts.

This means that even our own code has to be installed before we are able to use it. This seems a bit tedious but has some important advantages too:

1. Our scripts will consider our own algorithm(s) and external competitors both as packages to be imported. Putting these on equal footing enforces code quality (e.g., modularity, API-design, ...) and reproducibility.
2. If we build it like a package from the start on our local machine, the transition to an actual publishable package will be a lot smoother afterwards. In other words, we will try to get it right from the start.

### Installation instructions

To install, activate the conda environment and execute this line of code.

```bash
python setup.py develop # or `install`
```

What is the difference between `develop` or `install`? When you install the package with the `develop` flag, symlinks are created from yoru code to the python installation. That means that every time you change something in your codebase, the installed package in your python environment will also change. Typically, this is what you'd want: to see your changes reflected immediately.

The install option just copies your code as it is at time of installation and install the package in the python environment. This mimics what a third party would do.


1.5 CI (Travis)
---------------

Do not allow yourself to proceed without at least accumulating some tests. Therefore, we've set out to intigrate [CI](https://en.wikipedia.org/wiki/Continuous_integration) (i.e. [Travis](https://travis-ci.com)) right from the start.

Follow these steps:

1. Go to the [Travis](https://travis-ci.com/eliavw/mercs) page of this repo.
2. See if it ran.

**Note:** The tests depend on our **local dependency managment**. Why? Because we have full control of the Travis servers running our tests. Therefore, we can simply treat it as a computer we control. We only need to fall back on remote dependency managment if other people need to get our code up and running, without our intervention.



2 Distribution workflows
========================

This part is about publishing your project on PyPi.

2.1 Pypi
--------

Make your project publicly available on the Python Package Index, [PyPi](https://pypi.org/). To achieve this, we need **remote dependency managment**, since you want your software to run without forcing the users to recreate your conda environments. All dependencies have to be managed, automatically, during installation. To make this work, we need to do some extra work.

We follow the steps as outlined in the most basic (and official) [PyPi tutorial](https://packaging.python.org/tutorials/packaging-projects/).

### Generate distribution archives

Generate distribution packages for the package. These are archives that are uploaded to the Package Index and can be installed by pip.

```bash
python setup.py sdist bdist_wheel
```

After this, your package can be uploaded to the python package index. To see if it works on PyPi test server, do

```bash
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

and this will prompt some questions, but your package will end up in the index.

To make your package end up in the actual PyPi, the procedure is almost as simple, do


```bash
python -m twine upload --repository-url https://pypi.org/legacy/ dist/*
```

2.2 Docs
--------
Every good open source project at least consists of a bit of documentation. A part of this documentation is generated from decent docstrings you wrote together with your code.

### Tools

We will use [Mkdocs](https://www.mkdocs.org/), with its [material](https://squidfunk.github.io/mkdocs-material/) theme. This generates very nice webpages and is -in my humble opinion- a bit more modern than Sphinx (which is also good!).

The main upside of `mkdocs` is the fact that its source files are [markdown](https://en.wikipedia.org/wiki/Markdown), which is the most basic formatted text format there is. Readmes and even this deployment document are written in markdown itself. In that sense, we gain consistency, all the stuff that we want to communicate is written in markdown: 

- readme's in the repo
- text cells in jupyter notebooks
- source files for the documentation site

Which means that we can write everything once, and link it together. All the formats are the same, hence trivially compatible.

### Procedure

The cookiecutter already contains the [mkdocs.yml](mkdocs.yml) file, which is -unsurprisingly- the configuration file for your mkdocs project. Using this cookiecutter, you can focus on content. Alongside this configuration file, we also included a demo page; [index.md](./docs/index.md), which is the home page of the documentation website. 

For a test drive, you need to know some commands. To build your website (i.e., generate html starting from your markdown sources), you do

```bash
mkdocs build
```

To preview your website locally, you do

```bash
mkdocs serve
```

and surf to [localhost:8000](http://localhost:8000). Also note that this server will refresh whenever you alter something on disk (which is nice!), and hence does the build command automatically.

### Hosting on Github

Now, the last challenge is to make this website available over the internet. Luckily, mkdocs makes this [extremely easy](https://www.mkdocs.org/user-guide/deploying-your-docs/) when you want to host on [github pages](https://pages.github.com/)

```bash
mkdocs gh-deploy
```

and your site should be online at; [https://eliavw.github.io/mercs/](https://eliavw.github.io/mercs/). 

What happens under the hood is that a `mkdocs build` is executed, and then the resulting `site` directory is pushed to the `gh pages` branch in your repository. From that point on, github takes care of the rest.


### Repository Description

Often overlooked, but this is right on top of your repository and hence the absolute perfect place to link to your project website. Hence, a cookiecutter-generated sentence to put there would be;

> MERCS, cf. https://eliavw.github.io/mercs
