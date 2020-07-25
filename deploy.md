# mercs

Deployment information.

1 Development workflow
=======================

1.1 Clone the repository
-----------------

```bash
git clone https://github.com/systemallica/mercs
```

1.2 Create virtual environment and install the dependencies
----------------------

The project uses [Poetry](https://python-poetry.org) as dependency manager, which also creates a `virtualenv` automatically on install. So all you need to do is `poetry install`. 

You can now navigate to the `/tests` folder and write your own experiments.

1.3 Continuous Integration
---------------

Do not allow yourself to proceed without at least accumulating some tests. Therefore, we've set out to integrate [CI](https://en.wikipedia.org/wiki/Continuous_integration) right from the start.

CI has been set up with a [GitHub Action](https://github.com/features/actions) and will be triggered on every commit to master.

2 Distribution workflow
========================

This part is about publishing your project on PyPi.

2.1 PyPi
--------

As we have configured our project with Poetry, deploying to PyPi is as easy as:
1. `poetry build`
2. `poetry publish`
The terminal will ask for your PyPi username and password, and then publish the package.

2.2 Docs
--------
Every good open source project at least contains a bit of documentation. A part of this documentation is generated from decent docstrings you wrote together with your code.

### Tools

We will use [Mkdocs](https://www.mkdocs.org/), with its [material](https://squidfunk.github.io/mkdocs-material/) theme. This generates very nice webpages and is -in my humble opinion- a bit more modern than Sphinx (which is also good!).

The main upside of `mkdocs` is the fact that its source files are [markdown](https://en.wikipedia.org/wiki/Markdown), which is the most basic formatted text format there is. Readmes and even this deployment document are written in markdown itself. In that sense, we gain consistency, all the stuff that we want to communicate is written in markdown: 

- readme's in the repo
- text cells in jupyter notebooks
- source files for the documentation site

Which means that we can write everything once, and link it together. All the formats are the same, hence trivially compatible.

### Procedure

The project already contains the [mkdocs.yml](mkdocs.yml) file, which is -unsurprisingly- the configuration file for your mkdocs project. Using this, you can focus on content. Alongside this configuration file, we also included a demo page; [index.md](./docs/index.md), which is the home page of the documentation website. 

For a test drive, you need to use some commands. To build your website (i.e., generate html starting from your markdown sources), you do

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

and your site should be online at; [https://xxxx.github.io/mercs/](https://xxxx.github.io/mercs/). 

What happens under the hood is that a `mkdocs build` is executed, and then the resulting `site` directory is pushed to the `gh pages` branch in your repository. From that point on, github takes care of the rest.