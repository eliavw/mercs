# Project Structure

Layout for machine learning research projects.

```
.
+-- cli
    | Command line interface
+-- docs
    | Documentation
+-- note
    | Notebooks
+-- prod
    | Products = all kinds of outputs. Typically you put these in .gitignore
    +-- results
    +-- visuals
+-- resc
    | Rescources, usually config and data
    +-- config
    +-- data
        | Data typically goes through steps, keep track of them
        +--step-01
        +--step-02
+-- src
    +-- mercs
+-- test
|
+-- .gitignore
+-- environment.yml
+-- deploy.md
+-- readme.md
```
