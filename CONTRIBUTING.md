
# Contributing to Sudio

## Table of Contents

- [Bug Reports and Enhancement Requests](#bug-reports-and-enhancement-requests)
- [Finding an Issue to Contribute To](#finding-an-issue-to-contribute-to)
- [Submitting a Pull Request](#submitting-a-pull-request)
  - [Version Control, Git, and GitHub](#version-control-git-and-github)
  - [Creating a Fork of Sudio](#creating-a-fork-of-sudio)
  - [Creating a Feature Branch](#creating-a-feature-branch)
  - [Making Code Changes](#making-code-changes)
  - [Pushing Your Changes](#pushing-your-changes)
  - [Making a Pull Request](#making-a-pull-request)
  - [Updating Your Pull Request](#updating-your-pull-request)
  - [Updating the Development Environment](#updating-the-development-environment)
- [Tips for a Successful Pull Request](#tips-for-a-successful-pull-request)

---


## Bug Reports and Enhancement Requests

Bug reports and enhancement requests are essential for making Sudio more stable and feature-rich. When reporting an issue or request, please fill out the issue form fully on our [GitHub issues page](https://github.com/MrZahaki/sudio/issues/new/choose) to ensure clarity for others and the core development team. The issue will be open for comments and ideas from the community.

## Finding an Issue to Contribute To

If you are new to Sudio or open-source development, we recommend browsing the [GitHub issues tab](https://github.com/MrZahaki/sudio/issues) to find issues that interest you. Issues labeled [`Docs`](https://github.com/MrZahaki/sudio/issues?q=is%3Aopen+sort%3Aupdated-desc+label%3ADocs+no%3Aassignee) and [`good first issue`](https://github.com/MrZahaki/sudio/issues?q=is%3Aopen+sort%3Aupdated-desc+label%3A%22good+first+issue%22+no%3Aassignee) are typically suitable for new contributors.

Once you've found an interesting issue, assign it to yourself by commenting `take` on the issue. If you can't continue working on the issue, please unassign it so others can pick it up.

## Submitting a Pull Request

### Version Control, Git, and GitHub

Sudio is hosted on [GitHub](https://github.com/MrZahaki/sudio). To contribute, you need to sign up for a [free GitHub account](https://github.com/signup/free) and install [Git](https://git-scm.com/). Here are some resources for learning Git:
- [Git documentation](https://git-scm.com/doc)
- [GitHub documentation for contributing to projects](https://docs.github.com/en/get-started/quickstart/contributing-to-projects)

### Creating a Fork of Sudio

Fork the [Sudio repository](https://github.com/MrZahaki/sudio) to your GitHub account. Clone your fork to your local machine and add the upstream repository:

```shell
git clone https://github.com/your-username/sudio.git sudio-yourname
cd sudio-yourname
git remote add upstream https://github.com/MrZahaki/sudio.git
git fetch upstream
```

### Creating a Feature Branch

Ensure your local `main` branch is up-to-date with the upstream repository. Then create a feature branch:

```shell
git checkout main
git pull upstream main --ff-only
git checkout -b shiny-new-feature
```

### Making Code Changes

Follow the [development environment guidelines](#) to set up your environment. Make your changes and add the modified files:

```shell
git status
git add path/to/changed-file.py
```

Commit your changes with an explanatory message:

```shell
git commit -m "your commit message goes here"
```

### Pushing Your Changes

Push your feature branch to your forked repository:

```shell
git push origin shiny-new-feature
```

### Making a Pull Request

Navigate to your repository on GitHub, click the `Compare & pull request` button, review your changes, write a descriptive title and description, and then click `Send Pull Request`.

### Updating Your Pull Request

To update your pull request with changes from the `main` branch, fetch and merge the upstream changes:

```shell
git checkout shiny-new-feature
git fetch upstream
git merge upstream/main
git push origin shiny-new-feature
```

### Updating the Development Environment

Periodically update your local `main` branch with updates from the Sudio `main` branch and update your development environment:

```shell
git checkout main
git fetch upstream
git merge upstream/main
# Activate your virtual environment and update packages
python -m pip install --upgrade -r requirements-dev.txt
```

## Tips for a Successful Pull Request

- **Reference an open issue** for non-trivial changes.
- **Ensure you have appropriate tests** for your changes.
- **Keep your pull requests simple** and focused on one issue.
- **Ensure that CI checks are green** before requesting a review.
- **Regularly update your pull request** with the latest changes from `main`.

---

By contributing to Sudio, you are helping to create an open-source, easy-to-use digital audio processing library that blends real-time and non-real-time functionalities for diverse applications. Your support and contributions are invaluable.
