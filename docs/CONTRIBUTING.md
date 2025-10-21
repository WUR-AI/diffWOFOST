#

We welcome any kind of contribution to our software, from simple comment or
question to a full fledged [pull
request](https://help.github.com/articles/about-pull-requests/). Please read and
follow our [Code of Conduct](CODE_OF_CONDUCT.md).

A contribution can be one of the following cases:

1. you have a question;
1. you think you may have found a bug (including unexpected behavior);
1. you want to make some kind of change to the code base (e.g. to fix a bug, to add a new feature, to update documentation);
1. you want to make a new release of the code base.

The sections below outline the steps in each case.

## You have a question

1. use the search functionality [in
   issues](https://github.com/WUR-AI/diffwofost/issues) to see if someone
   already filed the same issue;
2. if your issue search did not yield any relevant results, make a new issue;
3. apply the "Question" label; apply other labels when relevant.

## You think you may have found a bug

1. use the search functionality [in
   issues](https://github.com/WUR-AI/diffwofost/issues) to see if someone
   already filed the same issue;
2. if your issue search did not yield any relevant results, make a new issue,
   making sure to provide enough information to the rest of the community to
   understand the cause and context of the problem.

## You want to make some kind of change to the code base

1. (**important**) announce your plan to the rest of the community *before you
   start working*. This announcement should be in the form of a (new) issue;
2. (**important**) wait until some kind of consensus is reached about your idea being a good idea;
3. follow the instruction in [developer_guide.md](developer_guide.md).

In case you feel like you've made a valuable contribution, but you don't know
how to write or run tests for it, or how to generate the documentation: don't
let this discourage you from making the pull request; we can help you! Just go
ahead and submit an issue and ask your questions.

## You want to make a new release of the code base

To create a release you need write permission on the repository.

1. Check the author list in `CITATION.cff`in
   the root of the repository.
2. Bump the version. The version can be manually changed in `pyproject.toml` in
   the root of the repository. Follow [Semantic Versioning](https://semver.org/)
   principles.

3. Go to the [GitHub release
   page](https://github.com/WUR-AI/diffwofost/releases). Press draft a new
   release button. Fill version, title and description field. Press the Publish
   Release button

4. This software automatically publish to PyPI using a release or publish
   workflow. Wait until [PyPi publish
   workflow](https://github.com/WUR-AI/diffwofost/actions/workflows/python-publish.yml)
   has completed and verify new release is on
   [PyPi](https://pypi.org/project/matchms/#history)
