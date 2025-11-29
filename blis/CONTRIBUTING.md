## How to contribute to BLIS

First, we want to thank you for your interest in contributing to BLIS! Please read through the following guidelines to help you better understand how to best contribute your potential bug report, bugfix, feature, etc.

#### **Did you find a bug?**

* **Check if the bug has already been reported** by searching on GitHub under [Issues](https://github.com/flame/blis/issues).

* If you can't find an open issue addressing the problem, please feel free to [open a new one](https://github.com/flame/blis/issues/new). Some things to keep in mind as you create your issue:
   * Be sure to include a **meaningful title**. Aim for a title that is neither overly general nor overly specific.
   * Putting some time into writing a **clear description** will help us understand your bug and how you found it.
   * You are welcome to include the BLIS version number (e.g. 0.3.2-15) if you wish, but please supplement it with the **actual git commit number** corresponding to the code that exhibits your reported behavior (the first seven or eight hex digits is fine).
   * Unless you are confident that it's not relevant, it's usually recommended that you **tell us how you configured BLIS** and **about your environment in general**. Your hardware microarchitecture, OS, compiler (including version), `configure` options, configuration target are all good examples of things to you may wish to include. If the bug involves elements of the build system such as bash or python functionality, please include those versions numbers, too.
   * If your bug involves behavior observed after linking to BLIS and running an application, please provide a minimally illustrative **code sample** that developers can run to (hopefully) reproduce the error or other concerning behavior.

#### **Did you write a patch that fixes a bug?**

If so, great, and thanks for your efforts! Please submit a new GitHub [pull request](https://github.com/flame/blis/pulls) with the patch.

* Ensure the PR description clearly describes the problem and solution. Include any relevant issue numbers, if applicable.

* Please limit your PR to addressing one issue at a time. For example, if you are fixing a bug and in the process you find a second, unrelated bug, please open a separate PR for the second bug (or, if the bugfix to the second bug is not obvious, you can simply open an [issue](https://github.com/flame/blis/issues/new) for the second bug).

* Before submitting new code, please read the [coding conventions](https://github.com/flame/blis/wiki/CodingConventions) guide to learn more about our preferred coding conventions. (It's unlikely that we will turn away your contributed code due to mismatched coding styles, but it will be **highly** appreciated by project maintainers since it will save them the time of digressing from their work--whether now or later--to reformat your code.)

#### **Did you fix whitespace or reformat code?**

Unlike some other projects, if you find code that does not abide by the project's [coding conventions](https://github.com/flame/blis/wiki/CodingConventions) and you would like to bring that code up to our standards, we will be happy to accept your contribution. Please note in the commit log the fixing of whitespace, formatting, etc. as applicable.

If you are making a more substantial contribution and in the vicinity of the affected code (i.e., within the same file) you stumble upon other code that works but could use some trivial changes or reformatting, you may combine the latter into the commit for the former. Just note in your commit log that you also fixed whitespace or applied reformatting.

#### **Do you intend to add a new feature or change an existing one?**

That's fine, we are interested to hear your ideas!

* You may wish to introduce your idea by opening an [issue](https://github.com/flame/blis/issues/new) to describe your new feature, or how an existing feature is not sufficiently general-purpose. This allows you the chance to open a dialogue with other developers, who may provide you with useful feedback.

* Before submitting new code, please read the [coding conventions](https://github.com/flame/blis/wiki/CodingConventions) guide to learn more about our preferred coding conventions. (See comments above regarding mismatched coding styles.)

#### **Do you have questions about the source code?**

* Feel free to ask questions on the [blis-devel mailing list](https://groups.google.com/d/forum/blis-devel). You'll have to join to post, but don't be shy! Most of the interesting discussion (outside of GitHub) happens on blis-devel. We also have a [blis-discuss mailing list](https://groups.google.com/d/forum/blis-discuss), but it is not the preferred venue for discussion these days.

Here at the BLIS project, we :heart: our community. :) Thanks for helping to make BLIS better!

Field

